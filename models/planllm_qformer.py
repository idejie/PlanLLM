import logging

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from .blip2.vit import build_vit
from .blip2.builder import build_qformer
from .criterions import VTC_VTM_Loss, get_sim,FocalLoss
from timm.models.layers import trunc_normal_
from models.schema.state_encoder import StateEncoder
from models.schema.state_decoder import StateDecoder
from models.schema.action_decoder import ActionDecoder
from models.utils import viterbi_path

logger = logging.getLogger(__name__)


class PlanLLM_qformer(nn.Module):
    """
    PlanLLM_qformer model.
    """
    def __init__(self, config, tokenizer):
        super(PlanLLM_qformer, self).__init__()
        # schema settings
        args=config
        self.att_heads = args.attn_heads
        self.mlp_ratio = args.mlp_ratio
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.uncertainty = args.uncertain
        self.dataset = args.dataset
        self.time_horz = args.max_traj_len
        self.embed_dim = args.embed_dim
        vis_input_dim=args.img_input_dim
        num_classes=args.num_action
        num_tasks=args.num_tasks
        lang_input_dim=args.text_input_dim
        self.state_task_pred = not args.no_state_task
        
        self.state_encoder = StateEncoder(
            vis_input_dim,
            lang_input_dim,
            self.embed_dim, 
            dropout = 0.4
        )
        state_interact_layer = nn.TransformerEncoderLayer(d_model=vis_input_dim,dim_feedforward=vis_input_dim, nhead=8,batch_first=True,dropout=self.dropout)
        
        self.state_interaction = nn.TransformerEncoder(state_interact_layer, num_layers=2)

        self.state_decoder = StateDecoder(
            embed_dim = self.embed_dim, 
            time_horz = self.time_horz, 
            att_heads = self.att_heads,
            mlp_ratio = self.mlp_ratio,
            num_layers = self.num_layers,
            dropout = self.dropout, 
            num_tasks = num_tasks,
            uncertainty = self.uncertainty,
            dataset = self.dataset,
            use_task_pred = self.state_task_pred
        )

        self.action_decoder = ActionDecoder(
            embed_dim = self.embed_dim,
            time_horz = self.time_horz,
            att_heads = self.att_heads,
            mlp_ratio = self.mlp_ratio,
            num_layers = self.num_layers,
            dropout = self.dropout, 
            num_classes = num_classes,
            img_input_dim = vis_input_dim,
            num_tasks = num_tasks,
            uncertainty = self.uncertainty
        )

        self.task_decoder = nn.Sequential(
            nn.Linear(self.embed_dim*2, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, num_tasks)
        )

        self.dropout = nn.Dropout(self.dropout)

        self.loss_action = nn.CrossEntropyLoss()
        self.loss_action_f = FocalLoss(gamma=1)
        self.loss_state = nn.CrossEntropyLoss()
        self.loss_state_pred = nn.MSELoss()
        self.loss_task = nn.CrossEntropyLoss()
        
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        
        self.config = config

        # self.vision_width = config.model.vision_encoder.d_model
        # self.text_width = config.model.text_encoder.d_model
        # self.embed_dim = config.model.embed_dim
        # self.agg_method = config.model.get("agg_method", "mean")

        if self.config.criterion.get('vtm_add_text_cls', False):
            logger.info('Use text [CLS] for matching: ADD')
        elif self.config.criterion.get('vtm_cat_text_cls', False):
            logger.info('Use text [CLS] for matching: CAT')

        # create modules. seperate vision_encoder and vision_temp_embed as
        # we wish to freeze vision_encoder

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)

        self.qformer, self.query_tokens = build_qformer(
            config.max_traj_len+1, self.embed_dim,
            config.model.get('qformer_hidden_dropout_prob', 0.1),
            config.model.get('qformer_attention_probs_dropout_prob', 0.1),
            config.model.get('drop_path_rate', 0.1),
        )
        self.qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.qformer.state_dict()
        for name, param in self.qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.qformer.config.hidden_size, self.embed_dim)
        self.text_proj = nn.Linear(self.qformer.config.hidden_size, self.embed_dim)
        if self.config.criterion.get('vtm_cat_text_cls', False):
            self.itm_head = nn.Linear(2 * self.qformer.config.hidden_size, 2)
        else:
            self.itm_head = nn.Linear(self.qformer.config.hidden_size, 2)

        # criterions
        self.loss_weight = config.criterion.loss_weight
        self.criterion_vtc_vtm = VTC_VTM_Loss(config.criterion.vtm_hard_neg)

        # init
        # trunc_normal_(self.vision_temp_embed)
        if self.config.model.vit_add_ln:
            self.vision_layernorm = nn.LayerNorm(self.embed_dim, eps=1e-12)
        else:
            self.vision_layernorm = nn.Identity()
        trunc_normal_(self.query_tokens)
        # self.vision_temp_embed = nn.Parameter(
        #     torch.zeros(1, args.max_traj_len, 1, self.embed_dim)
        # )
        self.agg_method = config.model.get("agg_method", "mean")


        self.vision_proj.apply(self._init_weights)
        self.text_proj.apply(self._init_weights)
        self.itm_head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward_once(
            self, 
            visual_features, 
            state_prompt_features, 
            tasks,
            text,
            idx
        ):
        '''Forward pass

        This function calls the state encoder, state decoder, task decoder, and
        action decoder to get the outputs.

        Args:
            visual_features:        Visual observations of procedures.  [batch_size, time_horizon, 2, vis_input_dim]
            state_prompt_features:  Descriptions of before and after state of all actions.     [num_action, num_prompts, lang_input_dim]
            tasks:                  Ground truth tasks.      [batch_size]
        
        Returns:
            outputs:                Dictionary of outputs.
        '''

        batch_size,T,N,d = visual_features.shape
        # add some noise to the visual features
    
        # Step 1: state encoding
        # (bx2xd),  (bxTxd),   (bx2x n_actions), (n_action,2xn_desc,d)
        state_feat_encode, inter_state_feat_encode, state_logits, state_prompt_features = \
            self.state_encoder(visual_features, state_prompt_features)
        # Step 2: state interaction
        state_feat_interacted = self.state_interaction(state_feat_encode)
        # Step 3: Q-Former
        state_feat_encode = state_feat_interacted
        vision_embeds, vision_query_embeds, vision_past_key_values = self.encode_vision(
            state_feat_encode, return_key_values=True
        )
        # print('vision_embeds',vision_embeds.shape,'vision_query_embeds',vision_query_embeds.shape)
        text_embeds, pooled_text_embeds = self.encode_text(text)

        # obtain vision and text representations.
        vision_proj = self.vision_proj(vision_query_embeds)  # [B, T+1, C]
        # vision_proj+=state_feat_encode
        # state_feat_encode = vision_proj
        text_proj = self.text_proj(pooled_text_embeds)  # [B, C]

        # calculate loss

        ## VTC loss
        if self.loss_weight.vtc != 0:
            # sim_idx: (sim_v2t, idx), to save computation
            loss_vtc = self.criterion_vtc_vtm.vtc_loss(
                vision_proj,
                text_proj,
                idx,
                self.temp,
                all_gather=True,
                agg_method=self.agg_method,
            )
        else:
            loss_vtc = torch.tensor(0)

        ## VTM loss
        if self.loss_weight.vtm != 0:
            loss_vtm = self.vtm_loss(
                text,
                vision_embeds,
                vision_proj,
                text_proj,
                idx,
            )
        else:
            loss_vtm = torch.tensor(0)

        ## CAP loss
        if self.loss_weight.cap != 0:
            loss_cap = self.cap_loss(
                text,
                vision_past_key_values,
            )
        else:
            loss_cap = torch.tensor(0)

        qformer_loss= dict(
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_cap=loss_cap * self.loss_weight.cap,
        )

        # Step 4.1: task prediction
        state_feat_concat = state_feat_encode.reshape(batch_size, -1)
        task_pred = self.task_decoder(state_feat_concat)
        # print('state_feat_encode',state_feat_encode.shape)
        # Step 4.2: state decoding
        state_feat_decode = self.state_decoder(
            state_feat_encode,
            state_prompt_features, 
            task_pred,
            vision_proj
        )
        state_feat_input = state_feat_decode

        # Step 5: action decoding
        prompt_features = state_prompt_features
        action_logits,action_emb = self.action_decoder(
            state_feat_input, 
            prompt_features, 
            tasks if self.training is True else task_pred.argmax(-1)
        )
        

        # Collect outputs
        outputs = self.process_outputs(state_prompt_features, 
                                       state_logits, 
                                       state_feat_decode, 
                                       action_logits,
                                       action_emb,
                                       task_pred)
        return outputs,qformer_loss

    def forward(self, visual_features, 
            state_prompt_features, 
            actions, 
            tasks, 
            transition_matrix=None,
            text_input=None,
            idx=None,
            ):
        '''Forward pass and loss calculation

        This function calls forward_once() to get the outputs, and then calls 
        forward_loss() to get processed labels and losses.

        Args:
            visual_features:        Visual observations of procedures.  [batch_size, time_horizon, 2, vis_input_dim]
            state_prompt_features:  Descriptions of before and after state of all actions. [num_action, num_prompts, lang_input_dim]
            actions:                Ground truth actions.     [batch_size, time_horizon]
            tasks:                  Ground truth tasks.       [batch_size]

        Returns:
            outputs: Dictionary of outputs.
            labels:  Dictionary of labels.
            losses:  Dictionary of losses.
        '''
        outputs,qformer_loss = self.forward_once(
            visual_features, 
            state_prompt_features, 
            tasks,
            text_input,
            idx
        )
        batch_size, T = actions.shape
        action_logits = outputs["action"].reshape(batch_size, T, -1)
        action_logits = torch.softmax(action_logits, -1)

        # viterbi decoding
        if transition_matrix is not None:
            pred_viterbi = []
            for i in range(batch_size):
                viterbi_rst = viterbi_path(transition_matrix, action_logits[i].permute(1, 0).detach().cpu().numpy())
                pred_viterbi.append(torch.from_numpy(viterbi_rst))
            pred_viterbi = torch.stack(pred_viterbi).cuda()
        else:
            pred_viterbi = None
        outputs["pred_viterbi"] = pred_viterbi

        # loss calculation
        labels, losses = self.forward_loss(outputs, actions, tasks)
        losses.update(qformer_loss)
        return outputs, labels, losses

        

    def freeze_module(self, m):
        m = m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m

    def encode_vision(self, vision_embeds, test=False, return_key_values=False):
        """encode image / videos as features.

        Args:
            vision_embeds (torch.Tensor): The input vision_embeds.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,N,C].
            - vision_past_key_values (torch.Tensor): The past key values of vision transformer.

        """
        vision_embeds = self.vision_layernorm(vision_embeds)  # [B, T*L, C]

        vision_atts = torch.ones(
            vision_embeds.shape[:-1], dtype=torch.long, device=vision_embeds.device
        )

        query_tokens = self.query_tokens.expand(vision_embeds.size(0), -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            use_cache=True,
            return_dict=True,
        )

        vision_query_embeds = query_output.last_hidden_state
        vision_past_key_values = query_output.past_key_values

        if return_key_values:  # This is used in this model cap loss
            return (
                vision_embeds,
                vision_query_embeds,
                vision_past_key_values,
            )
        else:  # This is to match retrieval.py #19
            return vision_embeds, vision_query_embeds

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min_val, max_val)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name
        logger.info(f"Build vision_encoder: {encoder_name}")
        if "vit" in encoder_name:
            vision_encoder = build_vit(self.config.model)
        else:
            raise ValueError(f"not implemented: {encoder_name}")

        if self.config.model.vit_add_ln:
            vision_layernorm = nn.LayerNorm(self.vision_width, eps=1e-12)
        else:
            vision_layernorm = nn.Identity()

        vision_temp_embed = nn.Parameter(
            torch.zeros(1, self.config.num_frames, 1, self.vision_width)
        )

        return vision_encoder, vision_layernorm, vision_temp_embed

    @torch.no_grad()
    def get_mask(self, sim, idx=None, idx_all=None):
        """get mask for sim matrix."""
        if idx is not None:
            idx = idx.view(-1, 1)
            idx_all = idx_all.view(1, -1) if idx_all is not None else idx.T
            mask = torch.eq(idx, idx_all).to(sim.device)
        else:
            rank = torch.distributed.get_rank()
            mask = torch.zeros_like(sim)
            bs = sim.size(0)
            mask[:, rank * bs : (rank + 1) * bs].fill_diagonal_(1)

        return mask.bool()

    def vtm_loss(
        self,
        text,
        vision_embeds,
        vision_proj,  # [B, L, C]
        text_proj,  # [B, C]
        idx,
    ):
        """vtm loss."""
        with torch.no_grad():
            sim_v2t, sim_t2v = get_sim(
                vision_proj, text_proj, self.temp, agg_method=self.agg_method
            )
            weights_v2t = F.softmax(sim_v2t, dim=1) + 1e-4  # (N, N)
            weights_t2v = F.softmax(sim_t2v, dim=1) + 1e-4

            mask = self.get_mask(sim_v2t, idx=idx).bool()
            weights_v2t.masked_fill_(mask, 0)
            weights_t2v.masked_fill_(mask, 0)
            weights_v2t = torch.nan_to_num_(
                weights_v2t, nan=1e-2, posinf=1e-2, neginf=1e-2
            )
            weights_t2v = torch.nan_to_num_(
                weights_t2v, nan=1e-2, posinf=1e-2, neginf=1e-2
            )

        # select a negative image for each text
        if self.config.criterion.vtm_hard_neg:
            vision_neg_indices = torch.multinomial(weights_t2v, 1).squeeze()
            text_neg_indices = torch.multinomial(weights_v2t, 1).squeeze()
        else:
            vision_neg_indices = self.get_rand_indices(mask, 1).squeeze()
            text_neg_indices = self.get_rand_indices(mask, 1).squeeze()

        vision_embeds_neg = vision_embeds[vision_neg_indices]  # [B, L, C]
        text_ids_neg = text.input_ids[text_neg_indices]  # [B, L]
        text_atts_neg = text.attention_mask[text_neg_indices]  # [B, L]

        # Concat vision pos and neg
        vision_embeds_pos_neg = torch.cat(
            [vision_embeds, vision_embeds_neg, vision_embeds], dim=0
        )  # [3B, L, C]
        vision_atts_pos_neg = torch.ones(
            vision_embeds_pos_neg.size()[:-1],
            dtype=torch.long,
            device=vision_embeds.device,
        )  # [3B, L]

        # Concat text pos and neg
        text_ids_pos_neg = torch.cat(
            [text.input_ids, text.input_ids, text_ids_neg], dim=0
        )
        text_atts_pos_neg = torch.cat(
            [text.attention_mask, text.attention_mask, text_atts_neg], dim=0
        )

        vl_embeddings = self.vtm_embed(
            text_ids=text_ids_pos_neg,
            text_atts=text_atts_pos_neg,
            vision_embeds=vision_embeds_pos_neg,
            vision_atts=vision_atts_pos_neg,
        )
        logits = self.itm_head(vl_embeddings)

        bs = logits.size(0) // 3
        vtm_labels = logits.new_ones(logits.size(0), dtype=torch.long)
        vtm_labels[bs:] = 0
        loss_vtm = F.cross_entropy(logits, vtm_labels)

        return loss_vtm

    def cap_loss(
        self,
        text,
        past_key_values,
    ):
        """caption loss."""
        text_ids = text.input_ids.clone()
        text_ids[:, 0] = self.tokenizer.bos_token_id
        labels = text_ids.masked_fill(text_ids == self.tokenizer.pad_token_id, -100)

        query_atts = torch.ones(
            text_ids.size(0),
            self.query_tokens.size(1),
            dtype=torch.long,
            device=text_ids.device,
        )
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
        cap_output = self.qformer(
            text_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            return_dict=True,
            labels=labels,
        )

        cap_loss = cap_output.loss

        return cap_loss

    def get_text_encoder(self):
        return None

    @torch.jit.ignore
    def no_weight_decay(self):
        """Do not apply weight decay on these parameters"""
        return {
            "query_tokens",
            "temp",
            "vision_temp_embed",
            "vision_encoder.class_embedding",
            "vision_encoder.positional_embedding",
        }

    def vtm_embed(self, text_ids, text_atts, vision_embeds, vision_atts):
        """vtm embedding."""
        query_tokens = self.query_tokens.expand(text_ids.size(0), -1, -1)
        query_atts = torch.ones(
            query_tokens.size()[:-1], dtype=torch.long, device=vision_embeds.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
        )
        if self.config.criterion.get('vtm_add_text_cls', False):
            tmp_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1)].mean(1)
            vl_embeddings = tmp_embeddings + output_itm.last_hidden_state[:, query_tokens.size(1)]
        elif self.config.criterion.get('vtm_cat_text_cls', False):
            tmp_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1)].mean(1)
            vl_embeddings = torch.cat([tmp_embeddings, output_itm.last_hidden_state[:, query_tokens.size(1)]], dim=1)
        else:
            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1)].mean(1)
        return vl_embeddings
    def forward_loss(self, outputs, actions, tasks):
        '''Loss calculation

        This function calculates the losses for state encoding, state decoding,
        action decoding, and task decoding.

        Args:
            outputs:    Dictionary of outputs.
            actions:    Ground truth actions.
            tasks:      Ground truth tasks.
        
        Returns:
            labels:     Dictionary of processed labels.
            losses:     Dictionary of losses.
        '''

        _, num_action = outputs["action"].shape
        embed_dim = self.embed_dim

        labels = self.process_labels(outputs, actions, tasks)

        losses = {}
        losses["state_encode"] = self.loss_state(
            outputs["state_encode"].reshape(-1, num_action), 
            labels["state"]
        )
        losses["state_decode"] = self.loss_state_pred(
            outputs["state_decode"].reshape(-1, embed_dim), 
            labels["state_decode"]
        )
        # constrastive loss
        action_emb = outputs['action_emb']

        B,T,D = action_emb.shape
        action_emb = action_emb.reshape(B*T,D)
        action_emb = F.normalize(action_emb, p=2, dim=1)
        sim = torch.matmul(action_emb, action_emb.T)
        label = torch.arange(B*T).to(action_emb.device)
        
        # sim = sim - torch.eye(sim.size(0)).cuda()
        # sim = sim - torch.max(sim, dim=1, keepdim=True)[0]
        # sim = torch.exp(sim)
        # sim = sim / torch.sum(sim, dim=1, keepdim=True)
        # sim = sim.log()
        losses["action_emb"] = F.cross_entropy(sim, label)
        losses["action"] = self.loss_action_f(outputs["action"].reshape(-1, num_action), labels["action"]) # self.loss_action(outputs["action"].reshape(-1, num_action), labels["action"])
        losses["task"] = self.loss_task(outputs["task"], labels["task"])

        return labels, losses


    def process_outputs(
            self, 
            state_prompt_features,
            state_logits, 
            state_feat_decode,
            action_logits,
            action_emb,
            task_pred,
            pred_viterbi = None,
        ):
        '''Process outputs

        This function processes the outputs from the forward pass.

        Args:
            state_prompt_features: Descriptions of before and after state of all actions.   [num_action, num_prompts, embed_dim]
            state_logits:          Similarity between visual and linguistic features for start and end states.  [batch_size, 2, num_action]
            state_feat_decode:     Decoded features of all states.  [batch_size, time_horizon+1, embed_dim]
            action_logits:         Predicted action logits.  [batch_size, time_horizon, num_action]
            task_pred:             Predicted tasks.     [batch_size, num_tasks]
            pred_viterbi:          Predicted actions using viterbi decoding.    [batch_size, time_horizon]

        Returns:
            outputs: Dictionary of processed outputs.
        '''

        batch_size, _, num_action = state_logits.shape

        outputs = {}
        outputs["state_encode"] = state_logits.reshape(-1, num_action)
        outputs["state_decode"] = state_feat_decode[:, 1:-1, :]
        outputs["action"] = action_logits.reshape(-1, num_action)
        outputs['action_emb'] = action_emb
        outputs["task"] = task_pred
        outputs["state_prompt_features"] = state_prompt_features
        outputs["pred_viterbi"] = pred_viterbi

        return outputs


    def process_labels(self, outputs, actions, tasks):
        labels = {}
        labels["state"] = actions[:, [0, -1]].reshape(-1)
        labels["action"] = actions.reshape(-1)
        labels["task"] = tasks
        labels["state_decode"] = self.process_state_prompts(outputs["state_prompt_features"], actions)

        return labels


    def process_state_prompts(self, state_prompt_features, actions):
        '''Process state prompts

        This function combines the language descriptions after the current action with
        the descriptions before the next action to get consistent descriptions for 
        each state.

`       Args:
            state_prompt_features: Descriptions of before and after state of all actions.   [num_action, num_prompts, embed_dim]
            actions:               Ground truth actions.    [batch_size, time_horizon]
        
        Returns:
            target_state_decode:   Reduced descriptions for each state.     [batch_size*(time_horizon-1), embed_dim]
        '''

        batch_size, time_horizon = actions.shape
        num_action, num_desc, embed_dim = state_prompt_features.shape
        actions = actions.reshape(-1)   # [batch_size*time_horizon]
        state_prompt_features = state_prompt_features[actions, :, :].reshape(batch_size, time_horizon, num_desc, embed_dim)

        before_state_prompt_feat = torch.cat([state_prompt_features[:, :, :num_desc//2, :], 
                                              state_prompt_features[:, -1:, num_desc//2:, : ]], 1)  # [batch_size, time_horizon+1, 3, embed_dim]
        
        after_state_prompt_feat  = torch.cat([state_prompt_features[:, :1, :num_desc//2, :], 
                                              state_prompt_features[:, :, num_desc//2:, :]], 1)     # [batch_size, time_horizon+1, 3, embed_dim]
    
        target_state_decode = torch.cat([before_state_prompt_feat, after_state_prompt_feat], 2)     # [batch_size, time_horizon+1, 6, embed_dim]
        target_state_decode = target_state_decode.mean(2)[:, 1:-1, :].reshape(-1, embed_dim)        # [batch_size*(time_horizon-1), embed_dim]

        return target_state_decode.clone().detach()
