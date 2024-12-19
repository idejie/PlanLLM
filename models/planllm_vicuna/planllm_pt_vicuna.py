import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from ..blip2.blip2 import Blip2Base, disabled_train
from transformers import LlamaTokenizer, LlamaConfig
from models.blip2.vit import build_vit
from models.blip2.builder import build_qformer
from models.criterions import VTC_VTM_Loss, get_sim,FocalLoss
from timm.models.layers import trunc_normal_
from models.schema.state_encoder import StateEncoder
from models.schema.state_decoder import StateDecoder
from models.schema.action_decoder import ActionDecoder
from models.utils import viterbi_path
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.easydict import EasyDict
logger = logging.getLogger(__name__)





def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    


class PlanLLM_pt_vicuna(Blip2Base):
    """
    VideoChat2 model.
    """
    def __init__(self, args,config):
        super().__init__()
        # schema settings
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
        
        
        
        self.config = config
        self.vision_layernorm = nn.LayerNorm(self.embed_dim, eps=1e-12)
        # pretrained_path
        vit_blip_model_path = config.get("vit_blip_model_path", None)
        llama_model_path = config.get("llama_model_path")
        freeze_qformer = config.get("freeze_qformer", True)
        # vit
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        # qformer
        num_query_token = args.max_traj_len+1
        qformer_hidden_dropout_prob = config.get("qformer_hidden_dropout_prob", 0.1)
        qformer_attention_probs_dropout_prob = config.get("qformer_attention_probs_dropout_prob", 0.1)
        qformer_drop_path_rate = config.get("qformer_drop_path_rate", 0.1)
        extra_num_query_token = config.get("extra_num_query_token", 32)
        # prompt
        prompt_path = config.get("prompt_path", "")
        img_prompt_path = config.get("img_prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 32)
        end_sym = config.get("end_sym", '\n')
        # debug
        debug = config.get("debug", False)
        use_flash_attention = config.get("use_flash_attention", False)

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.low_resource = low_resource
        
        self.qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.embed_dim,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
        )
        self.vision_proj = nn.Linear(self.qformer.config.hidden_size, self.embed_dim)
        self.qformer.bert.embeddings.word_embeddings = None
        self.qformer.bert.embeddings.position_embeddings = None
        for layer in self.qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.qformer.cls = None

        if vit_blip_model_path:
            logger.info(f"Load ViT and QFormer from {vit_blip_model_path}")
            state_dict = torch.load(vit_blip_model_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # import pdb;pdb.set_trace()
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info('Loading ViT and Q-Former Done')
        
        self.extra_num_query_token = extra_num_query_token
        if extra_num_query_token > 0:
            logger.info(f"Add extra {extra_num_query_token} tokens in QFormer")
            self.extra_query_tokens = nn.Parameter(
                torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
            )

        

        if freeze_qformer:
            logger.info("freeze qformer")
            for _, param in self.qformer.named_parameters():
                param.requires_grad = False
            self.qformer = self.qformer.eval()
            self.qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        logger.info('Loading LLAMA')
        # problem: do we need to set truncation_side="left"?
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        if not self.llama_tokenizer.pad_token:
            logger.info("Set pad_token")
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if use_flash_attention:
            logger.info("Use flash attention")
            from ..blip2.modeling_llama_mem import LlamaForCausalLM
        else:
            from ..blip2.modeling_llama import LlamaForCausalLM
        if debug:
            logger.info("Debug mode, build small LLAMA")
            llama_config = LlamaConfig.from_pretrained(llama_model_path)
            llama_config.hidden_size = 512
            llama_config.intermediate_size = 2048
            llama_config.num_attention_heads = 8
            llama_config.num_hidden_layers = 12
            llama_config.torch_dtype = torch.float16
            self.llama_model = LlamaForCausalLM(llama_config)
        else:
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.float16,
                )
                print('llama_model loaded',llama_model_path)

        logger.info("freeze LLAMA")
        for _, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.action_proj = nn.Linear(
            self.llama_model.config.hidden_size, self.embed_dim
        )
        llm_decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=8,batch_first=True)
        self.action_llm_deocer = nn.TransformerDecoder(llm_decoder_layer, num_layers=2)
        self.classifier_llm = nn.Linear(self.embed_dim, num_classes)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            self.prompt_list = self.process_prompt(prompt_path, prompt_template)
        else:
            self.prompt_list = []
        if img_prompt_path:
            self.img_prompt_list = self.process_prompt(img_prompt_path, prompt_template)
        else:
            self.img_prompt_list = []

    def process_prompt(self, prompt_path, prompt_template):
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
        prompt_list = [prompt_template.format(p) for p in filted_prompts]
        logger.info(f'Load {len(prompt_list)} training prompts')
        # logger.info(f'Prompt: {prompt_list}')
        return prompt_list

    def vit_to_cpu(self):
        self.vision_layernorm.to("cpu")
        self.vision_layernorm.float()
        self.vision_encoder.to("cpu")
        self.vision_encoder.float()

    def encode_img(self, image_embeds):
        device = image_embeds.device
        if self.low_resource:
            self.vit_to_cpu()
            image_embeds = image_embeds.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            if self.extra_num_query_token > 0:
                query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            else:
                query_tokens = self.query_tokens
            # print('qformer',query_tokens.shape,self.extra_query_tokens.shape)
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            
            query_output = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            vision_query_embeds = query_output.last_hidden_state
            # vision_past_key_values = query_output.past_key_values
            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return vision_query_embeds[:,:self.query_tokens.shape[1],...],inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt, use_image=False):
        if prompt:
            batch_size = img_embeds.shape[0]
            if use_image:
                p_before, p_after = prompt.split('<ImageHere>')
            else:
                p_before, p_after = prompt.split('<VideoHere>')
            # import pdb;pdb.set_trace()
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, visual_features, 
            state_prompt_features, 
            actions, 
            tasks, 
            transition_matrix=None,
            text_input=None,
            idx=None, instruction=None):
        batch_size,T,N,d = visual_features.shape
        # add some noise to the visual features
    
        # Step 1: state encoding
        # (bx2xd),  (bxTxd),   (bx2x n_actions), (n_action,2xn_desc,d)
        state_feat_encode, inter_state_feat_encode, state_logits, state_prompt_features = \
            self.state_encoder(visual_features, state_prompt_features)
        # Step 2: state interaction
        state_feat_interacted = self.state_interaction(state_feat_encode)
        
        # Step 3: Q-former
        
        img_embeds_action,img_embeds_llm, atts_img = self.encode_img(state_feat_interacted)
        img_embeds_action = self.vision_proj(img_embeds_action)  # [B, T+1, C]
        # print('qformer',img_embeds_action.shape)
        # import pdb;pdb.set_trace()
        
        # Step 4.1.1: task prediction
        state_feat_concat = state_feat_encode.reshape(batch_size, -1)
        task_pred = self.task_decoder(state_feat_concat)
        # Step 4.1.2: state decoding
        state_feat_decode = self.state_decoder(
            state_feat_encode,
            state_prompt_features, 
            task_pred,
            vision_proj=img_embeds_action,
        )
        state_feat_input = state_feat_decode
        
        # Step 4.2.1: LLM decoding
        use_image = True if T == 1 else False
        prompt = random.choice(self.prompt_list)

        if self.prompt_list:
            img_embeds_llm, atts_img = self.prompt_wrap(img_embeds_llm, atts_img, prompt, use_image)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in text_input]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(visual_features.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                    dtype=torch.long).to(visual_features.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds_llm.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds_llm, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            llm_outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True
            )
            llm_output_embeds = llm_outputs.hidden_states[-1]
            llm_output_embeds = self.action_proj(llm_output_embeds)
            
            # Step 4.1.4: action decoding
            prompt_features = state_prompt_features
            action_logits,action_emb,state_query = self.action_decoder(
                state_feat_input, 
                prompt_features, 
                tasks if self.training is False else task_pred.argmax(-1),
                return_query=True,
            )
            # step 4.2.2: knowledge fusion
            # batch_first
            state_query = state_query.permute(1,0,2)
            state_output_llm = self.action_llm_deocer(state_query, llm_output_embeds)
            
            state_action_logits_llm = self.classifier_llm(state_output_llm)

            # Select action tokens
            action_idx = list(range(1, self.time_horz*2, 2))    # index of action/state
            action_logits_llm = state_action_logits_llm[:, action_idx, :]
            state_output_llm = state_output_llm[:,action_idx,:]
            action_logits = action_logits+action_logits_llm   
            # Collect outputs
            outputs = self.process_outputs(state_prompt_features, 
                                        state_logits, 
                                        state_feat_decode, 
                                        action_logits,
                                        action_emb,
                                        task_pred)
            
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
            if self.training:
                losses['loss_llm'] = llm_outputs.loss
        
            return outputs, labels, losses

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