import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from models.blip2.blip2 import Blip2Base, disabled_train
from transformers import LlamaTokenizer, LlamaConfig

logger = logging.getLogger(__name__)
import logging

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from models.blip2.vit import build_vit
from models.blip2.builder import build_qformer
from models.criterions import VTC_VTM_Loss, get_sim,FocalLoss
from timm.models.layers import trunc_normal_
from models.schema.state_encoder import StateEncoder
from models.schema.state_decoder import StateDecoder
from models.schema.action_decoder import ActionDecoder
from models.utils import viterbi_path

logger = logging.getLogger(__name__)

class PlanLLM_it_vicuna(Blip2Base):
    """
    PlanLLM model.
    """
    def __init__(self, config):
        super().__init__()
        
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
        
        
        
        self.config = config


        # pretrained_path
        vit_blip_model_path = config.get("vit_blip_model_path", None)
        llama_model_path = config.get("llama_model_path")
        videochat2_model_path = config.get("videochat2_model_path", "")  
        freeze_qformer = config.get("freeze_qformer", True)
        # vit
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        # qformer
        num_query_token = config.max_traj_len+1
        qformer_hidden_dropout_prob = config.get("qformer_hidden_dropout_prob", 0.1)
        qformer_attention_probs_dropout_prob = config.get("qformer_attention_probs_dropout_prob", 0.1)
        qformer_drop_path_rate = config.get("qformer_drop_path_rate", 0.1)
        extra_num_query_token = config.get("extra_num_query_token", 32)
        self.qformer_text_input = config.get("qformer_text_input", True)
        # prompt
        max_txt_len = config.get("max_txt_len", 32)
        self.begin_signal = "###"
        self.role = ("Human", "Assistant")
        self.start_token = config.get("start_token", "<Video>")
        self.end_token = config.get("end_token", "</Video>")
        self.img_start_token = config.get("img_start_token", "<Image>")
        self.img_end_token = config.get("img_end_token", "</Image>")
        logger.info(f"Add instruction in qformer: {self.qformer_text_input}")
        # debug
        debug = config.get("debug", False)
        use_flash_attention = config.get("use_flash_attention", False)
        self.use_lora = config.get("use_lora", False)
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.low_resource = low_resource
        self.qformer, self.query_tokens = self.init_Qformer(
            num_query_token, config.vision_encoder.encoder_embed_dim,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
        )
        
        if not self.qformer_text_input:
            self.qformer.bert.embeddings.word_embeddings = None
            self.qformer.bert.embeddings.position_embeddings = None
            for layer in self.qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.qformer.resize_token_embeddings(len(self.tokenizer))
        self.qformer.cls = None

        if vit_blip_model_path:
            logger.info(f"Load ViT and QFormer from {vit_blip_model_path}")
            state_dict = torch.load(vit_blip_model_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
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
            logger.info("freeze Qformer")
            for _, param in self.qformer.named_parameters():
                param.requires_grad = False
            self.qformer = self.qformer.eval()
            self.qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        logger.info('Loading LLAMA')
        # problem: do we need to set truncation_side="left"?
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
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
                    device_map="auto",
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.float16,
                )

        logger.info("freeze LLAMA")
        for _, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading LLAMA Done')

        if self.use_lora:
            logger.info("Use lora")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()

        self.llama_proj = nn.Linear(
            self.qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len

        # load weights of VideoChat2
        if videochat2_model_path:
            logger.info(f"Load VideoChat2 from: {videochat2_model_path}")
            ckpt = torch.load(videochat2_model_path, map_location="cpu")
            if 'model' in ckpt.keys():
                msg = self.load_state_dict(ckpt['model'], strict=False)
            else:
                msg = self.load_state_dict(ckpt, strict=False)
            logger.info(msg)

    def vit_to_cpu(self):
        self.vision_layernorm.to("cpu")
        self.vision_layernorm.float()
        self.vision_encoder.to("cpu")
        self.vision_encoder.float()

    def encode_vis_feat(self, image_embeds, instruction):
        device = image_embeds.device
        if self.low_resource:
            self.vit_to_cpu()
            image_embeds = image_embeds.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, N, C]
            
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            if self.extra_num_query_token > 0:
                query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            else:
                query_tokens = self.query_tokens
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    instruction,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image_embeds.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                query_output = self.qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llama = self.llama_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        return inputs_llama, False
        
    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def forward(self,visual_features, 
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
        img_embeds, use_image = self.encode_vis_feat(state_feat_interacted, instruction)
        batch_size, img_len, _ = img_embeds.shape
        
        
        # Step 4.1: LLM decoding
        # mark the largest length
        # when padding, the attention mask will be 0
        max_len = 0
        input_embed_list = []
        p_before_len_list = []
        target_list = []
        # handle each prompt individually
        for idx, prompt in enumerate(text_input):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            # split the prompt via END_TOKEN
            end_token = self.img_end_token if use_image else self.end_token
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            if self.use_lora:
                p_before_embeds = self.llama_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids)
            else:
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.begin_signal + self.role[0] + ": "
            sep2 = self.begin_signal + self.role[1] + ": "
            raw_text = p_after.split(sep2)
            for idx in range(1, len(raw_text)):
                raw_text[idx] = sep2 + raw_text[idx]
            # the first raw_text contains system and question
            # the last raw_text only contains answer
            # rstrip() for the extra " "
            answer_targets = p_after_tokens.input_ids.clone()
            # target: "###Human:       ###Assistant: xxxxx. ###"
            system = raw_text[0].split(sep1)[0]
            system_len = self._get_text_len(system.rstrip())
            sep_len = self._get_text_len(sep1.rstrip())
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :system_len] = -100
            answer_targets[:, (system_len+sep_len):cur_len] = -100
            for text in raw_text[1:-1]: 
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]+sep1).rstrip())
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())
            assert cur_len == answer_targets.shape[1], f"The final length ({cur_len}) is not equal to the original prompt ({answer_targets.shape[1]}): {prompt}"

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)
        
        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len + 1, self.max_txt_len + img_len)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device) * self.llama_tokenizer.pad_token_id
        if self.use_lora:
            inputs_embeds = self.llama_model.base_model.model.model.embed_tokens(inputs_embeds)
        else:
            inputs_embeds = self.llama_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device).fill_(-100)
        # set bos_token
        inputs_embeds[:, :1] = self.llama_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len+1)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len+img_len+1):(input_len+1)] = target_list[idx][0, :(input_len-p_before_len-img_len)]
        
        with self.maybe_autocast():
            llm_outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        # Step 4.2.1: task prediction
        state_feat_concat = state_feat_encode.reshape(batch_size, -1)
        task_pred = self.task_decoder(state_feat_concat)
        # Step 4.2.2: state decoding
        state_feat_decode = self.state_decoder(
            state_feat_encode,
            state_prompt_features, 
            task_pred,
            vision_proj=img_embeds,
        )
        state_feat_input = state_feat_decode
        # Step 4.2.3: knowledge fusion

        # Step 4.2.4: action decoding
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