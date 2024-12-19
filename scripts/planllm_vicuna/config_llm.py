from configs.model import *


# ========================= schema setting ==========================
dataset = "niv" 
if dataset == "niv":
    model_name= "niv"
    num_layers = 2
    attn_heads = 32
    mlp_ratio = 2
    text_input_dim = 768
    img_input_dim = 512
    embed_dim=512
    max_traj_len=3
    num_action=48
    num_tasks=5
    epochs=500
    batch_size=256
    dropout=0.2
    step_size=40
    M=2
    aug_range=0
    no_state_task=True
    root_dir='dataset/niv'
    train_json= 'dataset/niv/niv_train.json'
    valid_json= 'dataset/niv/niv_valid.json'
    features_dir= 'data/niv_features/processed_data' 
    eval=False
    saved_path='checkpoints'
    last_epoch=1
    split='base'
    seed=42
    uncertain=False
    num_sample=1500
elif dataset =="coin":
    model_name= "coin"
    num_layers = 2
    attn_heads = 32
    mlp_ratio = 2
    text_input_dim = 768
    img_input_dim = 512
    embed_dim=512
    max_traj_len=3
    num_action=778
    num_tasks=180
    epochs=500
    batch_size=256
    dropout=0.2
    step_size=40
    M=2
    aug_range=0
    no_state_task=True
    root_dir='dataset/coin'
    train_json= 'dataset/coin/coin_train.json'
    valid_json= 'dataset/coin/coin_valid.json'
    features_dir= 'data/coin_features/full_npy'
    eval=False
    saved_path='checkpoints'
    last_epoch=1
    split='base'
    seed=3407
    uncertain=False
elif dataset=='crosstask':
    model_name= "crosstask"
    num_layers = 2
    attn_heads = 32
    mlp_ratio = 2
    text_input_dim = 768
    img_input_dim = 512
    embed_dim=512
    max_traj_len=3
    num_action=133
    num_tasks=18
    epochs=500
    batch_size=256
    dropout=0.2
    step_size=40
    M=2
    aug_range=0
    no_state_task=True
    root_dir='dataset/crosstask/crosstask_release'
    train_json= 'dataset/crosstask/cross_task_data_False.json'
    valid_json= 'dataset/crosstask/cross_task_data_True.json'
    features_dir= 'data/crosstask_features/processed_data'
    eval=False
    saved_path='checkpoints'
    last_epoch=1
    split='base'
    seed=3407
    uncertain=False
   
# =========================PlanLLM setting ==========================
stage='qformer'

# ========================= data ==========================
# train_corpus = "webvid10m_cc14m_plus"
# train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
# test_file = dict()
# test_types = []
num_workers = 6

stop_key = None

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
# batch_size = 4
max_txt_l = 512

pre_text = False

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=batch_size,
    batch_size_test=batch_size,
)

# ========================= model ==========================
model = dict(
    model_cls="PlanLLM_pt_vicuna",
    vit_blip_model_path=f"checkpoints/{dataset}/T3_model_qformer_best.pth",
    llama_model_path="pretrained/vicuna-7b-v0",
    freeze_vit=False,
    freeze_qformer=False,
    max_txt_len="${max_txt_l}",
    # vit
    low_resource=False,
    vision_encoder=dict(
        name="vit_l14",
        img_size=224, 
        patch_size=16, 
        d_model=1024,
        encoder_embed_dim=1024, 
        encoder_depth=24,
        encoder_num_heads=16, 
        drop_path_rate=0., 
        num_frames="${num_frames}",
        tubelet_size=1,
        use_checkpoint=False,
        checkpoint_num=0,
        pretrained="",
        return_index=-2,
        vit_add_ln=True,
    ),
    # prompt
    prompt_path="prompts/concise_description.txt",
    img_prompt_path="prompts/concise_image_description.txt",
    prompt_template="###Human: {} ###Assistant: ",
    end_sym="###",
    # qformer
    num_query_token=32,
    qformer_hidden_dropout_prob=0.1,
    qformer_attention_probs_dropout_prob=0.1,
    qformer_drop_path_rate=0.2,
    extra_num_query_token=64,
    # debug=True,
)

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=True,modules_lrs=dict(state_encoder=1e-4,state_decoder=1e-3,action_decoder=1e-3,task_decoder=1e-3)),
)

scheduler = dict(sched="xx", epochs=epochs, min_lr_multi=0.01, warmup_epochs=0.2)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="user",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="planLLM",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
seed = 42

save_latest = True
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?
distributed=False
# use_flash_attention=True