
from configs.model import *

# ========================= schema setting ==========================
dataset = "niv" 
max_traj_len=4
if dataset == "niv":
    model_name= "niv"
    num_layers = 2
    attn_heads = 32
    mlp_ratio = 2
    text_input_dim = 768
    img_input_dim = 512
    embed_dim=512
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
if dataset == 'niv':
    epochs=1000 # small dataset hard to fit
# =========================PlanLLM setting ==========================
stage='qformer'
num_workers = 6

stop_key = None

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
max_txt_l = 32

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
    max_txt_l=max_txt_l,
    batch_size=batch_size,
    batch_size_test=batch_size,
)

# ========================= model ==========================
text_enc = "bert"
model = dict(
    model_cls="PlanLLM_qformer",
    text_encoder="${TextEncoders[${text_enc}]}",
    vit_add_ln=True,
    embed_dim=768,
    temp=0.07,
    qformer_num_query_tokens=max_traj_len,
    agg_method="mean",
    drop_path_rate=0.2,
)

criterion = dict(
    loss_weight=dict(vtc=0.1, mlm=0.1, vtm=0.1, mvm=0.1, cap=0.1),  # 0: disabled.
    vtm_hard_neg=True,
    vtm_cat_text_cls=True
)

optimizer = dict(
    opt="adamW",
    lr=1e-4,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=0.01,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=True,modules_lrs=dict(state_encoder=1e-4,state_decoder=1e-3,action_decoder=1e-3,task_decoder=1e-3)),
)

scheduler = dict(sched="xx", epochs=epochs, min_lr_multi=0.01, warmup_epochs=100)

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
    project="planLlm",  # setup in your command line
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