import datetime
import logging
import time
from os.path import join
import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb

from dataset import MetaLoader,create_loader, create_stateful_sampler, create_sampler
from models import *
from utils.metrics import *
from models.utils import AverageMeter
from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)

def test(
        args,
        data_loader,
        model,
        logger,
        state_prompt_features,
        transition_matrix,
        e=0,
        device=torch.device("cuda"),
        is_train=False
    ):
    state_prompt_features  = torch.tensor(state_prompt_features).to(device, dtype=torch.float32).clone().detach()
    # losses
    losses_state  = AverageMeter()
    losses_action = AverageMeter()
    losses_state_pred = AverageMeter()
    losses_task = AverageMeter()

    # metrics for action
    action_acc1 = AverageMeter()
    action_acc5 = AverageMeter()
    action_sr   = AverageMeter()
    action_miou = AverageMeter()

    # metrics for viterbi
    viterbi_sr = AverageMeter()
    viterbi_acc1 = AverageMeter()
    viterbi_miou = AverageMeter()

    state_acc = AverageMeter()
    task_acc = AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            '''
            batch_states:  (batch_size, time_horizon, 2, embedding_dim)
            batch_actions: (batch_size, time_horizon)
            batch_prompts: (batch_size, 2*time_horizon, num_prompts, embedding_dim)
            '''
            batch_states, batch_actions, batch_tasks,batch_captions,idx = data
            batch_states = batch_states.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            batch_tasks = batch_tasks.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            # batch_captions = batch_captions.to(device, non_blocking=True)
            batch_size, _ = batch_actions.shape

            ## compute loss
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_tasks = batch_tasks.to(device)

            # with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs, labels, losses = model(
                visual_features=batch_states,
                state_prompt_features=state_prompt_features,
                actions=batch_actions,
                tasks=batch_tasks,
                text_input=batch_captions,
                idx=idx,
                transition_matrix = transition_matrix
            )
            loss = losses["action"] + losses["state_encode"] + losses["task"]*0.1+losses['action_emb'] + losses["state_decode"] * 0.1

            losses_state.update(losses["state_encode"].item(), batch_size)
            losses_action.update(losses["action"].item(), batch_size)
            losses_state_pred.update(losses["state_decode"].item(), batch_size)
            losses_task.update(losses["task"].item(), batch_size)

            ## metrics for state encoding
            acc_state = topk_accuracy(output=outputs["state_encode"].cpu(), target=labels["state"].cpu())
            state_acc.update(acc_state[0].item())

            ## computer accuracy for action prediction
            (acc1, acc5), sr, MIoU = \
                accuracy(outputs["action"].contiguous().view(-1, outputs["action"].shape[-1]).cpu(), 
                         labels["action"].contiguous().view(-1).cpu(), topk=(1, 5), max_traj_len=args.max_traj_len) 
            action_acc1.update(acc1.item(), batch_size)
            action_acc5.update(acc5.item(), batch_size)
            action_sr.update(sr.item(), batch_size)
            action_miou.update(MIoU, batch_size)

            # metrics for task prediction
            acc_task = topk_accuracy(output=outputs["task"].cpu(), target=labels["task"].cpu(), topk=[1])[0]
            task_acc.update(acc_task.item(), batch_size)

            # metrics for viterbi decoding
            pred_viterbi = outputs["pred_viterbi"].cpu().numpy()
            labels_viterbi = labels["action"].reshape(batch_size, -1).cpu().numpy().astype("int")
            sr_viterbi = success_rate(pred_viterbi, labels_viterbi, True)
            miou_viterbi = acc_iou(pred_viterbi, labels_viterbi, False).mean()
            acc_viterbi = mean_category_acc(pred_viterbi, labels_viterbi)
            viterbi_sr.update(sr_viterbi, batch_size)
            viterbi_acc1.update(acc_viterbi, batch_size)
            viterbi_miou.update(miou_viterbi, batch_size)
    if e% args.log_freq==0:
        logger.info("Epoch: {} State Loss: {:.2f} Top1 Acc: {:.2f}%"\
                    .format(e+1, losses_state.avg, state_acc.avg))
        logger.info("\tAction Loss: {:.2f}, SR: {:.2f}% Acc1: {:.2f}% Acc5: {:.2f}% MIoU: {:.2f}"\
                    .format(losses_action.avg,
                            action_sr.avg,
                            action_acc1.avg,
                            action_acc5.avg,
                            action_miou.avg))
        logger.info("\tViterbi, SR: {:.2f}% Acc: {:.2f}% MIoU: {:.2f}"\
                    .format(viterbi_sr.avg,
                            viterbi_acc1.avg,
                            viterbi_miou.avg))
        logger.info("\tTask Loss: {:.2f}, Acc1: {:.2f}%"\
                    .format(losses_task.avg, task_acc.avg))
        logger.info("\tState Pred Loss: {:.2f}"\
                    .format(losses_state_pred.avg))
        
    return viterbi_sr.avg, viterbi_acc1.avg, viterbi_miou.avg


def train(
    model,
    train_loaders,
    optimizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    state_prompt_features,transition_matrix
):
    state_prompt_features  = torch.tensor(state_prompt_features).to(device, dtype=torch.float32).clone().detach()
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=100, fmt="{value:.6f}"))
    loss_names =['action','state_encode','task','state_decode','loss_llm']

    media_types = get_media_types(train_loaders)

    for name in loss_names:
        for m in media_types:
            metric_logger.add_meter(
                f"{m}-{name}", SmoothedValue(window=100, fmt="{value:.4f}")
            )

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (media_type, (data)) in enumerate(iterator):
        batch_states, batch_actions, batch_tasks,batch_captions,idx = data
        batch_states = batch_states.to(device, non_blocking=True)
        batch_actions = batch_actions.to(device, non_blocking=True)
        batch_tasks = batch_tasks.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=config.fp16):
            outputs, labels, losses = model(
                visual_features=batch_states,
                state_prompt_features=state_prompt_features,
                actions=batch_actions,
                tasks=batch_tasks,
                text_input=batch_captions,
                idx=idx,
            )
            # print(losses.keys())
            loss = losses["action"] + losses["state_encode"] + losses["task"]+ losses["state_decode"] * 0.1
            loss+=losses['loss_llm']

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if config.optimizer.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        # logging
        for name in loss_names:
            value = losses[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{media_type}-{name}": value})
        ## compute accuracy for state encoding
        acc_state = topk_accuracy(output=outputs["state_encode"].cpu(), target=labels["state"].cpu())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(state_acc=acc_state[0].item())
        ## compute accuracy for action prediction
        (acc1, acc5), sr, MIoU = \
            accuracy(outputs["action"].contiguous().view(-1, outputs["action"].shape[-1]).cpu(), 
                        labels["action"].contiguous().view(-1).cpu(), topk=(1, 5), max_traj_len=config.max_traj_len) 
        metric_logger.update(action_acc1=acc1.item())
        metric_logger.update(action_acc5=acc5.item())
        metric_logger.update(action_sr=sr.item())
        metric_logger.update(action_MIoU=MIoU)
        acc_task = topk_accuracy(output=outputs["task"].cpu(), target=labels["task"].cpu(), topk=[1])[0]
        metric_logger.update(task_acc=acc_task.item())
        
        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

        if config.debug and global_step % 20 == 0:
            logger.info("debug mode, break training loop")
            break

        if config.debug and global_step % (2 * log_freq + 3) == 0:
            logger.info("debug mode, break training loop")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if epoch %config.log_freq==0:
        logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    return global_step

def parse_task_info(task_info_path):
    task_info = dict()
    with open(task_info_path, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 6):
            task_info[lines[i].strip()] = {
                "name": lines[i+1].strip(),
                "url": lines[i+2].strip(),
                "num_steps": int(lines[i+3].strip()),
                "steps": lines[i+4].strip().split(","),          
            }
    return task_info
def parse_annotation(anot_dir, task_info, idices_mapping):
    annotation = dict()
    action_collection = idices_mapping["action_idx"]
    reduced_action_collection = idices_mapping["rd_action_idx"] 
    task_collection = idices_mapping["task_idx"]

    for file in os.listdir(anot_dir):
        info = pd.read_csv(os.path.join(anot_dir, file), header=None)
        v_name = file.split(".")[0]
        task_id = v_name[:v_name.find("_")]
        video_id = v_name[v_name.find("_")+1:]
        annotation[video_id] = []
        for i in range(len(info)):
            action_id = int(info.iloc[i][0])
            task = task_info[task_id]["name"].strip()
            action = task_info[task_id]["steps"][action_id-1].strip()

            whole_action_id = action_collection["{}_{}".format(task, action)]
            reduced_action_id = reduced_action_collection[action]
            task_nid = task_collection[task]

            annotation[video_id].append({
                "task": task,
                "task_id": task_nid,
                "action": action,
                "action_id": whole_action_id,
                "reduced_action_id": reduced_action_id,
                "start": int(np.round(float(info.iloc[i][1]))),
                "end": int(np.round(float(info.iloc[i][2]))),
            })

    return annotation


def create_dataset(args):
    if args.dataset == 'crosstask':
        
        if args.split == 'base':
            from dataset.crosstask_dataloader import CrossTaskDataset as ProcedureDataset
        elif args.split == 'pdpp':
            # use PDPP data split and data sample
            from dataset.crosstask_dataloader_pdpp import CrossTaskDataset as ProcedureDataset
        elif args.split == 'p3iv':
            # use P3IV data split and data sample
            assert args.max_traj_len == 3, "Only the datasplit for max_traj_len = 3 is available."
            from dataset.crosstask_dataloader_p3iv import CrossTaskDataset as ProcedureDataset
        else:
            raise ValueError(f'No this split ({args.split == 'p3iv'}) for this dataset ({args.dataset})')
    
    elif args.dataset == 'coin':
        from dataset.coin_dataloader import CoinDataset as ProcedureDataset
    
    elif args.dataset == 'niv':
        from dataset.niv_dataloader import NivDataset as ProcedureDataset
        
    if args.dataset == 'crosstask':
        state_prompt_features = np.load(f'./data/state_description_features/crosstask_state_prompt_features.npy')
        
        task_info_path = os.path.join(args.root_dir, "tasks_primary.txt")
        task_info = parse_task_info(task_info_path)
        with open("data/crosstask_idices.json", "r") as f:
            idices_mapping = json.load(f)
        anot_dir = os.path.join(args.root_dir, "annotations")
        anot_info = parse_annotation(anot_dir, task_info, idices_mapping)

        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(anot_info, args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "train", M=args.M,stage=args.stage)
        
        logger.info("Loading valid data...")
        test_dataset = ProcedureDataset(anot_info, args.features_dir, state_prompt_features, 
                                        args.valid_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "valid", M=args.M,stage=args.stage)
        transition_matrix = train_dataset.transition_matrix
    
    elif args.dataset == "coin":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/coin_state_prompt_features.npy')
    
        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "train", M=args.M,stage=args.stage)
        
        logger.info("Loading valid data...")
        test_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.valid_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "valid", M=args.M,stage=args.stage)
        transition_matrix = train_dataset.transition_matrix
    elif args.dataset == "niv":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/niv_state_prompt_features.npy')

        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "train", M=args.M,stage=args.stage)
        
        logger.info("Loading valid data...")
        test_dataset = ProcedureDataset(args.features_dir, state_prompt_features,
                                        args.valid_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "valid", M=args.M,stage=args.stage)
        transition_matrix = train_dataset.transition_matrix
    
    else:
        raise ValueError(f'No this dataset ({args.dataset})')
    return [train_dataset], [test_dataset],state_prompt_features,transition_matrix

def setup_dataloaders(config, mode="pt"):
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode}")
    media_types = ['video']
    train_datasets, test_datasets,state_prompt_features,transition_matrix = create_dataset(config)
    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        samplers = create_sampler(
            train_datasets, [True], num_tasks, global_rank
        )
    else:
        samplers = [None]

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=[config.inputs.batch_size for k in media_types],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[None] * len(media_types),
    )  # [0]

    # test datasets, a mapping from dataset name to data loader
    test_loaders = create_loader(
        test_datasets,
        [None] * len(test_datasets),
        batch_size=[config.inputs.batch_size_test for d in test_datasets],
        num_workers=[config.num_workers] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[None] * len(test_datasets),
    )
    test_dataset_names=f'{config.dataset}_valid'
    test_name2loaders = {k: v for k, v in zip(test_dataset_names, test_loaders)}
    return train_loaders, test_name2loaders, media_types,state_prompt_features,transition_matrix



def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)


    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, test_name2loaders, train_media_types,state_prompt_features,transition_matrix = setup_dataloaders(
        config, mode=config.mode
    )
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = len(train_media_types) == 1
    
    model_cls = eval(config.model.get('model_cls', 'PlanLLM_pt_vicuna'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        find_unused_parameters=True,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    logger.info("Start training")
    start_time = time.time()
    max_SR = 0
    best = 0
    best_epoch = 0
    for epoch in range(start_epoch, config.scheduler.epochs):
        if not config.evaluate:
            global_step = train(
                model,
                train_loaders,
                optimizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
                state_prompt_features,transition_matrix
            )
        with torch.cuda.amp.autocast(enabled=config.fp16):
            
            model.eval()
            for test_name, test_loader in test_name2loaders.items():
                SR,acc,mIoU = test(config, 
                        test_loader, 
                        model, 
                        logger, 
                        state_prompt_features, 
                        transition_matrix, 
                        epoch, 
                        device,
                        is_train=True)
            result = {'SR':SR,'acc':acc,'mIoU':mIoU}
            logger.info(f'Epoch: {epoch},result: {result}')
        if is_main_process():
            param_grad_dic = {
                k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
            }
            state_dict = model_without_ddp.state_dict()
            for k in list(state_dict.keys()):
                if k in param_grad_dic.keys() and not param_grad_dic[k]:
                    # delete parameters that do not require gradient
                    del state_dict[k]
            torch.save(
                state_dict, 
                os.path.join(config.saved_path,
                    f"{config.dataset}/T{config.max_traj_len}_model_pt_last.pth"  
                )
            )
                
            if SR > max_SR:
                max_SR = SR
                best_result = {'SR':SR,'acc':acc,'mIoU':mIoU}
                log_save_path = f"{config.dataset}_T{config.max_traj_len}_model_pt_best.pth"  
                checkpoint_save_path = os.path.join(
                        config.saved_path, 
                        config.dataset,
                        f"T{config.max_traj_len}_model_pt_best.pth"
                    )
                
                torch.save(state_dict, checkpoint_save_path)        
                os.system(f"cp {checkpoint_save_path} {log_save_path}")
                logger.info(f'Epoch: {epoch},Best result: {best_result}')
            # logger.info(f"Epoch {epoch}")
            # param_grad_dic = {
            #     k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
            # }
            # state_dict = model_without_ddp.state_dict()
            # for k in list(state_dict.keys()):
            #     if k in param_grad_dic.keys() and not param_grad_dic[k]:
            #         # delete parameters that do not require gradient
            #         del state_dict[k]
            # save_obj = {
            #     "model": state_dict,
            #     "optimizer": optimizer.state_dict(),
            #     "scheduler": scheduler.state_dict(),
            #     "scaler": scaler.state_dict(),
            #     "config": config,
            #     "epoch": epoch,
            #     "global_step": global_step,
            # }
            # if config.get("save_latest", False):
            #     torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
            # else:
            #     torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

        if config.evaluate:
            break
        if config.distributed:
            dist.barrier()
        # dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    # logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()
    logger.info('='*50)
    logger.info(f'Best result: {best_result}')
    logger.info('='*50)


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
