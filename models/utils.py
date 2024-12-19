import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate

logger = logging.getLogger(__name__)


def _init_transformer_weights(module, initializer_range=0.02):
    """Initialize the weights. Copied from transformers ViT/Bert model init"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def load_temp_embed_with_mismatch(temp_embed_old, temp_embed_new, add_zero=True):
    """
    Add/Remove extra temporal_embeddings as needed.
    https://arxiv.org/abs/2104.00650 shows adding zero paddings works.

    temp_embed_old: (1, num_frames_old, 1, d)
    temp_embed_new: (1, num_frames_new, 1, d)
    add_zero: bool, if True, add zero, else, interpolate trained embeddings.
    """
    # TODO zero pad
    num_frms_new = temp_embed_new.shape[1]
    num_frms_old = temp_embed_old.shape[1]
    logger.info(f"Load temporal_embeddings, lengths: {num_frms_old}-->{num_frms_new}")
    if num_frms_new > num_frms_old:
        if add_zero:
            temp_embed_new[
                :, :num_frms_old
            ] = temp_embed_old  # untrained embeddings are zeros.
        else:
            temp_embed_new = interpolate_temporal_pos_embed(temp_embed_old, num_frms_new)
    elif num_frms_new < num_frms_old:
        temp_embed_new = temp_embed_old[:, :num_frms_new]
    else:  # =
        temp_embed_new = temp_embed_old
    return temp_embed_new


def load_temp_embed_with_mismatch(temp_embed_old, temp_embed_new, add_zero=True):
    """
    Add/Remove extra temporal_embeddings as needed.
    https://arxiv.org/abs/2104.00650 shows adding zero paddings works.

    temp_embed_old: (1, num_frames_old, 1, d)
    temp_embed_new: (1, num_frames_new, 1, d)
    add_zero: bool, if True, add zero, else, interpolate trained embeddings.
    """
    # TODO zero pad
    num_frms_new = temp_embed_new.shape[1]
    num_frms_old = temp_embed_old.shape[1]
    logger.info(f"Load temporal_embeddings, lengths: {num_frms_old}-->{num_frms_new}")
    if num_frms_new > num_frms_old:
        if add_zero:
            temp_embed_new[
                :, :num_frms_old
            ] = temp_embed_old  # untrained embeddings are zeros.
        else:
            temp_embed_new = interpolate_temporal_pos_embed(temp_embed_old, num_frms_new)
    elif num_frms_new < num_frms_old:
        temp_embed_new = temp_embed_old[:, :num_frms_new]
    else:  # =
        temp_embed_new = temp_embed_old
    return temp_embed_new


def interpolate_temporal_pos_embed(temp_embed_old, num_frames_new):
    """
    temp_embed_old: (1, num_frames_old, 1, d)
    Returns:
        temp_embed_new: (1, num_frames_new, 1, d)
    """
    temp_embed_old = temp_embed_old.squeeze(2).permute(
        0, 2, 1
    )  # (1, d, num_frames_old)
    temp_embed_new = F.interpolate(
        temp_embed_old, num_frames_new, mode="linear"
    )  # (1, d, num_frames_new)
    temp_embed_new = temp_embed_new.permute(0, 2, 1).unsqueeze(
        2
    )  # (1, num_frames_new, 1, d)
    return temp_embed_new


def interpolate_pos_embed(pos_embed_old, pos_embed_new, num_patches_new):
    """
    Args:
        pos_embed_old: (1, L_old, d), pre-trained
        pos_embed_new: (1, L_new, d), newly initialized, to be replaced by interpolated weights
        num_patches_new:
    """
    # interpolate position embedding
    embedding_size = pos_embed_old.shape[-1]
    num_extra_tokens = pos_embed_new.shape[-2] - num_patches_new
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_old.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches_new ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        # the extra tokens seems always at the beginning of the position embedding
        extra_tokens = pos_embed_old[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_old[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        interpolated_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        logger.info(f"reshape position embedding from {orig_size}**2 to {new_size}**2")
        return interpolated_pos_embed
    else:
        return pos_embed_old


def interpolate_pos_relative_bias_beit(state_dict_old, state_dict_new, patch_shape_new):
    """
    Args:
        state_dict_old: loaded state dict
        state_dict_new: state dict for model with new image size
        patch_shape_new: new model patch_shape
    ref: https://github.com/microsoft/unilm/blob/master/beit/run_class_finetuning.py
    """
    all_keys = list(state_dict_old.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            state_dict_old.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = state_dict_old[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = state_dict_new[key].size()
            dst_patch_shape = patch_shape_new
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1
            )
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                # logger.info("Position interpolate for %s from %dx%d to %dx%d" % (
                #     key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                # logger.info("Original positions = %s" % str(x))
                # logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(rel_pos_bias.device)
                    )

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict_old[key] = new_rel_pos_bias
    return state_dict_old


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*repeat_idx)
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

def img_text_similarlity(state_features, prompt_features, scale):
        ''' Compute the similarity between visual and linguistic features

        Args:
            state_features:     Input visual feature.   (batch, length, embedding_dim)
            prompt_features:    Input language feature. (batch, length, embedding_dim)
            scale:              Scale parameter.

        Returns:
            logits:             Similarity matrix.      (batch, length, length)
        '''

        embedding_dim = state_features.shape[-1]
        
        # flatten features
        state_features = state_features.reshape(-1, embedding_dim)
        prompt_features = prompt_features.reshape(-1, embedding_dim)

        # normalized features
        image_features = state_features / state_features.norm(dim=1, keepdim=True)
        text_features = prompt_features / prompt_features.norm(dim=1, keepdim=True)

        # similarity as logits
        logits = scale * image_features @ text_features.t()
        return logits

def retain_top_n(matrix, n):
    """
    保留每帧最大的 n 个类别概率，其余置为 0.
    
    Args:
        matrix (np.ndarray): 输入的 (T, N) 矩阵。
        n (int): 每帧保留的最大类别数。
        
    Returns:
        np.ndarray: 处理后的矩阵。
    """
    # 对每一行找到最大的 n 个值的索引
    top_n_indices = np.argsort(matrix, axis=1)[:, -n:]
    
    # 构造结果矩阵
    result = np.zeros_like(matrix)
    
    # 按行保留最大 n 个值
    for i in range(matrix.shape[0]):  # 遍历每帧
        result[i, top_n_indices[i]] = matrix[i, top_n_indices[i]]
    
    return result

def viterbi_path(transition, emission, prior=None, observation=None, return_likelihood=False):
    ''' Viterbi algorithm

    Search the most likely sequence of hidden states given the observations.

    Args:
        transition:     Transition matrix, where A[i][j] is the probability of 
                        transitioning from state i to state j.  (num_action, num_action)
        emission:       Emission matrix, where B[i][j] is the probability of 
                        emitting observation j from state i.    (num_action, horizon)
        prior:          Prior probabilities, where pi[i] is the probability of 
                        starting in state i.    (num_action)
        observation:    Sequence of observations.   (horizon)
        return_likelihood:  Whether to return the likelihood of the best path.  (default: False)
    
    Returns:
        best_path:      The most likely action sequence.    (horizon)
        best_path_prob: The likelihood of the best path.
    '''

    # Initialize trellis
    T = emission.shape[1]                       # time horizon
    N = transition.shape[0]                     # number of actions

    if observation is None:
        observation = np.arange(T)
    
    if prior is None:
        prior = np.ones((N,), dtype=np.float32) / N
    emission = retain_top_n(emission,3)
    trellis = np.zeros((T, N), dtype=np.float32)       # store the probabilities of each state at each time step
    backpointers = np.zeros((T, N), dtype=np.int32)    # store the indices of the most likely previous state at each time step
    
    # Calculate probabilities for first time step
    trellis[0] = prior * emission[:, observation[0]]
    
    # Calculate probabilities for subsequent time steps
    for t in range(1, T):
        temp = trellis[t-1].reshape((N, 1)) * transition
        trellis[t] = emission[:, observation[t]] * np.max(temp, axis=0)
        backpointers[t] = np.argmax(temp, axis=0)
    
    # Backtrack to find most likely sequence of hidden states
    best_path_prob = np.max(trellis[-1])
    best_path_pointer = np.argmax(trellis[-1])
    best_path = [best_path_pointer]
    for t in range(T-1, 0, -1):
        best_path_pointer = backpointers[t][best_path_pointer]
        best_path.insert(0, best_path_pointer)
    
    best_path = np.array(best_path)
    
    if return_likelihood:
        return best_path, best_path_prob
    else:
        return best_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

allgather_wgrad = AllGather.apply
