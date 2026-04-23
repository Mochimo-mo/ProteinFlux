# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import List

import torch
import torch.nn as nn
import numpy as np


def add(m1, m2, inplace):
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if(not inplace):
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def pts_to_distogram(pts, min_bin=2.3125, max_bin=21.6875, no_bins=64):
    boundaries = torch.linspace(
        min_bin, max_bin, no_bins - 1, device=pts.device
    )
    dists = torch.sqrt(
        torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1)
    )
    return torch.bucketize(dists, boundaries)


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


from einops import rearrange
import torch

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    """多维索引的 batched gather"""
    import torch
    import numpy as np
    
    return_numpy = isinstance(data, np.ndarray)
    
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(inds, np.ndarray):
        inds = torch.from_numpy(inds)
    
    inds = inds.long().to(data.device)
    
    if dim < 0:
        dim = len(data.shape) + dim
    
    # 展平索引的非批次维度
    original_shape = inds.shape
    inds_flat = inds.flatten(start_dim=no_batch_dims)
    
    # 为尾部维度添加并扩展
    trailing_dims = data.shape[dim + 1:]
    
    if len(trailing_dims) > 0:
        # 有尾部维度
        for _ in trailing_dims:
            inds_flat = inds_flat.unsqueeze(-1)
        
        # 构建扩展形状
        expand_shape = list(inds_flat.shape[:-len(trailing_dims)]) + list(trailing_dims)
        inds_flat = inds_flat.expand(expand_shape)
        
        # Gather 并恢复形状
        result = torch.gather(data, dim, inds_flat)
        result_shape = list(original_shape) + list(trailing_dims)
        result = result.view(result_shape)
    else:
        # 没有尾部维度
        result = torch.gather(data, dim, inds_flat)
        result = result.view(original_shape)
    
    if return_numpy:
        result = result.cpu().numpy()
    
    return result


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Tree of type {type(tree)} not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)