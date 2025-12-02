import os
from typing import Any, Tuple, Type
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
torch.nn.parallel.DataParallel

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.params:list[nn.Parameter] = list(params)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        # self.local_params = [param for i, param in enumerate(self.params) if i % self.world_size == self.rank]
        self.local_params:list[list[nn.Parameter]] = [[] for _ in range(self.world_size)]
        for i, param in enumerate(self.params):
            self.local_params[i % self.world_size].append(param)

        self.actual_size = [0 for _ in range(self.world_size)]
        for i in range(self.world_size):
            self.actual_size[i] = sum(x.numel() for x in self.local_params[i])
        self.max_size_ = max(self.actual_size)

        self.local_optimizer = optimizer_cls(self.local_params[self.rank], **kwargs)
        super().__init__(self.params, kwargs)

    def step(self, closure=None, **kwargs):
        self.local_optimizer.step(closure, **kwargs)

        # my_updated_params_data = [p.data for p in self.local_params[self.rank]] 
        # flat_param_buffer = _flatten_dense_tensors(my_updated_params_data)
        # local_size = flat_param_buffer.numel()

        # if self.max_size_:
        #     max_size = self.max_size_
        # else:
        #     max_size_tensor = torch.tensor([local_size], device=flat_param_buffer.device, dtype=torch.long)
        #     dist.all_reduce(max_size_tensor, op=dist.ReduceOp.MAX)
        #     max_size = max_size_tensor.item()
        #     self.max_size_ = max_size
        # # print(max_size, self.rank)
        
        # if local_size < max_size:
        #     padded_buffer = torch.zeros(max_size, dtype=flat_param_buffer.dtype, device=flat_param_buffer.device)
        #     padded_buffer[:local_size] = flat_param_buffer
        # else:
        #     padded_buffer = flat_param_buffer

        # all_rank_buffers = [
        #     torch.empty(max_size, dtype=padded_buffer.dtype, device=padded_buffer.device) 
        #     for _ in range(self.world_size)
        # ]
        # dist.all_gather(all_rank_buffers, padded_buffer)

        # for rank_id in range(self.world_size):
        #     if rank_id == self.rank:
        #         continue
        #     rank_buffer = all_rank_buffers[rank_id]
            
        #     target_params = self.local_params[rank_id]
            
        #     # actual_size = sum(p.numel() for p in target_params)
        #     actual_size = self.actual_size[rank_id]
        #     actual_rank_buffer = rank_buffer[:actual_size]
        #     actual_rank_buffer = _unflatten_dense_tensors(actual_rank_buffer, target_params)
        #     for new, old in zip(actual_rank_buffer, target_params):
        #         old.data.copy_(new)

        for i, param in enumerate(self.params):
            dist.broadcast(param.data,src=i % dist.get_world_size())

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)