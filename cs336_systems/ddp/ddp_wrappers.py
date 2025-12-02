# cs336_systems/ddp/ddp_wrappers.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.distributed as dist


class DDPIndividualParameters(nn.Module):
    """
    2.3.2: 逐参数异步 all-reduce 的 DDP 容器。

    特点：
    - 不做 flatten，不分配额外大 buffer -> 显存开销几乎只来自 param.grad 本身。
    - 使用 register_post_accumulate_grad_hook 在 backward 过程中就启动通信。
    - finish_gradient_synchronization() 会等待所有异步通信完成，并做 grad /= world_size。
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        assert dist.is_initialized(), "Process group must be initialized before creating DDPIndividualParameters."

        self.module = module
        self.world_size = dist.get_world_size()

        # 确保所有 rank 初始参数一致
        self._broadcast_parameters_from_rank0()

        # 存储 (handle, grad_tensor) 对，供 finish_gradient_synchronization 使用
        self._handles: List[Tuple[dist.Work, torch.Tensor]] = []

        # 注册逐参数 hook
        self._register_grad_hooks()

    # 让外部像普通 module 一样用 forward
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _broadcast_parameters_from_rank0(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _register_grad_hooks(self):
        """
        对每个需要 grad 的 parameter 注册 post-accumulate hook。
        当该 param 的 grad 计算完成时，立刻发起异步 all-reduce。
        """
        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            def _make_hook(p: torch.Tensor):
                # hook 的签名: hook(grad) -> None 或 grad
                def _hook(grad: torch.Tensor):
                    # grad 与 p.grad 是同一个 tensor
                    handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                    # 记录 handle 和 grad 引用，finish 时要 wait + /world_size
                    self._handles.append((handle, grad))
                    # 不替换 grad，按原样返回
                    return grad

                return _hook

            # 新 API: Tensor.register_post_accumulate_grad_hook
            param.register_post_accumulate_grad_hook(_make_hook(param))

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        """
        等待所有异步 all-reduce 完成，然后把对应的 grad /= world_size。
        在 optimizer.step() 之前调用。
        """
        for handle, grad in self._handles:
            # 等待通信完成
            handle.wait()
            # 把 sum / world_size 得到平均梯度
            grad /= self.world_size

        # 清空 handle 列表，供下一次 backward 使用
        self._handles.clear()

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)
