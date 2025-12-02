# cs336_systems/ddp/minimal_ddp.py

"""
最小 DDP 训练脚本（单机多卡, naive all-reduce）

特点：
- 每块 GPU（每个进程）上都有完整一份 BasicsTransformerLM
- backward 后调用 NaiveDDP.allreduce_gradients() 同步梯度
- 使用 OWT 的 owt_train.bin / owt_valid.bin
- 使用 cs336_basics.optimizer.get_cosine_lr 做手动 cosine lr 调度
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
torch.set_float32_matmul_precision('high')

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import get_cosine_lr, AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient


# -------------------------------
# 模型规格 & 配置
# -------------------------------

MODEL_SPECS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


@dataclass
class TrainConfig:
    model_size: str = "small"
    vocab_size: int = 40375        # 和你的 OWT 数据对齐
    seq_len: int = 128
    
    batch_size: int = 128
    
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    max_steps: int = 200
    log_interval: int = 10
    rope_theta: float = 10000.0
    use_fa2: bool = False

    data_dir: str = "/mnt/mnt/zjdx"   # 你的数据路径
    train_bin_name: str = "owt_train.bin"
    valid_bin_name: str = "owt_valid.bin"


# -------------------------------
# 数据相关
# -------------------------------

def load_bin_dataset(path: str, dtype: str = "uint16") -> npt.NDArray:
    """读取 owt_train.bin / owt_valid.bin 这样的 token 流。"""
    return np.memmap(path, dtype=dtype, mode="r")


def log_mem(tag: str, rank: int):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[rank {rank}] {tag}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")



def make_batch_fn(
    cfg: TrainConfig,
    device_str: str,
    train_data: npt.NDArray,
    valid_data: npt.NDArray,
    local_batch_size: int,
) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    """
    返回 batch_fn(split) -> (x, y)，这里的 local_batch_size 是“每个 rank 上的 batch size”。
    """
    def _batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            dataset = train_data
        elif split in ("val", "valid", "validation"):
            dataset = valid_data
        else:
            raise ValueError(f"Unknown split: {split}")

        return get_batch(
            dataset=dataset,
            batch_size=local_batch_size,
            context_length=cfg.seq_len,
            device=device_str,
        )

    return _batch



# -------------------------------
# 模型 & loss
# -------------------------------

def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model_size not in MODEL_SPECS:
        raise ValueError(f"Unknown model_size={cfg.model_size}")

    spec = MODEL_SPECS[cfg.model_size]

    model = BasicsTransformerLM(
        d_model=spec["d_model"],
        d_ff=spec["d_ff"],
        num_layers=spec["num_layers"],
        num_heads=spec["num_heads"],
        vocab_size=cfg.vocab_size,
        context_length=cfg.seq_len,
        rope_theta=cfg.rope_theta,
        use_fa2=cfg.use_fa2,
    )
    return model


# -------------------------------
# Naive DDP 实现
# -------------------------------

class NaiveDDP(nn.Module):
    """
    最简单的 DDP 容器：
    - 每个 rank 拥有完整一份模型
    - backward 完成后手动 all-reduce 所有 param.grad
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self._broadcast_parameters_from_rank0()

    def _broadcast_parameters_from_rank0(self):
        # 确保所有 rank 的初始参数一致
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def allreduce_gradients(self) -> float:
        """
        使用“单次 batched all-reduce”的方式同步梯度：
        1) 把所有 dense grad flatten 成一个大 tensor
        2) 对这个大 tensor 做一次 all-reduce
        3) / world_size 得到平均梯度
        4) unflatten 回各个 param.grad 中

        返回值：本次 all-reduce 的通信时间（秒），方便 benchmarking。
        """
        # 收集有 grad 的参数
        params, grads = [], []
        for p in self.module.parameters():
            if p.grad is None:
                continue
            params.append(p)
            grads.append(p.grad)

        if len(grads) == 0:
            return 0.0

        # 1. flatten 所有梯度到一个大 tensor
        flat_grads = _flatten_dense_tensors(grads)

        # 2. all-reduce（带 cuda 同步，便于测时间）
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()
        comm_time = time.perf_counter() - t0

        # 3. 平均梯度
        flat_grads /= self.world_size

        # 4. unflatten 回各个 grad tensor
        new_grads = _unflatten_dense_tensors(flat_grads, grads)
        for p, g in zip(params, new_grads):
            # 原地覆盖 grad，保持 optimizer.state 结构不变
            p.grad.copy_(g)

        return comm_time

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)


# -------------------------------
# DDP 训练核心
# -------------------------------

def setup_process_group(rank: int, world_size: int, backend: str = "nccl"):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "33333")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def ddp_train_worker(rank: int, world_size: int, cfg: TrainConfig):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    setup_process_group(rank, world_size, backend="nccl")

    if rank == 0:
        print(f"[DDP] world_size={world_size}")
        print(f"[DDP] global_batch_size={cfg.batch_size}")

    # ---- 计算 per-rank local batch size，按手册要求 shard ----
    assert cfg.batch_size % world_size == 0, (
        f"Global batch size {cfg.batch_size} must be divisible by "
        f"world_size {world_size} (manual requires n divisible by d)."
    )
    local_batch_size = cfg.batch_size // world_size
    if rank == 0:
        print(f"[DDP] local_batch_size per rank = {local_batch_size}")
        
        
    # 0. 刚刚 init 完 process group，看一下
    if rank == 0:
        log_mem("after dist.init_process_group", rank)

    # 1. 加载数据
    train_path = os.path.join(cfg.data_dir, cfg.train_bin_name)
    valid_path = os.path.join(cfg.data_dir, cfg.valid_bin_name)
    train_data = load_bin_dataset(train_path)
    valid_data = load_bin_dataset(valid_path)
    if rank == 0:
        log_mem("after loading datasets (CPU)", rank)

    # 2. 构建模型 & NaiveDDP
    model = build_model(cfg).to(device)
    ddp_model = NaiveDDP(model)
    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters())
        print(f"[rank {rank}] num_params = {num_params/1e9:.3f} B")
        log_mem("after building model + NaiveDDP", rank)


    # 3. 优化器
    optimizer = AdamW(
        ddp_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    if rank == 0:
        log_mem("after building optimizer", rank)
        
        
    # 4. 构建 batch 函数：这里传 local_batch_size
    device_str = f"cuda:{rank}"
    batch_fn = make_batch_fn(cfg, device_str, train_data, valid_data, local_batch_size)

    # 5. cosine lr 的参数
    warmup_iters = max(10, cfg.max_steps // 10)
    cosine_cycle_iters = cfg.max_steps
    max_lr = cfg.lr
    min_lr = cfg.min_lr

    # 6. 训练循环
    t0 = time.time()
    step_time_accum = 0.0
    comm_time_accum = 0.0
    
    
    for step in range(cfg.max_steps):
        if step == 0 and rank == 0:
            log_mem("before first iteration", rank)
            
        iter_start = time.time()

        it = step  # 0-based
        lr = get_cosine_lr(
            it=it,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        ddp_model.train()
        x, y = batch_fn("train")

        optimizer.zero_grad(set_to_none=True)

        logits = ddp_model(x)
        loss = cross_entropy(logits, y)

        loss.backward()

        # === 单次 batched all-reduce，同步梯度，并记录通信时间 ===
        comm_time = ddp_model.allreduce_gradients()

        clip_gradient(ddp_model.parameters(), max_norm=1.0)
        optimizer.step()

        # 结束本步，统计 step time
        torch.cuda.synchronize()
        iter_end = time.time()
        iter_time = iter_end - iter_start

        step_time_accum += iter_time
        comm_time_accum += comm_time

        global_step = step + 1

        # 只在 rank 0 打 log
        if rank == 0 and (global_step % cfg.log_interval == 0 or global_step == 1):
            elapsed = time.time() - t0
            t0 = time.time()

            # 简单 eval：rank0 自己拿一批 val
            ddp_model.eval()
            with torch.no_grad():
                x_val, y_val = batch_fn("val")
                logits_val = ddp_model(x_val)
                val_loss = cross_entropy(logits_val, y_val).item()

            # 平均每步总时间 & 通信时间（在最近 log_interval 步内）
            steps_in_window = cfg.log_interval if global_step > 1 else 1
            avg_iter_time = step_time_accum / steps_in_window
            avg_comm_time = comm_time_accum / steps_in_window

            print(
                f"[DDP step {global_step:5d}] "
                f"lr={lr:.6e} "
                f"train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} "
                f"iter_time={avg_iter_time:.4f}s "
                f"comm_time={avg_comm_time:.4f}s "
                f"({elapsed:.2f}s since last log)"
            )

            # 重置窗口累计
            step_time_accum = 0.0
            comm_time_accum = 0.0
        if step <= 50 and rank == 0:
            log_mem("after optimizer.step%d"%(step), rank)


    dist.destroy_process_group()
    if rank == 0:
        print("[DDP] Training done.")


def launch_ddp_training(world_size: int, cfg: TrainConfig):
    """
    用 mp.spawn 启动 world_size 个进程。
    """
    mp.spawn(
        ddp_train_worker,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True,
    )


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> tuple[TrainConfig, int]:
    parser = argparse.ArgumentParser(description="Naive DDP training script")

    parser.add_argument("--world-size", type=int, default=2, help="num of GPUs")

    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=list(MODEL_SPECS.keys()),
    )
    parser.add_argument("--vocab-size", type=int, default=40375)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="global batch size (will be sharded across GPUs)",
    )    
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--use-fa2", type=bool, default=False)

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/mnt/zjdx",
        help="directory containing owt_train.bin / owt_valid.bin",
    )
    parser.add_argument("--train-bin-name", type=str, default="owt_train.bin")
    parser.add_argument("--valid-bin-name", type=str, default="owt_valid.bin")

    args = parser.parse_args()

    cfg = TrainConfig(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        rope_theta=args.rope_theta,
        use_fa2=args.use_fa2,
        data_dir=args.data_dir,
        train_bin_name=args.train_bin_name,
        valid_bin_name=args.valid_bin_name,
    )

    return cfg, args.world_size


def main():
    cfg, world_size = parse_args()
    launch_ddp_training(world_size, cfg)


if __name__ == "__main__":
    main()
