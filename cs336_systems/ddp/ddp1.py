# cs336_systems/ddp/ddp_individual.py
"""
DDP 实现：2.3.2 Overlap backward + per-parameter gradient communication

特点：
- 使用 DDPIndividualParameters：
  - 对每个 param 注册 post_accumulate_grad_hook
  - 在 backward 过程中对每个 grad 做异步 all-reduce（不 flatten）
  - finish_gradient_synchronization() 中 wait 所有 handle 并做 grad /= world_size
- 不分配额外大 flatten buffer，显存占用接近 minimal DDP。
- 训练 loop 中统计每步 iter_time，便于 benchmark。
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable, Tuple, List, Optional

from contextlib import nullcontext

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

torch.set_float32_matmul_precision("high")

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import get_cosine_lr, AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_systems.ddp.wandb_config import init_wandb, log_training_metrics, log_validation_metrics


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
    vocab_size: int = 40375
    seq_len: int = 128

    # global batch size（会在各 rank 上 shard）
    batch_size: int = 128

    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    max_steps: int = 200
    log_interval: int = 10
    rope_theta: float = 10000.0
    use_fa2: bool = False
    use_bf16: bool = False
    
    data_dir: str = "/mnt/mnt/zjdx"
    train_bin_name: str = "owt_train.bin"
    valid_bin_name: str = "owt_valid.bin"


# -------------------------------
# 数据相关
# -------------------------------

def load_bin_dataset(path: str, dtype: str = "uint16") -> npt.NDArray:
    """读取 owt_train.bin / owt_valid.bin 这样的 token 流。"""
    return np.memmap(path, dtype=dtype, mode="r")


def make_batch_fn(
    cfg: TrainConfig,
    device_str: str,
    train_data: npt.NDArray,
    valid_data: npt.NDArray,
    local_batch_size: int,
    val_seqs: int = 1,   # 老师要求的 #valid_seqs
) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    """
    返回 batch_fn(split) -> (x, y)

    - train: 使用 per-rank 的 local_batch_size（global_batch_size / world_size）
    - val:   使用固定的 val_seqs
    """

    def _batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            dataset = train_data
            batch_size = local_batch_size
        elif split in ("val", "valid", "validation"):
            dataset = valid_data
            batch_size = val_seqs
        else:
            raise ValueError(f"Unknown split: {split}")

        return get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=cfg.seq_len,
            device=device_str,
        )

    return _batch


def run_validation_multi_step(
    ddp_model: nn.Module,
    cfg: TrainConfig,
    device_str: str,
    valid_data: npt.NDArray,
    rank: int,
    world_size: int,
    total_valid_seqs: int = 256,
    val_micro_batch_per_rank: int = 2,
) -> float:
    """
    在所有 rank 上一起做 validation：
    - 每个 rank 使用 val_micro_batch_per_rank 的 batch_size（比如 2）
    - 每个 rank 共处理 total_valid_seqs / world_size 个序列
    - 最终用 all_reduce 聚合成全局平均 loss

    这样：
        world_size=4, val_micro_batch_per_rank=2, total_valid_seqs=256
    -> 每个 rank 处理 64 个 seq，循环 32 次，每次 bs=2。
    """

    # 先算出每个 rank 该处理多少条 seq
    assert total_valid_seqs % world_size == 0, "total_valid_seqs 必须能被 world_size 整除"
    per_rank_valid_seqs = total_valid_seqs // world_size

    ddp_model.eval()
    local_loss_sum = 0.0
    local_count = 0

    with torch.no_grad():
        while local_count < per_rank_valid_seqs:
            cur_bs = min(val_micro_batch_per_rank, per_rank_valid_seqs - local_count)

            x_val, y_val = get_batch(
                dataset=valid_data,
                batch_size=cur_bs,
                context_length=cfg.seq_len,
                device=device_str,
            )

            autocast_ctx = (
                torch.amp.autocast_mode.autocast("cuda", dtype=torch.bfloat16)
                if cfg.use_bf16
                else nullcontext()
            )
            with autocast_ctx:
                logits_val = ddp_model(x_val)
                loss = cross_entropy(logits_val, y_val)

            local_loss_sum += loss.item() * cur_bs
            local_count += cur_bs



    # 在所有 rank 上聚合 loss_sum 和样本数
    loss_tensor = torch.tensor(local_loss_sum, device=device_str, dtype=torch.float32)
    count_tensor = torch.tensor(local_count, device=device_str, dtype=torch.float32)

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    global_avg_loss = (loss_tensor / count_tensor).item()
    return global_avg_loss


# -------------------------------
# 模型
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
# 2.3.2: 逐参数 overlap 的 DDP 容器
# -------------------------------

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
                # hook 的签名: hook(grad) -> grad or None
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
            handle.wait()
            grad /= self.world_size

        # 清空 handle 列表，供下一次 backward 使用
        self._handles.clear()

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

    # ---- per-rank local batch size，按 handout 要求 shard ----
    assert cfg.batch_size % world_size == 0, (
        f"Global batch size {cfg.batch_size} must be divisible by "
        f"world_size {world_size}."
    )
    local_batch_size = cfg.batch_size // world_size

    if rank == 0:
        print(f"[DDP] world_size={world_size}")
        print(f"[DDP] global_batch_size={cfg.batch_size}")
        print(f"[DDP] local_batch_size per rank = {local_batch_size}")

    # 1. 加载数据（CPU）
    train_path = os.path.join(cfg.data_dir, cfg.train_bin_name)
    valid_path = os.path.join(cfg.data_dir, cfg.valid_bin_name)
    train_data = load_bin_dataset(train_path)
    valid_data = load_bin_dataset(valid_path)

    # 2. 构建模型 & DDPIndividualParameters
    model = build_model(cfg).to(device)
    ddp_model = DDPIndividualParameters(model)

    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters())
        print(f"[rank {rank}] num_params = {num_params/1e9:.3f} B")

    # 3. 优化器
    optimizer = AdamW(
        ddp_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

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
    step_time_accum = 0.0  # 累计 iter_time，窗口内做平均

    for step in range(cfg.max_steps):
        iter_start = time.time()

        # 手动 cosine 学习率
        lr = get_cosine_lr(
            it=step,
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

        autocast_ctx = (
            torch.amp.autocast_mode.autocast("cuda", dtype=torch.bfloat16)
            if cfg.use_bf16
            else nullcontext()
        )
        with autocast_ctx:
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)

        loss.backward()

        ddp_model.finish_gradient_synchronization()
        clip_gradient(ddp_model.parameters(), max_norm=1.0)
        optimizer.step()


        # 统计 iter_time（包含 forward + backward + 通信 + optimizer.step）
        torch.cuda.synchronize()
        iter_end = time.time()
        iter_time = iter_end - iter_start
        step_time_accum += iter_time


        global_step = step + 1
        need_log = (global_step % cfg.log_interval == 0 or global_step == 1)


        if need_log:
            elapsed = time.time() - t0
            t0 = time.time()

            # --- 所有 rank 一起做 multi-step validation ---
            val_loss = run_validation_multi_step(
                ddp_model=ddp_model,
                cfg=cfg,
                device_str=device_str,
                valid_data=valid_data,
                rank=rank,
                world_size=world_size,
                total_valid_seqs=256,        # 老师要求 #valid_seqs=256
                val_micro_batch_per_rank=4,  # 比如每张卡一次跑 4 条 seq
            )

            # 只在 rank 0 打日志
            if rank == 0:
                steps_in_window = cfg.log_interval if global_step > 1 else 1
                avg_iter_time = step_time_accum / steps_in_window

                print(
                    f"[DDP-ind step {global_step:5d}] "
                    f"lr={lr:.6e} "
                    f"train_loss={loss.item():.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"iter_time={avg_iter_time:.4f}s "
                    f"({elapsed:.2f}s since last log)"
                )

                step_time_accum = 0.0


    dist.destroy_process_group()
    if rank == 0:
        print("[DDP-individual] Training done.")


def launch_ddp_training(world_size: int, cfg: TrainConfig):
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
    parser = argparse.ArgumentParser(
        description="DDP with per-parameter overlap (2.3.2) training script"
    )

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
    parser.add_argument("--use-fa2", action="store_true")
    parser.add_argument("--use-bf16", action="store_true")   # ← 新增


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
        use_bf16=args.use_bf16,      # ← 新增
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
