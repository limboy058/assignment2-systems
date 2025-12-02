"""
最小单卡训练脚本

功能：
- 从 cs336_basics.model 中构建 BasicsTransformerLM
- 从 cs336_basics.data 中拿 batch（如果没有就用随机数据兜底）
- 用一个简单的训练 loop 跑若干 step，打印 loss

后面做 DDP 时，只需要：
- 在 DDP worker 里调用 build_model / build_optimizer / training_loop
- 把 model 换成你的 NaiveDDP 包起来，在 backward 后加一行 allreduce_gradients()
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import numpy.typing as npt

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import get_cosine_lr, AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def load_bin_dataset(path: str, dtype: str = "uint16") -> npt.NDArray:
    """
    读取 owt_train.bin / owt_valid.bin 这样的 1D token 流。
    默认 dtype=uint16
    """
    return np.memmap(path, dtype=dtype, mode="r")


# -------------------------------
# 模型规格
# -------------------------------

MODEL_SPECS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


# -------------------------------
# 配置
# -------------------------------

@dataclass
class TrainConfig:
    model_size: str = "small"
    vocab_size: int = 10000
    seq_len: int = 128
    batch_size: int = 16
    lr: float = 3e-4           # 最大学习率（cosine 的 peak）
    min_lr: float = 3e-5       # cosine 收尾到的最小学习率
    weight_decay: float = 0.01
    max_steps: int = 100
    log_interval: int = 10
    device: str = "cuda"
    data_device: Optional[str] = None  # 如果想把数据放到别的 device
    train_split: str = "train"
    val_split: str = "val"
    rope_theta: float = 10000.0
    data_dir: str = "/mnt/mnt/zjdx"


# -------------------------------
# 模型 & 优化器
# -------------------------------

def build_model(cfg: TrainConfig) -> nn.Module:
    """
    用 cs336_basics.model 里的 BasicsTransformerLM 构建一个模型。
    """
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
    )
    return model


def build_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
) -> optim.Optimizer:
    """
    只返回 optimizer，lr 调度在 training_loop 里手动完成。
    """
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,                 # 先用最大 lr 初始化，之后每步会被 scheduler 覆盖
        weight_decay=cfg.weight_decay,
    )
    return optimizer


def make_batch_fn(
    cfg: TrainConfig,
    device: torch.device | str,
    train_data: npt.NDArray,
    valid_data: npt.NDArray,
) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    
    """
    返回一个函数：batch_fn(split) -> (x, y)
    根据 split 选用 train_data / valid_data，然后调用你自己的 get_batch。
    """
    
    # 把 torch.device 转成字符串，方便 "cuda" in device 判断
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device

    def _batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            dataset = train_data
        elif split in ("val", "valid", "validation"):
            dataset = valid_data
        else:
            raise ValueError(f"Unknown split: {split}")

        return get_batch(
            dataset=dataset,
            batch_size=cfg.batch_size,
            context_length=cfg.seq_len,
            device=device_str,
        )

    return _batch


# -------------------------------
# loss & 单步训练
# -------------------------------
def train_one_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    cfg: TrainConfig,
    *,
    split: str = "train",
) -> float:
    """
    跑一次前向 + 反向 + 更新，返回 loss（float）。
    这是后面 DDP 中也会复用的核心 step。
    """
    model.train()
    x, y = batch_fn(split)

    optimizer.zero_grad(set_to_none=True)

    logits = model(x)
    loss = cross_entropy(logits, y)

    loss.backward()

    clip_gradient(model.parameters(), max_norm=1.0)

    optimizer.step()

    return float(loss.detach().item())


@torch.no_grad()
def evaluate_one_step(
    model: nn.Module,
    batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    cfg: TrainConfig,
    *,
    split: str = "val",
) -> float:
    """
    简单的单 step eval，用来 sanity check。
    之后你做正式实验的时候可以写成多步平均。
    """
    model.eval()
    x, y = batch_fn(split)
    logits = model(x)
    loss = cross_entropy(logits, y)
    return float(loss.detach().item())


# -------------------------------
# 主训练 loop（带 cosine lr）
# -------------------------------

def training_loop(cfg: TrainConfig) -> None:
    """
    单卡训练主循环：
    - 构建 model / optimizer / batch_fn
    - 每步用 get_cosine_lr 手动调度学习率
    - 跑 max_steps 步训练
    - 每 log_interval 打印一次 loss
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[single_gpu_train] Using device: {device}")

    
    train_bin = os.path.join(cfg.data_dir, "owt_train.bin")
    valid_bin = os.path.join(cfg.data_dir, "owt_valid.bin")
    train_data = load_bin_dataset(train_bin)
    valid_data = load_bin_dataset(valid_bin)
    
    
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    batch_fn = make_batch_fn(cfg, device, train_data, valid_data)

    # 定义 cosine lr 的参数
    warmup_iters = max(10, cfg.max_steps // 10)
    cosine_cycle_iters = cfg.max_steps
    max_lr = cfg.lr
    min_lr = cfg.min_lr

    t0 = time.time()

    for step in range(cfg.max_steps):
        # step 从 0 开始，这里把它当成 it 传给 get_cosine_lr
        lr = get_cosine_lr(
            it=step,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        loss = train_one_step(model, optimizer, batch_fn, cfg, split=cfg.train_split)

        # 打 log
        global_step = step + 1
        if global_step % cfg.log_interval == 0 or global_step == 1:
            elapsed = time.time() - t0
            t0 = time.time()
            print(
                f"[step {global_step:5d}] lr={lr:.6e} "
                f"train_loss={loss:.4f} ({elapsed:.2f}s since last log)"
            )

            # 可选：顺便跑一下 val，看是否在下降
            try:
                val_loss = evaluate_one_step(
                    model, batch_fn, cfg, split=cfg.val_split
                )
                print(f"               val_loss={val_loss:.4f}")
            except Exception as e:
                # 如果你一开始没有准备好 val split，这里就先忽略
                print(f"               (val eval skipped: {e})")

    print("[single_gpu_train] Done.")


# -------------------------------
# CLI 入口
# -------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Minimal single-GPU training loop for cs336 BasicsTransformerLM",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=list(MODEL_SPECS.keys()),
        help="Model size: small / medium / large / xl / 2.7B",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device, e.g., cuda or cpu",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Name of training split for get_batch",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val",
        help="Name of validation split for get_batch",
    )

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
        device=args.device,
        train_split=args.train_split,
        val_split=args.val_split,
    )
    return cfg



if __name__ == "__main__":
    cfg = parse_args()
    training_loop(cfg)
