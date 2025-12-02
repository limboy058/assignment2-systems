from __future__ import annotations

import argparse
import os
os.environ['WANDB_API_KEY'] = 'bc3e2778d5116934b68809b6414d1812936a43a0'
os.environ["WANDB_CONSOLE"] = "wrap"

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
from cs336_systems.ddp.wandb_config import init_wandb, log_training_metrics, log_validation_metrics, finish_wandb, add_wandb_args
from cs336_systems.ddp.DDP import DDP
from cs336_systems.ddp.ShardedOptimizer import ShardedOptimizer



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

    # global batch size
    batch_size: int = 128

    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    max_steps: int = 200
    log_interval: int = 10
    rope_theta: float = 10000.0
    use_fa2: bool = False
    use_bf16: bool = False
    use_wandb: bool = False
    
    data_dir: str = "/mnt/mnt/zjdx"
    train_bin_name: str = "owt_train.bin"
    valid_bin_name: str = "owt_valid.bin"



def load_bin_dataset(path: str, dtype: str = "uint16") -> npt.NDArray:
    """读取 .bin"""
    return np.memmap(path, dtype=dtype, mode="r")



def make_batch_fn(
    cfg: TrainConfig,
    device_str: str,
    train_data: npt.NDArray,
    valid_data: npt.NDArray,
    local_batch_size: int,
    val_seqs: int = 256,
) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    """
    返回 batch_fn(split) -> (x, y)
    - train: local_batch_size
    - val:   (已弃用)
    """

    def _batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            dataset = train_data
            batch_size = local_batch_size
        elif split in ("val", "valid", "validation"):
            assert 0, "已弃用"
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
) -> tuple[float, tuple[int, int]]:
    """
    返回:
        global_avg_loss: 全局平均 val loss
        val_shape: (total_valid_seqs, seq_len)
    """
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

    loss_tensor = torch.tensor(local_loss_sum, device=device_str, dtype=torch.float32)
    count_tensor = torch.tensor(local_count, device=device_str, dtype=torch.float32)

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    global_avg_loss = (loss_tensor / count_tensor).item()
    # 总体上相当于 (total_valid_seqs, seq_len)
    val_shape = (total_valid_seqs, cfg.seq_len)
    return global_avg_loss, val_shape





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



# 已弃用! 你可以删除这段了(毕竟我们要调用全新版本)
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




def setup_process_group(rank: int, world_size: int, backend: str = "nccl"):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "33333")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def ddp_train_worker(rank: int, world_size: int, cfg: TrainConfig, args: argparse.Namespace):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    setup_process_group(rank, world_size, backend="nccl")

    assert cfg.batch_size % world_size == 0, (
        f"Global batch size {cfg.batch_size} must be divisible by "
        f"world_size {world_size}."
    )
    local_batch_size = cfg.batch_size // world_size

    if rank == 0:
        print(f"[DDP] world_size={world_size}")
        print(f"[DDP] global_batch_size={cfg.batch_size}")
        print(f"[DDP] local_batch_size per rank = {local_batch_size}")

    # 1. 加载数据
    train_path = os.path.join(cfg.data_dir, cfg.train_bin_name)
    valid_path = os.path.join(cfg.data_dir, cfg.valid_bin_name)
    train_data = load_bin_dataset(train_path)
    valid_data = load_bin_dataset(valid_path)

    # 2. 构建model & DDPmodel
    model = build_model(cfg).to(device)
    # ddp_model = DDPIndividualParameters(model)
    ddp_model = DDP(model, 64)

    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters())
        print(f"[rank {rank}] num_params = {num_params/1e9:.3f} B")

    # 3. 优化器
    # optimizer = AdamW(
    #     ddp_model.parameters(),
    #     lr=cfg.lr,
    #     weight_decay=cfg.weight_decay,
    # )
    
    optimizer = ShardedOptimizer(
        ddp_model.parameters(),
        AdamW,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    

    # 4. 初始化 wandb
    wandb_run = None
    if rank == 0 and args.use_wandb:
        wandb_run = init_wandb(
            args=args,
            model=ddp_model,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            watch_model=False,
            log_config=True,
        )

    # 5. checkpoint 相关
    ckpt_dir = os.path.join("/mnt/mnt/zjdx", "tyhcyq")
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    # 6. 采样数据函数
    device_str = f"cuda:{rank}"
    batch_fn = make_batch_fn(cfg, device_str, train_data, valid_data, local_batch_size)

    # 7. cosine lr 
    warmup_iters = min(1000, max(10, cfg.max_steps // 10))
    cosine_cycle_iters = min(25000, cfg.max_steps)
    max_lr = cfg.lr
    min_lr = cfg.min_lr



    # 8. 训练LOOP
    train_time_accum = 0.0
    steps_in_window = 0

    best_val_loss = float("inf")
    ckpt_dir = "/mnt/mnt/zjdx/tyhcyq"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 训练开始的 wall-clock，用作“背景时间”
    train_start_t = time.time()

    for step in range(cfg.max_steps):
        step_train_t0 = time.time()

        # 手动设置 lr
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

        # 记录 grad_norm（clip 前后，看你 nn_utils.clip_gradient 返回的是哪一个）
        grad_norm = clip_gradient(ddp_model.parameters(), max_norm=1.0)

        optimizer.step()

        step_train_time = time.time() - step_train_t0
        train_time_accum += step_train_time
        steps_in_window += 1

        global_step = step + 1
        need_log = (global_step % cfg.log_interval == 0 or global_step == 1)

        if need_log:
            # ====== validation 时间 ======
            val_t0 = time.time()
            val_loss, val_shape = run_validation_multi_step(
                ddp_model=ddp_model,
                cfg=cfg,
                device_str=device_str,
                valid_data=valid_data,
                rank=rank,
                world_size=world_size,
                total_valid_seqs=256,        # 符合要求的 valid_seqs=256
                val_micro_batch_per_rank=2,
            )
            val_time = time.time() - val_t0

            if rank == 0:
                # 背景时间：从训练开始到现在的 wall-clock 秒数
                wall_clock_s = time.time() - train_start_t
                wall_clock_str = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(time.time()),
                )

                # ===== wandb 时间 =====
                wandb_time = 0.0
                if wandb_run is not None:
                    t_w0 = time.time()
                    log_training_metrics(
                        wandb_run,
                        iteration=global_step,
                        loss=loss.item(),
                        learning_rate=lr,
                        grad_norm=grad_norm,
                        train_step_time=step_train_time,
                    )
                    log_validation_metrics(
                        wandb_run,
                        iteration=global_step,
                        val_loss=val_loss,
                        val_total_seqs=val_shape[0],
                        val_seq_len=val_shape[1],
                        val_step_time=val_time,
                    )
                    wandb_time = time.time() - t_w0

                # ===== ckpt 时间 =====
                t_ckpt0 = time.time()
                ckpt = {
                    "step": global_step,
                    "model_size": cfg.model_size,
                    "vocab_size": cfg.vocab_size,
                    "seq_len": cfg.seq_len,
                    "state_dict": ddp_model.module.state_dict(),
                    # "optimizer_state": optimizer.state_dict(),  # 需要的话再打开
                    "val_loss": val_loss,
                }

                last_path = os.path.join(ckpt_dir, "last.pt")
                torch.save(ckpt, last_path)

                is_new_best = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    torch.save(ckpt, best_path)
                    is_new_best = True

                ckpt_time = time.time() - t_ckpt0

                # ===== 打印平均 train_time（窗口内按 step 平均）=====
                avg_train_time = train_time_accum / steps_in_window

                # 第一个 log：背景时间 + loss + grad_norm + val shape
                print(
                    f"[DDP-ind step {global_step:5d}] "
                    f"time={wall_clock_str} "
                    f"(since_start={wall_clock_s:.1f}s) "
                    f"lr={lr:.6e} "
                    f"train_loss={loss.item():.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"val_shape={val_shape}"
                )

                # 第二个 log：各阶段耗时
                print(
                    f"[DDP-ind step {global_step:5d}] "
                    f"avg_train_time={avg_train_time:.4f}s, "
                    f"val_time(last_eval)={val_time:.4f}s, "
                    f"wandb_time={wandb_time:.4f}s, "
                    f"ckpt_time={ckpt_time:.4f}s, "
                    f"new_best={is_new_best}"
                )

                # 重置窗口计时
                train_time_accum = 0.0
                steps_in_window = 0




    dist.destroy_process_group()
    if rank == 0:
        print("[DDP-individual] Training done.")
        if wandb_run is not None:
            finish_wandb(wandb_run)




def launch_ddp_training(world_size: int, cfg: TrainConfig, args: argparse.Namespace):
    mp.spawn(
        ddp_train_worker,
        args=(world_size, cfg, args),
        nprocs=world_size,
        join=True,
    )





def parse_args() -> tuple[TrainConfig, int, argparse.Namespace]:
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--use-fa2", action="store_true")
    parser.add_argument("--use-bf16", action="store_true")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/mnt/zjdx",
        help="directory containing owt_train.bin / owt_valid.bin",
    )
    parser.add_argument("--train-bin-name", type=str, default="owt_train.bin")
    parser.add_argument("--valid-bin-name", type=str, default="owt_valid.bin")


    parser = add_wandb_args(parser)

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
        use_bf16=args.use_bf16,
        use_wandb=args.use_wandb,
        data_dir=args.data_dir,
        train_bin_name=args.train_bin_name,
        valid_bin_name=args.valid_bin_name,
    )

    return cfg, args.world_size, args



def main():
    cfg, world_size, args = parse_args()
    launch_ddp_training(world_size, cfg, args)


if __name__ == "__main__":
    main()
