import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "33333"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size):
    setup(rank, world_size)
    tensor_size = (3, 512, 10000)
    data = torch.rand(tensor_size, dtype=torch.float32, device="cuda")
    
    for _ in range(5):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        torch.cuda.synchronize()
        
    # 正式计时
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()

        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # 把每个 rank 的平均时间 all-gather 出来
    t_tensor = torch.tensor([sum(times) / len(times)], device=data.device)
    all_times = [torch.zeros_like(t_tensor) for _ in range(world_size)]
    dist.all_gather(all_times, t_tensor)
    
    if rank == 0:
        all_times_cpu = [t.item() for t in all_times]
        print(f"[allreduce] tensor_size={tensor_size}, "
              f"world_size={world_size}, per-rank mean times={all_times_cpu}, "
              f"global_mean={sum(all_times_cpu) / len(all_times_cpu):.6f}s")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)