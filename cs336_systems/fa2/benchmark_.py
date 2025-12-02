import torch
import triton
from cs336_systems.fa2_triton_bwd import FlashAttention2
from cs336_systems.fa2_torch import FlashAttention2_torch
import pandas as pd


def benchmark_attention(seq_len, d_head, dtype, use_triton=True):
    """
    Benchmark forward, backward, and end-to-end for FlashAttention
    
    Args:
        seq_len: sequence length
        d_head: head dimension
        dtype: torch.float32 or torch.bfloat16
        use_triton: if True use Triton impl, else use PyTorch impl
    
    Returns:
        dict with forward_time, backward_time, end_to_end_time in ms
    """
    # Generate random inputs (batch_size=1, is_causal=True)
    batch_size = 1
    q = torch.randn(batch_size, seq_len, d_head, device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_head, device='cuda', dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_head, device='cuda', dtype=dtype, requires_grad=True)
    
    # Choose implementation
    attn_func = FlashAttention2_triton.apply if use_triton else FlashAttention2_torch.apply
    attn_func = torch.compile(attn_func)
    
    # Benchmark forward only
    def forward_only():
        o = attn_func(q, k, v, True)
        return o
    
    # Benchmark backward only (need to run forward first)
    def backward_only():
        o = attn_func(q, k, v, True)
        grad_out = torch.randn_like(o)
        torch.autograd.grad(o, [q, k, v], grad_out)
    
    # Benchmark end-to-end
    def forward_backward():
        q.grad = None
        k.grad = None
        v.grad = None
        o = attn_func(q, k, v, True)
        loss = o.sum()
        loss.backward()
    
    # Run benchmarks
    forward_time = triton.testing.do_bench(forward_only, rep=100, warmup=30)
    backward_time = triton.testing.do_bench(backward_only, rep=100, warmup=30) - forward_time
    end_to_end_time = triton.testing.do_bench(forward_backward, rep=100, warmup=30)
    
    return {
        'forward_ms': forward_time,
        'backward_ms': backward_time,
        'end_to_end_ms': end_to_end_time
    }


def run_benchmarks():
    """
    Run all benchmarks and generate comparison table
    """
    # seq_lengths = [1024, 2048, 4096, 8192, 16384]
    seq_lengths = [16384]
    d_heads = [16]
    dtypes = [torch.bfloat16, torch.float32]
    
    results = []
    
    for seq_len in seq_lengths:
        for d_head in d_heads:
            for dtype in dtypes:
                dtype_name = 'bfloat16' if dtype == torch.bfloat16 else 'float32'
                
                print(f"Benchmarking seq_len={seq_len}, d_head={d_head}, dtype={dtype_name}")
                
                try:
                    # Benchmark Triton implementation
                    triton_results = benchmark_attention(seq_len, d_head, dtype, use_triton=True)
                    
                    # Benchmark PyTorch implementation
                    pytorch_results = benchmark_attention(seq_len, d_head, dtype, use_triton=False)
                    
                    # Store results
                    results.append({
                        'seq_len': seq_len,
                        'd_head': d_head,
                        'dtype': dtype_name,
                        'triton_forward_ms': triton_results['forward_ms'],
                        'triton_backward_ms': triton_results['backward_ms'],
                        'triton_end_to_end_ms': triton_results['end_to_end_ms'],
                        'pytorch_forward_ms': pytorch_results['forward_ms'],
                        'pytorch_backward_ms': pytorch_results['backward_ms'],
                        'pytorch_end_to_end_ms': pytorch_results['end_to_end_ms'],
                        'speedup_forward': pytorch_results['forward_ms'] / triton_results['forward_ms'],
                        'speedup_backward': pytorch_results['backward_ms'] / triton_results['backward_ms'],
                        'speedup_end_to_end': pytorch_results['end_to_end_ms'] / triton_results['end_to_end_ms'],
                    })
                    
                except Exception as e:
                    print(f"  Skipped due to error: {e}")
                    continue
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    # Print summary table
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('flash_attention_benchmark.csv', index=False)
    print("\nResults saved to flash_attention_benchmark.csv")
    
    return df


if __name__ == "__main__":
    df = run_benchmarks()