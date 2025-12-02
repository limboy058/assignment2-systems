import torch
import triton
from cs336_systems.fa2.fa2_triton import FlashAttention2
from cs336_systems.fa2.fa2_torch import FlashAttention2_torch
from cs336_systems.fa2.fa2_naive import MyAttention

torch.set_float32_matmul_precision('medium')


def test_timing_flash_forward_backward():
    n_heads = 32
    sequence_length = 8576
    d_head = 128
    
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device='cuda:1', dtype=torch.float32, requires_grad=True
    )

    flash = torch.compile(FlashAttention2.apply)

    def flash_forward_backward(causal = False):
        # q2 = q.clone(); q2.requires_grad_()
        # k2 = k.clone(); k2.requires_grad_()
        # v2 = v.clone(); v2.requires_grad_()
        # o = flash(q2, k2, v2, True)
        o = flash(q, k, v, causal)
        loss = o.sum()
        loss.backward()
        
    def flash_forward(causal = True):
        # q2 = q.clone(); q2.requires_grad_()
        # k2 = k.clone(); k2.requires_grad_()
        # v2 = v.clone(); v2.requires_grad_()
        # o = flash(q2, k2, v2, True)
        o = flash(q, k, v, causal)
    
    
    causal = True
    print("causal:",causal)
    results1 = triton.testing.do_bench(lambda: flash_forward_backward(causal), rep=10000, warmup=1000)
    print("all:",results1)
    # results2 = triton.testing.do_bench(lambda: flash_forward(causal), rep=10000, warmup=1000)
    # print("fwd:", results2)
    # print("bwd:",results1 - results2)

    
    
test_timing_flash_forward_backward()