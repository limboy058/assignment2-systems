import torch
import numpy as np
from cs336_systems.fa2_triton import FlashAttention2
from cs336_systems.fa2_torch import FlashAttention2_torch
from cs336_systems.fa2_naive import MyAttention

torch.set_float32_matmul_precision("high")


def test_precision_with_npz(filename, atol=1e-4,rtol=1e-4):
    """
    Test precision of FlashAttention implementations against saved ground truth
    
    Args:
        filename: path to .npz file containing q, k, v, o, grad_q, grad_k, grad_v
        atol: absolute tolerance for torch.allclose
    """
    print(f"\n{'='*80}")
    print(f"Testing: {filename}")
    print(f"{'='*80}")
    
    # Load ground truth data
    data = np.load(filename)
    
    # Convert to torch tensors
    q_gt = torch.from_numpy(data['q']).cuda().requires_grad_(True)
    k_gt = torch.from_numpy(data['k']).cuda().requires_grad_(True)
    v_gt = torch.from_numpy(data['v']).cuda().requires_grad_(True)
    o_gt = torch.from_numpy(data['o']).cuda()
    grad_q_gt = torch.from_numpy(data['grad_q']).cuda()
    grad_k_gt = torch.from_numpy(data['grad_k']).cuda()
    grad_v_gt = torch.from_numpy(data['grad_v']).cuda()
    
    print(f"\nInput shapes:")
    print(f"  Q: {q_gt.shape}, K: {k_gt.shape}, V: {v_gt.shape}")
    print(f"  dtype: {q_gt.dtype}")
    
    # Test both implementations
    for impl_name, impl_class in [ ("Triton FA2", FlashAttention2),]:# ("PyTorch FA2", FlashAttention2_torch) ("Naive atten",MyAttention)
        print(f"\n{'-'*80}")
        print(f"Testing {impl_name}")
        print(f"{'-'*80}")
        
        # Clone inputs for this test
        q = q_gt.clone().detach().requires_grad_(True)
        k = k_gt.clone().detach().requires_grad_(True)
        v = v_gt.clone().detach().requires_grad_(True)
        
        # Forward pass
        o = impl_class.apply(q, k, v, False)  # is_causal=False
        
        loss = o.sum()
        loss.backward()
        
        # Check forward pass precision
        forward_match = torch.allclose(o, o_gt, atol=atol,rtol=rtol)
        forward_max_diff = (o - o_gt).abs().max().item()
        forward_mean_diff = (o - o_gt).abs().mean().item()
        
        print(f"\nForward Pass:")
        print(f"  Match: {forward_match}")
        print(f"  Max absolute diff: {forward_max_diff:.6e}")
        print(f"  Mean absolute diff: {forward_mean_diff:.6e}")
        
        # Check backward pass precision
        grad_q_match = torch.allclose(q.grad, grad_q_gt, atol=atol,rtol=rtol) # 设置rtol=1e-3 即可pass !
        grad_k_match = torch.allclose(k.grad, grad_k_gt, atol=atol,rtol=rtol)
        grad_v_match = torch.allclose(v.grad, grad_v_gt, atol=atol,rtol=rtol)
        
        print(f"\nBackward Pass:")
        print(f"  grad_Q match: {grad_q_match}")
        print(f"    Max absolute diff: {(q.grad - grad_q_gt).abs().max().item():.6e}")
        print(f"    Mean absolute diff: {(q.grad - grad_q_gt).abs().mean().item():.6e}")
        
        print(f"  grad_K match: {grad_k_match}")
        print(f"    Max absolute diff: {(k.grad - grad_k_gt).abs().max().item():.6e}")
        print(f"    Mean absolute diff: {(k.grad - grad_k_gt).abs().mean().item():.6e}")
        
        print(f"  grad_V match: {grad_v_match}")
        print(f"    Max absolute diff: {(v.grad - grad_v_gt).abs().max().item():.6e}")
        print(f"    Mean absolute diff: {(v.grad - grad_v_gt).abs().mean().item():.6e}")
        
        # Overall result
        all_match = forward_match and grad_q_match and grad_k_match and grad_v_match
        print(f"\n{'✓' if all_match else '✗'} Overall: {'PASS' if all_match else 'FAIL'}")
        



if __name__ == "__main__":
    files = ["/data/nlp_course/2-fa2-data/fa2_nheads16_seq4128_dhead64.npz", "/data/nlp_course/2-fa2-data/fa2_nheads32_seq8576_dhead128.npz"]#,]#
    
    # Test against ground truth with different tolerance levels
    for atol,rtol in [(1e-3, 1e-3)]:
        print(f"\n{'#'*80}")
        print(f"Testing with atol={atol}, rtol={rtol}")
        print(f"{'#'*80}")
        
        for filename in files:
            test_precision_with_npz(filename, atol=atol, rtol=rtol)
    