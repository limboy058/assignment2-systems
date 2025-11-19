import math
import numpy as np
import torch

class MyAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # q/k/v: [..., S, D]; 允许任意前导维
        *leading, S_q, D = q.shape
        S_k = k.shape[-2]
        assert k.shape[:-2] == tuple(leading) and v.shape[:-2] == tuple(leading)
        assert k.shape[-1] == D and v.shape[-1] == D and v.shape[-2] == S_k

        # 展平前导维以便 bmm
        B = int(np.prod(leading)) if leading else 1
        # 半精度 -> fp32 以更稳定的 matmul/softmax
        q2 = q.reshape(B, S_q, D).to(torch.float32)
        k2 = k.reshape(B, S_k, D).to(torch.float32)
        v2 = v.reshape(B, S_k, D).to(torch.float32)

        scale = 1.0 / math.sqrt(D)
        scores = torch.bmm(q2, k2.transpose(1, 2)) * scale
        scores = scores - scores.amax(dim=-1, keepdim=True)  # 数值稳定性
        attn = torch.softmax(scores, dim=-1)
        o2 = torch.bmm(attn, v2)                     # fp32 结果

        # 保存反向需要的中间量（都用 fp32）
        ctx.save_for_backward(attn, q2, k2, v2)
        ctx.shape_ctx = (leading, S_q, S_k, D)
        return o2.reshape(*leading, S_q, D).to(q.dtype)  # 返回与输入相同的 dtype（半精度）

    @staticmethod
    def backward(ctx, grad_out):
        attn, q2, k2, v2 = ctx.saved_tensors
        leading, S_q, S_k, D = ctx.shape_ctx
        # 半精度的 grad -> fp32 计算梯度
        go2 = grad_out.reshape(-1, S_q, D).to(torch.float32)

        # grad_v
        grad_v2 = torch.bmm(attn.transpose(1, 2), go2)  # [B, S_k, D]
        # grad_attn
        grad_attn = torch.bmm(go2, v2.transpose(1, 2))  # [B, S_q, S_k]
        # softmax backward
        S = (grad_attn * attn).sum(dim=-1, keepdim=True)
        grad_scores = (grad_attn - S) * attn

        scale = 1.0 / math.sqrt(D)
        grad_q2 = torch.bmm(grad_scores, k2) * scale    # [B, S_q, D]
        grad_k2 = torch.bmm(grad_scores.transpose(1, 2), q2) * scale  # [B, S_k, D]

        # 还原形状并转回与输入一致的 dtype
        B = grad_q2.size(0)
        grad_q = grad_q2.reshape(*leading, S_q, D).to(grad_out.dtype)
        grad_k = grad_k2.reshape(*leading, S_k, D).to(grad_out.dtype)
        grad_v = grad_v2.reshape(*leading, S_k, D).to(grad_out.dtype)
        return grad_q, grad_k, grad_v,None