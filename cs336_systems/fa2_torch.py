import torch
import math

def flash_attention_backward(Q, K, V, O, L, dO, scale, is_causal=False):
    """
    FA-2 bwd in torch.compile   
    参考公式13-19
    Args:
        Q: (batch_size, seq_len_q, head_dim)
        K: (batch_size, seq_len_k, head_dim)
        V: (batch_size, seq_len_k, head_dim)
        O: (batch_size, seq_len_q, head_dim)
        L: (batch_size, seq_len_q)
        dO: (batch_size, seq_len_q, head_dim)
        scale: 1/sqrt(head_dim)
        is_causal: bool
    
    Returns:
        dQ: (batch_size, seq_len_q, head_dim)
        dK: (batch_size, seq_len_k, head_dim)
        dV: (batch_size, seq_len_k, head_dim)
    """
    batch_size, seq_len_q, head_dim = Q.shape
    seq_len_k = K.shape[1]
    
    D = (O * dO).sum(dim=-1, keepdim=True)  
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch_size, seq_len_q, seq_len_k)
    
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        S = S.masked_fill(causal_mask, float('-inf'))

    P = torch.exp(S - L.unsqueeze(-1))  #累积的L其实存在误差
    dV = torch.matmul(P.transpose(-2, -1), dO)  
    dP = torch.matmul(dO, V.transpose(-2, -1))  
    
    if is_causal:
        dP = dP.masked_fill(causal_mask, 0.0)
        
    dS = P * (dP - D) 
    # 理论上, 等价于dS = A * (dA - (A * dA).sum(dim=-1, keepdim=True))  
    
    # if is_causal:
    #     dS = dS.masked_fill(causal_mask, 0.0) 
        
    dQ = torch.matmul(dS, K) * scale  
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale 
    
    return dQ, dK, dV


class FlashAttention2_torch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FA-2 in PyTorch.
        话说uv test接口的_make_attn_inputs函数传入参数是(batch_size, seq_len, head_dim),
        但是评测的时候用(batch_size, heads, seq_len, head_dim), 实际上batch==1并且batch维度被消去了, 传入(heads, seq_len, head_dim), 

        batch_size和heads混用感觉并不是很严谨, 但为了适配test和评测接口, 我也先把heads当batch用了. 
        这样无法处理多batch且多heads情况, 日后再改.

        Args:
            Q: (batch_size/heads, seq_len_q, head_dim)
            K: (batch_size/heads, seq_len_k, head_dim)
            V: (batch_size/heads, seq_len_k, head_dim)
            is_causal: bool, whether to apply causal masking
        Returns:
            O: (batch_size/heads, seq_len_q, head_dim)
        """
        
        batch_size, seq_len_q, head_dim = Q.shape
        _, seq_len_k, _ = K.shape
        
        Bq = 16
        Bk = 16
        
        Tq = math.ceil(seq_len_q / Bq)
        Tk = math.ceil(seq_len_k / Bk)
        
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, seq_len_q, device=Q.device, dtype=Q.dtype)
        
       
        # for b in range(batch_size): # very low efficient, but 完全符合fa2逻辑
        #     for i in range(Tq):
        #         q_start = i * Bq
        #         q_end = min((i + 1) * Bq, seq_len_q)
        #         Qi = Q[b:b+1, q_start:q_end, :]  #(1, Bq, head_dim) 按照pdf的指导, 慢的钥匙
                
        #         Oi = torch.zeros_like(Qi)
        #         li = torch.zeros(1, q_end - q_start, device=Q.device, dtype=Q.dtype)
        #         mi = torch.full((1, q_end - q_start), float('-inf'), device=Q.device, dtype=Q.dtype)
                
        #         for j in range(Tk):
        #             k_start = j * Bk
        #             k_end = min((j + 1) * Bk, seq_len_k)
        #             Kj = K[b:b+1, k_start:k_end, :]  # (1, Bk, head_dim)
        #             Vj = V[b:b+1, k_start:k_end, :]  # (1, Bk, head_dim)
                    
        #             Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(head_dim) # (1, Bq, Bk)
                    
        #             # 对上三角矩阵mask为-inf, 其实相当保留下单角矩阵.
        #             if is_causal:
        #                 causal_mask = torch.triu(
        #                     torch.ones(q_end - q_start, k_end - k_start, device=Q.device),
        #                     diagonal=k_start - q_start + 1 # now 是可以保留注意力的
        #                 ).bool()
        #                 Sij = Sij.masked_fill(causal_mask, float('-inf'))
                    
        #             mi_new = torch.maximum(mi, Sij.max(dim=-1).values)
        #             Pij_tilde = torch.exp(Sij - mi_new.unsqueeze(-1))
        #             li_new = torch.exp(mi - mi_new) * li + Pij_tilde.sum(dim=-1)
        #             Oi = torch.exp(mi - mi_new).unsqueeze(-1) * Oi + torch.matmul(Pij_tilde, Vj)
                    
        #             mi = mi_new
        #             li = li_new
                
        #         Oi = Oi / li.unsqueeze(-1)
        #         Li = mi + torch.log(li)
                
        #         O[b:b+1, q_start:q_end, :] = Oi
        #         L[b:b+1, q_start:q_end] = Li
        
        # ctx.save_for_backward(Q, K, V, O, L)
        # ctx.is_causal = is_causal
        # ctx.scale = 1 / math.sqrt(head_dim)
        # return O
        
        for i in range(Tq):
            q_start = i * Bq
            q_end = min((i + 1) * Bq, seq_len_q)
            Qi = Q[:, q_start:q_end, :]  # (batch_size, Bq, head_dim)
            
            Oi = torch.zeros_like(Qi)
            li = torch.zeros(batch_size, q_end - q_start, device=Q.device, dtype=Q.dtype)
            mi = torch.full((batch_size, q_end - q_start), float('-inf'), device=Q.device, dtype=Q.dtype)
            
            for j in range(Tk):
                k_start = j * Bk
                k_end = min((j + 1) * Bk, seq_len_k)
                Kj = K[:, k_start:k_end, :]  # (batch_size, Bk, head_dim)
                Vj = V[:, k_start:k_end, :]  # (batch_size, Bk, head_dim)
                
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(head_dim) # (batch_size, Bq, Bk)
                
                # 对上三角矩阵mask为-inf, 其实相当保留下单角矩阵.
                if is_causal:
                    causal_mask = torch.triu(
                        torch.ones(q_end - q_start, k_end - k_start, device=Q.device),
                        diagonal=k_start - q_start + 1 # now 是可以保留注意力的
                    ).bool()
                    Sij = Sij.masked_fill(causal_mask, float('-inf'))
                
                mi_new = torch.maximum(mi, Sij.max(dim=-1).values)
                Pij_tilde = torch.exp(Sij - mi_new.unsqueeze(-1))
                li_new = torch.exp(mi - mi_new) * li + Pij_tilde.sum(dim=-1)
                Oi = torch.exp(mi - mi_new).unsqueeze(-1) * Oi + torch.matmul(Pij_tilde, Vj)
                
                mi = mi_new
                li = li_new
            
            Oi = Oi / li.unsqueeze(-1)
            Li = mi + torch.log(li)
            
            O[:, q_start:q_end, :] = Oi
            L[:, q_start:q_end] = Li
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = 1 / math.sqrt(head_dim)
        return O
    
    
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        dQ, dK, dV = flash_attention_backward(
            Q, K, V, O, L, grad_out, scale, is_causal
        )
        return dQ, dK, dV, None