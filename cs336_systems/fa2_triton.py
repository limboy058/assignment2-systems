import torch
import triton
import triton.language as tl
import math



# @triton.autotune(
#     configs=[
#         triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=4),
#         triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=4),
#         triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=8),
#     ],
#     key=['N_QUERIES', 'N_KEYS', 'D'],
# )
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    ) # 暗藏了一个转置
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_ptr = L_ptr + batch_index * stride_lb + query_tile_index * Q_TILE_SIZE * stride_lq
    
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1))  # (Q_TILE_SIZE, D)
    # 保证是 float32
    Oi = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    li = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    mi = tl.full([Q_TILE_SIZE], float('-inf'), dtype=tl.float32)
    
    q_start = query_tile_index * Q_TILE_SIZE
    
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if IS_CAUSAL: #不处理一定mask的块
        q_start = query_tile_index * Q_TILE_SIZE
        max_k_tile = tl.cdiv(q_start + Q_TILE_SIZE, K_TILE_SIZE)
        num_key_tiles = tl.minimum(num_key_tiles, max_k_tile)
        
    
    

    for j in range(num_key_tiles):
        k_start = j * K_TILE_SIZE
        
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1))  # (D, K_TILE_SIZE)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1))  # (K_TILE_SIZE, D)
        
        # Score = Q @ K^T / sqrt(d)
        Sij = tl.dot(Qi, Kj)  # allow_tf32=False (Q_TILE_SIZE, K_TILE_SIZE)
        Sij = Sij * scale

        if IS_CAUSAL:
            # 还原q和k的位置
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            # 在 q_idx < k_idx 处 mask. 广播
            causal_mask = q_idx[:, None] < k_idx[None, :]
            Sij = tl.where(causal_mask, float('-inf'), Sij)
        
        mi_new = tl.maximum(mi, tl.max(Sij, axis=1))
        Pij_ = tl.exp(Sij - mi_new[:, None])
        li_new = tl.exp(mi - mi_new) * li + tl.sum(Pij_, axis=1)
        
        # 先将 P 转换为 V 的 dtype
        #tl.device_print("Pij_dtype", Pij_.dtype.__dict__)  
        Pij_ = Pij_.to(Vj.dtype) 
        #tl.device_print("Pij_dtype_after", Pij_.dtype)  
        
        # alpha = tl.exp(mi - mi_new)[:, None]
        # Oi = alpha * Oi
        # Oi = Oi + tl.dot(Pij_, Vj) #, allow_tf32=False
        
        # 使用 acc ,更快
        Oi = tl.dot(Pij_, Vj, acc=tl.exp(mi - mi_new)[:, None] * Oi)
        
        mi = mi_new
        li = li_new
        
        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    Oi = Oi / li[:, None] # (64*64)
    #tl.device_print("Oi_shape", Oi.shape[1]) 
    Oi = Oi.to(O_block_ptr.type.element_ty) 
    tl.store(O_block_ptr, Oi, boundary_check=(0, 1))
    
    Li = mi + tl.log(li) # (64,)
    # tl.device_print("Li_shape", Li.shape[1])  
    l_offsets = tl.arange(0, Q_TILE_SIZE) * stride_lq  #我们只需要对L做一维上的操作, 而不是像Q做二维的操作, 所以直接使用原始的指针, 而非L_block_ptr(你也make不出来)
    q_mask = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)) < N_QUERIES #手动mask一下
    tl.store(L_ptr + l_offsets, Li, mask=q_mask)



@torch.compile
def flash_attention_backward_compiled(Q, K, V, O, L, dO, scale, is_causal=False):
    """
    FA-2 bwd in torch.compile   
    参考公式13-19. 关于注意力层的反向传播可能还有点点问题
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

    P = torch.exp(S - L.unsqueeze(-1))  
    #P = torch.softmax(S, dim=-1)
    dV = torch.matmul(P.transpose(-2, -1), dO)  
    dP = torch.matmul(dO, V.transpose(-2, -1))  

    dS = P * (dP - D) 
    # 理论上, 等价于dS = A * (dA - (A * dA).sum(dim=-1, keepdim=True))  
    if is_causal:
        dS = dS.masked_fill(causal_mask, 0.0)
        
    dQ = torch.matmul(dS, K) * scale  
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale 
    
    return dQ, dK, dV



class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FA-2 in Triton.

        PS: uv test接口的_make_attn_inputs函数传入参数是(batch_size, seq_len, head_dim),
        但是评测的时候用(batch_size, heads, seq_len, head_dim), 实际上batch==1并且batch维度被消去了, 传入(heads, seq_len, head_dim), 

        batch_size和heads混用感觉并不是很严谨, 但为了适配test和评测接口, 我也先把heads当batch用了. 
        这样无法处理多batch且多heads情况, 日后我再改吧.

        Args:
            Q: (batch_size/heads, seq_len_q, head_dim)
            K: (batch_size/heads, seq_len_k, head_dim)
            V: (batch_size/heads, seq_len_k, head_dim)
            is_causal: bool
        Returns:
            O: (batch_size/heads, seq_len_q, head_dim)
        """
        batch_size, seq_len_q, head_dim = Q.shape
        _, seq_len_k, _ = K.shape
        
        #如果设置为64无法通过1e-3评测.(max diff =  1.432449e-03)
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32
        
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, seq_len_q, device=Q.device, dtype=Q.dtype)
        
        scale = 1.0 / math.sqrt(head_dim)
        
        num_query_tiles = triton.cdiv(seq_len_q, Q_TILE_SIZE)
        
        grid = (num_query_tiles, batch_size, )
        
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_len_q, seq_len_k,
            scale,
            is_causal,
            head_dim,
            Q_TILE_SIZE,
            K_TILE_SIZE,
        )
        
        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        
        return O
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        dQ, dK, dV = flash_attention_backward_compiled(
            Q, K, V, O, L, grad_out, scale, is_causal
        )
        return dQ, dK, dV, None