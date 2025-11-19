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



@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    FlashAttention-2 Backward Kernel
    Grid: (num_key_tiles, batch_size)
    每个 program 处理一个 K tile，遍历所有 Q tiles
    """
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    k_start = key_tile_index * K_TILE_SIZE
    
    # 初始化 K, V block pointers
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # Load K(j), V(j)
    Kj = tl.load(K_block_ptr, boundary_check=(0, 1))
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1))
    
    Kj = Kj.to(tl.float32)
    Vj = Vj.to(tl.float32)
    
    # Initialize dK(j) = 0, dV(j) = 0
    dKj = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    dVj = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    
    num_query_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    
    # Causal masking: 只处理需要的 Q tiles
    if IS_CAUSAL:
        # 只有当 q_end > k_start 时才需要计算
        min_q_tile = tl.maximum(0, key_tile_index)
        start_q_tile = min_q_tile
    else:
        start_q_tile = 0
    
    # Loop over Q tiles
    for i in range(start_q_tile, num_query_tiles):
        q_start = i * Q_TILE_SIZE
        
        # 初始化 Q, O, dO block pointers
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        
        # Load Qi, Oi, dOi
        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1))
        Oi = tl.load(O_block_ptr, boundary_check=(0, 1))
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1))
        
        Qi = Qi.to(tl.float32)
        Oi = Oi.to(tl.float32)
        dOi = dOi.to(tl.float32)
        
        # Load Li, compute Di
        L_ptr_offset = batch_index * stride_lb + q_start * stride_lq
        l_offsets = tl.arange(0, Q_TILE_SIZE) * stride_lq
        q_mask = (q_start + tl.arange(0, Q_TILE_SIZE)) < N_QUERIES
        Li = tl.load(L_ptr + L_ptr_offset + l_offsets, mask=q_mask, other=0.0)
        
        # Compute Di = rowsum(dOi ◦ Oi)
        Di = tl.sum(dOi * Oi, axis=1)  # (Q_TILE_SIZE,)
        
        # Compute S(j)i = Qi @ Kj^T / sqrt(d)
        Sij = tl.dot(Qi, Kj.trans())  # (Q_TILE_SIZE, K_TILE_SIZE)
        Sij = Sij * scale
        
        # Apply causal mask
        if IS_CAUSAL:
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] < k_idx[None, :]
            Sij = tl.where(causal_mask, float('-inf'), Sij)
        
        # Compute P(j)i = exp(S(j)i - Li)
        Pij = tl.exp(Sij - Li[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        
        # Compute dV(j) += P(j)i^T @ dOi
        dVj += tl.dot(Pij.trans(), dOi)
        
        # Compute dP(j)i = dOi @ Vj^T
        dPij = tl.dot(dOi, Vj.trans())  # (Q_TILE_SIZE, K_TILE_SIZE)
        
        # Compute dS(j)i = P(j)i ◦ (dP(j)i - Di)
        dSij = Pij * (dPij - Di[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        
        # Apply causal mask to dSij
        if IS_CAUSAL:
            dSij = tl.where(causal_mask, 0.0, dSij)
        
        # Compute dQi = dS(j)i @ K(j)
        dQi = tl.dot(dSij, Kj) * scale  # (Q_TILE_SIZE, D)
        
        # Atomic add to dQ (需要原子操作)
        dQ_block_ptr = tl.make_block_ptr(
            dQ_ptr + batch_index * stride_dqb,
            shape=(N_QUERIES, D),
            strides=(stride_dqq, stride_dqd),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        
        # 使用 atomic_add（如果需要）
        # 由于每个 K tile 对应不同的 program，对同一个 Q tile 的 dQ 需要累加
        tl.atomic_add(dQ_block_ptr, dQi.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))
        
        # Compute dK(j) += dS(j)i^T @ Qi
        dKj += tl.dot(dSij.trans(), Qi) * scale
    
    # Write dK(j), dV(j) to global memory
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dKj = dKj.to(dK_block_ptr.type.element_ty)
    dVj = dVj.to(dV_block_ptr.type.element_ty)
    
    tl.store(dK_block_ptr, dKj, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj, boundary_check=(0, 1))


# 修改 FlashAttention2 的 backward 方法
class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # ... 保持原有 forward 代码不变 ...
        batch_size, seq_len_q, head_dim = Q.shape
        _, seq_len_k, _ = K.shape
        
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32
        
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, seq_len_q, device=Q.device, dtype=Q.dtype)
        
        scale = 1.0 / math.sqrt(head_dim)
        
        num_query_tiles = triton.cdiv(seq_len_q, Q_TILE_SIZE)
        
        grid = (num_query_tiles, batch_size, )
        
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_len_q, seq_len_k,
            scale, is_causal, head_dim,
            Q_TILE_SIZE, K_TILE_SIZE,
        )
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        
        return O
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE
        
        batch_size, seq_len_q, head_dim = Q.shape
        _, seq_len_k, _ = K.shape
        
        # 初始化梯度
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        num_key_tiles = triton.cdiv(seq_len_k, K_TILE_SIZE)
        
        grid = (num_key_tiles, batch_size, )
        
        flash_bwd_kernel[grid](
            Q, K, V, O, L,
            grad_out, dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            seq_len_q, seq_len_k,
            scale, is_causal, head_dim,
            Q_TILE_SIZE, K_TILE_SIZE,
        )
        
        return dQ, dK, dV, None