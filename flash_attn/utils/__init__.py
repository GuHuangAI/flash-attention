from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from flash_attn import flash_attn_varlen_func

__all__ = ["flash_attention"]


def _to_half_precision(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype in (torch.float16, torch.bfloat16):
        return x
    return x.to(dtype)


def _build_cu_seqlens(lens: Optional[Sequence[int]], max_len: int, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if lens is None:
        cu = torch.arange(0, (batch + 1) * max_len, max_len, dtype=torch.int32, device=device)
        lengths = torch.full((batch,), max_len, dtype=torch.int32, device=device)
    else:
        lengths = torch.as_tensor(lens, dtype=torch.int32, device=device)
        cu = torch.empty(batch + 1, dtype=torch.int32, device=device)
        cu[0] = 0
        torch.cumsum(lengths, dim=0, out=cu[1:])
    return cu, lengths


def _flatten_varlen(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat([x_i[:length] for x_i, length in zip(x, lengths.tolist())], dim=0)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[Sequence[int]] = None,
    k_lens: Optional[Sequence[int]] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Apply FlashAttention with optional custom attention mask.

    Args:
        q: ``(batch, seqlen_q, num_heads, head_dim)`` query tensor.
        k: ``(batch, seqlen_k, num_heads, head_dim)`` key tensor.
        v: ``(batch, seqlen_k, num_heads, value_dim)`` value tensor.
        q_lens: Optional per-example query lengths.
        k_lens: Optional per-example key/value lengths.
        attn_mask: Optional attention mask of shape ``(batch, num_heads, max_seqlen_q, max_seqlen_k)``.
            Bool masks mark valid positions with ``True``. Float masks are added to the attention
            logits directly.
        dropout_p: Dropout probability.
        softmax_scale: Optional scale factor for attention logits.
        q_scale: Optional scale applied directly to ``q`` before attention.
        causal: Whether to apply a causal mask.
        window_size: Local attention window. Only ``(-1, -1)`` is supported when ``attn_mask`` is provided.
        deterministic: Whether to use deterministic backward computation.
        dtype: Working dtype for non-half precision inputs.
    """

    if q.device.type != "cuda":
        raise ValueError("FlashAttention requires CUDA tensors")

    half_dtype = dtype
    if half_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("dtype must be float16 or bfloat16")

    batch, seqlen_q, num_heads_q, head_dim = q.shape
    seqlen_k = k.size(1)
    num_heads_k = k.size(2)

    if attn_mask is not None:
        if window_size != (-1, -1):
            raise ValueError("Custom attention masks do not support sliding window attention")
        if attn_mask.device != q.device:
            attn_mask = attn_mask.to(q.device)
        if attn_mask.dim() != 4:
            raise ValueError("attn_mask must have shape (batch, num_heads, seqlen_q, seqlen_k)")
        if attn_mask.shape[0] != batch or attn_mask.shape[1] != num_heads_q:
            raise ValueError("attn_mask batch/head dimensions must match queries")
        if attn_mask.shape[2] != seqlen_q or attn_mask.shape[3] != seqlen_k:
            raise ValueError("attn_mask spatial dimensions must match input sequence lengths")
        attn_mask = attn_mask.contiguous()
        if attn_mask.dtype not in (torch.bool, torch.float32):
            attn_mask = attn_mask.to(torch.float32)

    cu_q, q_lengths = _build_cu_seqlens(q_lens, seqlen_q, batch, q.device)
    cu_k, k_lengths = _build_cu_seqlens(k_lens, seqlen_k, batch, k.device)

    if q_lens is None:
        q_flat = _to_half_precision(q.reshape(batch * seqlen_q, num_heads_q, head_dim), half_dtype)
    else:
        q_flat = _to_half_precision(_flatten_varlen(q, q_lengths), half_dtype)

    if k_lens is None:
        k_flat = _to_half_precision(k.reshape(batch * seqlen_k, num_heads_k, head_dim), half_dtype)
        v_flat = _to_half_precision(v.reshape(batch * seqlen_k, num_heads_k, v.size(-1)), half_dtype)
    else:
        k_flat = _to_half_precision(_flatten_varlen(k, k_lengths), half_dtype)
        v_flat = _to_half_precision(_flatten_varlen(v, k_lengths), half_dtype)

    if q_scale is not None:
        q_flat = q_flat * q_scale

    out = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_q,
        cu_k,
        seqlen_q,
        seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=0.0,
        alibi_slopes=None,
        attn_mask=attn_mask,
        deterministic=deterministic,
        return_attn_probs=False,
        block_table=None,
    )

    out = out.to(q.dtype)
    if q_lens is None:
        return out.reshape(batch, seqlen_q, num_heads_q, out.size(-1))

    segments = []
    offset = 0
    value_dim = out.size(-1)
    for length in q_lengths.tolist():
        segments.append(out[offset : offset + length])
        offset += length
    padded = torch.zeros((batch, seqlen_q, num_heads_q, value_dim), dtype=out.dtype, device=out.device)
    for i, length in enumerate(q_lengths.tolist()):
        if length > 0:
            padded[i, :length] = segments[i]
    return padded.to(q.dtype)
