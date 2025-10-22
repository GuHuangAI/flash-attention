import pytest
import torch
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_func


def _prepare_varlen_inputs(q, k, v):
    batch, seqlen_q, nheads, dim = q.shape
    seqlen_k = k.size(1)
    cu_seqlens_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.arange(0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=q.device)
    q_flat = q.reshape(batch * seqlen_q, nheads, dim)
    k_flat = k.reshape(batch * seqlen_k, nheads, dim)
    v_flat = v.reshape(batch * seqlen_k, nheads, v.size(-1))
    return cu_seqlens_q, cu_seqlens_k, q_flat, k_flat, v_flat


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_mask_matches_sdpa(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = torch.device("cuda")
    batch, seqlen_q, seqlen_k, nheads, dim = 2, 11, 13, 4, 64

    base_q = torch.randn(batch, seqlen_q, nheads, dim, device=device, dtype=dtype)
    base_k = torch.randn(batch, seqlen_k, nheads, dim, device=device, dtype=dtype)
    base_v = torch.randn(batch, seqlen_k, nheads, dim, device=device, dtype=dtype)

    q_fa = base_q.clone().requires_grad_(True)
    k_fa = base_k.clone().requires_grad_(True)
    v_fa = base_v.clone().requires_grad_(True)

    q_ref_base = base_q.clone().requires_grad_(True)
    k_ref_base = base_k.clone().requires_grad_(True)
    v_ref_base = base_v.clone().requires_grad_(True)

    attn_mask = torch.rand(batch, nheads, seqlen_q, seqlen_k, device=device) > 0.4

    cu_q, cu_k, q_flat, k_flat, v_flat = _prepare_varlen_inputs(q_fa, k_fa, v_fa)

    out = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_q,
        cu_k,
        seqlen_q,
        seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        attn_mask=attn_mask,
        deterministic=True,
        return_attn_probs=False,
        block_table=None,
    )
    out = out.reshape(batch, seqlen_q, nheads, dim)

    q_ref = q_ref_base.permute(0, 2, 1, 3).reshape(batch * nheads, seqlen_q, dim)
    k_ref = k_ref_base.permute(0, 2, 1, 3).reshape(batch * nheads, seqlen_k, dim)
    v_ref = v_ref_base.permute(0, 2, 1, 3).reshape(batch * nheads, seqlen_k, dim)
    mask_ref = attn_mask.reshape(batch * nheads, seqlen_q, seqlen_k)
    ref = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=mask_ref,
        dropout_p=0.0,
        is_causal=False,
    )
    ref = ref.reshape(batch, nheads, seqlen_q, dim).permute(0, 2, 1, 3)

    atol = 5e-3 if dtype is torch.bfloat16 else 2e-3
    rtol = 5e-3 if dtype is torch.bfloat16 else 2e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)

    torch.testing.assert_close(
        q_fa.grad,
        q_ref.grad.reshape(batch, nheads, seqlen_q, dim).permute(0, 2, 1, 3),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        k_fa.grad,
        k_ref.grad.reshape(batch, nheads, seqlen_k, dim).permute(0, 2, 1, 3),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        v_fa.grad,
        v_ref.grad.reshape(batch, nheads, seqlen_k, dim).permute(0, 2, 1, 3),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_additive_mask(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = torch.device("cuda")
    batch, seqlen_q, seqlen_k, nheads, dim = 1, 7, 9, 3, 40

    q = torch.randn(batch, seqlen_q, nheads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads, dim, device=device, dtype=dtype)

    bool_mask = torch.rand(batch, nheads, seqlen_q, seqlen_k, device=device) > 0.25
    additive_mask = torch.where(bool_mask, torch.zeros(1, device=device), torch.full((), -1e4, device=device))

    cu_q, cu_k, q_flat, k_flat, v_flat = _prepare_varlen_inputs(q, k, v)

    out = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_q,
        cu_k,
        seqlen_q,
        seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        attn_mask=additive_mask,
        deterministic=True,
        return_attn_probs=False,
        block_table=None,
    ).reshape(batch, seqlen_q, nheads, dim)

    q_ref = q.permute(0, 2, 1, 3).reshape(batch * nheads, seqlen_q, dim)
    k_ref = k.permute(0, 2, 1, 3).reshape(batch * nheads, seqlen_k, dim)
    v_ref = v.permute(0, 2, 1, 3).reshape(batch * nheads, seqlen_k, dim)
    mask_ref = additive_mask.reshape(batch * nheads, seqlen_q, seqlen_k)
    ref = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=mask_ref,
        dropout_p=0.0,
        is_causal=False,
    ).reshape(batch, nheads, seqlen_q, dim).permute(0, 2, 1, 3)

    atol = 6e-3 if dtype is torch.bfloat16 else 2e-3
    rtol = 6e-3 if dtype is torch.bfloat16 else 2e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
