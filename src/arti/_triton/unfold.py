from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by dependency-isolation tests
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _palette_emit_kernel(
        weights_ptr,
        transformed_ptr,
        mix_ptr,
        scale_ptr,
        bias_ptr,
        output_ptr,
        stride_weights_batch,
        stride_weights_exposed,
        stride_weights_token,
        stride_transformed_batch,
        stride_transformed_operator,
        stride_transformed_token,
        stride_transformed_dim,
        stride_mix_exposed,
        stride_mix_operator,
        stride_slot_exposed,
        stride_slot_dim,
        stride_output_batch,
        stride_output_exposed,
        stride_output_dim,
        EXPOSED: tl.constexpr,
        TOKENS: tl.constexpr,
        OPERATORS: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_EXPOSED: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        batch = tl.program_id(0)
        exposed_block = tl.program_id(1)
        dim_block = tl.program_id(2)
        exposed_offsets = exposed_block * BLOCK_EXPOSED + tl.arange(0, BLOCK_EXPOSED)
        token_offsets = tl.arange(0, BLOCK_TOKENS)
        dim_offsets = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        exposed_mask = exposed_offsets < EXPOSED
        token_mask = token_offsets < TOKENS
        dim_mask = dim_offsets < DIM

        weight_offsets = (
            batch * stride_weights_batch
            + exposed_offsets[:, None] * stride_weights_exposed
            + token_offsets[None, :] * stride_weights_token
        )
        weights = tl.load(
            weights_ptr + weight_offsets,
            mask=exposed_mask[:, None] & token_mask[None, :],
            other=0.0,
        )
        accumulator = tl.zeros((BLOCK_EXPOSED, BLOCK_DIM), dtype=tl.float32)
        for operator in range(OPERATORS):
            transformed_offsets = (
                batch * stride_transformed_batch
                + operator * stride_transformed_operator
                + token_offsets[:, None] * stride_transformed_token
                + dim_offsets[None, :] * stride_transformed_dim
            )
            transformed = tl.load(
                transformed_ptr + transformed_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            operator_mix = tl.load(
                mix_ptr
                + exposed_offsets * stride_mix_exposed
                + operator * stride_mix_operator,
                mask=exposed_mask,
                other=0.0,
            )
            accumulator += (
                tl.dot(weights, transformed, input_precision="ieee")
                * operator_mix[:, None]
            )

        slot_offsets = (
            exposed_offsets[:, None] * stride_slot_exposed
            + dim_offsets[None, :] * stride_slot_dim
        )
        scale = tl.load(
            scale_ptr + slot_offsets,
            mask=exposed_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        bias = tl.load(
            bias_ptr + slot_offsets,
            mask=exposed_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        output_offsets = (
            batch * stride_output_batch
            + exposed_offsets[:, None] * stride_output_exposed
            + dim_offsets[None, :] * stride_output_dim
        )
        tl.store(
            output_ptr + output_offsets,
            accumulator * scale + bias,
            mask=exposed_mask[:, None] & dim_mask[None, :],
        )


def is_available() -> bool:
    return bool(triton is not None and torch.cuda.is_available())


def palette_values(
    x: Tensor,
    weights: Tensor,
    operators: Tensor,
    mix: Tensor,
    scale: Tensor,
    bias: Tensor,
) -> Tensor:
    """Execute the exact inference-only UnFold palette contraction."""

    if not is_available():
        raise RuntimeError("the optional Triton runtime is unavailable")
    if x.ndim != 3 or weights.ndim != 3 or operators.ndim != 3:
        raise ValueError("x, weights, and operators must be rank-3 tensors")
    batch, tokens, dim = x.shape
    weight_batch, exposed, weight_tokens = weights.shape
    operator_count, operator_in, operator_out = operators.shape
    if (weight_batch, weight_tokens) != (batch, tokens):
        raise ValueError("weights must have shape [B, E, N] matching x")
    if (operator_in, operator_out) != (dim, dim):
        raise ValueError("operators must have shape [R, D, D] matching x")
    if mix.shape != (exposed, operator_count):
        raise ValueError("mix must have shape [E, R]")
    if scale.shape != (exposed, dim) or bias.shape != (exposed, dim):
        raise ValueError("scale and bias must have shape [E, D]")
    tensors = (x, weights, operators, mix, scale, bias)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Triton palette tensors must be CUDA tensors")
    if any(tensor.device != x.device or tensor.dtype != x.dtype for tensor in tensors):
        raise ValueError("Triton palette tensors must share one device and dtype")
    if x.dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("Triton palette supports float32 and bfloat16")
    if tokens > 128:
        raise ValueError("Triton palette supports at most 128 input slots")
    if operator_count > 16:
        raise ValueError("Triton palette supports at most 16 operators")

    return _palette_values_unchecked(x, weights, operators, mix, scale, bias)


def _palette_values_unchecked(
    x: Tensor,
    weights: Tensor,
    operators: Tensor,
    mix: Tensor,
    scale: Tensor,
    bias: Tensor,
) -> Tensor:
    """Hot path for callers that already enforce the palette contract."""

    batch, tokens, dim = x.shape
    exposed = weights.shape[1]
    operator_count = operators.shape[0]
    transformed = torch.einsum("bnd,rdh->brnh", x, operators)
    output = torch.empty(batch, exposed, dim, device=x.device, dtype=x.dtype)
    block_tokens = max(16, triton.next_power_of_2(tokens))
    block_exposed = 16
    block_dim = 32
    grid = (batch, triton.cdiv(exposed, block_exposed), triton.cdiv(dim, block_dim))
    _palette_emit_kernel[grid](
        weights,
        transformed,
        mix,
        scale,
        bias,
        output,
        *weights.stride(),
        *transformed.stride(),
        *mix.stride(),
        *scale.stride(),
        *output.stride(),
        EXPOSED=exposed,
        TOKENS=tokens,
        OPERATORS=operator_count,
        DIM=dim,
        BLOCK_EXPOSED=block_exposed,
        BLOCK_TOKENS=block_tokens,
        BLOCK_DIM=block_dim,
        num_warps=4,
    )
    return output
