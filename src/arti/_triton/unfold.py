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
    def _query_context_kernel(
        x_ptr,
        keys_ptr,
        queries_ptr,
        mask_ptr,
        output_ptr,
        stride_x_batch,
        stride_x_token,
        stride_x_dim,
        stride_keys_batch,
        stride_keys_token,
        stride_keys_hidden,
        stride_queries_exposed,
        stride_queries_hidden,
        stride_mask_batch,
        stride_mask_token,
        stride_output_batch,
        stride_output_exposed,
        stride_output_dim,
        EXPOSED: tl.constexpr,
        TOKENS: tl.constexpr,
        DIM: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_EXPOSED: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        batch = tl.program_id(0)
        exposed_block = tl.program_id(1)
        exposed_offsets = exposed_block * BLOCK_EXPOSED + tl.arange(0, BLOCK_EXPOSED)
        token_offsets = tl.arange(0, BLOCK_TOKENS)
        dim_offsets = tl.arange(0, BLOCK_DIM)
        hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
        exposed_mask = exposed_offsets < EXPOSED
        token_bounds = token_offsets < TOKENS
        dim_mask = dim_offsets < DIM
        hidden_mask = hidden_offsets < HIDDEN

        query_offsets = (
            exposed_offsets[:, None] * stride_queries_exposed
            + hidden_offsets[None, :] * stride_queries_hidden
        )
        queries = tl.load(
            queries_ptr + query_offsets,
            mask=exposed_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        key_offsets = (
            batch * stride_keys_batch
            + token_offsets[None, :] * stride_keys_token
            + hidden_offsets[:, None] * stride_keys_hidden
        )
        keys = tl.load(
            keys_ptr + key_offsets,
            mask=hidden_mask[:, None] & token_bounds[None, :],
            other=0.0,
        )
        logits = tl.dot(queries, keys, input_precision="ieee") / tl.sqrt(
            HIDDEN * 1.0
        )
        valid_tokens = tl.load(
            mask_ptr + batch * stride_mask_batch + token_offsets * stride_mask_token,
            mask=token_bounds,
            other=0,
        ).to(tl.int1)
        valid = exposed_mask[:, None] & valid_tokens[None, :]
        logits = tl.where(valid, logits, -1.0e6)
        logits = logits - tl.max(logits, axis=1)[:, None]
        weights = tl.exp(logits) * valid.to(tl.float32)
        denominator = tl.sum(weights, axis=1)[:, None]
        weights = weights / tl.where(denominator > 0.0, denominator, 1.0)

        x_offsets = (
            batch * stride_x_batch
            + token_offsets[:, None] * stride_x_token
            + dim_offsets[None, :] * stride_x_dim
        )
        values = tl.load(
            x_ptr + x_offsets,
            mask=token_bounds[:, None] & dim_mask[None, :],
            other=0.0,
        )
        attended = tl.dot(weights.to(values.dtype), values, input_precision="ieee")
        output_offsets = (
            batch * stride_output_batch
            + exposed_offsets[:, None] * stride_output_exposed
            + dim_offsets[None, :] * stride_output_dim
        )
        tl.store(
            output_ptr + output_offsets,
            attended,
            mask=exposed_mask[:, None] & dim_mask[None, :],
        )

    @triton.jit
    def _slot_emit_kernel(
        attended_ptr,
        operators_ptr,
        mix_ptr,
        scale_ptr,
        bias_ptr,
        output_ptr,
        stride_attended_batch,
        stride_attended_exposed,
        stride_attended_dim,
        stride_operators_operator,
        stride_operators_input,
        stride_operators_output,
        stride_mix_exposed,
        stride_mix_operator,
        stride_slot_exposed,
        stride_slot_dim,
        stride_output_batch,
        stride_output_exposed,
        stride_output_dim,
        EXPOSED: tl.constexpr,
        OPERATORS: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_EXPOSED: tl.constexpr,
        BLOCK_INPUT: tl.constexpr,
        BLOCK_OUTPUT: tl.constexpr,
    ):
        batch = tl.program_id(0)
        exposed_block = tl.program_id(1)
        output_block = tl.program_id(2)
        exposed_offsets = exposed_block * BLOCK_EXPOSED + tl.arange(0, BLOCK_EXPOSED)
        input_offsets = tl.arange(0, BLOCK_INPUT)
        output_offsets = output_block * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
        exposed_mask = exposed_offsets < EXPOSED
        input_mask = input_offsets < DIM
        output_mask = output_offsets < DIM

        attended_offsets = (
            batch * stride_attended_batch
            + exposed_offsets[:, None] * stride_attended_exposed
            + input_offsets[None, :] * stride_attended_dim
        )
        attended = tl.load(
            attended_ptr + attended_offsets,
            mask=exposed_mask[:, None] & input_mask[None, :],
            other=0.0,
        )
        accumulator = tl.zeros((BLOCK_EXPOSED, BLOCK_OUTPUT), dtype=tl.float32)
        for operator in range(OPERATORS):
            operator_offsets = (
                operator * stride_operators_operator
                + input_offsets[:, None] * stride_operators_input
                + output_offsets[None, :] * stride_operators_output
            )
            operator_values = tl.load(
                operators_ptr + operator_offsets,
                mask=input_mask[:, None] & output_mask[None, :],
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
                tl.dot(attended, operator_values, input_precision="ieee")
                * operator_mix[:, None]
            )

        slot_offsets = (
            exposed_offsets[:, None] * stride_slot_exposed
            + output_offsets[None, :] * stride_slot_dim
        )
        scale = tl.load(
            scale_ptr + slot_offsets,
            mask=exposed_mask[:, None] & output_mask[None, :],
            other=0.0,
        )
        bias = tl.load(
            bias_ptr + slot_offsets,
            mask=exposed_mask[:, None] & output_mask[None, :],
            other=0.0,
        )
        destination = (
            batch * stride_output_batch
            + exposed_offsets[:, None] * stride_output_exposed
            + output_offsets[None, :] * stride_output_dim
        )
        tl.store(
            output_ptr + destination,
            accumulator * scale + bias,
            mask=exposed_mask[:, None] & output_mask[None, :],
        )

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


def _query_slot_values_unchecked(
    x: Tensor,
    keys: Tensor,
    queries: Tensor,
    mask: Tensor,
    operators: Tensor,
    mix: Tensor,
    scale: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fused inference query and exact slot-operator emission hot path."""

    batch, tokens, dim = x.shape
    exposed, hidden = queries.shape
    operator_count = operators.shape[0]
    attended = torch.empty(batch, exposed, dim, device=x.device, dtype=x.dtype)
    block_exposed = 16
    block_tokens = max(16, triton.next_power_of_2(tokens))
    block_dim = max(16, triton.next_power_of_2(dim))
    block_hidden = max(16, triton.next_power_of_2(hidden))
    query_grid = (batch, triton.cdiv(exposed, block_exposed))
    _query_context_kernel[query_grid](
        x,
        keys,
        queries,
        mask,
        attended,
        *x.stride(),
        *keys.stride(),
        *queries.stride(),
        *mask.stride(),
        *attended.stride(),
        EXPOSED=exposed,
        TOKENS=tokens,
        DIM=dim,
        HIDDEN=hidden,
        BLOCK_EXPOSED=block_exposed,
        BLOCK_TOKENS=block_tokens,
        BLOCK_DIM=block_dim,
        BLOCK_HIDDEN=block_hidden,
        num_warps=8 if dim >= 256 else 4,
        num_stages=1,
    )

    output = torch.empty_like(attended)
    block_output = 32
    emit_grid = (
        batch,
        triton.cdiv(exposed, block_exposed),
        triton.cdiv(dim, block_output),
    )
    _slot_emit_kernel[emit_grid](
        attended,
        operators,
        mix,
        scale,
        bias,
        output,
        *attended.stride(),
        *operators.stride(),
        *mix.stride(),
        *scale.stride(),
        *output.stride(),
        EXPOSED=exposed,
        OPERATORS=operator_count,
        DIM=dim,
        BLOCK_EXPOSED=block_exposed,
        BLOCK_INPUT=block_dim,
        BLOCK_OUTPUT=block_output,
        num_warps=4,
        num_stages=1,
    )
    return output, attended
