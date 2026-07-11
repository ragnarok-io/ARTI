"""Lightweight source integrity carriers for multi-source superposition.

The API is designed for LLM-friendly use: source identity is injected inside the
hidden dimension, token count does not change, and diagnostics can be summarized
by block instead of computed as a heavy per-token report.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn


SOURCE_INTEGRITY_MODES = ("off", "pilot", "summary", "full")


@dataclass(frozen=True)
class SourceIntegrityBasis:
    """Fixed carriers and low-rank integrity projections."""

    source_carrier: Tensor
    pilots: Tensor
    checksum_projection: Tensor


@dataclass(frozen=True)
class SourceIntegrityReport:
    """Summary diagnostics for source carrier health."""

    mode: str
    source_count: int
    payload_dim: int
    field_dim: int
    checksum_dim: int
    capacity_ratio: float
    reconstruction_error: Tensor
    pilot_self_score: Tensor
    pilot_cross_score: Tensor
    checksum_error: Tensor
    leakage_matrix: Tensor
    min_pilot_self_score: float
    max_pilot_cross_score: float
    max_checksum_error: float
    max_leakage: float
    valid: bool


class SourceIntegrityCarrier(nn.Module):
    """Module wrapper for fixed source carriers and integrity diagnostics.

    The carrier stores its basis as buffers, so it follows ``.to(device)`` and
    checkpoint serialization like ordinary PyTorch layers. Encoding keeps the
    token axis unchanged; the selected mode only controls diagnostic cost.
    """

    def __init__(
        self,
        source_count: int,
        payload_dim: int,
        field_dim: int | None = None,
        *,
        checksum_dim: int = 8,
        mode: str = "summary",
        block_size: int | None = 256,
        pilot_strength: float = 0.05,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _validate_mode(mode)
        basis = make_source_integrity_basis(
            source_count=source_count,
            payload_dim=payload_dim,
            field_dim=field_dim,
            checksum_dim=checksum_dim,
            dtype=dtype,
        )
        self.mode = mode
        self.block_size = block_size
        self.pilot_strength = pilot_strength
        self.register_buffer("source_carrier", basis.source_carrier)
        self.register_buffer("pilots", basis.pilots)
        self.register_buffer("checksum_projection", basis.checksum_projection)

    @property
    def basis(self) -> SourceIntegrityBasis:
        return SourceIntegrityBasis(
            source_carrier=self.source_carrier,
            pilots=self.pilots,
            checksum_projection=self.checksum_projection,
        )

    def encode_sources(self, x: Tensor, *, mask: Tensor | None = None, pilot: bool | None = None) -> Tensor:
        """Encode synchronized source streams ``[B, S, T, D]`` to ``[B, T, H]``."""

        strength = self.pilot_strength if (self._pilot_enabled(pilot)) else 0.0
        return superpose_sources(x, self.basis, mask=mask, pilot_strength=strength)

    def read_sources(self, field: Tensor) -> Tensor:
        """Read source streams from a carrier field."""

        return read_sources(field, self.basis)

    def encode_tokens(self, hidden: Tensor, source_ids: Tensor) -> Tensor:
        """Encode an LLM token stream without changing token count."""

        return encode_source_tokens(hidden, source_ids, self.basis)

    def decode_tokens(self, field: Tensor, source_ids: Tensor) -> Tensor:
        """Decode an LLM token stream using expected per-token source ids."""

        return decode_source_tokens(field, source_ids, self.basis)

    def report(
        self,
        original: Tensor,
        decoded: Tensor,
        *,
        mode: str | None = None,
        block_size: int | None = None,
    ) -> SourceIntegrityReport | None:
        """Return diagnostics for the selected cost mode.

        ``off`` returns ``None``. ``pilot`` and ``summary`` use block summaries;
        ``full`` checks every token. This keeps the same field representation
        while letting inference choose how much health checking to pay for.
        """

        selected_mode = self.mode if mode is None else mode
        _validate_mode(selected_mode)
        if selected_mode == "off":
            return None
        selected_block_size = block_size if block_size is not None else self.block_size
        if selected_mode == "full":
            selected_block_size = None
        strength = self.pilot_strength if selected_mode in {"pilot", "summary", "full"} else 0.0
        return source_integrity_report(
            original,
            decoded,
            self.basis,
            mode=selected_mode,
            block_size=selected_block_size,
            pilot_strength=strength,
        )

    def forward(self, x: Tensor, *, mask: Tensor | None = None, pilot: bool | None = None) -> Tensor:
        return self.encode_sources(x, mask=mask, pilot=pilot)

    def _pilot_enabled(self, pilot: bool | None) -> bool:
        if pilot is not None:
            return pilot
        return self.mode in {"pilot", "summary", "full"} and self.pilot_strength > 0


def make_source_integrity_basis(
    source_count: int,
    payload_dim: int,
    field_dim: int | None = None,
    *,
    checksum_dim: int = 8,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> SourceIntegrityBasis:
    """Create deterministic orthogonal source carriers and pilots.

    ``field_dim`` must be at least ``source_count * payload_dim`` for strict
    non-decoherent reconstruction. Larger fields leave unused capacity.
    """

    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if payload_dim <= 0:
        raise ValueError("payload_dim must be positive")
    if checksum_dim <= 0:
        raise ValueError("checksum_dim must be positive")
    resolved_field_dim = source_count * payload_dim if field_dim is None else field_dim
    if resolved_field_dim < source_count * payload_dim:
        raise ValueError("field_dim must be >= source_count * payload_dim for fixed orthogonal carriers")

    carrier = torch.zeros(source_count, payload_dim, resolved_field_dim, device=device, dtype=dtype)
    for source in range(source_count):
        start = source * payload_dim
        carrier[source, :, start : start + payload_dim] = torch.eye(payload_dim, device=device, dtype=dtype)

    pilots = torch.zeros(source_count, payload_dim, device=device, dtype=dtype)
    for source in range(source_count):
        pilots[source, source % payload_dim] = 1.0
    checksum_projection = _fixed_projection(checksum_dim, payload_dim, device=device, dtype=dtype)
    return SourceIntegrityBasis(source_carrier=carrier, pilots=pilots, checksum_projection=checksum_projection)


def superpose_sources(x: Tensor, basis: SourceIntegrityBasis, *, mask: Tensor | None = None, pilot_strength: float = 0.0) -> Tensor:
    """Superpose synchronized source streams into field tokens.

    ``x`` has shape ``[B, S, T, D]`` and returns ``[B, T, H]``. ``mask`` can be
    ``[B, S, T]`` and suppresses missing source samples.
    """

    if x.ndim != 4:
        raise ValueError("x must have shape [B, S, T, D]")
    carrier = basis.source_carrier.to(device=x.device, dtype=x.dtype)
    if x.shape[1] != carrier.shape[0] or x.shape[-1] != carrier.shape[1]:
        raise ValueError("x source/payload dimensions must match basis")
    pilots = basis.pilots.to(device=x.device, dtype=x.dtype)
    values = x + pilot_strength * pilots.unsqueeze(0).unsqueeze(2)
    values = values if mask is None else values * mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    return torch.einsum("bstd,sdh->bth", values, carrier)


def read_sources(field: Tensor, basis: SourceIntegrityBasis) -> Tensor:
    """Matched-filter readout from field tokens back to source streams."""

    if field.ndim != 3:
        raise ValueError("field must have shape [B, T, H]")
    carrier = basis.source_carrier.to(device=field.device, dtype=field.dtype)
    if field.shape[-1] != carrier.shape[-1]:
        raise ValueError("field dimension must match basis")
    return torch.einsum("bth,sdh->bstd", field, carrier)


def encode_source_tokens(hidden: Tensor, source_ids: Tensor, basis: SourceIntegrityBasis) -> Tensor:
    """LLM token-stream form: encode ``[B, N, D]`` with per-token source ids."""

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [B, N, D]")
    if source_ids.shape != hidden.shape[:2]:
        raise ValueError("source_ids must have shape [B, N]")
    carrier = basis.source_carrier.to(device=hidden.device, dtype=hidden.dtype)
    selected = carrier.index_select(0, source_ids.reshape(-1).to(torch.long)).reshape(*source_ids.shape, carrier.shape[1], carrier.shape[2])
    return torch.einsum("bnd,bndh->bnh", hidden, selected)


def decode_source_tokens(field: Tensor, source_ids: Tensor, basis: SourceIntegrityBasis) -> Tensor:
    """Decode LLM token-stream fields using the expected source id per token."""

    if field.ndim != 3:
        raise ValueError("field must have shape [B, N, H]")
    if source_ids.shape != field.shape[:2]:
        raise ValueError("source_ids must have shape [B, N]")
    carrier = basis.source_carrier.to(device=field.device, dtype=field.dtype)
    selected = carrier.index_select(0, source_ids.reshape(-1).to(torch.long)).reshape(*source_ids.shape, carrier.shape[1], carrier.shape[2])
    return torch.einsum("bnh,bndh->bnd", field, selected)


def source_basis_orthogonality_loss(basis: SourceIntegrityBasis | Tensor, *, cross_weight: float = 1.0) -> Tensor:
    """Return self/cross orthogonality loss for source carriers."""

    carrier = basis.source_carrier if isinstance(basis, SourceIntegrityBasis) else basis
    gram = torch.einsum("sdh,reh->srde", carrier, carrier)
    source_count, payload_dim = carrier.shape[:2]
    eye = torch.eye(payload_dim, device=carrier.device, dtype=carrier.dtype)
    self_loss = sum((gram[s, s] - eye).square().mean() for s in range(source_count))
    cross_loss = sum(gram[s, r].square().mean() for s in range(source_count) for r in range(source_count) if s != r)
    return self_loss + cross_weight * cross_loss


def source_integrity_report(
    original: Tensor,
    decoded: Tensor,
    basis: SourceIntegrityBasis,
    *,
    mode: str = "full",
    block_size: int | None = None,
    pilot_strength: float = 0.0,
    min_pilot_self_score: float = 0.95,
    max_pilot_cross_score: float = 0.10,
    max_checksum_error: float = 1e-3,
    max_leakage: float = 0.05,
) -> SourceIntegrityReport:
    """Summarize source reconstruction, pilot, checksum, and leakage health."""

    _validate_mode(mode)
    if mode == "off":
        raise ValueError("source_integrity_report does not run in off mode; use SourceIntegrityCarrier.report")
    if original.shape != decoded.shape or original.ndim != 4:
        raise ValueError("original and decoded must both have shape [B, S, T, D]")
    if block_size is not None and block_size <= 0:
        raise ValueError("block_size must be positive")
    carrier = basis.source_carrier.to(device=decoded.device, dtype=decoded.dtype)
    pilots = basis.pilots.to(device=decoded.device, dtype=decoded.dtype)
    checksum_projection = basis.checksum_projection.to(device=decoded.device, dtype=decoded.dtype)
    source_count, payload_dim, field_dim = carrier.shape
    checksum_dim = checksum_projection.shape[0]

    pilot_offset = pilot_strength * pilots.unsqueeze(0).unsqueeze(2)
    decoded_payload = decoded - pilot_offset
    pooled_original = _block_mean(original, block_size)
    pooled_decoded = _block_mean(decoded_payload, block_size)
    reconstruction_error = (pooled_decoded - pooled_original).square().mean(dim=(0, 2, 3)).sqrt()

    if pilot_strength > 0:
        pilot_residual = _block_mean(decoded - decoded_payload, block_size) / pilot_strength
        pilot_norm = torch.nn.functional.normalize(pilots, dim=-1)
        residual_norm = torch.nn.functional.normalize(pilot_residual, dim=-1)
        pilot_scores = torch.einsum("bsqd,rd->bsqr", residual_norm, pilot_norm).mean(dim=(0, 2))
        pilot_self = torch.diagonal(pilot_scores)
        pilot_cross = pilot_scores - torch.diag_embed(pilot_self)
    else:
        pilot_self = torch.ones(source_count, device=decoded.device, dtype=decoded.dtype)
        pilot_cross = torch.zeros(source_count, source_count, device=decoded.device, dtype=decoded.dtype)

    original_checksum = torch.einsum("bsqd,cd->bsqc", pooled_original, checksum_projection)
    decoded_checksum = torch.einsum("bsqd,cd->bsqc", pooled_decoded, checksum_projection)
    checksum_error = (decoded_checksum - original_checksum).square().mean(dim=(0, 2, 3)).sqrt()

    leakage = torch.einsum("sdh,rdh->sr", carrier, carrier).abs()
    leakage = leakage / max(1, payload_dim)
    leakage = leakage - torch.diag_embed(torch.diagonal(leakage))
    max_cross = float(pilot_cross.abs().max().item()) if source_count > 1 else 0.0
    max_leak = float(leakage.max().item()) if source_count > 1 else 0.0
    max_checksum = float(checksum_error.max().item())
    min_self = float(pilot_self.min().item())
    valid = min_self >= min_pilot_self_score and max_cross <= max_pilot_cross_score and max_checksum <= max_checksum_error and max_leak <= max_leakage
    return SourceIntegrityReport(
        mode=mode,
        source_count=source_count,
        payload_dim=payload_dim,
        field_dim=field_dim,
        checksum_dim=checksum_dim,
        capacity_ratio=field_dim / max(1, source_count * payload_dim),
        reconstruction_error=reconstruction_error,
        pilot_self_score=pilot_self,
        pilot_cross_score=pilot_cross,
        checksum_error=checksum_error,
        leakage_matrix=leakage,
        min_pilot_self_score=min_self,
        max_pilot_cross_score=max_cross,
        max_checksum_error=max_checksum,
        max_leakage=max_leak,
        valid=valid,
    )


def source_integrity_loss(report: SourceIntegrityReport) -> Tensor:
    """Small differentiable loss from report tensors."""

    return report.reconstruction_error.square().mean() + report.checksum_error.square().mean() + report.pilot_cross_score.square().mean()


def assert_source_integrity(report: SourceIntegrityReport) -> SourceIntegrityReport:
    """Raise if source integrity diagnostics mark the field invalid."""

    if not report.valid:
        raise ValueError(
            "source integrity failed: "
            f"min_pilot_self={report.min_pilot_self_score:.4f}, "
            f"max_pilot_cross={report.max_pilot_cross_score:.4f}, "
            f"max_checksum_error={report.max_checksum_error:.6f}, "
            f"max_leakage={report.max_leakage:.4f}"
        )
    return report


def _fixed_projection(rows: int, cols: int, *, device: torch.device | str | None, dtype: torch.dtype) -> Tensor:
    row = torch.arange(rows, device=device, dtype=dtype).unsqueeze(1) + 1
    col = torch.arange(cols, device=device, dtype=dtype).unsqueeze(0) + 1
    projection = torch.sin(row * col * 0.61803398875)
    return torch.nn.functional.normalize(projection, dim=-1)


def _validate_mode(mode: str) -> None:
    if mode not in SOURCE_INTEGRITY_MODES:
        allowed = ", ".join(SOURCE_INTEGRITY_MODES)
        raise ValueError(f"mode must be one of: {allowed}")


def _block_mean(x: Tensor, block_size: int | None) -> Tensor:
    if block_size is None:
        return x
    batch, sources, tokens, dim = x.shape
    blocks = (tokens + block_size - 1) // block_size
    padded = torch.nn.functional.pad(x, (0, 0, 0, blocks * block_size - tokens))
    return padded.reshape(batch, sources, blocks, block_size, dim).mean(dim=3)
