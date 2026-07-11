"""Pixel-shift visual scanning for rigid glyph tensors."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .visual_field import VisualField


DEFAULT_PIXEL_SHIFTS = ((0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5))


@dataclass(frozen=True)
class VisualScanConfig:
    """Serializable physical and neural configuration for :class:`VisualScan`."""

    low_size: tuple[int, int]
    channels: int = 1
    scale: int = 2
    shifts: tuple[tuple[float, float], ...] = DEFAULT_PIXEL_SHIFTS
    blur_sigma: float = 0.45
    noise_std: float = 0.0
    patch_size: tuple[int, int] = (4, 4)
    pulse_count: int = 4
    pulse_hidden_dim: int | None = None
    use_pulse: bool = True
    pulse_use_half: bool = True
    persistence_steps: int = 0
    persistence_decay: float = 0.85
    register_shifts: bool = True
    include_registered_fusion: bool = True
    registered_workspace_residual: bool = True
    include_shift: bool = False

    def __post_init__(self) -> None:
        if len(self.low_size) != 2 or min(self.low_size) <= 0:
            raise ValueError("low_size must contain two positive integers")
        if self.channels <= 0 or self.scale <= 0:
            raise ValueError("channels and scale must be positive")
        if not self.shifts or any(len(shift) != 2 for shift in self.shifts):
            raise ValueError("shifts must contain at least one (dy, dx) pair")
        if self.blur_sigma < 0 or self.noise_std < 0:
            raise ValueError("blur_sigma and noise_std must be non-negative")
        if len(self.patch_size) != 2 or min(self.patch_size) <= 0:
            raise ValueError("patch_size must contain two positive integers")
        if self.pulse_count <= 0:
            raise ValueError("pulse_count must be positive")
        if self.pulse_hidden_dim is not None and self.pulse_hidden_dim <= 0:
            raise ValueError("pulse_hidden_dim must be positive")
        if self.persistence_steps < 0:
            raise ValueError("persistence_steps must be non-negative")
        if not 0.0 < self.persistence_decay <= 1.0:
            raise ValueError("persistence_decay must be in (0, 1]")
        if not self.use_pulse and not self.registered_workspace_residual:
            raise ValueError("VisualScan requires use_pulse or registered_workspace_residual")
        if not self.use_pulse and not (self.register_shifts and self.include_registered_fusion):
            raise ValueError("carrier-only VisualScan requires registered fusion")

    @property
    def high_size(self) -> tuple[int, int]:
        return self.low_size[0] * self.scale, self.low_size[1] * self.scale

    @property
    def fragment_dim(self) -> int:
        pixels = self.channels * self.patch_size[0] * self.patch_size[1]
        return pixels + 5 + (2 if self.include_shift else 0)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["low_size"] = list(self.low_size)
        payload["shifts"] = [list(shift) for shift in self.shifts]
        payload["patch_size"] = list(self.patch_size)
        return payload

    def explain(self) -> dict[str, Any]:
        return {
            "mechanisms": {
                "phase": self.register_shifts,
                "coordinate_inverse": self.register_shifts,
                "virtual_interface": False,
                "pairwise_context": False,
                "recall": False,
                "virtual_recall": False,
                "half": self.use_pulse and self.pulse_use_half,
                "pulse": self.use_pulse,
                "carrier": self.registered_workspace_residual,
            },
            "required_inputs": {"frames": True, "shifts": self.register_shifts, "mask": False},
            "accepted_inputs": {"frames": True, "shifts": True, "mask": True},
            "capacities": {
                "pulse_slots": self.pulse_count if self.use_pulse else 0,
                "persistence_steps": self.persistence_steps,
            },
            "synthetic_context": False,
        }

    def diff(self, other: "VisualScanConfig") -> dict[str, dict[str, Any]]:
        if not isinstance(other, VisualScanConfig):
            raise TypeError("other must be a VisualScanConfig")
        left = self.to_dict()
        right = other.to_dict()
        return {key: {"self": left[key], "other": right[key]} for key in left if left[key] != right[key]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisualScanConfig":
        return cls(
            low_size=tuple(int(value) for value in payload["low_size"]),
            channels=int(payload.get("channels", 1)),
            scale=int(payload.get("scale", 2)),
            shifts=tuple(tuple(float(value) for value in shift) for shift in payload.get("shifts", DEFAULT_PIXEL_SHIFTS)),
            blur_sigma=float(payload.get("blur_sigma", 0.45)),
            noise_std=float(payload.get("noise_std", 0.0)),
            patch_size=tuple(int(value) for value in payload.get("patch_size", (4, 4))),
            pulse_count=int(payload.get("pulse_count", 4)),
            pulse_hidden_dim=None if payload.get("pulse_hidden_dim") is None else int(payload["pulse_hidden_dim"]),
            use_pulse=bool(payload.get("use_pulse", True)),
            pulse_use_half=bool(payload.get("pulse_use_half", True)),
            persistence_steps=int(payload.get("persistence_steps", 0)),
            persistence_decay=float(payload.get("persistence_decay", 0.85)),
            register_shifts=bool(payload.get("register_shifts", True)),
            include_registered_fusion=bool(payload.get("include_registered_fusion", True)),
            registered_workspace_residual=bool(payload.get("registered_workspace_residual", True)),
            include_shift=bool(payload.get("include_shift", False)),
        )


@dataclass(frozen=True)
class PixelShiftObservation:
    """Recorded low-resolution observations from the physical sampling model."""

    frames: Tensor
    shifts: Tensor
    mask: Tensor
    high_size: tuple[int, int]
    low_size: tuple[int, int]


@dataclass(frozen=True)
class VisualScanOutput:
    """Fixed Pulse workspace plus the exact fragments used to form it."""

    pulses: Tensor
    fragments: Tensor
    mask: Tensor
    survival: Tensor
    shifts: Tensor


class VisualScan(nn.Module):
    """Form a fixed Pulse workspace from recorded pixel-shift observations."""

    def __init__(self, config: VisualScanConfig) -> None:
        super().__init__()
        self.config = config
        self.field = VisualField(patch_size=config.patch_size, include_position=True)
        from .nn import Pulse

        hidden_dim = config.fragment_dim * 2 if config.pulse_hidden_dim is None else config.pulse_hidden_dim
        self.pulse = (
            Pulse(k=config.pulse_count, dim=config.fragment_dim, hidden_dim=hidden_dim, use_half=config.pulse_use_half)
            if config.use_pulse
            else None
        )

    def observe(
        self,
        high_res: Tensor,
        *,
        shifts: Tensor | None = None,
        noise_std: float | None = None,
        generator: torch.Generator | None = None,
    ) -> PixelShiftObservation:
        return pixel_shift_observe(high_res, self.config, shifts=shifts, noise_std=noise_std, generator=generator)

    def forward(
        self,
        frames: Tensor | PixelShiftObservation,
        shifts: Tensor | None = None,
        *,
        frame_mask: Tensor | None = None,
        return_info: bool = False,
    ) -> Tensor | VisualScanOutput:
        if isinstance(frames, PixelShiftObservation):
            if shifts is not None or frame_mask is not None:
                raise ValueError("shifts and frame_mask are already carried by PixelShiftObservation")
            observation = frames
            frame_tensor = observation.frames
            shifts = observation.shifts
            frame_mask = observation.mask
        else:
            frame_tensor = frames
        if frame_tensor.ndim != 5:
            raise ValueError("frames must have shape [B, T, C, H, W]")
        batch, scans, channels, height, width = frame_tensor.shape
        if channels != self.config.channels or (height, width) != self.config.low_size:
            raise ValueError(
                f"expected frames [B, T, {self.config.channels}, {self.config.low_size[0]}, {self.config.low_size[1]}]"
            )
        shifts = _normalize_shifts(shifts, batch, scans, frame_tensor)
        frame_mask = _normalize_frame_mask(frame_mask, batch, scans, frame_tensor.device)

        flattened = frame_tensor.reshape(batch * scans, channels, height, width)
        effective_mask = frame_mask
        if self.config.register_shifts:
            flattened = F.interpolate(
                flattened,
                size=self.config.high_size,
                mode="bicubic",
                align_corners=False,
            )
            flattened = translate_image(flattened, -shifts.reshape(-1, 2) * self.config.scale)
            if self.config.include_registered_fusion:
                fused = shift_and_add(
                    frame_tensor,
                    shifts,
                    scale=self.config.scale,
                    frame_mask=frame_mask,
                )
                flattened = torch.cat([flattened.reshape(batch, scans, channels, *self.config.high_size), fused.unsqueeze(1)], dim=1)
                flattened = flattened.reshape(batch * (scans + 1), channels, *self.config.high_size)
                effective_mask = torch.cat([frame_mask, frame_mask.any(dim=1, keepdim=True)], dim=1)
        field = self.field(flattened)
        effective_scans = effective_mask.shape[1]
        fragments = field.fragments.reshape(batch, effective_scans, field.fragments.shape[1], -1)
        patch_mask = field.mask.reshape(batch, effective_scans, field.mask.shape[1])
        if self.config.include_shift:
            feature_shifts = shifts
            if effective_scans != scans:
                feature_shifts = torch.cat([shifts, torch.zeros_like(shifts[:, :1])], dim=1)
            shift_features = feature_shifts.to(frame_tensor).unsqueeze(2).expand(-1, -1, fragments.shape[2], -1)
            fragments = torch.cat([fragments, shift_features], dim=-1)
        mask = patch_mask & effective_mask.unsqueeze(-1)
        survival = self._survival(batch, effective_scans, fragments.shape[2], frame_tensor) * mask.to(frame_tensor.dtype)
        pulse_q = survival.reshape(batch, -1) if self.config.persistence_steps > 0 else None
        flat_fragments = fragments.reshape(batch, -1, fragments.shape[-1])
        flat_mask = mask.reshape(batch, -1)
        pulses = None if self.pulse is None else self.pulse(flat_fragments, q=pulse_q, mask=flat_mask)
        if self.config.registered_workspace_residual and self.config.register_shifts and self.config.include_registered_fusion:
            carrier = fragments[:, -1].transpose(1, 2)
            carrier = F.adaptive_avg_pool1d(carrier, self.config.pulse_count).transpose(1, 2)
            pulses = carrier if pulses is None else pulses + carrier.to(dtype=pulses.dtype)
        if pulses is None:
            raise RuntimeError("VisualScan produced neither a Pulse nor a carrier workspace")
        if not return_info:
            return pulses
        return VisualScanOutput(
            pulses=pulses,
            fragments=flat_fragments,
            mask=flat_mask,
            survival=survival.reshape(batch, -1),
            shifts=shifts,
        )

    def _survival(self, batch: int, scans: int, patches: int, like: Tensor) -> Tensor:
        if self.config.persistence_steps <= 0:
            return torch.ones(batch, scans, patches, device=like.device, dtype=like.dtype)
        age = torch.arange(scans - 1, -1, -1, device=like.device, dtype=like.dtype)
        alive = age < self.config.persistence_steps
        values = torch.pow(torch.full_like(age, self.config.persistence_decay), age) * alive.to(like.dtype)
        return values.view(1, scans, 1).expand(batch, -1, patches)

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> "VisualScan":
        return cls(VisualScanConfig.from_dict(payload))


def pixel_shift_observe(
    high_res: Tensor,
    config: VisualScanConfig,
    *,
    shifts: Tensor | None = None,
    noise_std: float | None = None,
    generator: torch.Generator | None = None,
) -> PixelShiftObservation:
    """Apply ``y_t = D H T_delta x + epsilon`` to high-resolution glyphs."""

    if high_res.ndim != 4:
        raise ValueError("high_res must have shape [B, C, H*scale, W*scale]")
    if not high_res.is_floating_point():
        raise TypeError("high_res must be a floating-point Tensor")
    batch, channels, height, width = high_res.shape
    if channels != config.channels or (height, width) != config.high_size:
        raise ValueError(f"expected high_res shape [B, {config.channels}, {config.high_size[0]}, {config.high_size[1]}]")
    shift_tensor = _normalize_shifts(shifts, batch, len(config.shifts), high_res, default=config.shifts)
    scans = shift_tensor.shape[1]
    expanded = high_res.unsqueeze(1).expand(-1, scans, -1, -1, -1).reshape(batch * scans, channels, height, width)
    shifted = translate_image(expanded, shift_tensor.reshape(-1, 2) * config.scale)
    blurred = gaussian_blur(shifted, config.blur_sigma)
    frames = F.interpolate(blurred, size=config.low_size, mode="area")
    resolved_noise = config.noise_std if noise_std is None else float(noise_std)
    if resolved_noise < 0:
        raise ValueError("noise_std must be non-negative")
    if resolved_noise > 0:
        noise = torch.randn(frames.shape, device=frames.device, dtype=frames.dtype, generator=generator)
        frames = frames + resolved_noise * noise
    frames = frames.reshape(batch, scans, channels, *config.low_size)
    return PixelShiftObservation(
        frames=frames,
        shifts=shift_tensor,
        mask=torch.ones(batch, scans, device=high_res.device, dtype=torch.bool),
        high_size=config.high_size,
        low_size=config.low_size,
    )


def shift_and_add(
    frames: Tensor,
    shifts: Tensor,
    *,
    scale: int,
    frame_mask: Tensor | None = None,
) -> Tensor:
    """Classical registered shift-and-add reconstruction baseline."""

    if frames.ndim != 5 or scale <= 0:
        raise ValueError("frames must be [B, T, C, H, W] and scale must be positive")
    batch, scans, channels, height, width = frames.shape
    shifts = _normalize_shifts(shifts, batch, scans, frames)
    mask = _normalize_frame_mask(frame_mask, batch, scans, frames.device)
    upsampled = F.interpolate(
        frames.reshape(batch * scans, channels, height, width),
        size=(height * scale, width * scale),
        mode="bicubic",
        align_corners=False,
    )
    registered = translate_image(upsampled, -shifts.reshape(-1, 2) * scale)
    registered = registered.reshape(batch, scans, channels, height * scale, width * scale)
    weights = mask.to(frames.dtype).view(batch, scans, 1, 1, 1)
    return (registered * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def translate_image(image: Tensor, shift_high_pixels: Tensor) -> Tensor:
    """Translate batched images by recorded ``(dy, dx)`` sample offsets."""

    if image.ndim != 4 or shift_high_pixels.shape != (image.shape[0], 2):
        raise ValueError("image must be [B, C, H, W] and shifts must be [B, 2]")
    height, width = image.shape[-2:]
    theta = torch.eye(2, 3, device=image.device, dtype=image.dtype).unsqueeze(0).repeat(image.shape[0], 1, 1)
    theta[:, 0, 2] = 2.0 * shift_high_pixels[:, 1].to(image) / max(width - 1, 1)
    theta[:, 1, 2] = 2.0 * shift_high_pixels[:, 0].to(image) / max(height - 1, 1)
    grid = F.affine_grid(theta, image.shape, align_corners=True)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def gaussian_blur(image: Tensor, sigma: float) -> Tensor:
    """Depthwise Gaussian optical blur with a finite three-sigma kernel."""

    if sigma <= 0:
        return image
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel.view(1, 1, *kernel.shape).expand(image.shape[1], 1, -1, -1)
    return F.conv2d(image, kernel, padding=radius, groups=image.shape[1])


def _normalize_shifts(
    shifts: Tensor | None,
    batch: int,
    scans: int,
    like: Tensor,
    *,
    default: tuple[tuple[float, float], ...] | None = None,
) -> Tensor:
    if shifts is None:
        source = ((0.0, 0.0),) * scans if default is None else default
        shifts = torch.tensor(source, device=like.device, dtype=like.dtype)
    else:
        shifts = shifts.to(device=like.device, dtype=like.dtype)
    if shifts.ndim == 2:
        if shifts.shape != (scans, 2):
            raise ValueError(f"shifts must have shape [{scans}, 2] or [B, {scans}, 2]")
        shifts = shifts.unsqueeze(0).expand(batch, -1, -1)
    if shifts.shape != (batch, scans, 2):
        raise ValueError(f"shifts must have shape [{scans}, 2] or [{batch}, {scans}, 2]")
    return shifts


def _normalize_frame_mask(mask: Tensor | None, batch: int, scans: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(batch, scans, device=device, dtype=torch.bool)
    if mask.shape != (batch, scans):
        raise ValueError(f"frame_mask must have shape [{batch}, {scans}]")
    return mask.to(device=device, dtype=torch.bool)


__all__ = [
    "DEFAULT_PIXEL_SHIFTS",
    "VisualScanConfig",
    "PixelShiftObservation",
    "VisualScanOutput",
    "VisualScan",
    "pixel_shift_observe",
    "shift_and_add",
]
