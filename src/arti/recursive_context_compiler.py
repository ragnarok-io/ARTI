"""Native recursive context compilation inside a shared tensor backbone."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .tensor_schema import TensorSchema


@dataclass(frozen=True)
class ContextInput:
    value: Tensor
    valid: Tensor
    placement: Tensor


class TokenPositionEncoder(nn.Module):
    """Encode lexical positions before they enter the shared compiler."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if type(dim) is not int or dim < 1:
            raise ValueError("position encoder dimension must be a positive integer")
        self.dim = dim

    def _validate(self, positions: Tensor) -> None:
        if positions.ndim != 2 or (
            not positions.is_floating_point()
            and positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("positions must be numeric [B,L]")

    def config(self) -> dict:
        return {"dim": self.dim}


class SinusoidalTokenPositionEncoder(TokenPositionEncoder):
    """Explicit standard absolute position encoding for the lexical input."""

    _component_reference = "arti/token-position-sinusoidal@1"

    def __init__(self, dim: int, *, base: float) -> None:
        super().__init__(dim)
        if not math.isfinite(base) or base <= 1:
            raise ValueError("position base must be finite and greater than one")
        self.base = float(base)

    def forward(self, positions: Tensor) -> Tensor:
        self._validate(positions)
        dtype = torch.float64 if positions.dtype == torch.float64 else torch.float32
        frequencies = torch.exp(
            torch.arange(0, self.dim, 2, device=positions.device, dtype=dtype)
            * (-math.log(self.base) / self.dim)
        )
        angle = positions.to(dtype).unsqueeze(-1) * frequencies
        return torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)[..., : self.dim]

    def config(self) -> dict:
        return {**super().config(), "base": self.base}


class LearnedTokenPositionEncoder(TokenPositionEncoder):
    """Explicit learned absolute positions with a bounded maximum length."""

    _component_reference = "arti/token-position-learned@1"

    def __init__(self, dim: int, max_length: int) -> None:
        super().__init__(dim)
        if type(max_length) is not int or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, dim)

    def forward(self, positions: Tensor) -> Tensor:
        self._validate(positions)
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("learned positions must be integer [B,L]")
        if positions.numel() and (
            int(positions.min()) < 0 or int(positions.max()) >= self.max_length
        ):
            raise ValueError("learned positions exceed max_length")
        return self.embedding(positions.to(torch.long))

    def config(self) -> dict:
        return {**super().config(), "max_length": self.max_length}


class LexicalEmbedding(nn.Module):
    _component_reference = "arti/lexical-embedding@1"

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        trainable: bool = False,
        position_encoder: TokenPositionEncoder | Mapping[str, object],
    ) -> None:
        super().__init__()
        if vocab_size < 1 or dim < 1 or type(trainable) is not bool:
            raise ValueError("dimensions must be positive")
        self.vocab_size, self.dim, self.trainable = vocab_size, dim, trainable
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding.weight.requires_grad_(trainable)
        if position_encoder is None:
            raise ValueError(
                "position_encoder must be selected explicitly; "
                "use a registered TokenPositionEncoder"
            )
        if isinstance(position_encoder, Mapping):
            from .component_registry import resolve_component

            position_encoder = resolve_component(
                position_encoder["reference"], **position_encoder["config"]
            )
        if not isinstance(position_encoder, TokenPositionEncoder):
            raise TypeError("position_encoder must be a TokenPositionEncoder")
        if position_encoder.dim != dim:
            raise ValueError("position_encoder dimension must match lexical embedding")
        self.position_encoder = position_encoder

    def forward(self, tokens: Tensor, *, positions: Tensor | None = None) -> Tensor:
        if tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("tokens must be integer [B,L]")
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
        if (
            positions.shape not in ((1, tokens.shape[1]), tokens.shape)
            or positions.device != tokens.device
        ):
            raise ValueError("positions must be same-device [1,L] or [B,L]")
        value = self.embedding(tokens)
        return value + self.position_encoder(positions).to(value.dtype)

    def config(self) -> dict:
        from .component_registry import component_spec

        spec = component_spec(self.position_encoder)
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "trainable": self.trainable,
            "position_encoder": {"reference": spec.reference, "config": dict(spec.config)},
        }


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, expansion: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, expansion * dim), nn.GELU(), nn.Linear(expansion * dim, dim)
        )

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        batch, length, dim = x.shape
        q, k, v = (
            self.qkv(self.norm(x))
            .reshape(batch, length, 3, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        x = x + self.out(a.transpose(1, 2).reshape(batch, length, dim))
        x = x + self.ff(self.ff_norm(x))
        return torch.where(valid[..., None], x, 0.0)


class ContextPlacementEncoder(nn.Module):
    """Encode consumer-supplied relation coordinates into compiler inputs."""

    def __init__(self, relation_dim: int, dim: int) -> None:
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (relation_dim, dim)):
            raise ValueError("placement encoder dimensions must be positive integers")
        self.relation_dim, self.dim = relation_dim, dim

    def _validate(self, placement: Tensor) -> None:
        if (
            placement.ndim != 2
            or placement.shape[-1] != self.relation_dim
            or not placement.is_floating_point()
        ):
            raise ValueError("placement must be floating [B, relation_dim]")

    def config(self) -> dict:
        return {"relation_dim": self.relation_dim, "dim": self.dim}


class LinearContextPlacementEncoder(ContextPlacementEncoder):
    """Minimal learned relation encoding with no hidden frequency scale."""

    _component_reference = "arti/context-placement-linear@1"

    def __init__(self, relation_dim: int, dim: int) -> None:
        super().__init__(relation_dim, dim)
        self.projection = nn.Linear(relation_dim, dim, bias=False)

    def forward(self, placement: Tensor) -> Tensor:
        self._validate(placement)
        return self.projection(placement.to(self.projection.weight.dtype))


class FourierContextPlacementEncoder(ContextPlacementEncoder):
    """Explicit opt-in Fourier relation encoding for periodic coordinates."""

    _component_reference = "arti/context-placement-fourier@1"

    def __init__(
        self,
        relation_dim: int,
        dim: int,
        *,
        bands: int = 8,
        base: float = 10000.0,
    ) -> None:
        super().__init__(relation_dim, dim)
        if type(bands) is not int or bands < 2 or not math.isfinite(base) or base <= 1:
            raise ValueError("invalid Fourier placement configuration")
        self.bands, self.base = bands, float(base)
        self.projection = nn.Linear(2 * bands * relation_dim, dim, bias=False)

    def forward(self, placement: Tensor) -> Tensor:
        self._validate(placement)
        dtype = torch.float64 if placement.dtype == torch.float64 else torch.float32
        frequencies = torch.exp(
            torch.arange(self.bands, device=placement.device, dtype=dtype)
            * (-math.log(self.base) / max(1, self.bands - 1))
        )
        phase = placement.to(dtype)[..., None] * frequencies
        features = torch.stack((phase.sin(), phase.cos()), -1).flatten(-3)
        return self.projection(features.to(self.projection.weight.dtype))

    def config(self) -> dict:
        return {**super().config(), "bands": self.bands, "base": self.base}


class RecursiveContextCompiler(nn.Module):
    """Compile recursive C inputs and produce the ordinary downstream Y state."""

    _component_reference = "arti/recursive-context-compiler@1"

    def __init__(
        self,
        dim: int,
        context_width: int = 16,
        heads: int = 4,
        layers: int = 2,
        expansion: int = 4,
        relation_dim: int = 2,
        readout_layers: Sequence[int] | None = None,
        placement_encoder: ContextPlacementEncoder | Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if (
            any(
                type(v) is not int or v < 1
                for v in (dim, context_width, heads, layers, expansion, relation_dim)
            )
            or dim % heads
        ):
            raise ValueError("invalid context compiler dimensions")
        selected = tuple(range(1, layers + 1)) if readout_layers is None else tuple(readout_layers)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 1 <= i <= layers for i in selected)
        ):
            raise ValueError("invalid readout_layers")
        self.dim, self.context_width, self.heads = dim, context_width, heads
        self.layers, self.expansion, self.relation_dim = layers, expansion, relation_dim
        self.readout_layers = selected
        if placement_encoder is None:
            placement_encoder = LinearContextPlacementEncoder(relation_dim, dim)
        elif isinstance(placement_encoder, Mapping):
            from .component_registry import resolve_component

            placement_encoder = resolve_component(
                placement_encoder["reference"], **placement_encoder["config"]
            )
        if not isinstance(placement_encoder, ContextPlacementEncoder):
            raise TypeError("placement_encoder must be a ContextPlacementEncoder")
        if (placement_encoder.relation_dim, placement_encoder.dim) != (relation_dim, dim):
            raise ValueError("placement_encoder dimensions must match the compiler")
        self.placement_encoder = placement_encoder
        self.blocks = nn.ModuleList([_Block(dim, heads, expansion) for _ in range(layers)])
        self.context_seed = nn.Parameter(torch.randn(context_width, dim) / math.sqrt(dim))
        self.context_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in selected])
        self.context_out = nn.Linear(dim, dim)
        self.context_output_norm = nn.LayerNorm(dim)

    @property
    def width(self) -> int:
        return self.context_width

    @property
    def input_schema(self) -> TensorSchema:
        return TensorSchema("floating", "any", ("B", "L", self.dim), ("B", "L", "D"))

    @property
    def output_schema(self) -> TensorSchema:
        return TensorSchema("floating", "any", ("B", "M", self.dim), ("B", "M", "D"))

    def config(self) -> dict:
        return {
            "dim": self.dim,
            "context_width": self.context_width,
            "heads": self.heads,
            "layers": self.layers,
            "expansion": self.expansion,
            "relation_dim": self.relation_dim,
            "readout_layers": list(self.readout_layers),
            "placement_encoder": self._placement_encoder_config(),
        }

    def _placement_encoder_config(self) -> dict:
        from .component_registry import component_spec

        spec = component_spec(self.placement_encoder)
        return {"reference": spec.reference, "config": dict(spec.config)}

    def _network(self, channels: Sequence[ContextInput], context_width: int):
        if not channels:
            raise ValueError("context compiler requires inputs")
        first = channels[0].value
        if first.ndim != 3 or first.shape[-1] != self.dim:
            raise ValueError("values must be [B,L,dim]")
        b, dtype, device = first.shape[0], first.dtype, first.device
        values = []
        masks = []
        for c in channels:
            if (
                c.value.ndim != 3
                or c.value.shape[0] != b
                or c.value.shape[-1] != self.dim
                or c.value.dtype != dtype
                or c.value.device != device
            ):
                raise ValueError("incompatible channel value")
            if (
                c.valid.shape != c.value.shape[:2]
                or c.valid.dtype != torch.bool
                or c.valid.device != device
            ):
                raise ValueError("invalid channel mask")
            if c.placement.shape != (b, self.relation_dim) or c.placement.device != device:
                raise ValueError("invalid placement")
            values.append(
                torch.where(
                    c.valid[..., None],
                    c.value + self.placement_encoder(c.placement).to(dtype)[:, None],
                    0.0,
                )
            )
            masks.append(c.valid)
        x, valid = torch.cat(values, 1), torch.cat(masks, 1)
        source_length = x.shape[1]
        source_valid = valid
        row_valid = source_valid.any(1)
        seeds = self.context_seed[:context_width].to(dtype)[None].expand(b, -1, -1)
        # C is a native prefix of the shared trunk. In a causal host this makes
        # the compiled context visible to every later lexical position.
        x = torch.cat((seeds, x), 1)
        valid = torch.cat(
            (torch.ones(b, context_width, device=device, dtype=torch.bool), source_valid), 1
        )
        states = []
        for block in self.blocks:
            x = block(x, valid)
            states.append(x)
        return tuple(states), valid, row_valid, source_length

    def _context(self, states, row_valid, context_width):
        reads = [
            norm(states[layer - 1][:, :context_width])
            for norm, layer in zip(self.context_norms, self.readout_layers)
        ]
        out = self.context_output_norm(self.context_out(torch.stack(reads).mean(0)))
        return torch.where(row_valid[:, None, None], out, 0.0)

    def forward(
        self,
        channels: Sequence[Tensor | ContextInput],
        *,
        output: str = "context",
        context_width: int | None = None,
    ) -> Tensor:
        if output not in {"context", "downstream"}:
            raise ValueError("output must be context or downstream")
        width = self.context_width if context_width is None else context_width
        if type(width) is not int or not 1 <= width <= self.context_width:
            raise ValueError("context_width must be within the configured capacity")
        prepared = []
        for channel in channels:
            if isinstance(channel, Tensor):
                if channel.ndim != 3:
                    raise ValueError("captured tensors must be [B,L,D]")
                channel = ContextInput(
                    channel,
                    torch.ones(channel.shape[:2], device=channel.device, dtype=torch.bool),
                    channel.new_zeros(channel.shape[0], self.relation_dim),
                )
            if not isinstance(channel, ContextInput):
                raise TypeError("compiler inputs must be Tensors or ContextInput values")
            prepared.append(channel)
        states, valid, row_valid, source_length = self._network(prepared, width)
        if output == "context":
            return self._context(states, row_valid, width)
        positions = (torch.arange(source_length, device=valid.device) + width).expand(
            valid.shape[0], -1
        )
        index = torch.where(valid[:, width:], positions, -1).amax(1).clamp_min(width)
        y = states[-1][torch.arange(states[-1].shape[0], device=states[-1].device), index]
        return torch.where(row_valid[:, None], y, 0.0)

    def compile(
        self,
        channels: Sequence[Tensor | ContextInput],
        *,
        depth: int = 1,
        context_width: int | None = None,
    ) -> Tensor:
        """Deepen one chunk without creating a new abstraction level."""
        if type(depth) is not int or depth < 1:
            raise ValueError("depth must be a positive integer")
        current = self(channels, context_width=context_width)
        for _ in range(1, depth):
            current = self(
                (
                    ContextInput(
                        current,
                        torch.ones(current.shape[:2], device=current.device, dtype=torch.bool),
                        current.new_zeros(current.shape[0], self.relation_dim),
                    ),
                ),
                context_width=context_width,
            )
        return current


class LanguageHead(nn.Module):
    _component_reference = "arti/rcc-language-head@1"

    def __init__(self, dim: int, vocab_size: int) -> None:
        super().__init__()
        self.dim, self.vocab_size = dim, vocab_size
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 2 or value.shape[-1] != self.dim:
            raise ValueError("language head expects [B,dim]")
        return self.lm_head(self.norm(value))
