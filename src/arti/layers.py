"""Core tensor-in / tensor-out ARTI layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .config import ARTIConfig
from .functional import (
    as_sequence,
    apply_coord_frame_inverse,
    ensure_coord,
    ensure_mask,
    ensure_visibility,
    mask_coverage,
    masked_mean,
    masked_softmax,
    restore_input_rank,
)
from .init import init_arti_module
from .nn import Half
from .outputs import ARTIOutput
from .utils import assert_floating_tensor, detach_diagnostics


class ARTIPhaseMixer(nn.Module):
    """Mix cone-like latent receptors according to hidden state and coordinates."""

    def __init__(self, hidden_dim: int, coord_dim: int, operator_count: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.operator_count = operator_count
        self.hidden_dim = hidden_dim
        self.operators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(operator_count)
            ]
        )
        self.router = nn.Linear(hidden_dim + coord_dim, operator_count)
        self.coord_receptors = nn.Linear(coord_dim, operator_count * hidden_dim) if coord_dim > 0 else None

    def forward(self, z: Tensor, coord: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        route_input = torch.cat([z, coord], dim=-1)
        weights = torch.softmax(self.router(route_input), dim=-1)
        if self.coord_receptors is None:
            receptor_gain = torch.ones(*z.shape[:2], self.operator_count, self.hidden_dim, device=z.device, dtype=z.dtype)
        else:
            receptor_gain = 1.0 + torch.tanh(self.coord_receptors(coord)).view(*z.shape[:2], self.operator_count, self.hidden_dim)
        candidates = torch.stack([op(z * receptor_gain[:, :, index, :]) for index, op in enumerate(self.operators)], dim=-2)
        mixed = (candidates * weights.unsqueeze(-1)).sum(dim=-2)
        return mixed, weights, receptor_gain


class ARTIVirtualInterfaceMixer(nn.Module):
    """Fixed-size virtual interface for scalable token synchronization."""

    def __init__(
        self,
        hidden_dim: int,
        slots: int,
        *,
        recognition_mode: str = "explicit",
        recognition_threshold: float = 0.5,
        recognition_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.randn(slots, hidden_dim) * hidden_dim**-0.5)
        self.read = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.write = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim**-0.5

    def forward(self, z: Tensor, mask: Tensor, visibility: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        read_logits = torch.einsum("bnd,sd->bns", self.read(z), self.slots) * self.scale
        slot_read_weights = masked_softmax(read_logits, mask.unsqueeze(-1).expand_as(read_logits), dim=1)

        written = self.write(z)
        if visibility is None:
            interface_state = torch.einsum("bns,bnd->bsd", slot_read_weights, written)
            read_weights = slot_read_weights
        else:
            pair_weights = slot_read_weights.unsqueeze(1) * visibility.unsqueeze(-1).to(z.dtype)
            normalizer = pair_weights.sum(dim=2, keepdim=True).clamp_min(torch.finfo(z.dtype).eps)
            pair_weights = pair_weights / normalizer
            interface_state = torch.einsum("bnms,bmd->bnsd", pair_weights, written)
            read_weights = pair_weights.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1).to(z.dtype)

        write_logits = torch.einsum("bnd,bnsd->bns", z, interface_state) * self.scale if interface_state.ndim == 4 else torch.einsum("bnd,bsd->bns", z, interface_state) * self.scale
        write_weights = torch.softmax(write_logits, dim=-1)
        context = torch.einsum("bns,bnsd->bnd", write_weights, interface_state) if interface_state.ndim == 4 else torch.einsum("bns,bsd->bnd", write_weights, interface_state)
        return self.out(context), read_weights, write_weights


class ARTILatentRecallField(nn.Module):
    """Private latent recall slots used as low-channel internal condition."""

    def __init__(
        self,
        hidden_dim: int,
        slots: int,
        *,
        recognition_mode: str = "explicit",
        recognition_threshold: float = 0.5,
        recognition_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.bank = nn.Parameter(torch.randn(slots, hidden_dim) * hidden_dim**-0.5)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.external = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        if recognition_mode not in {"explicit", "alignment", "none"}:
            raise ValueError("recognition_mode must be 'explicit', 'alignment', or 'none'")
        self.recognition_mode = recognition_mode
        self.register_buffer("recognition_threshold", torch.tensor(float(recognition_threshold)), persistent=False)
        self.register_buffer("recognition_temperature", torch.tensor(float(recognition_temperature)), persistent=False)
        self.alignment_recognizer = nn.Linear(hidden_dim * 2, 1) if recognition_mode == "alignment" else None
        self.scale = hidden_dim**-0.5

    def forward(self, z: Tensor, mask: Tensor, recall: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bank = self.bank
        if recall is not None:
            if recall.ndim != 3 or recall.shape[0] != z.shape[0] or recall.shape[2] != z.shape[2]:
                raise ValueError("recall must have shape [B, K, H]")
            bank = torch.cat([bank.unsqueeze(0).expand(z.shape[0], -1, -1), recall.to(z)], dim=1)
            logits = torch.einsum("bnd,bkd->bnk", self.query(z), bank) * self.scale
            weights = torch.softmax(logits, dim=-1)
            context = torch.einsum("bnk,bkd->bnd", weights, bank)
            external_context = self.external(recall.to(z).mean(dim=1, keepdim=True)).expand_as(context)
            context = context + external_context
        else:
            logits = torch.einsum("bnd,kd->bnk", self.query(z), bank) * self.scale
            weights = torch.softmax(logits, dim=-1)
            context = torch.einsum("bnk,kd->bnd", weights, bank)

        query = self.query(z)
        similarity = torch.cosine_similarity(z, context, dim=-1, eps=1e-6)
        if self.recognition_mode == "explicit":
            temperature = self.recognition_temperature.to(z).clamp_min(torch.finfo(z.dtype).eps)
            recognition = torch.sigmoid((similarity - self.recognition_threshold.to(z)) / temperature)
        elif self.recognition_mode == "alignment":
            assert self.alignment_recognizer is not None
            recognition = torch.sigmoid(self.alignment_recognizer(torch.cat([query, context], dim=-1))).squeeze(-1)
        else:
            recognition = torch.ones_like(similarity)
        recognition = recognition * mask.to(z.dtype)
        gate = torch.sigmoid(self.gate(torch.cat([z, context], dim=-1))) * recognition.unsqueeze(-1)
        return gate * context, weights, gate, recognition


class ARTIDynamicStateLayer(nn.Module):
    """Runtime latent state update with phase, interface, recall, and gated residuals."""

    def __init__(self, config: ARTIConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.phase = ARTIPhaseMixer(hidden_dim, config.coord_dim, config.operator_count, config.dropout) if config.use_phase_mixer else None
        self.interface = ARTIVirtualInterfaceMixer(hidden_dim, config.interface_slots) if config.use_virtual_interface else None
        self.recall = (
            ARTILatentRecallField(
                hidden_dim,
                config.recall_slots,
                recognition_mode=config.recall_recognition_mode,
                recognition_threshold=config.recall_recognition_threshold,
                recognition_temperature=config.recall_recognition_temperature,
            )
            if config.use_recall and config.recall_steps > 0
            else None
        )
        self.recall_activation = Half() if config.recall_activation == "half" else nn.Identity()
        self.update_gate = nn.Linear(hidden_dim * 5 + config.coord_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)
        init_arti_module(self)

    def forward(
        self,
        z: Tensor,
        coord: Tensor,
        mask: Tensor,
        visibility: Tensor | None = None,
        recall: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        pairwise_visibility = ensure_visibility(visibility, mask) if self.config.use_pairwise_context else visibility

        if self.config.use_pairwise_context:
            visible_logits = torch.einsum("bnd,bmd->bnm", z, z) * (z.shape[-1] ** -0.5)
            visible_weights = masked_softmax(visible_logits, pairwise_visibility, dim=-1)
            visible_context = torch.einsum("bnm,bmd->bnd", visible_weights, z)
        else:
            visible_weights = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)
            visible_context = torch.zeros_like(z)

        if self.config.use_phase_mixer:
            assert self.phase is not None
            phase_context, operator_weights, phase_receptor_gain = self.phase(z, coord)
        else:
            phase_context = torch.zeros_like(z)
            operator_weights = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)
            phase_receptor_gain = torch.empty(z.shape[0], z.shape[1], 0, z.shape[-1], device=z.device, dtype=z.dtype)

        if self.config.use_virtual_interface:
            assert self.interface is not None
            interface_context, read_weights, write_weights = self.interface(z, mask, pairwise_visibility)
        else:
            interface_context = torch.zeros_like(z)
            read_weights = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)
            write_weights = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)

        recall_context = torch.zeros_like(z)
        recall_weights = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)
        recall_gate = torch.zeros_like(z)
        recall_recognition = torch.zeros(*z.shape[:2], device=z.device, dtype=z.dtype)
        raw_recall_context = torch.zeros_like(z)
        recall_steps = self.config.recall_steps if self.config.use_recall else 0
        for _ in range(recall_steps):
            assert self.recall is not None
            raw_recall_context, recall_weights, recall_gate, recall_recognition = self.recall(z, mask, recall)
            recall_context = self.recall_activation(raw_recall_context)
            z = z + self.dropout(recall_context)

        update_input = torch.cat([z, coord, phase_context, interface_context, visible_context, recall_context], dim=-1)
        gate = torch.sigmoid(self.update_gate(update_input)) * mask.unsqueeze(-1).to(z.dtype)
        candidate = phase_context + interface_context + visible_context + recall_context
        updated = self.norm(z + self.dropout(gate * candidate))
        updated = updated * mask.unsqueeze(-1).to(updated.dtype)

        diagnostics = {
            "operator_weights": operator_weights,
            "phase_receptor_gain": phase_receptor_gain,
            "interface_read_weights": read_weights,
            "interface_write_weights": write_weights,
            "visibility_weights": visible_weights,
            "recall_bank_weights": recall_weights,
            "recall_gate": recall_gate,
            "recall_recognition": recall_recognition,
            "recall_activation_half": torch.full(
                (z.shape[0],),
                1.0 if self.config.recall_activation == "half" and recall_steps > 0 else 0.0,
                device=z.device,
                dtype=z.dtype,
            ),
            "recall_activation_survival_ratio": recall_context.norm(dim=-1) / raw_recall_context.norm(dim=-1).clamp_min(torch.finfo(z.dtype).eps),
            "residual_gate": gate,
            "residual_norm": (updated - z).norm(dim=-1),
            "mask_coverage": mask_coverage(mask),
        }
        return updated, diagnostics, recall_context


class ARTILatentTensorLayer(nn.Module):
    """Project anonymous hidden tensors into a dynamic latent space."""

    def __init__(self, config: ARTIConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.in_proj = nn.Linear(config.input_dim, hidden_dim)
        self.state = ARTIDynamicStateLayer(config)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.virtual_recall_proj = nn.Linear(hidden_dim, hidden_dim) if config.use_virtual_recall else None
        if config.coord_dim > 0 and config.fallback_context in {"random_coord", "random_context"}:
            fallback_coord = torch.randn(config.fallback_slots, config.coord_dim) * (config.coord_dim**-0.5)
            self.register_buffer("fallback_coord_bank", fallback_coord)
        else:
            self.register_buffer("fallback_coord_bank", torch.empty(0, config.coord_dim))
        init_arti_module(self.in_proj)
        init_arti_module(self.out_proj)
        if self.virtual_recall_proj is not None:
            init_arti_module(self.virtual_recall_proj)

    def forward(
        self,
        x: Tensor,
        coord: Tensor | None = None,
        mask: Tensor | None = None,
        visibility: Tensor | None = None,
        recall: Tensor | None = None,
        frame_operators: Tensor | None = None,
        observer_coord: Tensor | None = None,
    ) -> ARTIOutput:
        assert_floating_tensor("x", x)
        seq, was_vector = as_sequence(x)
        batch, tokens, _ = seq.shape
        token_mask = ensure_mask(mask, batch, tokens, seq.device)
        if self.config.require_coord and coord is None:
            raise ValueError("coord is required by this ARTI configuration")
        token_coord = self._resolve_coord(coord, batch, tokens, seq.device, seq.dtype)
        if self.config.require_visibility and visibility is None:
            raise ValueError("visibility is required by this ARTI configuration")
        visibility = self._resolve_visibility(visibility, token_mask)

        seq_canonical = apply_coord_frame_inverse(
            seq,
            token_coord,
            self.config.coord_frame_mode,
            frame_operators,
            observer_coord=observer_coord,
        ) * token_mask.unsqueeze(-1).to(seq.dtype)
        z = self.in_proj(seq_canonical) * token_mask.unsqueeze(-1).to(seq.dtype)
        z, diagnostics, recall_influence = self.state(z, token_coord, token_mask, visibility, recall)
        y_seq = self.out_proj(z) * token_mask.unsqueeze(-1).to(z.dtype)
        virtual_seq = None
        recall_trace = None
        recall_prediction = None
        if self.virtual_recall_proj is not None:
            virtual_seq = self.virtual_recall_proj(z) * token_mask.unsqueeze(-1).to(z.dtype)
            recall_trace = virtual_seq
            recall_prediction = virtual_seq
        pooled = masked_mean(y_seq, token_mask, dim=1)
        diagnostics["pooled_mean"] = pooled.mean(dim=-1)
        diagnostics["pooled_std"] = pooled.std(dim=-1, unbiased=False)
        if recall_trace is not None and recall_prediction is not None:
            diagnostics["experiential_recall_trace_norm"] = recall_trace.norm(dim=-1)
            diagnostics["experiential_recall_prediction_norm"] = recall_prediction.norm(dim=-1)
            diagnostics["experiential_recall_familiarity"] = torch.cosine_similarity(
                recall_prediction,
                y_seq.detach(),
                dim=-1,
                eps=1e-6,
            ) * token_mask.to(y_seq.dtype)
        diagnostics["coord_frame_delta"] = (seq_canonical - seq).norm(dim=-1)
        diagnostics["observer_frame_active"] = torch.full(
            (batch,),
            1.0 if observer_coord is not None and self.config.coord_frame_mode != "none" else 0.0,
            device=seq.device,
            dtype=seq.dtype,
        )

        y = restore_input_rank(y_seq, was_vector and self.config.return_input_shape)
        virtual_y = None if virtual_seq is None else restore_input_rank(virtual_seq, was_vector and self.config.return_input_shape)
        trace = None if recall_trace is None else restore_input_rank(recall_trace, was_vector and self.config.return_input_shape)
        prediction = None if recall_prediction is None else restore_input_rank(recall_prediction, was_vector and self.config.return_input_shape)
        influence = restore_input_rank(recall_influence, was_vector and self.config.return_input_shape)
        return ARTIOutput(
            y=y,
            pooled=pooled,
            virtual_y=virtual_y,
            recall_trace=trace,
            recall_prediction=prediction,
            recall_influence=influence,
            diagnostics=detach_diagnostics(diagnostics),
        )

    def _resolve_coord(self, coord: Tensor | None, batch: int, tokens: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if coord is not None or self.config.fallback_context == "none" or self.config.coord_dim == 0:
            return ensure_coord(coord, batch, tokens, self.config.coord_dim, device, dtype)
        bank = self.fallback_coord_bank.to(device=device, dtype=dtype)
        if bank.shape[0] == 0:
            return ensure_coord(coord, batch, tokens, self.config.coord_dim, device, dtype)
        index = torch.arange(tokens, device=device) % bank.shape[0]
        return bank.index_select(0, index).unsqueeze(0).expand(batch, -1, -1)

    def _resolve_visibility(self, visibility: Tensor | None, mask: Tensor) -> Tensor | None:
        if visibility is not None or self.config.fallback_context != "random_context":
            return visibility
        return mask.unsqueeze(1) & mask.unsqueeze(2)


class ARTILayer(ARTILatentTensorLayer):
    """Convenience constructor for the default latent tensor layer."""

    def __init__(
        self,
        input_dim: int,
        coord_dim: int = 0,
        hidden_dim: int | None = None,
        operator_count: int = 4,
        interface_slots: int = 8,
        recall_slots: int = 4,
        recall_steps: int = 1,
        recall_activation: str = "half",
        recall_recognition_mode: str = "explicit",
        recall_recognition_threshold: float = 0.5,
        recall_recognition_temperature: float = 0.1,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_phase_mixer: bool = True,
        use_virtual_interface: bool = True,
        use_pairwise_context: bool = True,
        use_recall: bool = True,
        use_virtual_recall: bool = True,
        require_coord: bool = False,
        require_visibility: bool = False,
        coord_frame_mode: str = "none",
        fallback_context: str = "none",
        fallback_slots: int = 32,
    ) -> None:
        super().__init__(
            ARTIConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                coord_dim=coord_dim,
                operator_count=operator_count,
                interface_slots=interface_slots,
                recall_slots=recall_slots,
                recall_steps=recall_steps,
                recall_activation=recall_activation,
                recall_recognition_mode=recall_recognition_mode,
                recall_recognition_threshold=recall_recognition_threshold,
                recall_recognition_temperature=recall_recognition_temperature,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                use_phase_mixer=use_phase_mixer,
                use_virtual_interface=use_virtual_interface,
                use_pairwise_context=use_pairwise_context,
                use_recall=use_recall,
                use_virtual_recall=use_virtual_recall,
                require_coord=require_coord,
                require_visibility=require_visibility,
                coord_frame_mode=coord_frame_mode,
                fallback_context=fallback_context,
                fallback_slots=fallback_slots,
            )
        )
