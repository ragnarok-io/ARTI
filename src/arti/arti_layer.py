"""Stable tensor layer backed by the composable AdaptivePulse executor."""

from __future__ import annotations

from typing import Any, ClassVar

from torch import Tensor, nn

from .adaptive_pulse import AdaptivePulse, PulseOutput


class ARTILayer(nn.Module):
    """Tensor-in/tensor-out host for one versioned AdaptivePulse graph.

    The layer owns no duplicate mechanism mathematics. Observation, Half,
    Fold, Formula execution, UnFold, aggregation, and Bank updates remain
    modules of the supplied :class:`AdaptivePulse`. An empty Pulse is a true
    identity, which makes the default layer safe to insert before configuring
    an application-specific execution graph.
    """

    _component_reference: ClassVar[str] = "arti/layer@2"
    output_semantics: ClassVar[str] = "next_state"

    def __init__(self, pulse: AdaptivePulse | None = None) -> None:
        super().__init__()
        if pulse is not None and not isinstance(pulse, AdaptivePulse):
            raise TypeError("pulse must be an AdaptivePulse or None")
        self.pulse = AdaptivePulse() if pulse is None else pulse

    @property
    def manifest(self):
        """Return the immutable Pulse stage graph used by this layer."""

        return self.pulse.manifest

    def runtime_provenance(self) -> dict[str, object]:
        """Describe the Pulse attachment surface and its Federal boundary."""

        return {
            "surface": "adaptive-pulse",
            "layer_ref": self._component_reference,
            "operation_graph_ref": self.pulse._component_reference,
            "operation_graph_fingerprint": self.manifest.fingerprint,
            "federal_compiler_ref": "arti/federal-static-compiler@1",
            "federal_source_snapshot_required": True,
        }

    def run(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
        observed: Tensor | None = None,
        exposed: Tensor | None = None,
        intervened: Tensor | None = None,
        **pulse_inputs: Any,
    ) -> PulseOutput:
        """Execute the configured Pulse and retain its typed result."""

        return self.pulse.run_tensor(
            x,
            mask=mask,
            observed=observed,
            exposed=exposed,
            intervened=intervened,
            **pulse_inputs,
        )

    def forward(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
        observed: Tensor | None = None,
        exposed: Tensor | None = None,
        intervened: Tensor | None = None,
        return_info: bool = False,
        **pulse_inputs: Any,
    ) -> Tensor | tuple[Tensor, PulseOutput]:
        """Return the next tensor state, optionally with the typed Pulse result."""

        result = self.run(
            x,
            mask=mask,
            observed=observed,
            exposed=exposed,
            intervened=intervened,
            **pulse_inputs,
        )
        return (result.value, result) if return_info else result.value

    def extra_repr(self) -> str:
        enabled = ",".join(
            stage.stage_id for stage in self.manifest.stages if stage.component_ref is not None
        )
        return f"pulse={self.pulse._component_reference}, enabled={enabled or 'identity'}"


__all__ = ["ARTILayer"]
