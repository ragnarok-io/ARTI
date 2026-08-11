"""Contracts and validation helpers for Recall formulas.

This module deliberately contains no Recall routing, bank access, formula
implementation, registry, or serialization logic. A formula receives the
current state and one static factor tensor and returns the complete next state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from itertools import islice
from types import MappingProxyType
from typing import Final, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn


RecallOutputSemantics = Literal["next_state"]
MAX_RECALL_FORMULA_FACTORS: Final = 256
MAX_RECALL_FORMULA_PROBE_ELEMENTS: Final = 1_000_000

_COMPONENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_FORMULA_NAME_RE = re.compile(
    r"^[a-z][a-z0-9_]*(?:[./-][a-z][a-z0-9_]*)*$"
)
_INIT_KINDS: Final = frozenset({"zero", "normal"})
_FORBIDDEN_CONTROL_NAMES: Final = frozenset(
    {
        "configure_optimizer",
        "create_optimizer",
        "optimizer",
        "optimizer_step",
        "register_gradient_hook",
        "register_gradient_hooks",
        "set_requires_grad",
    }
)


def _validate_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _COMPONENT_RE.fullmatch(value):
        raise ValueError(
            f"{field} must match {_COMPONENT_RE.pattern!r}; received {value!r}"
        )
    return value


@dataclass(frozen=True)
class FactorSpec:
    """Stable metadata for one host-dimensional Recall factor.

    ``route`` identifies factors that may share a routing decision. ``identity``
    is the scalar factor value used by optional identity checks. ``init`` is
    descriptive initialization metadata; this protocol does not initialize a
    bank itself.
    """

    name: str
    route: str | None = None
    identity: float = 0.0
    init: str = "normal"
    init_scale: float | None = None

    def __post_init__(self) -> None:
        name = _validate_component(self.name, field="FactorSpec.name")
        route = name if self.route is None else self.route
        route = _validate_component(route, field="FactorSpec.route")
        object.__setattr__(self, "route", route)

        if isinstance(self.identity, bool) or not isinstance(self.identity, (int, float)):
            raise TypeError("FactorSpec.identity must be a finite real number")
        identity = float(self.identity)
        if not math.isfinite(identity):
            raise ValueError("FactorSpec.identity must be finite")
        object.__setattr__(self, "identity", identity)

        if self.init not in _INIT_KINDS:
            choices = ", ".join(sorted(_INIT_KINDS))
            raise ValueError(f"FactorSpec.init must be one of: {choices}")
        if self.init_scale is not None:
            if isinstance(self.init_scale, bool) or not isinstance(
                self.init_scale, (int, float)
            ):
                raise TypeError("FactorSpec.init_scale must be a finite positive number")
            init_scale = float(self.init_scale)
            if not math.isfinite(init_scale) or init_scale <= 0:
                raise ValueError("FactorSpec.init_scale must be finite and positive")
            object.__setattr__(self, "init_scale", init_scale)
        if self.init == "zero" and self.init_scale is not None:
            raise ValueError("zero initialization does not accept init_scale")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "route": self.route,
            "identity": self.identity,
            "init": self.init,
            "init_scale": self.init_scale,
        }


@dataclass(frozen=True)
class FormulaIdentity:
    """Stable semantic identity for a Recall formula."""

    name: str
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _FORMULA_NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"FormulaIdentity.name must match {_FORMULA_NAME_RE.pattern!r}; "
                f"received {self.name!r}"
            )
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("FormulaIdentity.version must be an integer")
        if self.version <= 0:
            raise ValueError("FormulaIdentity.version must be positive")

    @property
    def canonical_id(self) -> str:
        """Return the immutable ``name-vN`` identifier."""

        return f"{self.name}-v{self.version}"

    def __str__(self) -> str:
        return self.canonical_id

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class RecallFormulaContract:
    """Static contract implemented by a Recall formula module."""

    factors: tuple[FactorSpec, ...]
    identity: FormulaIdentity | None = None
    output_semantics: RecallOutputSemantics = "next_state"
    identity_preserving: bool = False
    api_version: int = 1

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        if not factors:
            raise ValueError("RecallFormulaContract requires at least one factor")
        if any(not isinstance(factor, FactorSpec) for factor in factors):
            raise TypeError("RecallFormulaContract.factors must contain FactorSpec values")
        names = tuple(factor.name for factor in factors)
        if len(set(names)) != len(names):
            raise ValueError("RecallFormulaContract factor names must be unique")
        object.__setattr__(self, "factors", factors)

        if self.identity is not None and not isinstance(self.identity, FormulaIdentity):
            raise TypeError("RecallFormulaContract.identity must be FormulaIdentity or None")
        if self.output_semantics != "next_state":
            raise ValueError(
                "Recall formulas must return next_state; delta output semantics are not supported"
            )
        if not isinstance(self.identity_preserving, bool):
            raise TypeError("RecallFormulaContract.identity_preserving must be bool")
        if isinstance(self.api_version, bool) or not isinstance(self.api_version, int):
            raise TypeError("RecallFormulaContract.api_version must be an integer")
        if self.api_version <= 0:
            raise ValueError("RecallFormulaContract.api_version must be positive")

    @property
    def factor_names(self) -> tuple[str, ...]:
        """Return factor names in their runtime tensor order."""

        return tuple(factor.name for factor in self.factors)

    @property
    def factor_count(self) -> int:
        """Return the static factor-axis size."""

        return len(self.factors)

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": None if self.identity is None else self.identity.to_dict(),
            "factors": [factor.to_dict() for factor in self.factors],
            "output_semantics": self.output_semantics,
            "identity_preserving": self.identity_preserving,
            "api_version": self.api_version,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RecallFormulaDescription:
    """Human-facing metadata for a formula contract."""

    contract: RecallFormulaContract
    summary: str
    legacy_value_composition: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contract, RecallFormulaContract):
            raise TypeError("RecallFormulaDescription.contract must be RecallFormulaContract")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("RecallFormulaDescription.summary must be non-empty")
        object.__setattr__(self, "summary", self.summary.strip())
        if self.legacy_value_composition is not None:
            _validate_component(
                self.legacy_value_composition,
                field="RecallFormulaDescription.legacy_value_composition",
            )


def _factor(
    name: str,
    *,
    route: str | None = None,
    identity: float = 0.0,
    init: str = "zero",
) -> FactorSpec:
    return FactorSpec(name=name, route=route, identity=identity, init=init)


_DELTA_CONTRACT = RecallFormulaContract(
    identity=FormulaIdentity("delta", 1),
    factors=(_factor("content"),),
    identity_preserving=True,
)

_AFFINE_CONTRACT = RecallFormulaContract(
    identity=FormulaIdentity("affine", 1),
    factors=(
        _factor("scale"),
        _factor("shift"),
    ),
    identity_preserving=True,
)

_STATE_CONTRACT = RecallFormulaContract(
    identity=FormulaIdentity("state", 1),
    factors=(
        _factor("coarse_content"),
        _factor("fine_content"),
        *tuple(_factor(f"modulation_{index:02d}") for index in range(13)),
        _factor("direction"),
        _factor("opacity"),
    ),
    identity_preserving=True,
)

BUILTIN_RECALL_FORMULAS: Final[Mapping[str, RecallFormulaDescription]] = MappingProxyType(
    {
        "delta-v1": RecallFormulaDescription(
            contract=_DELTA_CONTRACT,
            summary="One-factor Recall formula compatible with the legacy single mode.",
            legacy_value_composition="single",
        ),
        "affine-v1": RecallFormulaDescription(
            contract=_AFFINE_CONTRACT,
            summary="Two-factor Recall formula compatible with the legacy product mode.",
            legacy_value_composition="product",
        ),
        "state-v1": RecallFormulaDescription(
            contract=_STATE_CONTRACT,
            summary="Seventeen-factor Recall formula compatible with the legacy state mode.",
            legacy_value_composition="state",
        ),
    }
)

BUILTIN_RECALL_FORMULA_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "single": "delta-v1",
        "product": "affine-v1",
        "state": "state-v1",
    }
)


def _module_contract(module: nn.Module) -> RecallFormulaContract | None:
    contract = getattr(module, "recall_formula_contract", None)
    if contract is None:
        return None
    if not isinstance(contract, RecallFormulaContract):
        raise TypeError("module.recall_formula_contract must be RecallFormulaContract")
    return contract


def _resolve_factor_specs(
    module: nn.Module,
    *,
    factor_count: int | None,
    factor_names: Sequence[str] | None,
) -> tuple[FactorSpec, ...]:
    contract = _module_contract(module)
    contract_names = contract.factor_names if contract is not None else None
    module_names = getattr(module, "factor_names", None)

    if factor_count is not None:
        if isinstance(factor_count, bool) or not isinstance(factor_count, int):
            raise TypeError("factor_count must be an integer")
        if factor_count <= 0:
            raise ValueError("factor_count must be positive")
        if factor_count > MAX_RECALL_FORMULA_FACTORS:
            raise ValueError(
                "Recall formula factor count exceeds the validation limit of "
                f"{MAX_RECALL_FORMULA_FACTORS}"
            )

    def bounded_names(values: Sequence[str], *, field: str) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{field} must be a sequence, not a string")
        try:
            items = tuple(islice(iter(values), MAX_RECALL_FORMULA_FACTORS + 1))
        except TypeError as error:
            raise TypeError(f"{field} must be an iterable of names") from error
        if len(items) > MAX_RECALL_FORMULA_FACTORS:
            raise ValueError(
                "Recall formula factor count exceeds the validation limit of "
                f"{MAX_RECALL_FORMULA_FACTORS}"
            )
        return tuple(_validate_component(name, field=field) for name in items)

    explicit_names: tuple[str, ...] | None = None
    if factor_names is not None:
        explicit_names = bounded_names(factor_names, field="factor_names")
        if not explicit_names:
            raise ValueError("factor_names must not be empty")
        if len(set(explicit_names)) != len(explicit_names):
            raise ValueError("factor_names must be unique")

    declared_names: tuple[str, ...] | None = None
    if module_names is not None:
        declared_names = bounded_names(module_names, field="module.factor_names")
        if not declared_names or len(set(declared_names)) != len(declared_names):
            raise ValueError("module.factor_names must be non-empty and unique")

    candidates = [
        names for names in (explicit_names, contract_names, declared_names) if names is not None
    ]
    if any(len(names) > MAX_RECALL_FORMULA_FACTORS for names in candidates):
        raise ValueError(
            "Recall formula factor count exceeds the validation limit of "
            f"{MAX_RECALL_FORMULA_FACTORS}"
        )
    if candidates and any(names != candidates[0] for names in candidates[1:]):
        raise ValueError("factor_names disagree with the module's declared Recall contract")
    names = candidates[0] if candidates else None

    if names is not None and factor_count is not None and len(names) != factor_count:
        raise ValueError(
            f"factor_count={factor_count} does not match {len(names)} factor_names"
        )
    if names is None:
        if factor_count is None:
            raise ValueError(
                "provide factor_count or factor_names, or declare a Recall formula contract"
            )
        names = tuple(f"factor_{index}" for index in range(factor_count))

    if contract is not None:
        return contract.factors
    return tuple(FactorSpec(name=name) for name in names)


def _deterministic_factors(x: Tensor, factor_count: int) -> Tensor:
    shape = (*x.shape[:-1], factor_count, x.shape[-1])
    count = math.prod(shape)
    return torch.linspace(
        -0.25,
        0.25,
        count,
        device=x.device,
        dtype=x.dtype,
    ).reshape(shape)


def _validate_formula_output(output: object, x: Tensor) -> Tensor:
    if not isinstance(output, Tensor):
        raise TypeError(
            "Recall formula must return one Tensor containing next_state; "
            "delta, tuple, and mapping outputs are not supported"
        )
    if output.shape != x.shape:
        raise ValueError(
            f"Recall formula next_state shape must be {tuple(x.shape)}, "
            f"received {tuple(output.shape)}"
        )
    if output.device != x.device:
        raise ValueError(
            f"Recall formula next_state device must be {x.device}, received {output.device}"
        )
    if output.dtype != x.dtype:
        raise ValueError(
            f"Recall formula next_state dtype must be {x.dtype}, received {output.dtype}"
        )
    if not bool(torch.isfinite(output).all()):
        raise ValueError("Recall formula next_state must contain only finite values")
    return output


def _module_state_changed(module: nn.Module, snapshot: Mapping[str, Tensor]) -> bool:
    current = module.state_dict()
    return set(current) != set(snapshot) or any(
        not torch.equal(current[name], value) for name, value in snapshot.items()
    )


def check_recall_formula(
    module: nn.Module,
    x: Tensor,
    *,
    factor_count: int | None = None,
    factor_names: Sequence[str] | None = None,
    test_identity: bool = False,
    identity_atol: float = 1e-6,
    identity_rtol: float = 1e-5,
) -> RecallFormulaContract:
    """Validate a custom Recall formula against the tensor contract.

    The runtime call is always ``module(x, factors)`` where ``factors`` has
    shape ``[..., F, D]``. The single returned tensor is always interpreted as
    the complete next state, never as a delta.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("module must be an instance of torch.nn.Module")
    for name in _FORBIDDEN_CONTROL_NAMES:
        if hasattr(module, name):
            raise ValueError(
                "Recall formula must not own optimizer, gradient-hook, or "
                f"requires-grad control; remove {name!r}"
            )
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim < 1 or x.shape[-1] <= 0:
        raise ValueError("x must have shape [..., D] with D > 0")
    if not x.is_floating_point():
        raise TypeError("x must have a floating-point dtype")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x must contain only finite values")
    if not isinstance(test_identity, bool):
        raise TypeError("test_identity must be bool")
    for value, name in (
        (identity_atol, "identity_atol"),
        (identity_rtol, "identity_rtol"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite non-negative number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    declared_semantics = getattr(module, "output_semantics", "next_state")
    if declared_semantics != "next_state":
        raise ValueError(
            "Recall formulas must declare output_semantics='next_state'; "
            "delta output semantics are ambiguous and unsupported"
        )

    factors = _resolve_factor_specs(
        module,
        factor_count=factor_count,
        factor_names=factor_names,
    )
    if len(factors) > MAX_RECALL_FORMULA_FACTORS:
        raise ValueError(
            "Recall formula factor count exceeds the validation limit of "
            f"{MAX_RECALL_FORMULA_FACTORS}"
        )
    probe_elements = x.numel() * len(factors)
    if probe_elements > MAX_RECALL_FORMULA_PROBE_ELEMENTS:
        raise ValueError(
            "Recall formula validation probe exceeds the element limit of "
            f"{MAX_RECALL_FORMULA_PROBE_ELEMENTS}"
        )
    contract = _module_contract(module)
    if contract is None:
        contract = RecallFormulaContract(
            factors=factors,
            output_semantics="next_state",
            identity_preserving=test_identity,
        )

    probe_state = x.detach().clone()
    original_probe_state = probe_state.clone()
    probe_factors = _deterministic_factors(probe_state, len(factors))
    module_state = {
        name: value.detach().clone() for name, value in module.state_dict().items()
    }
    training = module.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        with torch.no_grad():
            output = module(probe_state, probe_factors)
            input_mutated = not torch.equal(probe_state, original_probe_state)
            state_mutated = _module_state_changed(module, module_state)
            if input_mutated or state_mutated:
                raise ValueError(
                    "Recall formula must not mutate its input or module state during forward"
                )
            _validate_formula_output(output, probe_state)

            if test_identity:
                identity_factors = torch.empty_like(probe_factors)
                for index, factor in enumerate(factors):
                    identity_factors[..., index, :].fill_(factor.identity)
                identity_input = original_probe_state.clone()
                identity_output = _validate_formula_output(
                    module(identity_input, identity_factors),
                    identity_input,
                )
                if not torch.equal(
                    identity_input, original_probe_state
                ) or _module_state_changed(module, module_state):
                    raise ValueError(
                        "Recall formula must not mutate its input or module state during forward"
                    )
                if not torch.allclose(
                    identity_output,
                    original_probe_state,
                    atol=float(identity_atol),
                    rtol=float(identity_rtol),
                ):
                    max_error = (identity_output - original_probe_state).abs().max().item()
                    raise ValueError(
                        "Recall formula failed its next_state identity check: "
                        f"maximum absolute error was {max_error:.6g}"
                    )
    finally:
        module.load_state_dict(module_state, strict=False)
        module.train(training)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    return contract


__all__ = [
    "BUILTIN_RECALL_FORMULA_ALIASES",
    "BUILTIN_RECALL_FORMULAS",
    "FactorSpec",
    "FormulaIdentity",
    "MAX_RECALL_FORMULA_FACTORS",
    "MAX_RECALL_FORMULA_PROBE_ELEMENTS",
    "RecallFormulaContract",
    "RecallFormulaDescription",
    "RecallOutputSemantics",
    "check_recall_formula",
]
