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

from .recall_registry import RecallFormulaId


RecallOutputSemantics = Literal["next_state"]
RecallFormulaComposition = Literal["custom", "single", "product", "state"]
RecallFormulaVectorization = Literal["scalar_vmap", "batched"]
RecallFormulaLayout = Literal["generic", "contiguous_mfd"]
RecallFormulaAccumulation = Literal["activation", "float32"]
MAX_RECALL_FORMULA_FACTORS: Final = 256
MAX_RECALL_FORMULA_PROBE_ELEMENTS: Final = 1_000_000
RECALL_FORMULA_CONTRACT_API_VERSION: Final = 2
RECALL_FORMULA_LOCK_VERSION: Final = 2

_COMPONENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_COMPOSITIONS: Final = frozenset({"custom", "single", "product", "state"})
_VECTORIZATION_MODES: Final = frozenset({"scalar_vmap", "batched"})
_LAYOUTS: Final = frozenset({"generic", "contiguous_mfd"})
_ACCUMULATION_DTYPES: Final = frozenset({"activation", "float32"})
_DTYPE_NAMES: Final = frozenset(
    {"floating", "float16", "bfloat16", "float32", "float64"}
)
_DTYPE_TO_NAME: Final = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}
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


def formula_dtype_supported(dtype: torch.dtype, supported_dtypes: Sequence[str]) -> bool:
    """Return whether a Formula execution contract admits ``dtype``."""

    dtype_name = _DTYPE_TO_NAME.get(dtype)
    if dtype_name is None:
        return False
    supported = frozenset(supported_dtypes)
    return "floating" in supported or dtype_name in supported


def _validate_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _COMPONENT_RE.fullmatch(value):
        raise ValueError(
            f"{field} must match {_COMPONENT_RE.pattern!r}; received {value!r}"
        )
    return value


@dataclass(frozen=True)
class RecallFormulaExecutionSpec:
    """Static execution and compiler capabilities of a Formula.

    This is metadata only. It never selects a backend or installs a compiler;
    the runtime remains responsible for checking the declaration and choosing
    eager or compiled execution.
    """

    vectorization: RecallFormulaVectorization = "scalar_vmap"
    layout: RecallFormulaLayout = "generic"
    supported_dtypes: tuple[str, ...] = ("floating",)
    accumulation_dtype: RecallFormulaAccumulation = "activation"
    supports_autograd: bool = True
    supports_compile: bool = False
    supports_triton: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.vectorization not in _VECTORIZATION_MODES:
            raise ValueError(
                "RecallFormulaExecutionSpec.vectorization must be 'scalar_vmap' or 'batched'"
            )
        if self.layout not in _LAYOUTS:
            raise ValueError(
                "RecallFormulaExecutionSpec.layout must be 'generic' or 'contiguous_mfd'"
            )
        dtypes = tuple(self.supported_dtypes)
        if not dtypes or any(dtype not in _DTYPE_NAMES for dtype in dtypes):
            raise ValueError(
                "RecallFormulaExecutionSpec.supported_dtypes contains an unsupported dtype"
            )
        if len(set(dtypes)) != len(dtypes):
            raise ValueError("RecallFormulaExecutionSpec.supported_dtypes must be unique")
        if tuple(sorted(dtypes)) != dtypes:
            raise ValueError(
                "RecallFormulaExecutionSpec.supported_dtypes must be sorted"
            )
        object.__setattr__(self, "supported_dtypes", dtypes)
        if self.accumulation_dtype not in _ACCUMULATION_DTYPES:
            raise ValueError(
                "RecallFormulaExecutionSpec.accumulation_dtype must be 'activation' or 'float32'"
            )
        for value, name in (
            (self.supports_autograd, "supports_autograd"),
            (self.supports_compile, "supports_compile"),
            (self.supports_triton, "supports_triton"),
            (self.deterministic, "deterministic"),
        ):
            if type(value) is not bool:
                raise TypeError(f"RecallFormulaExecutionSpec.{name} must be bool")
        if self.supports_triton and self.layout != "contiguous_mfd":
            raise ValueError(
                "Triton-capable Formula execution requires layout='contiguous_mfd'"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "vectorization": self.vectorization,
            "layout": self.layout,
            "supported_dtypes": list(self.supported_dtypes),
            "accumulation_dtype": self.accumulation_dtype,
            "supports_autograd": self.supports_autograd,
            "supports_compile": self.supports_compile,
            "supports_triton": self.supports_triton,
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RecallFormulaExecutionSpec":
        if not isinstance(value, Mapping):
            raise TypeError("RecallFormulaExecutionSpec payload must be a mapping")
        required = {
            "vectorization",
            "layout",
            "supported_dtypes",
            "accumulation_dtype",
            "supports_autograd",
            "supports_compile",
            "supports_triton",
            "deterministic",
        }
        if set(value) != required:
            raise ValueError(
                "RecallFormulaExecutionSpec payload contains missing or unknown fields"
            )
        dtypes = value["supported_dtypes"]
        if not isinstance(dtypes, Sequence) or isinstance(dtypes, (str, bytes)):
            raise TypeError("supported_dtypes must be a sequence")
        return cls(
            vectorization=value["vectorization"],
            layout=value["layout"],
            supported_dtypes=tuple(dtypes),
            accumulation_dtype=value["accumulation_dtype"],
            supports_autograd=value["supports_autograd"],
            supports_compile=value["supports_compile"],
            supports_triton=value["supports_triton"],
            deterministic=value["deterministic"],
        )


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

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FactorSpec":
        """Restore a factor declaration without importing executable code."""

        if not isinstance(value, Mapping):
            raise TypeError("FactorSpec payload must be a mapping")
        required = {"name", "route", "identity", "init", "init_scale"}
        if set(value) != required:
            raise ValueError(
                "FactorSpec payload must contain exactly name, route, identity, "
                "init, and init_scale"
            )
        return cls(
            name=value["name"],
            route=value["route"],
            identity=value["identity"],
            init=value["init"],
            init_scale=value["init_scale"],
        )


@dataclass(frozen=True)
class RecallFormulaContract:
    """Static contract implemented by a Recall formula module."""

    factors: tuple[FactorSpec, ...]
    identity: RecallFormulaId | None = None
    output_semantics: RecallOutputSemantics = "next_state"
    identity_preserving: bool = False
    api_version: int = RECALL_FORMULA_CONTRACT_API_VERSION
    composition: RecallFormulaComposition = "custom"
    capabilities: tuple[str, ...] = ("torch.eager",)
    execution: RecallFormulaExecutionSpec = RecallFormulaExecutionSpec()

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

        if self.identity is not None and not isinstance(self.identity, RecallFormulaId):
            raise TypeError("RecallFormulaContract.identity must be RecallFormulaId or None")
        if self.output_semantics != "next_state":
            raise ValueError(
                "Recall formulas must return next_state; delta output semantics are not supported"
            )
        if not isinstance(self.identity_preserving, bool):
            raise TypeError("RecallFormulaContract.identity_preserving must be bool")
        if isinstance(self.api_version, bool) or not isinstance(self.api_version, int):
            raise TypeError("RecallFormulaContract.api_version must be an integer")
        if self.api_version != RECALL_FORMULA_CONTRACT_API_VERSION:
            raise ValueError(
                "unsupported RecallFormulaContract.api_version="
                f"{self.api_version!r}; expected {RECALL_FORMULA_CONTRACT_API_VERSION}"
            )
        if self.composition not in _COMPOSITIONS:
            choices = ", ".join(sorted(_COMPOSITIONS))
            raise ValueError(
                f"RecallFormulaContract.composition must be one of: {choices}"
            )
        capabilities = tuple(self.capabilities)
        if any(
            not isinstance(capability, str)
            or _CAPABILITY_RE.fullmatch(capability) is None
            for capability in capabilities
        ):
            raise ValueError(
                "RecallFormulaContract.capabilities must contain lowercase capability names"
            )
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("RecallFormulaContract.capabilities must be unique")
        if tuple(sorted(capabilities)) != capabilities:
            raise ValueError(
                "RecallFormulaContract.capabilities must be sorted for stable serialization"
            )
        object.__setattr__(self, "capabilities", capabilities)
        if not isinstance(self.execution, RecallFormulaExecutionSpec):
            raise TypeError(
                "RecallFormulaContract.execution must be RecallFormulaExecutionSpec"
            )

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
            "composition": self.composition,
            "capabilities": list(self.capabilities),
            "execution": self.execution.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RecallFormulaContract":
        """Restore a pure-data contract and reject schema drift."""

        if not isinstance(value, Mapping):
            raise TypeError("RecallFormulaContract payload must be a mapping")
        required = {
            "identity",
            "factors",
            "output_semantics",
            "identity_preserving",
            "api_version",
            "composition",
            "capabilities",
            "execution",
        }
        if set(value) != required:
            raise ValueError(
                "RecallFormulaContract payload contains missing or unknown fields"
            )
        identity_payload = value["identity"]
        identity = None
        if identity_payload is not None:
            identity = RecallFormulaId.from_dict(identity_payload)
        factor_payloads = value["factors"]
        if not isinstance(factor_payloads, Sequence) or isinstance(
            factor_payloads, (str, bytes)
        ):
            raise TypeError("RecallFormulaContract.factors must be a sequence")
        capabilities = value["capabilities"]
        if not isinstance(capabilities, Sequence) or isinstance(capabilities, (str, bytes)):
            raise TypeError("RecallFormulaContract.capabilities must be a sequence")
        return cls(
            factors=tuple(FactorSpec.from_dict(item) for item in factor_payloads),
            identity=identity,
            output_semantics=value["output_semantics"],
            identity_preserving=value["identity_preserving"],
            api_version=value["api_version"],
            composition=value["composition"],
            capabilities=tuple(capabilities),
            execution=RecallFormulaExecutionSpec.from_dict(value["execution"]),
        )


@dataclass(frozen=True)
class RecallFormulaLock:
    """Code-free binding of a Formula contract to one concrete Recall shape.

    The contract describes the mathematical interface. The lock fixes factor
    order, hidden width, slot count, and backend for one instantiated layer.
    It contains no module, optimizer, or tensor values, so it can be inspected
    and compared before loading weights.
    """

    contract: RecallFormulaContract
    hidden_dim: int
    slots: int
    backend: str = "torch"
    lock_version: int = RECALL_FORMULA_LOCK_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.contract, RecallFormulaContract):
            raise TypeError("RecallFormulaLock.contract must be RecallFormulaContract")
        if type(self.lock_version) is not int or self.lock_version != RECALL_FORMULA_LOCK_VERSION:
            raise ValueError(
                "unsupported RecallFormulaLock.lock_version; "
                f"expected {RECALL_FORMULA_LOCK_VERSION}"
            )
        for value, name in ((self.hidden_dim, "hidden_dim"), (self.slots, "slots")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.slots % self.contract.factor_count:
            raise ValueError(
                "slots must be divisible by the Formula factor count: "
                f"{self.slots} % {self.contract.factor_count} != 0"
            )
        if self.backend != "torch":
            raise ValueError("RecallFormulaLock.backend currently supports only 'torch'")

    @classmethod
    def bind(
        cls,
        contract: RecallFormulaContract,
        *,
        hidden_dim: int,
        slots: int,
        backend: str = "torch",
    ) -> "RecallFormulaLock":
        """Create the deterministic shape/backend binding for a Formula."""

        return cls(
            contract=contract,
            hidden_dim=hidden_dim,
            slots=slots,
            backend=backend,
        )

    @property
    def factor_count(self) -> int:
        return self.contract.factor_count

    @property
    def contract_fingerprint(self) -> str:
        return self.contract.fingerprint

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self.contract.factor_names

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self.contract.capabilities

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate(
        self,
        contract: RecallFormulaContract,
        *,
        hidden_dim: int,
        slots: int,
        backend: str = "torch",
    ) -> None:
        """Validate that a runtime Formula instance matches this lock."""

        if contract.fingerprint != self.contract_fingerprint:
            raise ValueError("Formula contract fingerprint does not match the lock")
        if hidden_dim != self.hidden_dim or slots != self.slots:
            raise ValueError("Formula shape does not match the lock")
        if backend != self.backend:
            raise ValueError("Formula backend does not match the lock")

    def to_dict(self) -> dict[str, object]:
        return {
            "lock_version": self.lock_version,
            "contract": self.contract.to_dict(),
            "contract_fingerprint": self.contract_fingerprint,
            "hidden_dim": self.hidden_dim,
            "slots": self.slots,
            "backend": self.backend,
            "layout": {
                "state": "[..., D]",
                "factors": "[..., F, D]",
                "factor_axis": -2,
            },
            "capabilities": list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RecallFormulaLock":
        """Restore and verify a lock without resolving or importing a Formula."""

        if not isinstance(value, Mapping):
            raise TypeError("RecallFormulaLock payload must be a mapping")
        required = {
            "lock_version",
            "contract",
            "contract_fingerprint",
            "hidden_dim",
            "slots",
            "backend",
            "layout",
            "capabilities",
        }
        if set(value) != required:
            raise ValueError("RecallFormulaLock payload contains missing or unknown fields")
        if value["layout"] != {
            "state": "[..., D]",
            "factors": "[..., F, D]",
            "factor_axis": -2,
        }:
            raise ValueError("RecallFormulaLock layout is unsupported")
        contract = RecallFormulaContract.from_dict(value["contract"])
        lock = cls(
            contract=contract,
            hidden_dim=value["hidden_dim"],
            slots=value["slots"],
            backend=value["backend"],
            lock_version=value["lock_version"],
        )
        if value["contract_fingerprint"] != lock.contract_fingerprint:
            raise ValueError("RecallFormulaLock contract fingerprint is invalid")
        capabilities = value["capabilities"]
        if not isinstance(capabilities, Sequence) or isinstance(capabilities, (str, bytes)):
            raise TypeError("RecallFormulaLock.capabilities must be a sequence")
        if tuple(capabilities) != lock.capabilities:
            raise ValueError("RecallFormulaLock capabilities do not match its contract")
        return lock


@dataclass(frozen=True)
class _BuiltinFormulaDefinition:
    """Internal implementation metadata for a built-in Formula."""

    contract: RecallFormulaContract
    summary: str
    composition: str

    def __post_init__(self) -> None:
        if not isinstance(self.contract, RecallFormulaContract):
            raise TypeError("_BuiltinFormulaDefinition.contract must be RecallFormulaContract")
        if not self.summary.strip():
            raise ValueError("_BuiltinFormulaDefinition.summary must be non-empty")
        _validate_component(self.composition, field="_BuiltinFormulaDefinition.composition")


def _factor(
    name: str,
    *,
    route: str | None = None,
    identity: float = 0.0,
    init: str = "zero",
) -> FactorSpec:
    return FactorSpec(name=name, route=route, identity=identity, init=init)


_DELTA_CONTRACT = RecallFormulaContract(
    identity=RecallFormulaId.parse("arti/delta@1"),
    factors=(_factor("content"),),
    identity_preserving=True,
    composition="single",
)

_AFFINE_CONTRACT = RecallFormulaContract(
    identity=RecallFormulaId.parse("arti/affine@1"),
    factors=(
        _factor("scale"),
        _factor("shift"),
    ),
    identity_preserving=True,
    composition="product",
)

_STATE_CONTRACT = RecallFormulaContract(
    identity=RecallFormulaId.parse("arti/state@1"),
    factors=(
        _factor("coarse_content"),
        _factor("fine_content"),
        *tuple(_factor(f"modulation_{index:02d}") for index in range(13)),
        _factor("direction"),
        _factor("opacity"),
    ),
    identity_preserving=True,
    composition="state",
)

BUILTIN_RECALL_FORMULAS: Final[Mapping[str, _BuiltinFormulaDefinition]] = MappingProxyType(
    {
        "arti/delta@1": _BuiltinFormulaDefinition(
            contract=_DELTA_CONTRACT,
            summary="One-factor next-state Recall Formula.",
            composition="single",
        ),
        "arti/affine@1": _BuiltinFormulaDefinition(
            contract=_AFFINE_CONTRACT,
            summary="Two-factor affine next-state Recall Formula.",
            composition="product",
        ),
        "arti/state@1": _BuiltinFormulaDefinition(
            contract=_STATE_CONTRACT,
            summary="Seventeen-factor structured next-state Recall Formula.",
            composition="state",
        ),
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


def validate_formula(
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
    if not formula_dtype_supported(x.dtype, contract.execution.supported_dtypes):
        raise TypeError(
            "Recall formula execution contract does not support "
            f"dtype {x.dtype}; supported dtypes are "
            f"{contract.execution.supported_dtypes!r}"
        )

    probe_state = x.detach().clone()
    probe_factors = _deterministic_factors(probe_state, len(factors))
    module_state = {
        name: value.detach().clone() for name, value in module.state_dict().items()
    }
    training = module.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        with torch.no_grad():
            flat_state = probe_state.reshape(-1, probe_state.shape[-1])
            flat_factors = probe_factors.reshape(-1, len(factors), probe_state.shape[-1])

            def run_abi(state: Tensor, factor_values: Tensor) -> Tensor:
                if contract.execution.vectorization == "batched":
                    return module(state, factor_values)
                # Probe with independent randomness so a Formula can surface
                # its own exception. Deterministic contracts are rejected by
                # the repeated-output check below if they actually use it.
                return torch.vmap(module, randomness="different")(
                    state,
                    factor_values,
                )

            def run_with_probe_rng(state: Tensor, factor_values: Tensor) -> Tensor:
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_state)
                return run_abi(state, factor_values)

            abi_state = flat_state.detach().clone()
            abi_factors = flat_factors.detach().clone()
            output = run_abi(abi_state, abi_factors)
            input_mutated = not torch.equal(abi_state, flat_state)
            state_mutated = _module_state_changed(module, module_state)
            if input_mutated or state_mutated:
                raise ValueError(
                    "Recall formula must not mutate its input or module state during forward"
                )
            _validate_formula_output(output, flat_state)

            if contract.execution.vectorization == "batched":
                # A batched Formula is an optimization of independent rows,
                # not permission to mix samples or padded tokens. Probe that
                # invariant by changing only the second row while resetting
                # RNG state for stochastic contracts.
                seed_state = flat_state[:1]
                seed_factors = flat_factors[:1]
                batched_state = torch.cat((seed_state, seed_state + 0.5), dim=0)
                batched_factors = torch.cat(
                    (seed_factors, seed_factors + 0.125), dim=0
                )
                batched_output = _validate_formula_output(
                    run_with_probe_rng(batched_state, batched_factors),
                    batched_state,
                )
                changed_state = batched_state.clone()
                changed_factors = batched_factors.clone()
                changed_state[1].add_(17.0)
                changed_factors[1].sub_(11.0)
                changed_output = _validate_formula_output(
                    run_with_probe_rng(changed_state, changed_factors),
                    changed_state,
                )
                if not torch.allclose(
                    batched_output[0],
                    changed_output[0],
                    atol=float(identity_atol),
                    rtol=float(identity_rtol),
                ):
                    raise ValueError(
                        "batched Recall formulas must be row-independent; "
                        "changing one batch/token row changed another row"
                    )

            if contract.execution.deterministic:
                repeat_state = flat_state.detach().clone()
                repeat_output = _validate_formula_output(
                    run_abi(repeat_state, flat_factors),
                    flat_state,
                )
                if not torch.equal(output, repeat_output):
                    raise ValueError(
                        "deterministic Recall formulas must return the same output "
                        "for repeated identical inputs"
                    )
                if not torch.equal(repeat_state, flat_state) or _module_state_changed(
                    module, module_state
                ):
                    raise ValueError(
                        "Recall formula must not mutate its input or module state during forward"
                    )

            if test_identity or contract.identity_preserving:
                identity_factors = torch.empty_like(flat_factors)
                for index, factor in enumerate(factors):
                    identity_factors[..., index, :].fill_(factor.identity)
                identity_input = flat_state.detach().clone()
                identity_output = _validate_formula_output(
                    run_abi(identity_input, identity_factors),
                    identity_input,
                )
                if not torch.equal(
                    identity_input, flat_state
                ) or _module_state_changed(module, module_state):
                    raise ValueError(
                        "Recall formula must not mutate its input or module state during forward"
                    )
                if not torch.allclose(
                    identity_output,
                    flat_state,
                    atol=float(identity_atol),
                    rtol=float(identity_rtol),
                ):
                    max_error = (identity_output - flat_state).abs().max().item()
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
    "BUILTIN_RECALL_FORMULAS",
    "FactorSpec",
    "MAX_RECALL_FORMULA_FACTORS",
    "MAX_RECALL_FORMULA_PROBE_ELEMENTS",
    "RecallFormulaContract",
    "RecallFormulaExecutionSpec",
    "RecallFormulaLock",
    "RecallOutputSemantics",
    "formula_dtype_supported",
    "validate_formula",
]
