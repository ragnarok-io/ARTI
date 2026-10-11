"""Explicit installed Python dependencies for workflow configurations."""

from dataclasses import dataclass, field
from functools import partial
import importlib
import json
from typing import Any, ClassVar


@dataclass(frozen=True)
class PythonFactoryConfig:
    reference: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    kind: ClassVar[str] = "Python"

    def __post_init__(self):
        parts = self.reference.split(":") if isinstance(self.reference, str) else []
        if len(parts) != 2 or any(
            not part or not all(name.isidentifier() for name in part.split(".")) for part in parts
        ):
            raise ValueError(
                f"{self.kind} factory reference must use installed.module:callable syntax"
            )
        if not isinstance(self.kwargs, dict) or any(
            not isinstance(key, str) for key in self.kwargs
        ):
            raise ValueError(f"{self.kind} factory kwargs must be a JSON object")
        object.__setattr__(self, "kwargs", json.loads(json.dumps(self.kwargs, allow_nan=False)))

    def to_dict(self):
        return {"reference": self.reference, "kwargs": json.loads(json.dumps(self.kwargs))}

    def resolve(self):
        module, attribute = self.reference.split(":")
        try:
            target = importlib.import_module(module)
            for name in attribute.split("."):
                target = getattr(target, name)
        except (ImportError, AttributeError) as error:
            raise ValueError(
                f"cannot import {self.kind} factory {self.reference!r}: {error}"
            ) from error
        if not callable(target):
            raise ValueError(f"{self.kind} factory {self.reference!r} is not callable")
        return partial(target, **self.to_dict()["kwargs"])


__all__ = ["PythonFactoryConfig"]
