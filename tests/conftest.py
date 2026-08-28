from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch


@pytest.fixture
def paired_rng() -> Callable[[Callable[[], Any], Callable[[], Any]], tuple[Any, Any]]:
    """Run paired probes from the same CPU/CUDA random state."""

    def run(first: Callable[[], Any], second: Callable[[], Any]) -> tuple[Any, Any]:
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        expected = first()
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        actual = second()
        return expected, actual

    return run
