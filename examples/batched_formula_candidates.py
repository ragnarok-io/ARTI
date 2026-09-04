"""Low-level independent candidate execution with the existing QueryV4 arena.

Run from the repository: python examples/batched_formula_candidates.py
This illustrates execution, not ranking, search, or automatic persistence.
"""

from __future__ import annotations

import torch

from predecessor_bank_plasticity import build_query


def main() -> None:
    query = build_query()
    root = query.initial_bank_state()
    # _arena is the existing alpha scheduler's low-level execution state.
    branches = tuple(query._arena({"x": torch.tensor([[float(i), -1.0, 0.5]])},
                                 bank_state=root) for i in range(1, 6))
    serial = branches
    grouped = branches
    with torch.no_grad():
        for candidate in query.candidates:
            serial = query.execute_many(tuple((candidate, arena) for arena in serial), serial=True)
            grouped = query.execute_many(
                tuple((candidate, arena) for arena in grouped), chunk_size=2,
            )
        for reference, actual in zip(serial, grouped, strict=True):
            torch.testing.assert_close(reference.values.get("output"), actual.values.get("output"))
            for left, right in zip(reference.committed_state().values,
                                   actual.committed_state().values, strict=True):
                torch.testing.assert_close(left, right)
        for before, after in zip(root.values, query.initial_bank_state().values, strict=True):
            assert torch.equal(before, after)
    print("Five independent branches: outputs and Bank proposals match serial execution.")
    print("Persistent Bank unchanged; no automatic winner selection or commit.")


if __name__ == "__main__":
    main()
