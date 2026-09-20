"""Run the lightweight component identity and state-contract release gate."""

from __future__ import annotations

import json

import arti


REQUIRED_REFS = {
    "arti/layer@3",
    "arti/layer@1",
    "arti/classic-layer@1",
    "arti/half@1",
    "arti/fold@1",
    "arti/fold@2",
    "arti/unfold@1",
    "arti/unfold@2",
    "arti/pulse@1",
    "arti/pulse@2",
    "arti/pulse-legacy@1",
    "arti/recall@2",
    "arti/recall@3",
    "arti/recall@4",
    "arti/recall-executor@2",
    "arti/recall-state@1",
    "arti/updater@1",
    "arti/affine-updater@1",
    "arti/normalized-updater@1",
    "arti/stacked-updater@1",
    "arti/fusion-pulse@1",
}


def main() -> int:
    catalog = arti.component_catalog()
    component_items = [
        item
        for item in catalog
        if isinstance(item.get("contract"), dict)
        and isinstance(item["contract"].get("semantic_key"), str)
    ]
    semantic_keys = {item["contract"]["semantic_key"] for item in component_items}
    missing = sorted(REQUIRED_REFS - semantic_keys)
    if missing:
        raise SystemExit(f"lifecycle catalog is missing semantic keys: {missing}")
    by_ref = {item["contract"]["semantic_key"]: item for item in component_items}
    expected_lifecycles = {
        "arti/layer@3": "alpha",
        "arti/pulse@2": "stable",
        "arti/fold@2": "stable",
        "arti/unfold@2": "stable",
        "arti/recall@4": "stable",
        "arti/recall-executor@2": "stable",
        "arti/layer@1": "legacy",
    }
    for reference, expected in expected_lifecycles.items():
        actual = by_ref[reference]["lifecycle"]
        if actual != expected:
            raise SystemExit(
                f"{reference} lifecycle is {actual!r}, expected {expected!r}"
            )
    aliases: dict[str, str] = {}
    for item in component_items:
        for alias in [*item["aliases"], *item["deprecated_aliases"]]:
            owner = aliases.setdefault(alias, item["ref"])
            if owner != item["ref"]:
                raise SystemExit(f"alias is owned by multiple components: {alias!r}")

    model = arti.Pulse(k=2, dim=4).eval()
    contract = arti.component_state_contract(model, model.state_dict(), scope="all")
    arti.validate_component_state_contract(contract, state_dict=model.state_dict(), model=model)
    state = arti.mechanisms.RecallState.zeros(1, 3, 4, dtype=next(model.parameters()).dtype)
    if not arti.component_ref(state).startswith("arti/recall-state@sha256:"):
        raise SystemExit("RecallState does not expose its hashed artifact reference")
    print(json.dumps({"ok": True, "components": len(catalog), "state_contract": contract["fingerprint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
