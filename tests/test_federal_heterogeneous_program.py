from __future__ import annotations

from copy import deepcopy

from safetensors.torch import load_file, save_file
import torch

from arti.component_registry import component_provenance, validate_component_provenance
from benchmarks.train_federal_heterogeneous_program import (
    build_runtime,
    evaluate,
    hard_query_probe,
    sample_task,
    train_queries,
    HeterogeneousProgramWorkspace,
)


def test_two_bank_formula_program_trains_and_round_trips(tmp_path) -> None:
    device = torch.device("cpu")
    root, leaf = train_queries(
        seed=47,
        leaf_steps=100,
        root_steps=180,
        batch_size=96,
        device=device,
    )
    probe = hard_query_probe(
        root,
        leaf,
        seed=90_048,
        count=32,
        device=device,
    )
    assert probe["root_step_1"] == 1.0
    assert probe["root_step_2"] == 1.0
    assert probe["leaf_step_1"] == 1.0
    assert probe["leaf_step_2"] == 1.0

    result = evaluate(
        root,
        leaf,
        seed=90_048,
        count=16,
        device=device,
    )
    assert result["mse"] == 0.0
    assert result["path_accuracy"] == 1.0
    assert result["root_query_fingerprint"] != result["leaf_query_fingerprint"]

    runtime = build_runtime(root, leaf)
    provenance = component_provenance(runtime)
    assert validate_component_provenance(provenance) == provenance
    programs = [
        component
        for component in provenance["components"]
        if component["ref"] == "arti/bank-local-formula-program@1"
    ]
    assert len(programs) == 2
    dependencies = {
        dependency
        for component in provenance["components"]
        for dependency in component["dependencies"]
    }
    assert "arti/formula-atom-scale@1" in dependencies
    assert "arti/formula-atom-permute@1" in dependencies
    assert "arti/formula-atom-gather@1" in dependencies
    assert "arti/formula-atom-scatter@1" in dependencies

    artifact = tmp_path / "federal-program.safetensors"
    save_file(
        {
            name: value.detach().contiguous().cpu()
            for name, value in runtime.state_dict().items()
        },
        str(artifact),
    )
    restored = build_runtime(deepcopy(root), deepcopy(leaf))
    restored.load_state_dict(load_file(str(artifact), device="cpu"), strict=True)
    workspace = HeterogeneousProgramWorkspace()
    batch = sample_task(
        count=1,
        seed=91_001,
        device=device,
        workspace=workspace,
    )
    expected = runtime(batch.value)["value"]
    actual = restored(batch.value)["value"]
    torch.testing.assert_close(actual, expected)
