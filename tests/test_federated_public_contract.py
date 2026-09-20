import torch
from torch import nn

import arti
from arti import mechanisms
from benchmarks.federal_refine_architecture_search import ArchitectureTaskBank, ExperimentConfig


def _program() -> mechanisms.FederatedProgram:
    config = ExperimentConfig(
        tasks=2,
        depths=(1, 2),
        steps=2,
        batch_size=2,
        candidates=2,
        eval_size_per_task=2,
        max_seconds=30.0,
    )
    bank = ArchitectureTaskBank(config, task_id=0, seed=17)
    program = bank.sealed_program(refine_steps=2)
    assert isinstance(program, mechanisms.RoutedProgram)
    return mechanisms.FederatedProgram(
        {program.program_id: program},
        terminal_abi=bank.terminal_abi(),
        root_program_ids=(program.program_id,),
        max_levels=1,
        max_k=16,
    )


def test_program_is_the_public_artilayer_and_attachment_contract() -> None:
    program = _program()
    layer = arti.ARTILayer(
        program=program,
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
    )
    provenance = layer.runtime_provenance()

    assert provenance["surface"] == "program-runtime"
    assert provenance["program_ref"] == arti.canonical_contract_reference(
        "arti/federated-program@1"
    )
    assert arti.component_ref(layer).startswith("arti/layer@sha256:")
    assert layer.program is program
    assert program.root_program_ids == tuple(
        item.program_id for item in program.programs.values()
    )
    assert not hasattr(mechanisms, "FederalRecallV3")
    assert not hasattr(mechanisms, "TensorViewFormulaProgram")
    assert hasattr(arti.legacy, "FederalRecallV3")
    assert hasattr(arti.legacy, "TensorViewFormulaProgram")

    model = nn.Sequential(nn.Linear(16, 16))
    attached = arti.ARTI.attach(model, layer, layers="0")
    result = attached(torch.randn(1, 16))

    assert result.shape == (1, 16)
