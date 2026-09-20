from __future__ import annotations

import torch

from arti import mechanisms

from benchmarks.federal_refine_architecture_search import (
    ArchitectureTaskBank,
    ExperimentConfig,
    _beam_rollout,
    _sample_rollout,
    _training_loss,
    _view,
    make_targets,
)


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        tasks=2,
        depths=(1, 2, 4),
        steps=4,
        batch_size=2,
        candidates=4,
        eval_size_per_task=2,
        max_seconds=30.0,
    )


def test_architecture_bank_uses_real_shape_and_fold_formula_atoms() -> None:
    config = tiny_config()
    bank = ArchitectureTaskBank(config, task_id=0, seed=11)
    atoms = {
        instruction.atom_ref
        for action in bank.actions
        for instruction in action.action.program.instructions
    }

    assert "arti/formula-atom-reshape@1" in atoms
    assert "arti/formula-atom-contract@1" in atoms
    assert "arti/formula-atom-gather@1" in atoms
    assert "arti/formula-atom-scatter@1" in atoms
    assert {bank.output_layout(index) for index in range(len(bank.actions))} == {
        "vector",
        "plane",
        "volume",
    }


def test_sampled_k_wide_rollout_uses_exact_depth_and_backpropagates() -> None:
    config = tiny_config()
    bank = ArchitectureTaskBank(config, task_id=0, seed=13)
    targets = make_targets(config, device=torch.device("cpu"), seed=17)
    generator = torch.Generator(device="cpu").manual_seed(19)
    x, target, _mode = targets.sample(0, count=2, generator=generator)

    value, log_probability, path, layouts = _sample_rollout(
        bank,
        x,
        refine_steps=4,
        candidates=4,
        generator=generator,
    )
    assert value.shape == (8, config.dim)
    assert log_probability.shape == (8,)
    assert path.shape == (8, 4)
    assert layouts.shape == (8, 4)

    loss, _metrics = _training_loss(
        bank,
        x,
        target,
        refine_steps=4,
        candidates=4,
        generator=generator,
        route_weight=config.route_weight,
    )
    loss.backward()
    gradients = [parameter.grad for parameter in bank.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)


def test_hard_beam_returns_one_winner_and_exact_route_length() -> None:
    config = tiny_config()
    bank = ArchitectureTaskBank(config, task_id=0, seed=23)
    x = torch.randn(3, config.dim)

    output, score, path, layouts = _beam_rollout(
        bank,
        x,
        refine_steps=4,
        max_k=4,
    )

    assert output.shape == x.shape
    assert score.shape == (3,)
    assert path.shape == (3, 4)
    assert layouts.shape == (3, 4)
    assert torch.isfinite(output).all()
    assert torch.isfinite(score).all()


def test_target_family_contains_independent_dense_and_structured_operators() -> None:
    config = tiny_config()
    targets = make_targets(config, device=torch.device("cpu"), seed=27)

    assert targets.matrices.shape == (config.tasks, 2, config.dim, config.dim)
    identity = torch.eye(config.dim)
    for task in range(config.tasks):
        for mode in range(2):
            matrix = targets.matrices[task, mode]
            torch.testing.assert_close(matrix @ matrix.T, identity, atol=1e-5, rtol=1e-5)
    assert not torch.equal(targets.matrices[0], targets.matrices[1])


def test_query_can_be_sealed_into_the_current_federal_runtime() -> None:
    config = tiny_config()
    bank = ArchitectureTaskBank(config, task_id=0, seed=29)
    program = bank.sealed_program(refine_steps=2)

    assert program.local_refine.min_steps == 3
    assert program.local_refine.max_steps == 3
    assert program.query.signature.member_ids == (*bank.action_ids, "exit")
    assert all(not parameter.requires_grad for parameter in program.query.parameters())


def test_sealed_bank_executes_inside_federal_recall_v3() -> None:
    config = tiny_config()
    bank = ArchitectureTaskBank(config, task_id=0, seed=31)
    program = bank.sealed_program(refine_steps=2)
    runtime = mechanisms.FederalRecallV3(
        {program.bank_id: program},
        terminal_abi=bank.terminal_abi(),
        root_bank_ids=(program.bank_id,),
        max_levels=1,
        max_k=16,
    )

    output, trace = runtime(
        _view("vector", torch.randn(1, config.dim)),
        return_trace=True,
    )

    assert output["value"].shape == (1, config.dim)
    assert trace.winner_paths[0].startswith(program.bank_id)
