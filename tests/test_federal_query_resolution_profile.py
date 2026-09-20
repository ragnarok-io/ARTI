from __future__ import annotations

import argparse

import torch

from benchmarks import train_federal_query_resolution_profile as experiment


def test_leaf_query_resolution_is_independent_of_terminal_value_width() -> None:
    model = experiment.FederalResolutionModel(
        seed=17,
        parent_rank=16,
        leaf_rank=2,
    )

    assert model.parent.rank == 16
    assert all(query.rank == 2 for query in model.leaves)
    assert model.terminal_values.shape == (
        experiment.DOMAIN_COUNT,
        experiment.CHOICE_COUNT,
        experiment.VALUE_DIM,
    )
    assert experiment.VALUE_DIM > model.leaf_rank


def test_resolution_profile_smoke(tmp_path) -> None:
    result = experiment.run(
        argparse.Namespace(
            seeds=[17],
            steps=4,
            batch_size=32,
            test_size=64,
            noise=0.72,
            ks=[1, 2],
            device="cpu",
            output=tmp_path / "results.json",
            plot=tmp_path / "accuracy.svg",
        )
    )

    variants = {row["variant"] for row in result["summary"]}
    assert {
        "funnel",
        "uniform_high",
        "inverted",
        "uniform_low",
        "wrong_leaf_query",
        "flat",
    } <= variants
    assert result["training_signal"] == "final 64D tensor MSE only"
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "accuracy.svg").is_file()


def test_funnel_has_lower_leaf_capacity_and_runtime_cost_than_uniform_high() -> None:
    funnel = experiment.FederalResolutionModel(
        seed=17,
        parent_rank=16,
        leaf_rank=2,
    )
    uniform_high = experiment.FederalResolutionModel(
        seed=17,
        parent_rank=16,
        leaf_rank=16,
    )

    assert experiment.parameter_count(funnel) < experiment.parameter_count(uniform_high)
    assert funnel.runtime_query_macs(k=4) < uniform_high.runtime_query_macs(k=4)


def test_sample_target_is_a_real_high_dimensional_terminal_value() -> None:
    model = experiment.FederalResolutionModel(
        seed=29,
        parent_rank=16,
        leaf_rank=2,
    )
    batch = model.sample(32, seed=101, noise=0.72)

    expected = model.terminal_values[batch.domain, batch.choice]
    torch.testing.assert_close(batch.target, expected)
