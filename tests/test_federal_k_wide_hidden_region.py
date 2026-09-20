from argparse import Namespace

import torch

from benchmarks import train_federal_k_wide_hidden_region as EXPERIMENT


def test_hidden_region_batch_is_balanced_and_preserves_signal_payload() -> None:
    batch = EXPERIMENT.sample_task(
        count=32,
        feature_dim=8,
        seed=13,
        device=torch.device("cpu"),
        balanced=True,
    )
    assert torch.bincount(batch.signal_shard, minlength=8).tolist() == [4] * 8
    pairs = batch.value.reshape(32, 8, 2, 8)
    rows = torch.arange(32)
    selected = pairs[rows, batch.signal_shard]
    assert torch.equal(selected.reshape(32, 1, 16), batch.target)


def test_k_wide_hidden_region_smoke(tmp_path) -> None:
    result = EXPERIMENT.run(
        Namespace(
            seeds=(7,),
            steps=50,
            batch_size=64,
            test_size=32,
            feature_dim=8,
            hidden_dim=16,
            device="cpu",
            output=tmp_path / "results.json",
            plot=tmp_path / "curve.svg",
        )
    )
    trial = result["trials"][0]
    assert result["passed"] is True
    assert trial["k_curve"]["8"]["exact_rate"] == 1.0
    assert trial["k_curve"]["1"]["exact_rate"] == 0.125
    assert trial["state_blind_k8"]["exact_rate"] == 0.125
    assert trial["untrained_k8"]["exact_rate"] == 0.125
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "curve.svg").is_file()
