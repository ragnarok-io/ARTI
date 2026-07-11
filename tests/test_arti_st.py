from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

import arti
from arti import ARTILayer, LiteralSequenceDecoder


def core_layer() -> ARTILayer:
    return ARTILayer(input_dim=6, hidden_dim=8, coord_dim=2, dropout=0.0).eval()


def test_arti_st_core_layer_round_trip_preserves_output(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = core_layer()
    x = torch.randn(2, 4, 6)
    coord = torch.randn(2, 4, 2)
    expected = model(x, coord=coord).y.detach()

    saved = arti.save(model, tmp_path / "arti.st")
    restored = core_layer()
    loaded = arti.load(saved.weights_path, model=restored)
    actual = restored(x, coord=coord).y.detach()

    assert torch.allclose(actual, expected)
    assert loaded.model is restored
    assert loaded.device == "cpu"
    assert saved.weights_path.name == "arti.st"
    assert saved.manifest_path.name == "arti.json"
    assert saved.lock_path.name == "arti.lock.json"
    assert loaded.manifest["architecture"]["class_name"] == "ARTILayer"


def test_arti_st_literal_decoder_resources_are_strictly_separated(tmp_path: Path) -> None:
    torch.manual_seed(5)
    decoder = LiteralSequenceDecoder(context_dim=7, vocab_tensor_dim=5, hidden_dim=9, key_dim=6).eval()
    context = torch.randn(2, 7)
    vocab = torch.randn(2, 8, 5)
    expected = decoder(context, vocab, steps=4, batched_vocab=True).logits.detach()
    glyphs = {"visible": torch.randn(4, 1, 14, 16), "control": torch.randn(2, 3)}
    vocab_metadata = {"items": ["a", "b", "<eos>"], "font": "Example Sans"}

    saved = arti.save(
        decoder,
        tmp_path / "decoder.st",
        glyph_tensors=glyphs,
        vocab_metadata=vocab_metadata,
    )
    restored = LiteralSequenceDecoder(context_dim=7, vocab_tensor_dim=5, hidden_dim=9, key_dim=6).eval()
    loaded = arti.load(saved.weights_path, model=restored)

    assert torch.allclose(restored(context, vocab, steps=4, batched_vocab=True).logits, expected)
    assert loaded.glyph_tensors is not None
    assert torch.equal(loaded.glyph_tensors["visible"], glyphs["visible"])
    assert loaded.vocab_metadata == vocab_metadata
    assert saved.glyphs_path is not None and saved.glyphs_path.name == "decoder.glyphs.st"
    assert saved.vocab_path is not None and saved.vocab_path.name == "decoder.vocab.json"
    assert loaded.manifest["architecture"]["config"]["context_dim"] == 7
    assert loaded.manifest["architecture"]["config"]["condition_on_vocab"] is True
    with safe_open(saved.weights_path, framework="pt", device="cpu") as handle:
        assert set(handle.metadata() or {}) == {"format", "format_version", "kind", "scope"}
        assert all("glyph" not in key and "vocab" not in key for key in handle.keys())


def test_arti_st_sha256_detects_weight_corruption(tmp_path: Path) -> None:
    saved = arti.save(nn.Linear(4, 3), tmp_path / "arti.st")
    data = bytearray(saved.weights_path.read_bytes())
    data[-1] ^= 1
    saved.weights_path.write_bytes(data)

    try:
        arti.load(saved.weights_path)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("corrupted arti.st should fail integrity validation")


def test_arti_st_manifest_hash_and_future_major_version_are_checked(tmp_path: Path) -> None:
    saved = arti.save(nn.Linear(4, 3), tmp_path / "arti.st")
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    manifest["package_version"] = "2.0.0"
    saved.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(saved.manifest_path.read_bytes()).hexdigest()
    lock = json.loads(saved.lock_path.read_text(encoding="utf-8"))
    lock["manifest_sha256"] = digest
    lock["files"]["manifest"]["sha256"] = digest
    saved.lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    try:
        arti.load(saved.weights_path)
    except ValueError as exc:
        assert "major version" in str(exc)
    else:
        raise AssertionError("future major package should fail compatibility validation")


def test_arti_st_accepts_pre_public_0_2_artifact(tmp_path: Path) -> None:
    saved = arti.save(nn.Linear(4, 3), tmp_path / "legacy.st")
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    manifest["package_version"] = "0.2.0"
    saved.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(saved.manifest_path.read_bytes()).hexdigest()
    lock = json.loads(saved.lock_path.read_text(encoding="utf-8"))
    lock["manifest_sha256"] = digest
    lock["files"]["manifest"]["sha256"] = digest
    saved.lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    loaded = arti.load(saved.weights_path)

    assert loaded.manifest["package_version"] == "0.2.0"


def test_arti_st_checkpoint_resume_matches_continuous_training(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    x = torch.randn(5, 4)
    target = torch.randn(5, 2)
    _training_step(model, optimizer, scheduler, x, target)
    saved = arti.save(
        model,
        tmp_path / "resume.st",
        optimizer=optimizer,
        scheduler=scheduler,
        training_state={"step": 1, "best": 0.25, "rng": torch.arange(4)},
    )

    restored = nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 2))
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-2)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.8)
    loaded = arti.load(
        saved.weights_path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert loaded.training_state is not None
    assert loaded.training_state["step"] == 1
    assert torch.equal(loaded.training_state["rng"], torch.arange(4))
    _training_step(model, optimizer, scheduler, x, target)
    _training_step(restored, restored_optimizer, restored_scheduler, x, target)
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.allclose(actual, expected)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_arti_st_legacy_pt_migration_uses_tensor_state(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = nn.Linear(3, 2)
    legacy = tmp_path / "legacy.pt"
    torch.save({"state_dict": model.state_dict(), "epoch": 4}, legacy)

    migrated = arti.migrate_pt(legacy, tmp_path / "arti.st", model=nn.Linear(3, 2))
    restored = nn.Linear(3, 2)
    result = arti.load(migrated.weights_path, model=restored)

    assert result.manifest["legacy_migration"]["selected_state_key"] == "state_dict"
    assert result.manifest["legacy_migration"]["source_sha256"]
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(actual, expected)


def test_arti_st_rejects_non_tensor_legacy_payload(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pt"
    torch.save({"message": "not a state dict", "epoch": 2}, legacy)

    try:
        arti.migrate_pt(legacy, tmp_path / "arti.st")
    except ValueError as exc:
        assert "tensor-only state dictionary" in str(exc)
    else:
        raise AssertionError("non-tensor legacy payload should not migrate")


def test_arti_st_migrates_real_legacy_fit_adapter_artifact(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    legacy = arti.fit(model, sample_batch=torch.randn(3, 4), target_modules="0").export(tmp_path / "adapter.pt")

    migrated = arti.migrate_pt(legacy, tmp_path / "arti.st")
    loaded = arti.load(migrated.weights_path)

    assert loaded.manifest["weight_scope"] == "trainable"
    assert loaded.manifest["legacy_migration"]["selected_state_key"] == "adapter_state_dict"
    assert loaded.state_dict
    assert all("adapter" in key for key in loaded.state_dict)


def test_fit_result_exports_trainable_arti_st_by_default(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    result = arti.fit(model, sample_batch=torch.randn(3, 4), target_modules="0")

    saved = result.export_st(tmp_path / "arti.st")
    loaded = arti.load(saved.weights_path)

    assert loaded.manifest["weight_scope"] == "trainable"
    assert loaded.state_dict
    assert all("adapter" in key for key in loaded.state_dict)


def test_arti_st_rejects_wrong_target_architecture_by_default(tmp_path: Path) -> None:
    saved = arti.save(nn.Linear(4, 3), tmp_path / "arti.st")

    try:
        arti.load(saved.weights_path, model=nn.Sequential(nn.Linear(4, 3)))
    except ValueError as exc:
        assert "architecture does not match" in str(exc)
    else:
        raise AssertionError("wrong target architecture should fail before state loading")


def test_arti_st_supports_metadata_only_training_checkpoint(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    saved = arti.save(model, tmp_path / "arti.st", training_state={"epoch": 3, "note": "warmup"})

    loaded = arti.load(saved.weights_path, model=nn.Linear(2, 2))

    assert loaded.training_state == {"epoch": 3, "note": "warmup"}
    assert saved.checkpoint_path is not None
    assert saved.checkpoint_metadata_path is not None


def test_arti_st_cuda_map_location_moves_model_when_available(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        return
    model = nn.Linear(4, 3)
    saved = arti.save(model, tmp_path / "arti.st")
    restored = nn.Linear(4, 3)

    loaded = arti.load(saved.weights_path, model=restored, map_location="cuda")

    assert loaded.device == "cuda"
    assert next(restored.parameters()).is_cuda
    assert all(tensor.is_cuda for tensor in loaded.state_dict.values())


def _training_step(model, optimizer, scheduler, x, target) -> None:
    loss = torch.nn.functional.mse_loss(model(x), target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
