from __future__ import annotations

import pytest
import torch

from arti import (
    assert_source_integrity,
    decode_source_tokens,
    encode_source_tokens,
    make_source_integrity_basis,
    read_sources,
    SourceIntegrityCarrier,
    source_basis_orthogonality_loss,
    source_integrity_report,
    superpose_sources,
)


def test_source_integrity_superpose_and_read_reconstructs() -> None:
    torch.manual_seed(3)
    basis = make_source_integrity_basis(source_count=2, payload_dim=4, checksum_dim=2)
    x = torch.randn(2, 2, 5, 4)

    field = superpose_sources(x, basis)
    decoded = read_sources(field, basis)
    report = source_integrity_report(x, decoded, basis, block_size=2)

    assert field.shape == (2, 5, 8)
    assert torch.allclose(decoded, x)
    assert float(source_basis_orthogonality_loss(basis)) == 0.0
    assert report.valid
    assert_source_integrity(report)


def test_source_integrity_pilot_mode_validates_source_identity() -> None:
    basis = make_source_integrity_basis(source_count=2, payload_dim=4, checksum_dim=2)
    x = torch.zeros(1, 2, 3, 4)

    field = superpose_sources(x, basis, pilot_strength=0.1)
    decoded = read_sources(field, basis)
    report = source_integrity_report(x, decoded, basis, pilot_strength=0.1)

    assert report.valid
    assert float(report.min_pilot_self_score) > 0.99
    assert float(report.max_pilot_cross_score) == 0.0


def test_source_integrity_detects_wrong_basis_readout() -> None:
    basis = make_source_integrity_basis(source_count=2, payload_dim=3, checksum_dim=2)
    x = torch.zeros(1, 2, 2, 3)
    x[:, 0, :, 0] = 1.0
    x[:, 1, :, 1] = 1.0
    field = superpose_sources(x, basis)
    wrong_basis = make_source_integrity_basis(source_count=2, payload_dim=3, checksum_dim=2)
    wrong_basis = type(wrong_basis)(
        source_carrier=wrong_basis.source_carrier.flip(0),
        pilots=wrong_basis.pilots,
        checksum_projection=wrong_basis.checksum_projection,
    )

    decoded = read_sources(field, wrong_basis)
    report = source_integrity_report(x, decoded, basis)

    assert not report.valid
    with pytest.raises(ValueError, match="source integrity failed"):
        assert_source_integrity(report)


def test_source_integrity_swap_cases_are_distinct() -> None:
    basis = make_source_integrity_basis(source_count=2, payload_dim=2)
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    case_a = torch.stack([a, b]).reshape(1, 2, 1, 2)
    case_b = torch.stack([b, a]).reshape(1, 2, 1, 2)

    field_a = superpose_sources(case_a, basis)
    field_b = superpose_sources(case_b, basis)
    naive_a = case_a.sum(dim=1)
    naive_b = case_b.sum(dim=1)

    assert not torch.allclose(field_a, field_b)
    assert torch.allclose(naive_a, naive_b)


def test_source_integrity_token_stream_keeps_token_count() -> None:
    basis = make_source_integrity_basis(source_count=3, payload_dim=4)
    hidden = torch.randn(2, 6, 4)
    source_ids = torch.tensor([[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0]])

    field = encode_source_tokens(hidden, source_ids, basis)
    decoded = decode_source_tokens(field, source_ids, basis)

    assert field.shape == (2, 6, 12)
    assert decoded.shape == hidden.shape
    assert torch.allclose(decoded, hidden)


def test_source_integrity_carrier_modes_and_buffers() -> None:
    carrier = SourceIntegrityCarrier(source_count=2, payload_dim=4, checksum_dim=2, mode="summary", block_size=2, pilot_strength=0.1)
    x = torch.randn(2, 2, 5, 4)

    field = carrier(x)
    decoded = carrier.read_sources(field)
    summary = carrier.report(x, decoded)
    full = carrier.report(x, decoded, mode="full")
    pilot = carrier.report(x, decoded, mode="pilot")

    assert field.shape == (2, 5, 8)
    assert summary is not None and summary.mode == "summary" and summary.valid
    assert full is not None and full.mode == "full" and full.valid
    assert pilot is not None and pilot.mode == "pilot" and pilot.valid
    assert carrier.report(x, decoded, mode="off") is None
    assert "source_carrier" in carrier.state_dict()


def test_source_integrity_carrier_token_api_keeps_token_axis() -> None:
    carrier = SourceIntegrityCarrier(source_count=2, payload_dim=3, mode="off")
    hidden = torch.randn(1, 7, 3)
    source_ids = torch.tensor([[0, 1, 0, 1, 1, 0, 1]])

    field = carrier.encode_tokens(hidden, source_ids)
    decoded = carrier.decode_tokens(field, source_ids)

    assert field.shape[:2] == hidden.shape[:2]
    assert decoded.shape == hidden.shape
    assert torch.allclose(decoded, hidden)


@pytest.mark.parametrize("source_count", [4, 8, 16])
def test_source_integrity_multisource_reconstructs_and_reports(source_count: int) -> None:
    carrier = SourceIntegrityCarrier(
        source_count=source_count,
        payload_dim=source_count,
        checksum_dim=min(8, source_count),
        mode="summary",
        block_size=4,
        pilot_strength=0.1,
    )
    x = torch.eye(source_count).reshape(1, source_count, 1, source_count).repeat(2, 1, 8, 1)

    field = carrier.encode_sources(x)
    decoded = carrier.read_sources(field)
    report = carrier.report(x, decoded)

    assert field.shape == (2, 8, source_count * source_count)
    assert torch.allclose(decoded - carrier.pilot_strength * carrier.pilots.reshape(1, source_count, 1, source_count), x)
    assert report is not None and report.valid
    assert float(report.leakage_matrix.max()) == 0.0


def test_source_integrity_multisource_token_ids_batch_encode() -> None:
    source_count = 8
    carrier = SourceIntegrityCarrier(source_count=source_count, payload_dim=5, mode="off")
    hidden = torch.randn(3, 17, 5)
    source_ids = torch.arange(17).remainder(source_count).repeat(3, 1)

    field = carrier.encode_tokens(hidden, source_ids)
    decoded = carrier.decode_tokens(field, source_ids)

    assert field.shape == (3, 17, source_count * 5)
    assert torch.allclose(decoded, hidden)


def test_source_integrity_multisource_localizes_missing_and_corrupt_source() -> None:
    source_count = 8
    carrier = SourceIntegrityCarrier(source_count=source_count, payload_dim=source_count, mode="summary", block_size=2, pilot_strength=0.1)
    x = torch.eye(source_count).reshape(1, source_count, 1, source_count).repeat(1, 1, 6, 1)
    mask = torch.ones(1, source_count, 6, dtype=torch.bool)
    missing_source = 5
    mask[:, missing_source] = False

    missing_decoded = carrier.read_sources(carrier.encode_sources(x, mask=mask))
    missing_report = carrier.report(x, missing_decoded)
    assert missing_report is not None
    assert not missing_report.valid
    assert int(missing_report.reconstruction_error.argmax()) == missing_source

    decoded = carrier.read_sources(carrier.encode_sources(x))
    corrupted_source = 3
    decoded[:, corrupted_source, 2:4, :] += 0.5
    corruption_report = carrier.report(x, decoded)
    assert corruption_report is not None
    assert not corruption_report.valid
    assert int(corruption_report.reconstruction_error.argmax()) == corrupted_source
