from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_public_tabular", ROOT / "benchmarks" / "prepare_public_tabular.py")
assert SPEC is not None
prepare_public_tabular = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["prepare_public_tabular"] = prepare_public_tabular
SPEC.loader.exec_module(prepare_public_tabular)


def registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "id": "uci_breast_cancer_wisconsin_diagnostic",
                        "source": "UCI",
                        "download_url": "https://example.com/wdbc.data",
                        "license": "terms",
                        "target": "diagnosis",
                        "expected_feature_count": 30,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def wdbc_row(row_id: str, diagnosis: str, value: float) -> str:
    features = [str(value + index) for index in range(30)]
    return ",".join([row_id, diagnosis, *features])


def test_prepare_wdbc_converts_to_adapter_csv(tmp_path: Path) -> None:
    raw = tmp_path / "wdbc.data"
    raw.write_text("\n".join([wdbc_row("1", "M", 1.0), wdbc_row("2", "B", 2.0)]), encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry(registry_path)
    output_csv = tmp_path / "wdbc.csv"

    payload = prepare_public_tabular.prepare(
        "uci_breast_cancer_wisconsin_diagnostic", raw, output_csv, registry_path
    )

    lines = output_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("diagnosis,f0,f1")
    assert lines[1].startswith("1,1.0,2.0")
    assert lines[2].startswith("0,2.0,3.0")
    assert payload["rows"] == 2
    assert payload["source_rows"] == 2
    assert payload["feature_count"] == 30
    assert payload["status"] == "prepared_not_evaluated"
    assert "--external-public" in payload["adapter_command"]


def test_prepare_wdbc_rejects_wrong_column_count(tmp_path: Path) -> None:
    raw = tmp_path / "wdbc.data"
    raw.write_text("1,M,1.0\n", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry(registry_path)

    with pytest.raises(ValueError, match="expected 32"):
        prepare_public_tabular.prepare(
            "uci_breast_cancer_wisconsin_diagnostic",
            raw,
            tmp_path / "out.csv",
            registry_path,
        )


def test_prepare_openml_phishing_converts_arff(tmp_path: Path) -> None:
    raw = tmp_path / "phishing.arff"
    raw.write_text(
        "\n".join(
            [
                "@relation phishing",
                "@attribute having_IP_Address {-1,1}",
                "@attribute URL_Length {-1,0,1}",
                "@attribute Result {-1,1}",
                "@data",
                "1,-1,1",
                "-1,0,-1",
            ]
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "id": "openml_phishing_websites",
                        "source": "OpenML",
                        "download_url": "https://example.com/phishing.arff",
                        "license": "Public",
                        "target": "Result",
                        "expected_feature_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = prepare_public_tabular.prepare(
        "openml_phishing_websites",
        raw,
        tmp_path / "phishing.csv",
        registry_path,
    )

    lines = (tmp_path / "phishing.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Result,f0,f1"
    assert lines[1] == "1,1.0,-1.0"
    assert lines[2] == "0,-1.0,0.0"
    assert payload["target"] == "Result"
    assert payload["feature_count"] == 2


def test_prepare_openml_phishing_can_write_fixed_subsample(tmp_path: Path) -> None:
    raw = tmp_path / "phishing.arff"
    rows = ["1,-1,1" if index % 2 == 0 else "-1,0,-1" for index in range(10)]
    raw.write_text(
        "\n".join(
            [
                "@relation phishing",
                "@attribute having_IP_Address {-1,1}",
                "@attribute URL_Length {-1,0,1}",
                "@attribute Result {-1,1}",
                "@data",
                *rows,
            ]
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "id": "openml_phishing_websites",
                        "source": "OpenML",
                        "download_url": "https://example.com/phishing.arff",
                        "license": "Public",
                        "target": "Result",
                        "expected_feature_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = prepare_public_tabular.prepare(
        "openml_phishing_websites",
        raw,
        tmp_path / "phishing.csv",
        registry_path,
        max_rows=4,
    )

    assert payload["rows"] == 4
    assert payload["source_rows"] == 10
    assert payload["max_rows"] == 4
