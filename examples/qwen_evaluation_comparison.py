"""Compare saved candidate/baseline Qwen receipts without running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--comparison", choices=("exact_text", "numeric"))
    parser.add_argument("--pairing", choices=("exact", "intersection"), default="exact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = arti.compare_causal_lm_evaluations(
        args.candidate,
        args.baseline,
        comparison=args.comparison,
        pairing=args.pairing,
        output=args.output,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "predictions"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
