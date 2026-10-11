"""Inspect or package a saved Qwen recipe export without training or downloading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--package", type=Path, help="Fresh destination; omit for inspection only.")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.package is not None:
        result = arti.package_causal_lm_deployment(
            args.model, args.package, include_base=args.include_base, base_model=args.base_model
        )
    else:
        result = arti.inspect_causal_lm_deployment(
            args.model, base_model=args.base_model, output=args.report
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
