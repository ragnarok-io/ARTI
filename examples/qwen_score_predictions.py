"""Rescore existing generated replies without a model or new inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arti import score_causal_lm_predictions
from arti.recipe_data import read_jsonl
from arti.recipe_workflow import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--comparison", choices=("exact_text", "numeric"), default="exact_text")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = list(read_jsonl(args.predictions))
    generated = [row for row in records if "generation" in row]
    report = score_causal_lm_predictions(generated, comparison=args.comparison)
    report["source_records"] = len(records)
    report["non_generated_records"] = len(records) - len(generated)
    if args.output is not None:
        write_json(args.output, report)
    print(
        json.dumps(
            {
                "metrics": report["metrics"],
                "source_records": len(records),
                "non_generated_records": report["non_generated_records"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
