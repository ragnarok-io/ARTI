"""Generate source-backed ARTI documentation pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from arti import (
    check_generated_docs,
    generate_capabilities_markdown,
    write_generated_docs,
)

__all__ = ["generate_capabilities_markdown", "main"]


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "reference" / "capabilities.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate source-backed ARTI docs.")
    parser.add_argument("--check", action="store_true", help="Fail if generated docs are not up to date.")
    parser.add_argument("--phases", type=int, default=16, help="Observer-phase preview coordinate dimension.")
    parser.add_argument("--output", type=Path, default=TARGET, help="Generated Markdown output path.")
    args = parser.parse_args()

    if args.check:
        try:
            check_generated_docs(args.output, phases=args.phases)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Generated docs are up to date: {args.output}")
        return
    write_generated_docs(args.output, phases=args.phases)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
