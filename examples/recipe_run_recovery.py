"""Inspect or recover an existing classification/Qwen task run.

Run: python examples/recipe_run_recovery.py outputs/first --plan-only
Uses the public CLI lifecycle; no new dataset or training loop is created.
"""

from __future__ import annotations

import sys

from arti.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["recipe", "resume-run", *sys.argv[1:]]))
