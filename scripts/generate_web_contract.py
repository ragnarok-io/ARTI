"""Generate TypeScript bindings from the canonical Python Web contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from arti.web import write_typescript_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_typescript_contract(args.output)


if __name__ == "__main__":
    main()
