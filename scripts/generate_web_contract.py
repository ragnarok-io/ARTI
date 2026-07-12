"""Generate TypeScript bindings from the canonical Python Web contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arti.web import write_artifact_typescript, write_typescript_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, help="generate artifact-specific types from an export manifest")
    args = parser.parse_args()
    if args.manifest is None:
        write_typescript_contract(args.output)
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        write_artifact_typescript(manifest, args.output)


if __name__ == "__main__":
    main()
