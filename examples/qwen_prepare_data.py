"""Prepare existing Qwen supervision without loading a host or training.

Run: python examples/qwen_prepare_data.py recipe.toml --output prepared.json
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import arti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="Use an already saved local tokenizer.")
    parser.add_argument("--cache-dir", type=Path, help="Override the configured preparation cache for this command.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = arti.CausalLMRecipeConfig.read(args.config)
    if args.cache_dir is not None:
        config = replace(config, cache_dir=args.cache_dir)
    tokenizer = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    report = arti.prepare_causal_lm_data(config, tokenizer=tokenizer, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
