#!/usr/bin/env python
"""Download Meddies/meddies-pii Hugging Face configs to local JSONL."""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO / "data/bioes-v2/hf_configs"
DEFAULT_REPO_ID = "Meddies/meddies-pii"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download every config and split from a Hugging Face dataset repo to local JSONL files for offline analysis."
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for <config>.<split>.jsonl files.",
    )
    return parser.parse_args(argv)


def download_configs(repo_id: str, output_dir: Path) -> dict[str, dict[str, int]]:
    from datasets import get_dataset_config_names, load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    configs = get_dataset_config_names(repo_id)
    print(f"{len(configs)} configs: {configs}", flush=True)

    manifest: dict[str, dict[str, int]] = {}
    for config in configs:
        dataset = load_dataset(repo_id, config)
        manifest[config] = {}
        for split in dataset:
            output_path = output_dir / f"{config}.{split}.jsonl"
            dataset[split].to_json(str(output_path), force_ascii=False, lines=True)
            row_count = int(dataset[split].num_rows)
            manifest[config][str(split)] = row_count
            print(
                f"  {config}.{split}: {row_count:,} rows -> {output_path.name}",
                flush=True,
            )

    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = download_configs(args.repo_id, args.output_dir)
    (args.output_dir / "_download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total = sum(row_count for splits in manifest.values() for row_count in splits.values())
    print(
        f"DONE: {total:,} rows across {len(manifest)} configs -> {args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
