#!/usr/bin/env python
"""Convert downloaded Meddies/meddies-pii configs to Meddies Labels JSONL."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.training.bioes.data.inline_tags import convert_config_row

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO / "data/bioes-v2/hf_configs"
DEFAULT_OUTPUT_DIR = REPO / "data/bioes-v2/internal/hf_configs"

DEFAULT_CONFIGS = (
    "burmese",
    "chinese",
    "english",
    "filipino",
    "french",
    "german",
    "indonesian",
    "japanese",
    "korean",
    "laos",
    "malay",
    "portuguese",
    "russian",
    "spanish",
    "tamil",
    "thai",
    "vietnamese",
    "vietnamese-translated",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Convert local Hugging Face config JSONL exports from inline tags to Meddies Labels span records."),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing <config>.train.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for <config>.{LEGACY_ARTIFACT_TOKEN}.jsonl outputs.",
    )
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        choices=DEFAULT_CONFIGS,
        help="Config to convert. May be passed more than once. Defaults to all configs.",
    )
    return parser.parse_args(argv)


def convert_configs(
    source_dir: Path,
    output_dir: Path,
    configs: Sequence[str],
) -> dict[str, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}

    for config in configs:
        source_path = source_dir / f"{config}.train.jsonl"
        if not source_path.exists():
            print(f"  SKIP {config}: no file", flush=True)
            continue

        output_path = output_dir / f"{config}.{LEGACY_ARTIFACT_TOKEN}.jsonl"
        kept = 0
        dropped = 0
        spans = 0
        with (
            source_path.open(encoding="utf-8") as source,
            output_path.open("w", encoding="utf-8") as output,
        ):
            for index, line in enumerate(source):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                try:
                    row = json.loads(stripped_line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                record = convert_config_row(row, uid=f"{config}-{index}")
                if record is None:
                    dropped += 1
                    continue
                label = record["label"]
                spans += len(label) if isinstance(label, list) else 0
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

        summary[config] = {"kept": kept, "dropped": dropped, "spans": spans}
        print(
            f"  {config}: {kept:,} rows, {spans:,} spans ({dropped} dropped)",
            flush=True,
        )

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configs = tuple(args.configs) if args.configs else DEFAULT_CONFIGS
    summary = convert_configs(args.source_dir, args.output_dir, configs)
    grand_total = sum(item["kept"] for item in summary.values())
    print(f"DONE: {grand_total:,} rows -> {args.output_dir}", flush=True)
    (args.output_dir / "_convert_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
