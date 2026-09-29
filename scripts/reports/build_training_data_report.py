#!/usr/bin/env python
from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.training.bioes.reports.training_data_breakdown import (
    TrainingDataSourceConfig,
    scan_training_data_source,
    write_training_data_breakdown_report,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO / "reports" / "2026-06-11-training-data-breakdown.html"


def _glob(rel: str) -> tuple[Path, ...]:
    if "*" in rel:
        return tuple(sorted(REPO.glob(rel)))
    return (REPO / rel,)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the BIOES training-data breakdown HTML report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_PATH,
        help=f"Report path. Default: {REPORT_PATH}",
    )
    return parser.parse_args(argv)


def _source_configs() -> tuple[TrainingDataSourceConfig, ...]:
    return (
        TrainingDataSourceConfig(
            "Xiaomi all-language augmented (flagship)",
            "internal-augmented",
            (
                "Largest internal set. 17 langs, multi-format (HTML/JSON/markdown is intentional). Span "
                "encoding needs a parser pass before use."
            ),
            _glob(f"data/bioes-v2/base/all_lang_unique_augmented-20260530.train.{LEGACY_ARTIFACT_TOKEN}.jsonl"),
            "clean-first",
        ),
        TrainingDataSourceConfig(
            "MIMO general-domain 20k (vi+en)",
            "internal-augmented",
            "General-domain augmentation, Vietnamese + English.",
            _glob("data/bioes-v2/synthetic/mimo_general_20k/accepted.vi.jsonl")
            + _glob("data/bioes-v2/synthetic/mimo_general_20k/accepted.en.jsonl"),
            "clean-first",
        ),
        TrainingDataSourceConfig(
            "GRPO hard-examples (Gemini-Pro)",
            "internal-grpo",
            "Gemini-Pro generated. GRPO approach dropped -> repurpose as high-quality general data.",
            _glob("data/hard-examples/gemini_results.jsonl"),
            "convert-first",
        ),
        TrainingDataSourceConfig(
            "Nemotron health (raw)",
            "external-raw",
            (
                "Raw nvidia/Nemotron-PII health slice — the ONLY external company_name source. Needs "
                "legacy PII-label conversion (premap wired)."
            ),
            _glob("data/bioes-v2/external/nvidia-health.jsonl"),
            "convert-first",
        ),
        TrainingDataSourceConfig(
            "OpenCodeZen clinical daily (synthetic)",
            "internal-synthetic",
            "Free daily-schedule synthetic bulk (vi+en), already migrated into bioes-v2.",
            _glob("data/bioes-v2/synthetic/opencode_zen/*.jsonl"),
            "include",
        ),
        TrainingDataSourceConfig(
            "MIMO new-data vi (synthetic)",
            "internal-synthetic",
            "Stray repo-root synthetic, migrated into bioes-v2.",
            _glob("data/bioes-v2/synthetic/mimo_new_data_vi.jsonl"),
            "include",
        ),
        TrainingDataSourceConfig(
            "ai4privacy + gretel (external)",
            "external-raw",
            "On HuggingFace, not local. ai4privacy 1.5m+500k (drop 1m), gretel ~8k (en). Needs HF token to stage + count.",
            (),
            "convert-first",
        ),
        TrainingDataSourceConfig(
            "Meddies/meddies-pii 19 configs (HF)",
            "internal-augmented",
            (
                "Canonical 17-lang set + nvidia-health/non-health + vietnamese-translated. On HF — needs "
                "token to stage + count."
            ),
            (),
            "clean-first",
        ),
        TrainingDataSourceConfig(
            "label-review batches (old schema)",
            "misc",
            "Feb-2026 prediction dumps, OLD 7-label schema. Unusable without re-annotation.",
            _glob("data/label-review/batch-eval.jsonl"),
            "exclude",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scanned = [scan_training_data_source(source) for source in _source_configs()]
    output_path = write_training_data_breakdown_report(args.output, scanned)
    print(f"wrote {output_path}")
    for source in scanned:
        if source.existing_paths:
            print(f"  {source.rows:>8,}  {source.role:<20} {source.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
