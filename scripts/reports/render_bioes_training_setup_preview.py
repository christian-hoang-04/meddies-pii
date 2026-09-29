#!/usr/bin/env python
from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
from pathlib import Path

from meddies_pii.training.bioes.reports.training_setup_preview import (
    DEFAULT_DATASET_SUMMARY_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMOKE_RESULT_JSON,
    DEFAULT_SPLIT_SUMMARY_JSON,
    DEFAULT_TRAIN_JSONL,
    DEFAULT_VALIDATION_JSONL,
    render_html,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the pre-launch HTML setup report for the Meddies BIOES H100 run.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--validation-jsonl", type=Path, default=DEFAULT_VALIDATION_JSONL)
    parser.add_argument("--split-summary-json", type=Path, default=DEFAULT_SPLIT_SUMMARY_JSON)
    parser.add_argument("--dataset-summary-json", type=Path, default=DEFAULT_DATASET_SUMMARY_JSON)
    parser.add_argument(
        "--smoke-result-json",
        type=Path,
        default=DEFAULT_SMOKE_RESULT_JSON,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()

    output = render_html(
        train_jsonl=args.train_jsonl,
        validation_jsonl=args.validation_jsonl,
        split_summary_json=args.split_summary_json,
        dataset_summary_json=args.dataset_summary_json,
        smoke_result_json=args.smoke_result_json,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
    )
    print(output)


if __name__ == "__main__":
    main()
