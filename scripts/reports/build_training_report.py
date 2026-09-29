#!/usr/bin/env python
from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
from pathlib import Path

from meddies_pii.training.bioes.report import (
    TrainingReportOptions,
    build_training_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a static HTML pre-launch report for a Meddies Labels BIOES training bundle.",
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--validation-jsonl",
        type=Path,
        default=None,
        help="Optional held-out validation JSONL to include in split/examples breakdown.",
    )
    parser.add_argument(
        "--split-summary-json",
        type=Path,
        default=None,
        help="Optional split summary JSON with train/validation counts.",
    )
    parser.add_argument(
        "--dataset-summary-json",
        type=Path,
        default=None,
        help="Optional dataset summary JSON with all-row/source/language policy details.",
    )
    parser.add_argument("--audit-jsonl", type=Path, default=None)
    parser.add_argument("--adversarial-jsonl", type=Path, default=None)
    parser.add_argument("--modal-result-json", type=Path, default=None)
    parser.add_argument("--todo-md", type=Path, default=None)
    parser.add_argument("--progress-md", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/training-readiness/training_readiness_report.html"),
    )
    parser.add_argument("--title", default="Meddies Labels BIOES Training Readiness Report")
    parser.add_argument("--rows-per-label", type=int, default=3)
    parser.add_argument("--audit-samples-per-rule", type=int, default=3)
    parser.add_argument("--row-table-limit", type=int, default=0)
    parser.add_argument(
        "--include-all-rows",
        action="store_true",
        help="Inline every training row as a compact snippet. This makes a large HTML file.",
    )
    parser.add_argument("--snippet-context", type=int, default=180)
    args = parser.parse_args()

    output = build_training_report(
        TrainingReportOptions(
            train_jsonl=args.train_jsonl,
            validation_jsonl=args.validation_jsonl,
            summary_json=args.summary_json,
            split_summary_json=args.split_summary_json,
            dataset_summary_json=args.dataset_summary_json,
            audit_jsonl=args.audit_jsonl,
            adversarial_jsonl=args.adversarial_jsonl,
            modal_result_json=args.modal_result_json,
            todo_md=args.todo_md,
            progress_md=args.progress_md,
            output_html=args.output,
            title=args.title,
            rows_per_label=args.rows_per_label,
            audit_samples_per_rule=args.audit_samples_per_rule,
            row_table_limit=args.row_table_limit,
            include_all_rows=args.include_all_rows,
            snippet_context=args.snippet_context,
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
