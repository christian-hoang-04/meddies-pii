#!/usr/bin/env python
from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from pathlib import Path

from meddies_pii.training.bioes.eval.audit import audit_records, summarize_issues
from meddies_pii.training.bioes.reports.eval_prediction_audit import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREDICTIONS_JSON,
    read_prediction_records,
    write_eval_prediction_audit_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BIOES validation predictions for likely gold-label issues.")
    parser.add_argument("--predictions-json", type=Path, default=DEFAULT_PREDICTIONS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-issues", type=int, default=500)
    args = parser.parse_args()

    records = read_prediction_records(args.predictions_json)
    issues = audit_records(records)
    summary = summarize_issues(issues)
    html_path = write_eval_prediction_audit_outputs(
        predictions_json=args.predictions_json,
        output_dir=args.output_dir,
        max_issues=args.max_issues,
        records=records,
        issues=issues,
    )
    print(f"records={len(records)} issues={len(issues)}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(f"HTML audit: file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
