"""Render local demo files from already-validated inline records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.jsonl import write_jsonl

if TYPE_CHECKING:
    from meddies_pii.annotations.inline_records import ValidationResult


@dataclass(frozen=True)
class DemoResult:
    validation: ValidationResult
    output_dir: Path
    plain_path: Path
    spans_path: Path
    report_path: Path


def write_demo_output(validation: ValidationResult, output_dir: str | Path) -> DemoResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plain_path = output_path / "plain.jsonl"
    spans_path = output_path / "spans.jsonl"
    report_path = output_path / "report.json"

    write_jsonl(
        plain_path,
        (
            {
                "id": record.record_id,
                "language": record.language,
                "text": record.plain_text,
            }
            for record in validation.records
        ),
    )
    write_jsonl(
        spans_path,
        (
            {
                "id": record.record_id,
                "language": record.language,
                "text": record.plain_text,
                "spans": [asdict(span) for span in record.spans],
            }
            for record in validation.records
        ),
    )
    report_path.write_text(
        json.dumps(
            {
                "records": len(validation.records),
                "spans": validation.span_count,
                "labels": list(validation.labels),
                "files": {
                    "plain": str(plain_path),
                    "spans": str(spans_path),
                    "report": str(report_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return DemoResult(
        validation=validation,
        output_dir=output_path,
        plain_path=plain_path,
        spans_path=spans_path,
        report_path=report_path,
    )
