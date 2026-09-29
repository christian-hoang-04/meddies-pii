"""Facade for static BIOES training-readiness reports.

The public Interface stays here for CLI/tests, while scanning, highlighting,
and HTML rendering live behind the `reports` Module for better locality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .reports import (
    LabelExample,
    RowPreview,
    TrainingReportOptions,
    TrainingScan,
    highlight_spans,
    render_training_report_html,
    scan_adversarial_jsonl,
    scan_audit_jsonl,
    scan_labeled_jsonl,
    scan_training_jsonl,
    snippet_html,
)
from .reports.json_narrowing import load_optional_json_object

if TYPE_CHECKING:
    from pathlib import Path


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def build_training_report(options: TrainingReportOptions) -> Path:
    """Build a single static HTML report and return its path.

    Returns:
        The path of the written report. Every input JSON is loaded optionally, so a run that
        produced only some of its summaries still renders a report covering what exists rather
        than failing on the first absent file.

    """
    summary = load_optional_json_object(options.summary_json)
    split_summary = load_optional_json_object(options.split_summary_json)
    dataset_summary = load_optional_json_object(options.dataset_summary_json)
    modal = load_optional_json_object(options.modal_result_json)
    scan = scan_training_jsonl(options)
    validation_scan = (
        scan_labeled_jsonl(
            options.validation_jsonl,
            rows_per_label=options.rows_per_label,
            row_table_limit=options.row_table_limit,
            include_all_rows=options.include_all_rows,
            snippet_context=options.snippet_context,
        )
        if options.validation_jsonl
        else None
    )
    audit_samples = scan_audit_jsonl(options.audit_jsonl, options.audit_samples_per_rule) if options.audit_jsonl else {}
    adversarial = scan_adversarial_jsonl(options.adversarial_jsonl) if options.adversarial_jsonl else []
    todo_text = _read_text(options.todo_md)
    progress_text = _read_text(options.progress_md)

    html = render_training_report_html(
        options=options,
        scan=scan,
        validation_scan=validation_scan,
        summary=summary,
        split_summary=split_summary,
        dataset_summary=dataset_summary,
        audit_samples=audit_samples,
        adversarial=adversarial,
        modal=modal,
        todo_text=todo_text,
        progress_text=progress_text,
    )
    options.output_html.parent.mkdir(parents=True, exist_ok=True)
    options.output_html.write_text(html, encoding="utf-8")
    return options.output_html


__all__ = [
    "LabelExample",
    "RowPreview",
    "TrainingReportOptions",
    "TrainingScan",
    "build_training_report",
    "highlight_spans",
    "render_training_report_html",
    "scan_adversarial_jsonl",
    "scan_audit_jsonl",
    "scan_labeled_jsonl",
    "scan_training_jsonl",
    "snippet_html",
]
