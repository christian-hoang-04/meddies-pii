"""Persist the BIOES evaluation-prediction audit and public report API."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meddies_pii.training.bioes.eval.audit import (
    EvalAuditIssue,
    audit_records,
    summarize_issues,
)

from .eval_prediction_artifacts import (
    read_prediction_records,
    write_adjudication_batches,
    write_jsonl,
)
from .eval_prediction_audit_html import (
    SCRIPT,
    STYLE,
    render_eval_prediction_audit_html,
)
from .eval_prediction_metrics import metric_scenarios

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PREDICTIONS_JSON",
    "SCRIPT",
    "STYLE",
    "metric_scenarios",
    "read_prediction_records",
    "render_eval_prediction_audit_report",
    "write_adjudication_batches",
    "write_eval_prediction_audit_outputs",
]

DEFAULT_PREDICTIONS_JSON = Path("reports/bioes-eval/predictions/lfm25-bioes-step150-full-validation-predictions.json")
DEFAULT_OUTPUT_DIR = Path("reports/bioes-eval-audit/lfm25-bioes-step150-full-validation-audit")


def render_eval_prediction_audit_report(
    *,
    records: Sequence[Mapping[str, Any]],
    issues: Sequence[EvalAuditIssue],
    output_dir: Path,
    predictions_json: Path,
    max_issues: int,
) -> Path:
    """Render the audit index and return its path.

    Returns:
        The path of the written ``index.html``, already on disk when it comes back. The
        generation timestamp is stamped at render time, so two calls with identical inputs
        produce byte-different files.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    html = render_eval_prediction_audit_html(
        records=records,
        issues=issues,
        predictions_json=predictions_json,
        max_issues=max_issues,
        generated_at=datetime.now(UTC).isoformat(),
    )
    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def write_eval_prediction_audit_outputs(
    *,
    predictions_json: Path,
    output_dir: Path,
    max_issues: int = 500,
    records: Sequence[Mapping[str, Any]] | None = None,
    issues: Sequence[EvalAuditIssue] | None = None,
) -> Path:
    """Write the machine-readable audit queue, batches, and HTML index.

    Returns:
        The path of the HTML index, which is written LAST -- the JSON summary, the metric
        scenarios and the batches all land before it, so an existing index implies the
        machine-readable outputs beside it are complete. Passing ``records`` or ``issues``
        skips the corresponding read or audit pass, which is how a caller that already has
        them avoids re-reading the predictions file.

    """
    records = list(records) if records is not None else read_prediction_records(predictions_json)
    issues = list(issues) if issues is not None else audit_records(records)
    summary = summarize_issues(issues)
    scenarios = metric_scenarios(records, issues)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metric_scenarios.json").write_text(
        json.dumps(scenarios, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    issue_rows = [issue.to_dict() for issue in issues]
    write_jsonl(output_dir / "review_queue.jsonl", issue_rows)
    write_jsonl(
        output_dir / "gold_add_candidates.jsonl",
        [issue.to_dict() for issue in issues if issue.recommended_action == "review_gold_add_predicted_span"],
    )
    write_jsonl(
        output_dir / "regex_candidates.jsonl",
        [issue.to_dict() for issue in issues if issue.recommended_action == "review_regex_candidate"],
    )
    write_jsonl(
        output_dir / "nested_candidate_conflicts.jsonl",
        [issue.to_dict() for issue in issues if issue.recommended_action == "review_nested_candidate_conflict"],
    )
    write_adjudication_batches(output_dir, issues)
    write_jsonl(
        output_dir / "gold_remove_or_relabel_candidates.jsonl",
        [issue.to_dict() for issue in issues if issue.recommended_action == "review_gold_remove_or_relabel"],
    )
    return render_eval_prediction_audit_report(
        records=records,
        issues=issues,
        output_dir=output_dir,
        predictions_json=predictions_json,
        max_issues=max_issues,
    )
