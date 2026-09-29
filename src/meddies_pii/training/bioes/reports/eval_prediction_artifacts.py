"""Persist deterministic evaluation-audit review artifacts."""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from meddies_pii.training.bioes.eval.audit import EvalAuditIssue


def read_prediction_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        msg = f"Prediction payload has no records list: {path}"
        raise ValueError(msg)
    return [record for record in records if isinstance(record, dict)]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _issue_key(issue: EvalAuditIssue) -> tuple[str, str, str, str, int, int, str]:
    return (
        issue.uid,
        issue.issue_type,
        issue.recommended_action,
        issue.label,
        issue.start,
        issue.end,
        issue.text,
    )


def _sorted_issue_rows(issues: Sequence[EvalAuditIssue]) -> list[dict[str, object]]:
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    return [
        issue.to_dict()
        for issue in sorted(
            issues,
            key=lambda item: (
                severity_rank[item.severity],
                item.uid,
                item.start,
                item.label,
                item.issue_type,
            ),
        )
    ]


def write_adjudication_batches(output_dir: Path, issues: Sequence[EvalAuditIssue]) -> None:
    batch_dir = output_dir / "adjudication_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for old_file in batch_dir.glob("*.jsonl"):
        old_file.unlink()

    selected: set[tuple[str, str, str, str, int, int, str]] = set()
    batch_specs: list[tuple[str, str, Any]] = [
        (
            "01_metric_predicted_gold_add.jsonl",
            "Model-predicted spans absent from gold; these are the only add-span rows used by the metric what-if.",
            lambda issue: issue.recommended_action == "review_gold_add_predicted_span",
        ),
        (
            "02_company_regex_candidates.jsonl",
            "Organization/facility/insurer candidates not predicted by the model and absent from gold.",
            lambda issue: issue.recommended_action == "review_regex_candidate" and issue.label == "company_name",
        ),
        (
            "03_gold_remove_or_relabel.jsonl",
            "Gold spans that violate current label policy, usually lab ranges or field-prefix leakage.",
            lambda issue: issue.recommended_action == "review_gold_remove_or_relabel",
        ),
        (
            "04_label_mismatch_review.jsonl",
            "Prediction/gold overlaps with different labels.",
            lambda issue: issue.recommended_action == "review_gold_or_prediction_label",
        ),
        (
            "05_boundary_policy_review.jsonl",
            "Same-label overlaps with different boundaries.",
            lambda issue: issue.recommended_action == "review_boundary_policy",
        ),
        (
            "06_private_url_secret_review.jsonl",
            "Privacy-critical private_url and secret rows across all remaining actions.",
            lambda issue: issue.label in {"private_url", "secret"},
        ),
        (
            "07_model_miss_review.jsonl",
            "Gold spans not predicted and not already covered by a policy problem.",
            lambda issue: issue.recommended_action == "review_model_miss",
        ),
        (
            "08_regex_candidate_remainder.jsonl",
            "Remaining regex-only candidates; lower priority than metric-affecting add spans.",
            lambda issue: issue.recommended_action == "review_regex_candidate",
        ),
        (
            "09_nested_candidate_conflicts.jsonl",
            "Regex candidates that overlap existing gold/predicted spans with another label.",
            lambda issue: issue.recommended_action == "review_nested_candidate_conflict",
        ),
        (
            "10_model_false_positive_review.jsonl",
            "Predicted spans absent from gold that currently fail label-specific policy checks.",
            lambda issue: issue.recommended_action == "review_model_false_positive",
        ),
    ]

    summaries: list[dict[str, object]] = []
    for file_name, description, predicate in batch_specs:
        batch = [issue for issue in issues if _issue_key(issue) not in selected and predicate(issue)]
        selected.update(_issue_key(issue) for issue in batch)
        rows = _sorted_issue_rows(batch)
        write_jsonl(batch_dir / file_name, rows)
        summaries.append({"file": file_name, "rows": len(rows), "description": description})

    remainder = [issue for issue in issues if _issue_key(issue) not in selected]
    if remainder:
        rows = _sorted_issue_rows(remainder)
        write_jsonl(batch_dir / "99_unbatched_review.jsonl", rows)
        summaries.append({
            "file": "99_unbatched_review.jsonl",
            "rows": len(rows),
            "description": "Rows not captured by the named priority batches.",
        })

    (batch_dir / "summary.json").write_text(
        json.dumps({"batches": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# BIOES validation adjudication batches",
        "",
        "These batches are reviewer queues, not automatic gold edits.",
        "",
        "Priority:",
        "",
        "1. `01_metric_predicted_gold_add.jsonl` changes the checkpoint metric if accepted.",
        (
            "2. `02_company_regex_candidates.jsonl` exposes missing organization/facility gold spans "
            "that the model may also miss."
        ),
        "3. `03_gold_remove_or_relabel.jsonl` removes obvious gold-policy violations.",
        (
            "4. Boundary, label, `private_url`, and `secret` batches lock the policy before any "
            "corrected validation artifact is frozen."
        ),
        "",
        (
            "Review rule: accept only spans that match the adjudication policy; do not copy model "
            "predictions into gold without human confirmation."
        ),
        "",
        "## Batch counts",
        "",
    ]
    readme.extend(f"- `{item['file']}` — {item['rows']} rows. {item['description']}" for item in summaries)
    (batch_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
