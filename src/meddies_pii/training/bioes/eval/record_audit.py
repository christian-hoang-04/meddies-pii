from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from meddies_pii.json_types import is_str_mapping

from .audit_models import (
    EvalAuditIssue,
    Severity,
    SpanAuditRecord,
    _candidate_exact_match,
    _context,
    _overlap,
    _overlapping_spans,
    _same_boundary,
    _span_covered,
)
from .candidate_detection import find_label_candidates
from .span_validation import _looks_valid_for_label, _suspicious_gold_reason


def _parse_spans(raw_spans: object) -> list[SpanAuditRecord]:
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
        return []
    parsed: list[SpanAuditRecord] = []
    for raw_span in raw_spans:
        if not is_str_mapping(raw_span):
            continue
        parsed.append(SpanAuditRecord.from_mapping(raw_span))
    return parsed


# reason: issue keeps uid/evidence at its adapter seam; bundling would hide required inputs.
def _issue(  # ruff: ignore[too-many-arguments]
    *,
    uid: str,
    issue_type: str,
    severity: Severity,
    label: str,
    start: int,
    end: int,
    text: str,
    reason: str,
    recommended_action: str,
    document_text: str,
    evidence: dict[str, object],
) -> EvalAuditIssue:
    return EvalAuditIssue(
        uid=uid,
        issue_type=issue_type,
        severity=severity,
        label=label,
        start=start,
        end=end,
        text=text,
        reason=reason,
        recommended_action=recommended_action,
        context=_context(document_text, start, end),
        evidence=evidence,
    )


# reason: audit record owns parse spans and exact match together; splitting would fragment diagnostics.
def audit_record(record: Mapping[str, object]) -> list[EvalAuditIssue]:  # ruff: ignore[complex-structure,too-many-branches,too-many-locals]
    uid = str(record.get("uid", "unknown"))
    text = str(record.get("text", ""))
    gold_spans = _parse_spans(record.get("gold_spans", []))
    predicted_spans = _parse_spans(record.get("predicted_spans", []))
    issues: list[EvalAuditIssue] = []

    for candidate in find_label_candidates(text):
        if _candidate_exact_match(candidate, gold_spans):
            continue
        if _candidate_exact_match(candidate, predicted_spans):
            continue
        overlapping_gold = _overlapping_spans(candidate, gold_spans)
        overlapping_predicted = _overlapping_spans(candidate, predicted_spans)
        if overlapping_gold or overlapping_predicted:
            if candidate.severity == "low":
                continue
            if candidate.label in {"date", "phone_number"} and any(
                span.label in {"address", "date", "id_number", "private_url", "secret"}
                for span in (*overlapping_gold, *overlapping_predicted)
            ):
                continue
            issues.append(
                _issue(
                    uid=uid,
                    issue_type="nested_candidate_conflict",
                    severity="medium",
                    label=candidate.label,
                    start=candidate.start,
                    end=candidate.end,
                    text=candidate.text,
                    reason=candidate.reason,
                    recommended_action="review_nested_candidate_conflict",
                    document_text=text,
                    evidence={
                        "candidate": candidate.to_dict(),
                        "overlapping_gold_spans": [span.to_dict() for span in overlapping_gold],
                        "overlapping_predicted_spans": [span.to_dict() for span in overlapping_predicted],
                    },
                ),
            )
            continue
        issues.append(
            _issue(
                uid=uid,
                issue_type="unlabeled_candidate",
                severity=candidate.severity,
                label=candidate.label,
                start=candidate.start,
                end=candidate.end,
                text=candidate.text,
                reason=candidate.reason,
                recommended_action="review_regex_candidate",
                document_text=text,
                evidence={"candidate": candidate.to_dict()},
            ),
        )

    exact_gold = {(span.label, span.start, span.end): span for span in gold_spans}
    exact_pred = {(span.label, span.start, span.end): span for span in predicted_spans}
    gold_only = [span for key, span in exact_gold.items() if key not in exact_pred]
    pred_only = [span for key, span in exact_pred.items() if key not in exact_gold]

    for span in gold_only:
        reason = _suspicious_gold_reason(span)
        if reason is not None:
            issues.append(
                _issue(
                    uid=uid,
                    issue_type="suspicious_gold_span",
                    severity="high",
                    label=span.label,
                    start=span.start,
                    end=span.end,
                    text=span.text,
                    reason=reason,
                    recommended_action="review_gold_remove_or_relabel",
                    document_text=text,
                    evidence={"gold_span": span.to_dict()},
                ),
            )

    for predicted in pred_only:
        overlapping_gold_same_label = [
            span
            for span in gold_spans
            if span.label == predicted.label
            and _overlap(predicted.start, predicted.end, span.start, span.end) > 0
            and not _same_boundary(predicted, span)
        ]
        if overlapping_gold_same_label:
            issues.append(
                _issue(
                    uid=uid,
                    issue_type="boundary_mismatch",
                    severity="medium",
                    label=predicted.label,
                    start=predicted.start,
                    end=predicted.end,
                    text=predicted.text,
                    reason="prediction_overlaps_gold_same_label_with_different_boundary",
                    recommended_action="review_boundary_policy",
                    document_text=text,
                    evidence={
                        "predicted_span": predicted.to_dict(),
                        "overlapping_gold_spans": [span.to_dict() for span in overlapping_gold_same_label],
                    },
                ),
            )
            continue

        overlapping_gold_other_label = [
            span
            for span in gold_spans
            if span.label != predicted.label and _overlap(predicted.start, predicted.end, span.start, span.end) > 0
        ]
        if overlapping_gold_other_label:
            issues.append(
                _issue(
                    uid=uid,
                    issue_type="label_mismatch",
                    severity="high",
                    label=predicted.label,
                    start=predicted.start,
                    end=predicted.end,
                    text=predicted.text,
                    reason="prediction_overlaps_gold_with_different_label",
                    recommended_action="review_gold_or_prediction_label",
                    document_text=text,
                    evidence={
                        "predicted_span": predicted.to_dict(),
                        "overlapping_gold_spans": [span.to_dict() for span in overlapping_gold_other_label],
                    },
                ),
            )
            continue

        recommended_action = (
            "review_gold_add_predicted_span" if _looks_valid_for_label(predicted) else "review_model_false_positive"
        )
        issues.append(
            _issue(
                uid=uid,
                issue_type="prediction_not_in_gold",
                severity=("high" if recommended_action == "review_gold_add_predicted_span" else "medium"),
                label=predicted.label,
                start=predicted.start,
                end=predicted.end,
                text=predicted.text,
                reason="predicted_span_absent_from_gold",
                recommended_action=recommended_action,
                document_text=text,
                evidence={"predicted_span": predicted.to_dict()},
            ),
        )

    for gold in gold_only:
        if _span_covered(gold, predicted_spans):
            continue
        if _suspicious_gold_reason(gold) is not None:
            continue
        issues.append(
            _issue(
                uid=uid,
                issue_type="gold_not_predicted",
                severity="medium",
                label=gold.label,
                start=gold.start,
                end=gold.end,
                text=gold.text,
                reason="gold_span_absent_from_prediction",
                recommended_action="review_model_miss",
                document_text=text,
                evidence={"gold_span": gold.to_dict()},
            ),
        )

    deduped: dict[tuple[str, str, str, int, int, str, str], EvalAuditIssue] = {}
    for issue in issues:
        deduped[
            issue.uid,
            issue.issue_type,
            issue.recommended_action,
            issue.start,
            issue.end,
            issue.label,
            issue.text,
        ] = issue

    return sorted(
        deduped.values(),
        key=lambda issue: (
            {"high": 0, "medium": 1, "low": 2}[issue.severity],
            issue.uid,
            issue.start,
            issue.issue_type,
        ),
    )


def audit_records(records: Sequence[Mapping[str, object]]) -> list[EvalAuditIssue]:
    issues: list[EvalAuditIssue] = []
    for record in records:
        issues.extend(audit_record(record))
    return issues


def summarize_issues(issues: Sequence[EvalAuditIssue]) -> dict[str, Any]:
    by_type = Counter(issue.issue_type for issue in issues)
    by_action = Counter(issue.recommended_action for issue in issues)
    by_label = Counter(issue.label for issue in issues)
    by_severity = Counter(issue.severity for issue in issues)
    return {
        "issues": len(issues),
        "by_type": dict(sorted(by_type.items())),
        "by_action": dict(sorted(by_action.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_severity": dict(sorted(by_severity.items())),
    }
