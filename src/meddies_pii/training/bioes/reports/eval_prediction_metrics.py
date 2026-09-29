"""Compute exact-span evaluation-audit metric scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.reports.span_records import (
    SpanKey,
    report_span_key,
    report_span_mappings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from meddies_pii.training.bioes.eval.audit import EvalAuditIssue


def _span_key(span: Mapping[str, object]) -> SpanKey:
    return report_span_key(span)


def _prf(
    *,
    true_positive: int,
    pred_total: int,
    gold_total: int,
) -> dict[str, float | int]:
    precision = true_positive / pred_total if pred_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "pred_total": pred_total,
        "gold_total": gold_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def metric_scenarios(
    records: Sequence[Mapping[str, Any]],
    issues: Sequence[EvalAuditIssue],
) -> dict[str, Any]:
    add_to_gold: dict[str, set[SpanKey]] = {}
    remove_from_gold: dict[str, set[SpanKey]] = {}
    for issue in issues:
        if issue.issue_type == "prediction_not_in_gold" and issue.recommended_action == "review_gold_add_predicted_span":
            span = issue.evidence.get("predicted_span")
            if is_str_mapping(span):
                add_to_gold.setdefault(issue.uid, set()).add(_span_key(span))
        if issue.issue_type == "suspicious_gold_span" and issue.recommended_action == "review_gold_remove_or_relabel":
            span = issue.evidence.get("gold_span")
            if is_str_mapping(span):
                remove_from_gold.setdefault(issue.uid, set()).add(_span_key(span))

    def score(*, adjusted: bool) -> dict[str, float | int]:
        true_positive = 0
        pred_total = 0
        gold_total = 0
        for record in records:
            uid = str(record.get("uid", "unknown"))
            gold_spans = report_span_mappings(record.get("gold_spans", []))
            predicted_spans = report_span_mappings(record.get("predicted_spans", []))
            gold = {_span_key(span) for span in gold_spans}
            predicted = {_span_key(span) for span in predicted_spans}
            if adjusted:
                gold = (gold - remove_from_gold.get(uid, set())) | add_to_gold.get(uid, set())
            true_positive += len(gold & predicted)
            pred_total += len(predicted)
            gold_total += len(gold)
        return _prf(
            true_positive=true_positive,
            pred_total=pred_total,
            gold_total=gold_total,
        )

    return {
        "raw_exact": score(adjusted=False),
        "if_accept_prediction_gold_adds_and_suspicious_gold_removals": score(adjusted=True),
        "adjustment_counts": {
            "prediction_spans_to_add_to_gold": sum(len(values) for values in add_to_gold.values()),
            "gold_spans_to_remove_or_relabel": sum(len(values) for values in remove_from_gold.values()),
        },
        "note": (
            "Adjusted metrics are a what-if for human-reviewed gold fixes only: "
            "model-predicted spans marked likely missing from gold are added, and "
            "suspicious gold spans are removed. Unlabeled candidates not predicted "
            "by the model are not added in this scenario."
        ),
    }
