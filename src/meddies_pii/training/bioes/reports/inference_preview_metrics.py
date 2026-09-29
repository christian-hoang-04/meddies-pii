"""Exact-span comparison and aggregate metrics for inference previews."""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack is an optional extra; importing it at module load would make the package unimportable without
# reason: it.
from collections.abc import Mapping, Sequence
from typing import Any

from meddies_pii.training.bioes.reports.span_records import (
    SpanKey,
    report_span_from_mapping,
    report_span_key,
    report_span_record,
)


def _span_key(span: Mapping[str, object]) -> SpanKey:
    return report_span_key(span)


def _span_record(span: Mapping[str, object]) -> dict[str, object]:
    return report_span_record(span)


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 0.0 if denominator == 0 else 2 * precision * recall / denominator


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def preview_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for record in records:
        exact = record.get("exact", {})
        if not isinstance(exact, Mapping):
            continue
        true_positive += int(exact.get("true_positive", 0))
        false_positive += int(exact.get("false_positive", 0))
        false_negative += int(exact.get("false_negative", 0))
    precision = _safe_rate(true_positive, true_positive + false_positive)
    recall = _safe_rate(true_positive, true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _span_records_by_key(
    spans: Sequence[Mapping[str, object]],
) -> dict[SpanKey, list[dict[str, object]]]:
    by_key: dict[SpanKey, list[dict[str, object]]] = {}
    for span in spans:
        by_key.setdefault(_span_key(span), []).append(_span_record(span))
    return by_key


def _repeat_span_records(
    by_key: Mapping[SpanKey, Sequence[Mapping[str, object]]],
    keys: Sequence[SpanKey],
) -> list[dict[str, object]]:
    used: dict[SpanKey, int] = {}
    rows = []
    for key in keys:
        index = used.get(key, 0)
        used[key] = index + 1
        candidates = by_key.get(key, ())
        if candidates:
            rows.append(dict(candidates[min(index, len(candidates) - 1)]))
    return rows


# reason: compare spans coordinates safe rate with records by; extra seams would duplicate totals or escaping.
def compare_spans(  # ruff: ignore[too-many-locals]
    gold_spans: Sequence[Mapping[str, object]],
    predicted_spans: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    from collections import Counter

    gold_counts = Counter(_span_key(span) for span in gold_spans)
    predicted_counts = Counter(_span_key(span) for span in predicted_spans)
    true_positive_keys = sorted((gold_counts & predicted_counts).elements())
    false_positive_keys = sorted((predicted_counts - gold_counts).elements())
    false_negative_keys = sorted((gold_counts - predicted_counts).elements())
    precision = _safe_rate(len(true_positive_keys), len(true_positive_keys) + len(false_positive_keys))
    recall = _safe_rate(len(true_positive_keys), len(true_positive_keys) + len(false_negative_keys))
    gold_by_key = _span_records_by_key(gold_spans)
    predicted_by_key = _span_records_by_key(predicted_spans)
    boundary_mismatches = []
    gold_by_boundary = {}
    for span in gold_spans:
        gold_span = report_span_from_mapping(span)
        gold_by_boundary[gold_span.boundary_key] = gold_span.label
    for span in predicted_spans:
        predicted_span = report_span_from_mapping(span)
        boundary = predicted_span.boundary_key
        gold_label = gold_by_boundary.get(boundary)
        predicted_label = predicted_span.label
        if gold_label is not None and gold_label != predicted_label:
            boundary_mismatches.append({
                "start": boundary[0],
                "end": boundary[1],
                "text": boundary[2],
                "gold_label": gold_label,
                "predicted_label": predicted_label,
            })
    return {
        "true_positive": len(true_positive_keys),
        "false_positive": len(false_positive_keys),
        "false_negative": len(false_negative_keys),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "tp_spans": _repeat_span_records(gold_by_key, true_positive_keys),
        "fp_spans": _repeat_span_records(predicted_by_key, false_positive_keys),
        "fn_spans": _repeat_span_records(gold_by_key, false_negative_keys),
        "boundary_label_mismatches": boundary_mismatches,
    }
