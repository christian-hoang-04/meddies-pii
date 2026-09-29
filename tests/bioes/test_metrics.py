from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
from math import isclose

from meddies_pii.evaluation.span_metrics import (
    SpanMetricBlock,
    containment_span_prf_by_label,
    exact_span_prf_by_label,
)
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import PII_LABELS

GOLD_BY_DOC: dict[str, tuple[CharSpan, ...]] = {
    "doc1": (
        CharSpan(start=0, end=4, text="John", label="human_name"),
        CharSpan(start=10, end=18, text="555-0100", label="phone_number"),
        CharSpan(start=20, end=30, text="2026-06-09", label="date"),
    ),
    "doc2": (
        CharSpan(start=0, end=4, text="Jane", label="human_name"),
        CharSpan(start=40, end=50, text="2026-06-10", label="date"),
    ),
}
"""Two docs, three labels.

``date`` appears only in gold with zero predictions, guarding against per-label silently dropping a zero-prediction label.

"""

PREDICTED_BY_DOC: dict[str, tuple[CharSpan, ...]] = {
    "doc1": (
        CharSpan(start=0, end=4, text="John", label="human_name"),
        CharSpan(start=10, end=18, text="555-0100", label="phone_number"),
    ),
    "doc2": (
        CharSpan(start=0, end=4, text="Jane", label="human_name"),
        CharSpan(start=60, end=64, text="Mike", label="human_name"),
        CharSpan(start=70, end=78, text="555-0200", label="phone_number"),
    ),
}


def _assert_prf(block: SpanMetricBlock, *, precision: float, recall: float, f1: float) -> None:
    assert isclose(block["precision"], precision, abs_tol=1e-9)
    assert isclose(block["recall"], recall, abs_tol=1e-9)
    assert isclose(block["f1"], f1, abs_tol=1e-9)


def test_exact_span_prf_by_label_micro_averages_per_label() -> None:
    """human_name: tp=2, pred=3, gold=2 -> P=2/3, R=1.0, F1=0.8.

    phone_number: tp=1, pred=2, gold=1 -> P=0.5, R=1.0, F1=2/3.

    """
    result = exact_span_prf_by_label(PREDICTED_BY_DOC, GOLD_BY_DOC)

    _assert_prf(result["human_name"], precision=2 / 3, recall=1.0, f1=0.8)
    _assert_prf(result["phone_number"], precision=0.5, recall=1.0, f1=2 / 3)


def test_exact_span_prf_by_label_zero_prediction_label_is_zeroed_not_dropped() -> None:
    """Date appears in gold (2 spans) but has zero predictions.

    Must be present with explicit zeros, never a KeyError and never silently omitted.

    """
    result = exact_span_prf_by_label(PREDICTED_BY_DOC, GOLD_BY_DOC)

    assert "date" in result
    _assert_prf(result["date"], precision=0.0, recall=0.0, f1=0.0)
    assert result["date"]["gold_total"] == 2.0
    assert result["date"]["pred_total"] == 0.0


def test_exact_span_prf_by_label_keys_are_allowed_labels() -> None:
    result = exact_span_prf_by_label(PREDICTED_BY_DOC, GOLD_BY_DOC)
    assert set(result) <= set(PII_LABELS)


def test_containment_span_prf_by_label_micro_averages_per_label() -> None:
    result = containment_span_prf_by_label(PREDICTED_BY_DOC, GOLD_BY_DOC)

    _assert_prf(result["human_name"], precision=2 / 3, recall=1.0, f1=0.8)
    _assert_prf(result["phone_number"], precision=0.5, recall=1.0, f1=2 / 3)


def test_containment_span_prf_by_label_zero_prediction_label_is_zeroed_not_dropped() -> None:
    result = containment_span_prf_by_label(PREDICTED_BY_DOC, GOLD_BY_DOC)

    assert "date" in result
    _assert_prf(result["date"], precision=0.0, recall=0.0, f1=0.0)
    assert result["date"]["gold_total"] == 2.0
    assert result["date"]["pred_total"] == 0.0
