from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
import string
from math import isclose
from typing import cast

import pytest

from meddies_pii.evaluation.span_metrics import (
    SpanMode,
    coerce_exact_span_report,
    containment_span_report_by_doc,
    containment_span_slice_report_by_doc,
    exact_span_prf,
    exact_span_report_by_doc,
    exact_span_slice_report_by_doc,
    typed_span_counts_by_label,
)
from meddies_pii.json_types import is_str_mapping
from meddies_pii.spans import CharSpan


def _span(start: int, end: int, label: str = "human_name") -> CharSpan:
    text = string.digits
    return CharSpan(start=start, end=end, text=text[start:end], label=label)


def test_exact_report_keeps_duplicate_counts_and_exposes_label_confusion() -> None:
    report = exact_span_report_by_doc(
        {"one": (_span(0, 4), _span(0, 4), _span(5, 8, "date"))},
        {"one": (_span(0, 4), _span(5, 8, "human_name"))},
    )

    assert report["typed"] == {
        "tp": 1.0,
        "pred_total": 3.0,
        "gold_total": 2.0,
        "precision": 1 / 3,
        "recall": 0.5,
        "f1": 0.4,
    }
    assert report["untyped"] == {
        "tp": 2.0,
        "pred_total": 3.0,
        "gold_total": 2.0,
        "precision": 2 / 3,
        "recall": 1.0,
        "f1": 0.8,
    }
    assert report["untyped_minus_typed"] == {
        "precision": 1 / 3,
        "recall": 0.5,
        "f1": 0.4,
    }


def test_containment_report_credits_geometry_without_erasing_label_error() -> None:
    """A narrow prediction earns precision credit only.

    It still misses recall because the larger gold span is not contained in the prediction.

    """
    report = containment_span_report_by_doc(
        {"one": (_span(2, 8, "date"),)},
        {"one": (_span(0, 10, "human_name"),)},
    )

    assert report["typed"]["precision"] == 0.0
    assert report["typed"]["recall"] == 0.0
    assert report["untyped"]["precision"] == 1.0
    assert report["untyped"]["recall"] == 0.0
    assert report["untyped_minus_typed"] == {
        "precision": 1.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_exact_prf_rejects_unknown_mode_and_zero_denominators_are_zero() -> None:
    assert exact_span_prf([], []) == {
        "tp": 0.0,
        "pred_total": 0.0,
        "gold_total": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    with pytest.raises(ValueError, match="Unsupported exact-span mode"):
        exact_span_prf([_span(0, 1)], [_span(0, 1)], mode=cast("SpanMode", "other"))


def test_slice_reports_retain_declared_empty_slice_and_ignore_unknown_docs() -> None:
    predicted = {"one": (_span(0, 4),)}
    gold = {"one": (_span(0, 4),)}
    slices = {"one": ("happy",), "not-loaded": ("empty",)}

    exact = exact_span_slice_report_by_doc(predicted, gold, slices, slice_names=("empty", "happy"))
    containment = containment_span_slice_report_by_doc(predicted, gold, slices, slice_names=("empty", "happy"))

    assert exact["empty"]["support_docs"] == 1
    assert exact["empty"]["support_gold_spans"] == 0
    assert exact["happy"]["typed"]["f1"] == 1.0
    assert containment["empty"]["support_predicted_spans"] == 0
    assert containment["happy"]["typed"]["precision_tp"] == 1.0


def test_coerce_legacy_report_builds_containment_fallbacks_and_preserves_nested_values() -> None:
    legacy = coerce_exact_span_report({
        "tp": "2",
        "pred_total": 4,
        "gold_total": 5,
        "precision": "0.5",
        "recall": 0.4,
        "f1": 4 / 9,
        "slices": {
            "challenge": {
                "support_docs": 3,
                "typed": {
                    "tp": 1,
                    "pred_total": 2,
                    "gold_total": 2,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                },
                "untyped": {
                    "tp": 2,
                    "pred_total": 2,
                    "gold_total": 2,
                    "precision": 1,
                    "recall": 1,
                    "f1": 1,
                },
            },
        },
    })

    assert legacy["typed"]["tp"] == 2.0
    assert legacy["untyped"] == legacy["typed"]
    assert legacy["containment_span"]["precision_tp"] == 2.0
    challenge = legacy["containment_span"]["slices"]["challenge"]
    assert is_str_mapping(challenge)
    assert challenge["support_docs"] == 3
    challenge_typed = challenge["typed"]
    assert is_str_mapping(challenge_typed)
    assert challenge_typed["precision_tp"] == 1.0
    assert challenge["untyped_minus_typed"] == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_coerce_current_report_uses_explicit_containment_delta_and_slices() -> None:
    report = coerce_exact_span_report({
        "typed": {
            "tp": 1,
            "pred_total": 2,
            "gold_total": 2,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        },
        "untyped": {
            "tp": 2,
            "pred_total": 2,
            "gold_total": 2,
            "precision": 1,
            "recall": 1,
            "f1": 1,
        },
        "containment_span": {
            "typed": {
                "precision_tp": 1,
                "recall_tp": 2,
                "pred_total": 3,
                "gold_total": 4,
                "precision": 1 / 3,
                "recall": 0.5,
                "f1": 0.4,
            },
            "untyped": {
                "precision_tp": 3,
                "recall_tp": 4,
                "pred_total": 3,
                "gold_total": 4,
                "precision": 1,
                "recall": 1,
                "f1": 1,
            },
            "untyped_minus_typed": {
                "precision": "0.7",
                "recall": "0.5",
                "f1": "0.6",
            },
            "slices": {"adversarial": {"opaque": True}},
        },
    })

    containment = report["containment_span"]
    assert containment["typed"]["precision_tp"] == 1.0
    assert containment["untyped_minus_typed"] == {
        "precision": 0.7,
        "recall": 0.5,
        "f1": 0.6,
    }
    assert containment["slices"] == {"adversarial": {"opaque": True}}
    assert isclose(report["untyped_minus_typed"]["f1"], 0.5)


def test_typed_span_counts_by_label_preserves_exact_duplicates_and_containment_geometry() -> None:
    exact, containment = typed_span_counts_by_label(
        (
            _span(0, 4, "human_name"),
            _span(1, 3, "human_name"),
            _span(5, 8, "date"),
            _span(5, 8, "unknown"),
        ),
        (_span(0, 4, "human_name"), _span(6, 8, "date")),
    )

    assert exact["human_name"] == {"tp": 1, "pred_total": 2, "gold_total": 1}
    assert containment["human_name"] == {
        "precision_tp": 2,
        "recall_tp": 1,
        "pred_total": 2,
        "gold_total": 1,
    }
    assert exact["date"] == {"tp": 0, "pred_total": 1, "gold_total": 1}
    assert containment["date"] == {
        "precision_tp": 0,
        "recall_tp": 1,
        "pred_total": 1,
        "gold_total": 1,
    }
