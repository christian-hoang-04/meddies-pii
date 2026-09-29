from __future__ import annotations

import pytest

from meddies_pii.spans import CharSpan
from meddies_pii.training.bioes.eval.harness import (
    classify_adversarial_slices,
    new_slice_filter_report,
    record_slice_event,
)


def test_classifier_identifies_repeated_names_and_minimal_name_fragments() -> None:
    text = "Review A. Smith before Smith calls."
    first = text.index("Smith")
    second = text.rindex("Smith")
    spans = (
        CharSpan(first, first + 5, "Smith", "human_name"),
        CharSpan(second, second + 5, "Smith", "human_name"),
        CharSpan(0, 0, "", "human_name"),
        CharSpan(0, 1, " ", "human_name"),
    )

    slices = classify_adversarial_slices(text, spans)

    assert "minimal_name_fragment" in slices
    assert "repeated_surface" in slices


def test_classifier_omits_optional_slices_from_plain_text() -> None:
    assert classify_adversarial_slices("Routine clinical note.") == ("all",)


@pytest.mark.parametrize(
    ("stage", "count_key", "reason_key"),
    [
        ("attempted", "attempted_docs", None),
        ("source_accepted", "source_accepted_docs", None),
        ("source_rejected", "source_filtered_docs", "unknown"),
        ("preparation_attempted", "preparation_attempted_docs", None),
        ("prepared", "prepared_docs", None),
        ("preparation_rejected", "preparation_filtered_docs", "alignment"),
    ],
)
def test_slice_report_counts_each_filter_stage(
    stage: str,
    count_key: str,
    reason_key: str | None,
) -> None:
    report = new_slice_filter_report()

    record_slice_event(
        report,
        ("all", "not-a-registered-slice"),
        stage=stage,
        reason="alignment" if stage == "preparation_rejected" else None,
    )

    assert report["all"][count_key] == 1
    if reason_key == "unknown":
        assert report["all"]["source_rejection_reasons"] == {"unknown": 1}
    elif reason_key == "alignment":
        assert report["all"]["preparation_rejection_reasons"] == {"alignment": 1}


def test_slice_report_accumulates_named_rejection_reasons() -> None:
    report = new_slice_filter_report()

    record_slice_event(
        report,
        ("all",),
        stage="source_rejected",
        reason="label_mismatch",
    )
    record_slice_event(
        report,
        ("all",),
        stage="source_rejected",
        reason="label_mismatch",
    )

    assert report["all"]["source_rejection_reasons"] == {"label_mismatch": 2}


def test_slice_event_without_report_is_a_no_op() -> None:
    assert record_slice_event(None, ("all",), stage="attempted") is None


def test_slice_report_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unsupported slice filter stage: 'invalid'"):
        record_slice_event(new_slice_filter_report(), ("all",), stage="invalid")
