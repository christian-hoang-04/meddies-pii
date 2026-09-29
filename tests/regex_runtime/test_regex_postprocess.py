from __future__ import annotations

from typing import TYPE_CHECKING

from meddies_pii.regex_runtime.postprocess import merge_tiered_candidates
from meddies_pii.regex_runtime.rules import RegexCandidate, RegexTier
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from meddies_pii.taxonomy import PiiLabel

_TEXT = "Email: alpha@example.invalid"


def _candidate(start: int, end: int, label: PiiLabel, tier: RegexTier) -> RegexCandidate:
    return RegexCandidate(CharSpan(start, end, _TEXT[start:end], label), tier, "probe")


def test_snap_candidate_without_an_overlapping_model_span_adds_nothing() -> None:
    candidate = _candidate(7, 28, "email_address", "SNAP")

    result = merge_tiered_candidates(_TEXT, (), (candidate,))

    assert result.spans == ()
    assert result.conflicts == ()


def test_snap_candidate_widens_an_overlapping_model_span_of_its_label() -> None:
    model_span = CharSpan(7, 20, "alpha@example", "email_address")
    candidate = _candidate(7, 28, "email_address", "SNAP")

    result = merge_tiered_candidates(_TEXT, (model_span,), (candidate,))

    assert result.spans == (CharSpan(7, 28, "alpha@example.invalid", "email_address"),)


def test_snap_candidate_never_erases_an_overlapping_model_span_of_another_label() -> None:
    model_span = CharSpan(7, 12, "alpha", "human_name")
    candidate = _candidate(7, 28, "email_address", "SNAP")

    result = merge_tiered_candidates(_TEXT, (model_span,), (candidate,))

    assert result.spans == (
        model_span,
        CharSpan(12, 28, "@example.invalid", "email_address"),
    )
    assert [conflict.reason for conflict in result.conflicts] == ["different_label_trimmed_remainder"]


def test_structured_candidate_emits_each_non_overlapping_remainder() -> None:
    text = "abc__def__ghi@example"
    first_model = CharSpan(0, 3, "abc", "human_name")
    second_model = CharSpan(10, 13, "ghi", "human_name")
    candidate = RegexCandidate(CharSpan(0, len(text), text, "email_address"), "AUTH", "probe")

    result = merge_tiered_candidates(text, (first_model, second_model), (candidate,))

    assert result.spans == (
        first_model,
        CharSpan(3, 10, "__def__", "email_address"),
        second_model,
        CharSpan(13, len(text), "@example", "email_address"),
    )
    assert [conflict.reason for conflict in result.conflicts] == [
        "different_label_trimmed_remainder",
        "different_label_trimmed_remainder",
    ]


def test_structured_candidate_drops_a_remainder_shorter_than_three_characters() -> None:
    text = "abc@x"
    model_span = CharSpan(0, 3, "abc", "human_name")
    candidate = RegexCandidate(CharSpan(0, len(text), text, "email_address"), "AUTH", "probe")

    result = merge_tiered_candidates(text, (model_span,), (candidate,))

    assert result.spans == (model_span,)
    assert [conflict.reason for conflict in result.conflicts] == ["different_label_trimmed_remainder"]


def test_company_candidate_with_a_different_label_overlap_is_discarded() -> None:
    text = "Acme Hospital"
    model_span = CharSpan(0, 4, "Acme", "human_name")
    candidate = RegexCandidate(CharSpan(0, len(text), text, "company_name"), "AUTH", "probe")

    result = merge_tiered_candidates(text, (model_span,), (candidate,))

    assert result.spans == (model_span,)
    assert [conflict.reason for conflict in result.conflicts] == ["different_label_preserves_model"]


def test_context_candidate_adds_a_span_where_the_model_found_nothing() -> None:
    candidate = _candidate(7, 28, "email_address", "CONTEXT")

    result = merge_tiered_candidates(_TEXT, (), (candidate,))

    assert result.spans == (CharSpan(7, 28, "alpha@example.invalid", "email_address"),)
