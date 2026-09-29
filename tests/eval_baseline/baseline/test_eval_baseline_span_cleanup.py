from __future__ import annotations

from meddies_pii.eval_baseline.baseline.span_cleanup import clean_spans
from meddies_pii.spans import CharSpan


def _span(text: str, start: int, end: int, label: str) -> CharSpan:
    return CharSpan(start=start, end=end, text=text[start:end], label=label)


def test_trims_leading_whitespace_and_trailing_punctuation() -> None:
    text = "Hi  Mister, x"
    raw = [_span(text, 2, 11, "human_name")]
    out = clean_spans(text, raw)
    assert len(out) == 1
    assert out[0].text == "Mister"
    assert text[out[0].start : out[0].end] == out[0].text


def test_merges_adjacent_same_label_separated_by_whitespace() -> None:
    text = "Mister Enie Idilia is here"
    raw = [
        _span(text, 0, 6, "human_name"),
        _span(text, 7, 11, "human_name"),
        _span(text, 12, 18, "human_name"),
    ]
    out = clean_spans(text, raw)
    assert len(out) == 1
    assert out[0].text == "Mister Enie Idilia"


def test_does_not_merge_across_non_whitespace_gap() -> None:
    text = "Mister, John"
    raw = [_span(text, 0, 6, "human_name"), _span(text, 8, 12, "human_name")]
    out = clean_spans(text, raw)
    assert len(out) == 2


def test_does_not_merge_different_labels() -> None:
    text = "Mister 1999"
    raw = [_span(text, 0, 6, "human_name"), _span(text, 7, 11, "date")]
    out = clean_spans(text, raw)
    assert len(out) == 2


def test_drops_span_that_is_all_punctuation_or_whitespace() -> None:
    text = "a , b"
    raw = [_span(text, 1, 4, "human_name")]
    out = clean_spans(text, raw)
    assert out == []


def test_preserves_internal_whitespace_after_merge() -> None:
    text = "Lorong K Telok Kurau"
    raw = [_span(text, 0, 8, "address"), _span(text, 9, 20, "address")]
    out = clean_spans(text, raw)
    assert len(out) == 1
    assert out[0].text == "Lorong K Telok Kurau"
