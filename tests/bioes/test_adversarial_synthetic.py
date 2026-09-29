"""An Arabic numeral adjacent to spelled digit-words, as in "012 ba bốn".

The slicer only matched pure same-language word runs, so mixed forms (a phone written half in numerals, half spoken)
slipped out of the hard-case slice.

a stray numeral with no adjacent spelled digit-word is not the hard case.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.data.adversarial_synthetic import (
    build_adversarial_pii_label_examples,
)
from meddies_pii.training.bioes.eval.harness import classify_adversarial_slices

if TYPE_CHECKING:
    from collections.abc import Mapping


def test_adversarial_synthetic_examples_have_valid_offsets_and_pii_label_labels() -> None:
    examples = build_adversarial_pii_label_examples()

    assert len(examples) >= 6
    for idx, record in enumerate(examples):
        _, text, spans = parse_labeled_record(record, default_id=f"synthetic-{idx}")
        for span in spans:
            assert text[span.start : span.end] == span.text


def test_adversarial_synthetic_examples_cover_missing_opf_style_slices() -> None:
    observed: set[str] = set()
    for record in build_adversarial_pii_label_examples():
        _, text, spans = parse_labeled_record(record)
        observed.update(classify_adversarial_slices(text, spans))

    assert {
        "at_dot_obfuscation",
        "line_breaks",
        "spacing",
        "symbol_substitution",
        "emoji_word_replacement",
        "phonetic_alphabet",
        "digit_words",
    }.issubset(observed)


def _info_id(record: Mapping[str, object]) -> object:
    """Return a record's ``info.id``, narrowing the nested block where it is read.

    `build_adversarial_pii_label_examples` returns `list[dict[str, object]]`, so the
    `info` block arrives as `object` and cannot be read by key until the shape is stated.

    Returns:
        The ``id`` of the record's ``info`` block, or None when the record carries no
        readable ``info`` mapping.

    """
    info = record.get("info")
    return info.get("id") if is_str_mapping(info) else None


def test_adversarial_synthetic_includes_public_url_date_negative() -> None:
    negative = [
        record
        for record in build_adversarial_pii_label_examples()
        if _info_id(record) == "synthetic-public-url-date-negative"
    ]

    assert len(negative) == 1
    assert negative[0]["label"] == []


def test_digit_words_slice_catches_mixed_numeral_and_word_forms() -> None:
    assert "digit_words" in classify_adversarial_slices("Số ĐT: 012 ba bốn")
    assert "digit_words" in classify_adversarial_slices("code one 2 three")


def test_digit_words_slice_still_catches_pure_word_runs() -> None:
    assert "digit_words" in classify_adversarial_slices("gọi ba bốn năm ngay")


def test_digit_words_slice_ignores_plain_numerals_in_prose() -> None:
    assert "digit_words" not in classify_adversarial_slices("I have 3 cats")
    assert "digit_words" not in classify_adversarial_slices("see page 2 of 3")
