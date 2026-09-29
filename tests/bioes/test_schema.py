"""Tests for the BIOES schema (training/bioes/schema.py).

Behavior tests against the public interface of the schema module:
the 37-class BIOES vocabulary, the JSONL record contract for parsing
and emitting predictions/redactions, label validation, and the
placeholder formatter. No mocks — everything here is pure-function.

--- vocabulary shape ---------------------------------------------------------.

--- placeholder --------------------------------------------------------------.

--- validate_label -----------------------------------------------------------.

--- parse_labeled_record (spans form) ----------------------------------------.

--- parse_labeled_record (label form) ----------------------------------------.

--- prediction_record --------------------------------------------------------.

--- redaction_record ---------------------------------------------------------.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import pytest

from meddies_pii.annotations.span_records import (
    BACKGROUND_LABEL,
    BIOES_LABELS,
    ENTITY_LABELS,
    OUTPUT_SCHEMA_VERSION,
    SPAN_CLASS_NAMES,
    build_bioes_label_space,
    build_label_to_id,
    parse_labeled_record,
    placeholder,
    prediction_record,
    redaction_record,
    validate_label,
)
from meddies_pii.json_types import is_str_mapping
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Mapping


def _mapping_field(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return a record field the caller reads by key, asserting it really is a nested mapping.

    `prediction_record` and `redaction_record` return `OrderedDict[str, object]`, so every
    field reads as `object`. Narrowing here pins what these tests rely on; changing the
    builders' return type is a parked decision, because `OrderedDict` equality is
    order-sensitive and a plain dict's is not.

    Returns:
        The named field, narrowed to a mapping the caller can read by string key.

    """
    value = payload[key]
    assert is_str_mapping(value), f"{key} must be a mapping, got {type(value).__name__}"
    return value


def test_entity_labels_are_the_nine_meddies_labels() -> None:
    assert ENTITY_LABELS == (
        "address",
        "company_name",
        "date",
        "email_address",
        "human_name",
        "id_number",
        "phone_number",
        "private_url",
        "secret",
    )


def test_bioes_labels_have_37_classes() -> None:
    assert len(BIOES_LABELS) == 37


def test_bioes_labels_start_with_background_then_b_i_e_s_per_entity() -> None:
    """Per entity, the 4 boundary tags appear in B/I/E/S order."""
    assert BIOES_LABELS[0] == BACKGROUND_LABEL == "O"
    for index, label in enumerate(ENTITY_LABELS):
        offset = 1 + index * 4
        assert BIOES_LABELS[offset] == f"B-{label}"
        assert BIOES_LABELS[offset + 1] == f"I-{label}"
        assert BIOES_LABELS[offset + 2] == f"E-{label}"
        assert BIOES_LABELS[offset + 3] == f"S-{label}"


def test_span_class_names_is_background_plus_entities() -> None:
    assert (BACKGROUND_LABEL, *ENTITY_LABELS) == SPAN_CLASS_NAMES
    assert len(SPAN_CLASS_NAMES) == 10


def test_build_bioes_label_space_is_pure_function() -> None:
    custom = build_bioes_label_space(("foo", "bar"))
    assert custom == (
        "O",
        "B-foo",
        "I-foo",
        "E-foo",
        "S-foo",
        "B-bar",
        "I-bar",
        "E-bar",
        "S-bar",
    )
    assert len(custom) == 9


def test_build_label_to_id_round_trips() -> None:
    label_to_id = build_label_to_id(BIOES_LABELS)
    assert label_to_id["O"] == 0
    assert len(set(label_to_id.values())) == len(BIOES_LABELS)
    for label, idx in label_to_id.items():
        assert BIOES_LABELS[idx] == label


def test_placeholder_uppercases_label() -> None:
    assert placeholder("human_name") == "<HUMAN_NAME>"


def test_placeholder_normalizes_punctuation() -> None:
    assert placeholder("date-of-birth") == "<DATE_OF_BIRTH>"


def test_placeholder_handles_empty_label() -> None:
    assert placeholder("") == "<REDACTED>"
    assert placeholder("___") == "<REDACTED>"


def test_validate_label_accepts_each_meddies_label() -> None:
    for label in ENTITY_LABELS:
        validate_label(label, field="test")


def test_validate_label_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="meddies taxonomy"):
        validate_label("private_person", field="test")


def test_validate_label_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="meddies taxonomy"):
        validate_label("", field="test")


def test_parse_labeled_record_spans_form_typed() -> None:
    record = {
        "text": "Patient John Doe visited.",
        "spans": {"human_name: John Doe": [[8, 16]]},
        "info": {"id": "doc-1"},
    }
    example_id, text, spans = parse_labeled_record(record)
    assert example_id == "doc-1"
    assert text == "Patient John Doe visited."
    assert len(spans) == 1
    assert spans[0].start == 8
    assert spans[0].end == 16
    assert spans[0].label == "human_name"
    assert spans[0].text == "John Doe"


def test_parse_labeled_record_uses_default_id_when_info_missing() -> None:
    record = {"text": "abc", "spans": {}}
    example_id, _, _ = parse_labeled_record(record, default_id="fallback")
    assert example_id == "fallback"


def test_parse_labeled_record_typed_rejects_unknown_label() -> None:
    record = {
        "text": "Patient.",
        "spans": {"private_person: Patient": [[0, 7]]},
    }
    with pytest.raises(ValueError, match="meddies taxonomy"):
        parse_labeled_record(record, eval_mode="typed")


def test_parse_labeled_record_untyped_accepts_unknown_label() -> None:
    record = {
        "text": "Patient.",
        "spans": {"private_person: Patient": [[0, 7]]},
    }
    _, _, spans = parse_labeled_record(record, eval_mode="untyped")
    assert spans[0].label == "private_person"


def test_parse_labeled_record_label_form() -> None:
    record = {
        "text": "Call 555-0100.",
        "label": [{"category": "phone_number", "start": 5, "end": 13}],
    }
    _, _, spans = parse_labeled_record(record)
    assert spans[0].label == "phone_number"
    assert spans[0].text == "555-0100"


def test_parse_labeled_record_rejects_non_string_text() -> None:
    with pytest.raises(ValueError, match="text must be a string"):
        parse_labeled_record({"text": 42})


def test_parse_labeled_record_rejects_unsupported_eval_mode() -> None:
    invalid_mode = cast("Literal['typed', 'untyped']", "bogus")
    with pytest.raises(ValueError, match="unsupported eval_mode"):
        parse_labeled_record({"text": "abc"}, eval_mode=invalid_mode)


def test_prediction_record_groups_spans_by_label_text_key() -> None:
    spans = [
        CharSpan(start=0, end=4, text="John", label="human_name"),
        CharSpan(start=10, end=14, text="John", label="human_name"),
        CharSpan(start=20, end=28, text="555-0100", label="phone_number"),
    ]
    record = prediction_record(example_id="doc-1", text="...", spans=spans)
    assert record["example_id"] == "doc-1"
    predicted = _mapping_field(record, "predicted_spans")
    assert predicted["human_name: John"] == [[0, 4], [10, 14]]
    assert predicted["phone_number: 555-0100"] == [[20, 28]]


def test_redaction_record_typed_uses_per_label_placeholders() -> None:
    text = "Call John Doe at 555-0100."
    spans = [
        CharSpan(start=5, end=13, text="John Doe", label="human_name"),
        CharSpan(start=17, end=25, text="555-0100", label="phone_number"),
    ]
    record = redaction_record(text=text, spans=spans, output_mode="typed")
    assert record["redacted_text"] == "Call <HUMAN_NAME> at <PHONE_NUMBER>."
    summary = _mapping_field(record, "summary")
    assert summary["span_count"] == 2
    assert summary["by_label"] == {"human_name": 1, "phone_number": 1}


def test_redaction_record_redacted_mode_collapses_to_single_placeholder() -> None:
    text = "Call John Doe at 555-0100."
    spans = [
        CharSpan(start=5, end=13, text="John Doe", label="human_name"),
        CharSpan(start=17, end=25, text="555-0100", label="phone_number"),
    ]
    record = redaction_record(text=text, spans=spans, output_mode="redacted")
    assert record["redacted_text"] == "Call <REDACTED> at <REDACTED>."
    assert _mapping_field(record, "summary")["by_label"] == {"redacted": 2}


def test_redaction_record_skips_overlapping_spans() -> None:
    """First span covers full name; second is contained within.

    Only the longer span lands; the inner one is skipped (its start < cursor).

    """
    text = "John Doe."
    spans = [
        CharSpan(start=0, end=8, text="John Doe", label="human_name"),
        CharSpan(start=0, end=4, text="John", label="human_name"),
    ]
    record = redaction_record(text=text, spans=spans)
    assert _mapping_field(record, "summary")["span_count"] == 1


def test_redaction_record_rejects_unsupported_output_mode() -> None:
    invalid_mode = cast("Literal['typed', 'redacted']", "invalid")
    with pytest.raises(ValueError, match="unsupported output_mode"):
        redaction_record(text="abc", spans=[], output_mode=invalid_mode)


def test_redaction_record_carries_schema_version() -> None:
    record = redaction_record(text="hi", spans=[])
    assert record["schema_version"] == OUTPUT_SCHEMA_VERSION
