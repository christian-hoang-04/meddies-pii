from __future__ import annotations

import pytest

from meddies_pii.spans import CharSpan, char_span_from_value, char_span_to_dict


def test_char_span_round_trips_through_persisted_json_value() -> None:
    span = CharSpan(start=4, end=8, text="Ngoc", label="human_name")

    persisted = char_span_to_dict(span)

    assert persisted == {
        "start": 4,
        "end": 8,
        "text": "Ngoc",
        "label": "human_name",
    }
    assert char_span_from_value(persisted) == span


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"start": "4", "end": 8, "text": "Ngoc", "label": "human_name"},
        {"start": 4, "end": True, "text": "Ngoc", "label": "human_name"},
        {"start": 4, "end": 8, "text": "Ngoc"},
    ],
)
def test_char_span_parser_fails_closed_for_malformed_values(value: object) -> None:
    assert char_span_from_value(value) is None
