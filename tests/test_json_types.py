from __future__ import annotations

from meddies_pii.json_types import as_json_object, is_json_value


def test_accepts_nested_json_value() -> None:
    value = {
        "record": [
            {"text": "Vietnamese clinical note", "score": 0.2},
            {"labels": ["human_name", None]},
        ],
    }

    assert is_json_value(value)
    assert as_json_object(value) == value


def test_rejects_non_string_object_keys() -> None:
    assert not is_json_value({1: "invalid JSON object key"})
    assert as_json_object({1: "invalid JSON object key"}) is None


def test_rejects_non_json_nested_object() -> None:
    assert not is_json_value({"nested": object()})
    assert as_json_object({"nested": object()}) is None
