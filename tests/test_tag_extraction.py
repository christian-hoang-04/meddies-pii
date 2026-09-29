from __future__ import annotations

import json

from meddies_pii.tags import extract_entities


def test_extract_entities_basic() -> None:
    text = json.dumps({"human_name": ["John Doe"], "address": ["123 Main St"]})

    assert extract_entities(text) == {
        ("John Doe", "human_name"),
        ("123 Main St", "address"),
    }


def test_extract_entities_empty_or_malformed_json() -> None:
    assert extract_entities("{}") == set()
    assert extract_entities("") == set()
    assert extract_entities("not valid json") == set()
    assert extract_entities("{broken") == set()


def test_extract_entities_collapses_duplicates_and_keeps_labels() -> None:
    text = json.dumps({"human_name": ["John", "Jane", "John"]})

    assert extract_entities(text) == {("John", "human_name"), ("Jane", "human_name")}
