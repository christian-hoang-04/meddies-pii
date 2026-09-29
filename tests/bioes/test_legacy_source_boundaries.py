from __future__ import annotations

from meddies_pii.training.bioes.data.legacy_to_pii_labels import (
    ConvertedLegacyRow,
    convert_legacy_row_to_pii_labels,
)


def _label_texts(row: ConvertedLegacyRow) -> list[str]:
    return [span.text for span in row.labels]


def test_migration_uses_source_uid_and_plain_text_fallbacks() -> None:
    from_text = convert_legacy_row_to_pii_labels({
        "source": "source-uid",
        "raw": "[Alice]<human_name>",
        "text": "[Alice]<human_name>",
    })
    from_tagged_raw = convert_legacy_row_to_pii_labels(
        {
            "raw": "[Bob]<human_name>",
            "text": "Bob",
        },
        default_uid="fallback-uid",
    )
    empty = convert_legacy_row_to_pii_labels({"raw": None, "text": None})

    assert from_text.uid == "source-uid"
    assert from_text.text == "Alice"
    assert _label_texts(from_text) == ["Alice"]
    assert from_tagged_raw.uid == "fallback-uid"
    assert from_tagged_raw.text == "Bob"
    assert _label_texts(from_tagged_raw) == ["Bob"]
    # reason: the converter was handed `{"raw": None, "text": None}`, so the claim is that it returns
    # reason: the empty STRING. The rule's `not empty.text` would also accept the `None` it was given.
    assert empty.text == ""  # ruff: ignore[compare-to-empty-string]
    assert empty.labels == ()


def test_migration_filters_malformed_original_spans_and_accepts_alias_fields() -> None:
    migrated = convert_legacy_row_to_pii_labels({
        "uid": "mixed-original-spans",
        "raw": "Alice",
        "spans": [
            None,
            {"label": 7, "text": "Alice"},
            {
                "category": "human_name",
                "value": "Alice",
                "start": 0,
                "end": 5,
            },
            {"label": "human_name", "text": "Missing"},
        ],
    })

    assert migrated.status == "kept"
    assert _label_texts(migrated) == ["Alice"]


def test_migration_unwraps_unlabeled_brackets_around_inline_tags() -> None:
    migrated = convert_legacy_row_to_pii_labels({
        "uid": "bracket-placeholders",
        "text": "[header] [Alice]<human_name> trailing [",
    })

    assert migrated.text == "[header] Alice trailing ["
    assert _label_texts(migrated) == ["Alice"]


def test_migration_ignores_unclosed_inline_label() -> None:
    migrated = convert_legacy_row_to_pii_labels({
        "uid": "unclosed-label",
        "raw": "Alice",
        "text": "[Alice]<human_name",
    })

    assert migrated.status == "kept"
    assert migrated.labels == ()


def test_migration_relocates_multiple_inline_spans_when_plain_text_differs() -> None:
    migrated = convert_legacy_row_to_pii_labels({
        "uid": "relocated",
        "raw": "Prefix Alice and Bob",
        "text": "[Alice]<human_name> and [Bob]<human_name>",
    })

    assert _label_texts(migrated) == ["Alice", "Bob"]
    assert [(span.start, span.end) for span in migrated.labels] == [(7, 12), (17, 20)]


def test_migration_discards_inline_span_absent_from_plain_text() -> None:
    migrated = convert_legacy_row_to_pii_labels({
        "uid": "missing-inline-value",
        "raw": "Bob",
        "text": "[Alice]<human_name>",
    })

    assert migrated.text == "Bob"
    assert migrated.labels == ()
