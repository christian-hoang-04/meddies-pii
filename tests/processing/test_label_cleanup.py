from __future__ import annotations

from meddies_pii.processing.label_cleanup import (
    apply_quality_filters,
    clean_bracket_artifacts,
)


def test_clean_bracket_artifacts_removes_wrappers_drops_empty_and_preserves_input() -> None:
    source = {
        "human_name": ["[{}<Nguyen Van A>]{}", "[]", 7],
        "metadata": {"source": "fixture"},
    }

    cleaned = clean_bracket_artifacts(source)

    assert cleaned == {
        "human_name": ["Nguyen Van A", 7],
        "metadata": {"source": "fixture"},
    }
    assert source == {
        "human_name": ["[{}<Nguyen Van A>]{}", "[]", 7],
        "metadata": {"source": "fixture"},
    }
    assert cleaned is not source


def test_apply_quality_filters_rejects_non_pii_values_and_preserves_valid_values() -> None:
    source = {
        "id_number": [
            "https://patient.example/private",
            "www.example.org",
            "192.168.1.1",
            "00:11:22:33:44:55",
            "Mozilla/5.0",
            " digitally signed ",
            " 123 ",
            "Patient-1234",
            "MRN-2026-0001",
        ],
        "date": [
            " 12 ",
            "pending",
            "23:59 PM",
            "235959",
            "240000",
            "2026-07-29",
        ],
        "human_name": [
            " ",
            "attending physician",
            "Dr.",
            "A",
            "王",
            "Nguyen\nVan A",
            "{template}",
            "Nguyen Van A",
        ],
    }

    filtered = apply_quality_filters(source)

    assert filtered == {
        "id_number": ["MRN-2026-0001"],
        "date": ["240000", "2026-07-29"],
        "human_name": ["王", "Nguyen Van A"],
    }
    assert source["id_number"][-1] == "MRN-2026-0001"
    assert source["date"] == [
        " 12 ",
        "pending",
        "23:59 PM",
        "235959",
        "240000",
        "2026-07-29",
    ]


def test_apply_quality_filters_deduplicates_stably_and_preserves_non_list_values() -> None:
    source = {
        "id_number": "external identifier",
        "date": "external date",
        "human_name": "external name",
        "email_address": ["a@example.test", "a@example.test", 3, 3],
    }

    filtered = apply_quality_filters(source)

    assert filtered == {
        "id_number": "external identifier",
        "date": "external date",
        "human_name": "external name",
        "email_address": ["a@example.test", 3],
    }
    assert source["email_address"] == ["a@example.test", "a@example.test", 3, 3]


def test_apply_quality_filters_accepts_missing_supported_labels() -> None:
    assert apply_quality_filters({}) == {}
