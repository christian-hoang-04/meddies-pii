from __future__ import annotations

import json

from meddies_pii.annotations.span_records import BIOES_LABELS, ENTITY_LABELS
from meddies_pii.annotations.tagged_text import parse_tagged_text
from meddies_pii.training.bioes.data.preparation import _select_source_rows
from meddies_pii.training.bioes.eval.harness import classify_adversarial_slices


def test_bioes_schema_exposes_pii_label_as_37_token_classes() -> None:
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
    assert len(BIOES_LABELS) == 37


def test_parse_tagged_text_accepts_private_url_and_secret() -> None:
    parsed = parse_tagged_text("Portal [https://portal.example/p/123?token=abc]<private_url> uses [sk-live-123]<secret>.")

    assert [(span.text, span.label) for span in parsed.spans] == [
        ("https://portal.example/p/123?token=abc", "private_url"),
        ("sk-live-123", "secret"),
    ]


def test_source_selection_prefers_hf_tagged_text_over_plain_raw() -> None:
    rows: list[dict[str, object]] = [
        {
            "uid": "hf-row-shape",
            "text": "Patient [John Doe]<human_name> visited.",
            "raw": "Patient John Doe visited.",
            "label": json.dumps({"human_name": ["John Doe"]}),
        },
    ]

    selected, stats = _select_source_rows(
        rows,
        limit=1,
        allow_label_repairs=False,
        require_label_json=True,
    )

    assert stats.accepted == 1
    assert selected[0].parsed.raw == "Patient John Doe visited."
    assert selected[0].parsed.spans[0].text == "John Doe"


def test_source_selection_accepts_pii_label_label_list_records() -> None:
    text = "Portal https://portal.example/patients/42?token=abc uses sk-live-123."
    url_start = text.index("https://")
    secret_start = text.index("sk-live")
    rows: list[dict[str, object]] = [
        {
            "uid": "pii_label-jsonl-row",
            "text": text,
            "label": [
                {
                    "category": "private_url",
                    "start": url_start,
                    "end": url_start + len("https://portal.example/patients/42?token=abc"),
                },
                {
                    "category": "secret",
                    "start": secret_start,
                    "end": secret_start + len("sk-live-123"),
                },
            ],
        },
    ]

    selected, stats = _select_source_rows(
        rows,
        limit=1,
        allow_label_repairs=False,
        require_label_json=True,
    )

    assert stats.accepted == 1
    assert selected[0].parsed.raw == text
    assert [(span.text, span.label) for span in selected[0].parsed.spans] == [
        ("https://portal.example/patients/42?token=abc", "private_url"),
        ("sk-live-123", "secret"),
    ]


def test_adversarial_slices_cover_opf_formatting_axes() -> None:
    text = (
        "Email: charlie oscar lima echo at golf mike alpha india lima dot com\n"
        "Secret: tok_ 7Qz .P9x 2Kj\n"
        "Site: www[dot]patient-portal[dot]com\n"
        "Phone: two six eight - seven two two - one zero four nine\n"
        "Address: 482 Maple 🛣️, Apt 3B"
    )

    slices = set(classify_adversarial_slices(text))

    assert {
        "line_breaks",
        "spacing",
        "at_dot_obfuscation",
        "phonetic_alphabet",
        "digit_words",
        "emoji_word_replacement",
    }.issubset(slices)
