"""Inline ``[value]<label>`` parsing for the Meddies/meddies-pii configs.

The vietnamese-translated config rows carry no `language` field; the config name supplies it so they aren't bucketed as
"unknown" (which the gate resolver drops from the coverage grid and the language entropy).

"""

from __future__ import annotations

from meddies_pii.training.bioes.data.inline_tags import (
    convert_config_row,
    parse_inline_tagged,
)


def test_strips_tags_to_clean_text() -> None:
    text = "Facture: [FAC-88472]<id_number> du [15/11/2023]<date>."
    clean, _ = parse_inline_tagged(text)
    assert clean == "Facture: FAC-88472 du 15/11/2023."


def test_spans_carry_exact_offsets() -> None:
    text = "Dr. [Jean Dubois]<human_name> at [Hopital X]<company_name>."
    clean, spans = parse_inline_tagged(text)
    assert [(s["category"], s["text"]) for s in spans] == [
        ("human_name", "Jean Dubois"),
        ("company_name", "Hopital X"),
    ]
    for span in spans:
        assert clean[span["start"] : span["end"]] == span["text"]


def test_drops_non_pii_label_label() -> None:
    text = "Age [45]<age> and name [Lee]<human_name>."
    _, spans = parse_inline_tagged(text)
    assert [s["category"] for s in spans] == ["human_name"]


def test_no_tags_returns_clean_text_no_spans() -> None:
    clean, spans = parse_inline_tagged("Plain clinical note, no PII tags.")
    assert clean == "Plain clinical note, no PII tags."
    assert spans == []


def test_convert_config_row_carries_metadata() -> None:
    row = {
        "text": "Call [Lee]<human_name>.",
        "language": "French",
        "document_type": "MEDICAL_BILL",
        "text_format": "FHIR_JSON",
        "edge_cases": ["Spoken at/dot email.", "Truncated id."],
    }
    rec = convert_config_row(row, uid="cfg-1")
    assert rec is not None
    assert rec["text"] == "Call Lee."
    info = rec["info"]
    assert info["language"] == "French"
    assert info["text_format"] == "FHIR_JSON"
    assert info["edge_cases"] == ["Spoken at/dot email.", "Truncated id."]
    label = rec["label"]
    assert label[0]["category"] == "human_name"


def test_convert_config_row_none_when_no_spans() -> None:
    assert convert_config_row({"text": "No PII here."}, uid="cfg-2") is None


def test_convert_config_row_default_language_fallback() -> None:
    rec = convert_config_row(
        {"text": "Call [Lee]<human_name>."},
        uid="cfg-3",
        default_language="Vietnamese",
    )
    assert rec is not None
    info = rec["info"]
    assert info["language"] == "Vietnamese"


def test_convert_config_row_row_language_beats_default() -> None:
    rec = convert_config_row(
        {"text": "Call [Lee]<human_name>.", "language": "Tamil"},
        uid="cfg-4",
        default_language="Vietnamese",
    )
    assert rec is not None
    info = rec["info"]
    assert info["language"] == "Tamil"
