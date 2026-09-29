"""Nemotron org-placeholder → company_name converter-layer pre-map.

ADR 0008 §4: Nemotron is the ONLY external company_name source. Its 500k-schema
rows tag organisations with `ORGANISATIONPLACEHOLDER`, a token absent from the
shared LABEL_MAP — so without a Nemotron-specific pre-map, external
company_name supply is ~0. These tests pin the pre-map at the converter layer.
"""

from __future__ import annotations

from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.training.bioes.data.mixed import convert_nemotron_row


def test_nemotron_organisation_placeholder_maps_to_company_name() -> None:
    text = "Acme Clinic treated the patient on 2026-06-11."
    row = {
        "uid": "org-1",
        "locale": "us",
        "domain": "Healthcare",
        "text": text,
        "spans": repr([
            {
                "start": 0,
                "end": 11,
                "text": "Acme Clinic",
                "label": "ORGANISATIONPLACEHOLDER",
            },
        ]),
    }

    record, dropped = convert_nemotron_row(row, default_uid="row-0")

    assert record is not None
    assert dropped.get("organisationplaceholder", 0) == 0
    _example_id, _text, spans = parse_labeled_record(record)
    assert [(span.text, span.label) for span in spans] == [
        ("Acme Clinic", "company_name"),
    ]


def test_nemotron_organization_us_spelling_maps_to_company_name() -> None:
    text = "St. Mary Hospital admitted three patients today."
    row = {
        "uid": "org-2",
        "locale": "us",
        "text": text,
        "spans": repr([
            {
                "start": 0,
                "end": 17,
                "text": "St. Mary Hospital",
                "label": "ORGANIZATIONPLACEHOLDER",
            },
        ]),
    }

    record, dropped = convert_nemotron_row(row, default_uid="row-0")

    assert record is not None
    assert dropped.get("organizationplaceholder", 0) == 0
    _example_id, _text, spans = parse_labeled_record(record)
    assert [(span.label) for span in spans] == ["company_name"]
