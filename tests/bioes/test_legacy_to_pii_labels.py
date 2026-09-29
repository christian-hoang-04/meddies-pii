from __future__ import annotations

import json
from typing import TYPE_CHECKING

from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.historical_artifacts import legacy_jsonl_locator
from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.data.legacy_to_pii_labels import (
    convert_legacy_row_to_pii_labels,
    write_legacy_conversion_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


def _labels_by_category(row: dict[str, object]) -> dict[str, list[str]]:
    labels = row["label"]
    assert isinstance(labels, list)
    by_category: dict[str, list[str]] = {}
    for span in labels:
        assert is_str_mapping(span)
        by_category.setdefault(str(span["category"]), []).append(str(span["text"]))
    return by_category


def test_migration_remaps_preserved_url_and_secret_fine_labels() -> None:
    url = "https://portal.hospital.vn/patients/123/results?token=abc"
    # reason: this synthetic credential is the PII the labeller under test has to find, so it is the
    # reason: fixture's payload rather than a secret in use. Renaming it would stop describing the row.
    secret = "sk-live-123"  # ruff: ignore[hardcoded-password-string]
    mrn = "MRN-42"
    raw = f"Use portal {url} and API key {secret}. MRN {mrn}."
    row = {
        "uid": "remap-row",
        "raw": raw,
        "text": (f"Use portal [{url}]<id_number> and API key [{secret}]<id_number>. MRN [{mrn}]<id_number>."),
        "spans": repr([
            {"label": "url", "text": url},
            {"label": "api_key", "text": secret},
            {"label": "medical_record_number", "text": mrn},
        ]),
        "label": json.dumps({"id_number": [url, secret, mrn]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert _labels_by_category(migrated.as_record()) == {
        "private_url": [url],
        "secret": [secret],
        "id_number": [mrn],
    }
    assert {decision.rule for decision in migrated.decisions} >= {
        "private_url_context",
        "secret_fine_label",
        "id_fine_label",
    }


def test_migration_drops_public_url_from_training_labels() -> None:
    url = "https://moh.gov.vn/guidelines/diabetes"
    raw = f"Public diabetes guideline is published at {url}."
    row = {
        "uid": "public-url-row",
        "raw": raw,
        "text": f"Public diabetes guideline is published at [{url}]<id_number>.",
        "spans": repr([{"label": "url", "text": url}]),
        "label": json.dumps({"id_number": [url]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert migrated.as_record()["label"] == []
    assert migrated.decisions[0].decision == "drop"
    assert migrated.decisions[0].rule == "public_url_context"


def test_migration_drops_public_url_even_when_source_label_was_address() -> None:
    url = "ftp://files.pharmahealth.com/policies/data_integrity_policy.pdf"
    raw = f"Audit policy reports are stored at {url}."
    row = {
        "uid": "public-url-address-row",
        "raw": raw,
        "text": f"Audit policy reports are stored at [{url}]<address>.",
        "spans": repr([{"label": "address", "text": url}]),
        "label": json.dumps({"address": [url]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert migrated.as_record()["label"] == []
    assert migrated.decisions[0].decision == "drop"
    assert migrated.decisions[0].rule == "public_url_context"


def test_migration_does_not_treat_training_sessions_as_private_url_context() -> None:
    url = "https://regulatorystrategy.com"
    raw = f"Public regulatory strategy training sessions are posted at {url}."
    row = {
        "uid": "training-session-url-row",
        "raw": raw,
        "text": f"Public regulatory strategy training sessions are posted at [{url}]<url>.",
        "spans": repr([{"label": "url", "text": url}]),
        "label": json.dumps({"url": [url]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert migrated.as_record()["label"] == []
    assert migrated.decisions[0].rule == "public_url_context"


def test_migration_does_not_treat_later_secure_verb_as_private_url_context() -> None:
    url = "https://darkreading.com/search?query=email&type=article"
    raw = f"For further reading, visit {url}. Please secure affected systems."
    row = {
        "uid": "further-reading-url-row",
        "raw": raw,
        "text": f"For further reading, visit [{url}]<url>. Please secure affected systems.",
        "spans": repr([{"label": "url", "text": url}]),
        "label": json.dumps({"url": [url]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert migrated.as_record()["label"] == []
    assert migrated.decisions[0].rule == "public_url_context"


def test_migration_drops_public_website_even_when_near_private_context_words() -> None:
    url = "https://cdc.gov"
    raw = f"Medical record number MRN-42 was reviewed using public guidelines provided at {url}."
    row = {
        "uid": "public-cdc-row",
        "raw": raw,
        "text": (
            f"Medical record number [MRN-42]<id_number> was reviewed using public guidelines provided at [{url}]<url>."
        ),
        "spans": repr([
            {"label": "medical_record_number", "text": "MRN-42"},
            {"label": "url", "text": url},
        ]),
        "label": json.dumps({"id_number": ["MRN-42", url]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert _labels_by_category(migrated.as_record()) == {"id_number": ["MRN-42"]}
    assert migrated.decisions[1].rule == "public_url_context"


def test_migration_does_not_match_signed_inside_designed_for_public_website() -> None:
    url = "https://biotechinnovations.com"
    row = {
        "uid": "designed-public-website-row",
        "raw": f"Products are designed for lab workflows. Visit our website at {url}.",
        "text": f"Products are designed for lab workflows. Visit our website at [{url}]<url>.",
        "spans": repr([{"label": "url", "text": url}]),
        "label": json.dumps({"url": [url]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert migrated.as_record()["label"] == []
    assert migrated.decisions[0].rule == "public_url_context"


def test_migration_drops_malformed_structured_spans_before_training() -> None:
    bad_address = (
        '\n  {"url": "http://hl7.org/fhir/StructureDefinition/patient-birthPlace", "valueAddress": {"text": "[Hải Phòng'
    )
    row = {
        "uid": "malformed-fhir-span-row",
        "raw": f"FHIR extension captured badly: {bad_address}",
        "text": f"FHIR extension captured badly: [{bad_address}]<address>",
        "spans": repr([{"label": "address", "text": bad_address}]),
        "label": json.dumps({"address": [bad_address]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert migrated.as_record()["label"] == []
    assert migrated.decisions[0].rule == "malformed_structured_span"


def test_migration_drops_public_investor_or_report_url_even_with_results_context() -> None:
    url = "ftp://documents.pharmaexperts.com/clinical_trial_data/2023_report.pdf"
    raw = f"Detailed data analyses and results can be accessed via the public report URL {url}."
    row = {
        "uid": "public-report-url-row",
        "raw": raw,
        "text": f"Detailed data analyses and results can be accessed via the public report URL [{url}]<id_number>.",
        "spans": repr([{"label": "url", "text": url}]),
        "label": json.dumps({"id_number": [url]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert migrated.as_record()["label"] == []
    assert migrated.decisions[0].rule == "public_url_context"


def test_migration_quarantines_ambiguous_url_rows() -> None:
    url = "https://example.com/opaque"
    row = {
        "uid": "ambiguous-url-row",
        "raw": f"Reference: {url}",
        "text": f"Reference: [{url}]<id_number>",
        "spans": repr([{"label": "url", "text": url}]),
        "label": json.dumps({"id_number": [url]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "quarantined"
    assert migrated.quarantine_reason == "ambiguous_url"


def test_migration_keeps_supported_fine_label_aliases_and_ignores_non_pii_labels() -> None:
    row = {
        "uid": "fine-label-row",
        "raw": "Patient Ada lives in Rome and works as an analyst.",
        "text": "Patient [Ada]<human_name> lives in [Rome]<address> and works as an analyst.",
        "spans": repr([
            {"label": "first_name", "text": "Ada"},
            {"label": "city", "text": "Rome"},
            {"label": "occupation", "text": "analyst"},
        ]),
        "label": json.dumps({"human_name": ["Ada"], "address": ["Rome"]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert _labels_by_category(migrated.as_record()) == {
        "human_name": ["Ada"],
        "address": ["Rome"],
    }
    assert migrated.decisions[-1].decision == "drop"
    assert migrated.decisions[-1].rule == "ignored_unsupported_source_label"


def test_migration_recovers_collapsed_secret_without_original_spans() -> None:
    # reason: this synthetic credential is the PII the labeller under test has to find, so it is the
    # reason: fixture's payload rather than a secret in use. Renaming it would stop describing the row.
    secret = "River2025!"  # ruff: ignore[hardcoded-password-string]
    row = {
        "uid": "collapsed-secret-row",
        "text": f"The portal password is [{secret}]<id_number>.",
        "raw": f"The portal password is {secret}.",
        "label": json.dumps({"id_number": [secret]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert _labels_by_category(migrated.as_record()) == {"secret": [secret]}
    assert migrated.decisions[0].rule == "secret_context"


def test_migration_fallback_offsets_survive_unlabeled_bracketed_placeholders() -> None:
    row = {
        "uid": "unlabeled-bracket-shift-row",
        "text": (
            "<header>[Hospital Santa Maria]</header>"
            "<region>[Vila Velha, Espírito Santo]<address></region>"
            "<patient>[Maria Alves da Costa]<human_name></patient>"
        ),
        "raw": (
            "<header>Hospital Santa Maria</header>"
            "<region>Vila Velha, Espírito Santo</region>"
            "<patient>Maria Alves da Costa</patient>"
        ),
        "label": json.dumps({
            "address": ["Vila Velha, Espírito Santo"],
            "human_name": ["Maria Alves da Costa"],
        }),
    }

    migrated = convert_legacy_row_to_pii_labels(row)
    record = migrated.as_record()
    _, text, spans = parse_labeled_record(record)

    assert migrated.status == "kept"
    assert [(span.text, span.label, text[span.start : span.end]) for span in spans] == [
        ("Vila Velha, Espírito Santo", "address", "Vila Velha, Espírito Santo"),
        ("Maria Alves da Costa", "human_name", "Maria Alves da Costa"),
    ]


def test_migration_does_not_promote_neighboring_admin_id_to_secret() -> None:
    mrn = "MRN-42"
    # reason: this synthetic credential is the PII the labeller under test has to find, so it is the
    # reason: fixture's payload rather than a secret in use. Renaming it would stop describing the row.
    secret = "River2025!"  # ruff: ignore[hardcoded-password-string]
    row = {
        "uid": "neighbor-secret-row",
        "text": f"MRN [{mrn}]<id_number>. The portal password is [{secret}]<id_number>.",
        "raw": f"MRN {mrn}. The portal password is {secret}.",
        "label": json.dumps({"id_number": [mrn, secret]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert _labels_by_category(migrated.as_record()) == {
        "id_number": [mrn],
        "secret": [secret],
    }


def test_migration_drops_public_date_but_keeps_encounter_date() -> None:
    public = "2024-01-15"
    private = "2024-02-20"
    row = {
        "uid": "date-row",
        "text": (f"Guideline published on [{public}]<date>. Patient admitted on [{private}]<date>."),
        "raw": f"Guideline published on {public}. Patient admitted on {private}.",
        "label": json.dumps({"date": [public, private]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert _labels_by_category(migrated.as_record()) == {"date": [private]}
    assert [(d.text, d.decision, d.rule) for d in migrated.decisions] == [
        (public, "drop", "public_date_context"),
        (private, "keep", "same_label"),
    ]


def test_migration_drops_schedule_time_ranges_and_durations_as_dates() -> None:
    schedule = "6:00 AM - 11:00 AM"
    duration = "5-7 business days"
    raw = f"Public room service hours are {schedule}. Refunds are processed within {duration}."
    row = {
        "uid": "public-schedule-duration-row",
        "raw": raw,
        "text": (f"Public room service hours are [{schedule}]<date>. Refunds are processed within [{duration}]<date>."),
        "label": json.dumps({"date": [schedule, duration]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert migrated.as_record()["label"] == []
    assert [(d.text, d.decision, d.rule) for d in migrated.decisions] == [
        (schedule, "drop", "non_private_date_value"),
        (duration, "drop", "non_private_date_value"),
    ]


def test_migration_does_not_promote_medication_context_or_price_to_secret() -> None:
    rx_code = "RX-LA-884572"
    price = "250,000"
    row = {
        "uid": "medication-context-row",
        "text": (f"Amlodipine 5mg dispensing code [{rx_code}]<id_number>. Medication price [{price}]<id_number> VND."),
        "raw": (f"Amlodipine 5mg dispensing code {rx_code}. Medication price {price} VND."),
        "label": json.dumps({"id_number": [rx_code, price]}),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "kept"
    assert _labels_by_category(migrated.as_record()) == {"id_number": [rx_code, price]}


def test_migration_quarantines_rows_with_overlapping_training_spans() -> None:
    raw = "Vendor Indonesia Entertainment Ventures submitted the report."
    row = {
        "uid": "overlap-row",
        "raw": raw,
        "text": raw,
        "spans": repr([
            {
                "label": "company_name",
                "text": "Indonesia Entertainment Ventures",
            },
            {"label": "country", "text": "Indonesia"},
        ]),
        "label": json.dumps({
            "company_name": ["Indonesia Entertainment Ventures"],
            "address": ["Indonesia"],
        }),
    }

    migrated = convert_legacy_row_to_pii_labels(row)

    assert migrated.status == "quarantined"
    assert migrated.quarantine_reason == "overlapping_spans"
    assert migrated.as_record()["label"] == []


def test_write_legacy_conversion_artifacts_emits_jsonl_and_summary(tmp_path: Path) -> None:
    # reason: this synthetic credential is the PII the labeller under test has to find, so it is the
    # reason: fixture's payload rather than a secret in use. Renaming it would stop describing the row.
    secret = "sk-live-123"  # ruff: ignore[hardcoded-password-string]
    rows = [
        {
            "uid": "kept",
            "text": f"API key [{secret}]<id_number>.",
            "raw": f"API key {secret}.",
            "label": json.dumps({"id_number": [secret]}),
        },
        {
            "uid": "quarantined",
            "text": "[https://example.com/opaque]<id_number>",
            "raw": "https://example.com/opaque",
            "spans": repr([{"label": "url", "text": "https://example.com/opaque"}]),
            "label": json.dumps({"id_number": ["https://example.com/opaque"]}),
        },
    ]
    output_path = tmp_path / legacy_jsonl_locator("train")
    audit_path = tmp_path / "audit.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = write_legacy_conversion_artifacts(
        rows,
        output_path=output_path,
        audit_path=audit_path,
        summary_path=summary_path,
    )

    assert summary["rows"] == 2
    assert summary["kept_rows"] == 1
    assert summary["quarantined_rows"] == 1
    assert output_path.read_text().count("\n") == 1
    assert audit_path.read_text().count("\n") == 2
    assert json.loads(summary_path.read_text()) == summary
