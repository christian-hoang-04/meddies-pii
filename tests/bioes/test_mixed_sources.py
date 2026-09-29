"""The 8 ai4privacy alias-gap labels now route via the shared LABEL_MAP.

ORGANISATION/BANKNAME/HOSPITALNAME -> company_name, ACCOUNTNUM/POLICYNUM/ MACADDRESS -> id_number, USERNAME/DOCTORNAME ->
human_name.

Adjacent TITLE+GIVENNAME+SURNAME merge into ONE human_name span (matching meddies-pii-v2's "Dr. Jane Doe = one span"); TIME
still -> date.

A TITLE with no adjacent given/surname is NOT a name → dropped, not tagged.

title->human_name (ai4privacy-only) and time->date (gretel/ai4privacy-only) are source-scoped premaps; nemotron still drops
bare title AND bare time via the shared drop-sets, so neither premap may leak across sources.

Adjacent first_name+last_name → one human_name span; user_name stays its own human_name span (not a name component to
merge).

first_name and last_name in separate structured fields (not adjacent) stay as two human_name spans — the merge must not
join across a field gap.

Two gretel spellings the shared LABEL_MAP lacked (alias gaps) now route to id_number: credit_card_number (vs
credit_debit_card) and unique_identifier (vs unique_id). A card number identifies; cvv/pin are the secret part.

Gretel-scoped premap: the shared drop-set drops bare `time`, but gretel rows retain it as `date` (project decision: only
private_url is removed).

Nemotron keeps bare `time` dropped (in the shared NEMOTRON_DROP_LABELS): we prioritize dates; a datetime still maps via
date_time->date, but a bare clock time is not a date and is left untagged.

A near-duplicate differing only by internal whitespace / case must dedupe to one row — otherwise a normalized eval row
leaks into training (the raw-sha1 gap). normalize_text strips + lowercases + collapses whitespace.

English-only corpus: every gretel row is 'en', which passes the gate; a non-17-language row would be dropped by
is_supported_meddies_language.

A gretel-shaped row routed through the build dispatch must produce PII-label spans — proving convert_gretel_row is actually
wired into convert_external (ADR 0008 §1: gretel source dispatch).

Local-first read (ADR 0008 §1): JSONL files under external/ route to the right converter by filename stem — nvidia-health →
nemotron, gretel* → gretel, ai4privacy* → ai4privacy. No HuggingFace call.

"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import json
from collections import Counter
from typing import TYPE_CHECKING

from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.data.mixed import (
    convert_ai4privacy_row,
    convert_gretel_row,
    convert_nemotron_row,
    is_supported_meddies_language,
    language_bucket,
    map_external_label_to_pii_label,
    summarize_records,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _mapping_field(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return a summary field the caller reads by key, asserting it really is a nested mapping.

    `RecordSummary.asdict` returns `dict[str, object]`, so each per-label count block reads
    as `object` and cannot be subscripted until the shape is stated.

    Returns:
        The named field, narrowed to a mapping the caller can read by string key.

    """
    value = payload[key]
    assert is_str_mapping(value), f"{key} must be a mapping, got {type(value).__name__}"
    return value


def test_map_external_label_to_pii_label_returns_canonical_labels() -> None:
    assert map_external_label_to_pii_label("EMAIL") == "email_address"
    assert map_external_label_to_pii_label("not-a-pii-label") is None


def test_convert_ai4privacy_row_maps_openpii_labels_to_pii_label() -> None:
    """GIVENNAME + SURNAME are adjacent → merged into one human_name span."""
    text = "Alice Nguyen lives at 42 Main St and uses alice@example.com."
    row = {
        "source_text": text,
        "uid": 123,
        "language": "en",
        "split": "train",
        "privacy_mask": [
            {"label": "GIVENNAME", "start": 0, "end": 5, "value": "Alice"},
            {"label": "SURNAME", "start": 6, "end": 12, "value": "Nguyen"},
            {"label": "STREET", "start": 22, "end": 32, "value": "42 Main St"},
            {"label": "EMAIL", "start": 42, "end": 59, "value": "alice@example.com"},
            {"label": "AGE", "start": 0, "end": 0, "value": ""},
        ],
    }

    record, dropped = convert_ai4privacy_row(
        row,
        dataset_id="ai4privacy/pii-masking-openpii-1m",
        default_uid="row-0",
    )

    assert record is not None
    assert dropped == Counter({"age": 1})
    _example_id, parsed_text, spans = parse_labeled_record(record)
    assert parsed_text == text
    assert [(span.text, span.label) for span in spans] == [
        ("Alice Nguyen", "human_name"),
        ("42 Main St", "address"),
        ("alice@example.com", "email_address"),
    ]
    assert record["info"]["domain_bucket"] == "general"
    assert record["info"]["language_bucket"] == "en"


def test_convert_ai4privacy_row_maps_new_aliases_to_pii_label() -> None:
    text = (
        "Dr Smith at Mercy Hospital and First Bank; policy POL-7, "
        "acct ACC-12345, user jdoe99, org Acme Ltd, mac 00:1A:2B:3C:4D:5E."
    )
    masks = [
        {"label": "DOCTORNAME", "value": "Dr Smith"},
        {"label": "HOSPITALNAME", "value": "Mercy Hospital"},
        {"label": "BANKNAME", "value": "First Bank"},
        {"label": "POLICYNUM", "value": "POL-7"},
        {"label": "ACCOUNTNUM", "value": "ACC-12345"},
        {"label": "USERNAME", "value": "jdoe99"},
        {"label": "ORGANISATION", "value": "Acme Ltd"},
        {"label": "MACADDRESS", "value": "00:1A:2B:3C:4D:5E"},
    ]
    row = {
        "source_text": text,
        "uid": "a1",
        "language": "en",
        "split": "train",
        "privacy_mask": masks,
    }

    record, _dropped = convert_ai4privacy_row(row, dataset_id="ai4privacy/pii-masking-openpii-1.5m", default_uid="row-0")

    assert record is not None
    _id, _text, spans = parse_labeled_record(record)
    by_text = {s.text: s.label for s in spans}
    assert by_text == {
        "Dr Smith": "human_name",
        "Mercy Hospital": "company_name",
        "First Bank": "company_name",
        "POL-7": "id_number",
        "ACC-12345": "id_number",
        "jdoe99": "human_name",
        "Acme Ltd": "company_name",
        "00:1A:2B:3C:4D:5E": "id_number",
    }


def test_convert_ai4privacy_row_merges_title_given_surname_and_maps_time() -> None:
    text = "Patient Dr. Jane Doe arrived at 3:00 PM."
    row = {
        "source_text": text,
        "uid": "a2",
        "language": "en",
        "split": "train",
        "privacy_mask": [
            {"label": "TITLE", "start": 8, "end": 11, "value": "Dr."},
            {"label": "GIVENNAME", "start": 12, "end": 16, "value": "Jane"},
            {"label": "SURNAME", "start": 17, "end": 20, "value": "Doe"},
            {"label": "TIME", "start": 32, "end": 39, "value": "3:00 PM"},
        ],
    }

    record, dropped = convert_ai4privacy_row(row, dataset_id="ai4privacy/pii-masking-openpii-1.5m", default_uid="row-0")

    assert record is not None
    assert "title" not in dropped
    assert "time" not in dropped
    _id, _text, spans = parse_labeled_record(record)
    assert {s.text: s.label for s in spans} == {
        "Dr. Jane Doe": "human_name",
        "3:00 PM": "date",
    }


def test_convert_ai4privacy_row_drops_standalone_title() -> None:
    text = "Title field: Mr."
    row = {
        "source_text": text,
        "uid": "a3",
        "language": "en",
        "split": "train",
        "privacy_mask": [{"label": "TITLE", "start": 13, "end": 16, "value": "Mr."}],
    }

    record, dropped = convert_ai4privacy_row(row, dataset_id="ai4privacy/pii-masking-openpii-1.5m", default_uid="row-0")

    assert record is None
    assert dropped["title"] == 1


def test_nemotron_still_drops_title_and_time_proving_scope() -> None:
    text = "Mr Jones met at 3:00 PM."
    row = {
        "uid": "n3",
        "locale": "us",
        "domain": "x",
        "text": text,
        "spans": [
            {"start": 0, "end": 2, "text": "Mr", "label": "title"},
            {"start": 16, "end": 23, "text": "3:00 PM", "label": "time"},
        ],
    }

    record, dropped = convert_nemotron_row(row, default_uid="row-0")

    assert record is None
    assert dropped["title"] == 1
    assert dropped["time"] == 1


def test_convert_nemotron_row_merges_adjacent_first_last_keeps_username() -> None:
    text = "Patient Jason Smith logged in as jsmith92."
    row = {
        "uid": "n4",
        "locale": "us",
        "domain": "x",
        "text": text,
        "spans": [
            {"start": 8, "end": 13, "text": "Jason", "label": "first_name"},
            {"start": 14, "end": 19, "text": "Smith", "label": "last_name"},
            {"start": 33, "end": 41, "text": "jsmith92", "label": "user_name"},
        ],
    }

    record, _dropped = convert_nemotron_row(row, default_uid="row-0")

    assert record is not None
    _id, _text, spans = parse_labeled_record(record)
    assert {s.text: s.label for s in spans} == {
        "Jason Smith": "human_name",
        "jsmith92": "human_name",
    }


def test_convert_nemotron_row_derives_healthcare_vs_general_domain_bucket() -> None:
    base = {
        "uid": "n-dom",
        "locale": "us",
        "text": "Patient Jason Smith seen.",
        "spans": [
            {"start": 8, "end": 13, "text": "Jason", "label": "first_name"},
            {"start": 14, "end": 19, "text": "Smith", "label": "last_name"},
        ],
    }
    for domain in ("Healthcare", "Health", "Healthcare Providers", "Pharmaceuticals"):
        record, _ = convert_nemotron_row({**base, "domain": domain}, default_uid="r")
        assert record is not None
        assert record["info"]["domain_bucket"] == "healthcare", domain
    for domain in ("Biotechnology", "Disability", "Life", "Fitness", "Finance"):
        record, _ = convert_nemotron_row({**base, "domain": domain}, default_uid="r")
        assert record is not None
        assert record["info"]["domain_bucket"] == "general", domain


def test_convert_nemotron_row_keeps_nonadjacent_names_separate() -> None:
    text = "First: Jason | Last: Smith"
    row = {
        "uid": "n5",
        "locale": "us",
        "domain": "x",
        "text": text,
        "spans": [
            {"start": 7, "end": 12, "text": "Jason", "label": "first_name"},
            {"start": 21, "end": 26, "text": "Smith", "label": "last_name"},
        ],
    }

    record, _dropped = convert_nemotron_row(row, default_uid="row-0")

    assert record is not None
    _id, _text, spans = parse_labeled_record(record)
    assert sorted((s.text, s.label) for s in spans) == [
        ("Jason", "human_name"),
        ("Smith", "human_name"),
    ]


def test_convert_nemotron_row_drops_non_pii_and_generic_url() -> None:
    text = "Jason was born 1987-05-22. Visit https://example.com."
    row = {
        "uid": "n1",
        "locale": "us",
        "domain": "Identity Verification Services",
        "text": text,
        "spans": repr([
            {"start": 0, "end": 6, "text": "Jason", "label": "first_name"},
            {
                "start": 16,
                "end": 26,
                "text": "1987-05-22",
                "label": "date_of_birth",
            },
            {
                "start": 34,
                "end": 53,
                "text": "https://example.com",
                "label": "url",
            },
            {"start": 0, "end": 0, "text": "", "label": "occupation"},
        ]),
    }

    record, dropped = convert_nemotron_row(row, default_uid="row-0")

    assert record is not None
    assert dropped["url"] == 1
    assert dropped["occupation"] == 1
    _example_id, _text, spans = parse_labeled_record(record)
    assert [(span.text, span.label) for span in spans] == [
        ("Jason", "human_name"),
        ("1987-05-22", "date"),
    ]


def test_summarize_records_counts_domain_language_source_and_labels() -> None:
    record, _ = convert_ai4privacy_row(
        {
            "source_text": "Call Bob at 555-1212.",
            "uid": "x",
            "language": "en",
            "split": "train",
            "privacy_mask": [
                {"label": "GIVENNAME", "start": 5, "end": 8, "value": "Bob"},
                {"label": "TELEPHONENUM", "start": 12, "end": 20, "value": "555-1212"},
            ],
        },
        dataset_id="ai4privacy/open-pii-masking-500k-ai4privacy",
        default_uid="x",
    )
    assert record is not None

    summary = summarize_records([record]).asdict()

    assert summary["rows"] == 1
    assert summary["domain_bucket_counts"] == {"general": 1}
    assert summary["language_bucket_counts"] == {"en": 1}
    label_counts = _mapping_field(summary, "label_counts")
    assert label_counts["human_name"] == 1
    assert label_counts["phone_number"] == 1


def test_language_bucket_treats_vietnamese_translated_source_as_vi() -> None:
    assert language_bucket(language="UNKNOWN", source="vietnamese-translated") == "vi"


def test_supported_meddies_language_rejects_non_target_external_language() -> None:
    assert is_supported_meddies_language("fr")
    assert is_supported_meddies_language("us")
    assert not is_supported_meddies_language("bg")
    assert not is_supported_meddies_language("nl")


def test_convert_gretel_row_reads_label_from_types_and_value_from_entity() -> None:
    """Gretel `healthcare` industry -> healthcare domain_bucket (the 2-value split)."""
    text = "Patient Urvashi Jaggi, MRN MRN-99812, email urvashi@example.com."
    row = {
        "uid": "g1",
        "domain": "healthcare",
        "text": text,
        "entities": repr([
            {"entity": "Urvashi Jaggi", "types": ["name"]},
            {"entity": "MRN-99812", "types": ["medical_record_number"]},
            {"entity": "urvashi@example.com", "types": ["email"]},
        ]),
    }

    record, dropped = convert_gretel_row(row, default_uid="row-0")

    assert record is not None
    assert dropped == Counter()
    _example_id, parsed_text, spans = parse_labeled_record(record)
    assert parsed_text == text
    assert [(span.text, span.label) for span in spans] == [
        ("Urvashi Jaggi", "human_name"),
        ("MRN-99812", "id_number"),
        ("urvashi@example.com", "email_address"),
    ]
    assert record["info"]["domain_bucket"] == "healthcare"


def test_convert_gretel_row_derives_healthcare_vs_general_domain_bucket() -> None:
    base = {
        "uid": "g-dom",
        "text": "Patient MRN-1 seen.",
        "entities": [{"entity": "MRN-1", "types": ["medical_record_number"]}],
    }
    for domain in (
        "healthcare",
        "healthcare-administration",
        "pharmaceuticals-biotechnology",
    ):
        record, _ = convert_gretel_row({**base, "domain": domain}, default_uid="r")
        assert record is not None
        assert record["info"]["domain_bucket"] == "healthcare", domain
    for domain in ("biometrics", "insurance", "finance", "support"):
        record, _ = convert_gretel_row({**base, "domain": domain}, default_uid="r")
        assert record is not None
        assert record["info"]["domain_bucket"] == "general", domain


def test_convert_gretel_row_maps_phi_identifiers_to_id_number() -> None:
    text = "SSN 111-22-3333 and plan HPB-7781 on file."
    row = {
        "uid": "g2",
        "domain": "healthcare",
        "text": text,
        "entities": [
            {"entity": "111-22-3333", "types": ["ssn"]},
            {"entity": "HPB-7781", "types": ["health_plan_beneficiary_number"]},
        ],
    }

    record, _dropped = convert_gretel_row(row, default_uid="row-0")

    assert record is not None
    _example_id, _text, spans = parse_labeled_record(record)
    assert [(span.text, span.label) for span in spans] == [
        ("111-22-3333", "id_number"),
        ("HPB-7781", "id_number"),
    ]


def test_convert_gretel_row_maps_credit_card_and_unique_identifier_to_id_number() -> None:
    text = "Card 4111-1111-1111-1111 ref UID-PRWBO4TB on file."
    row = {
        "uid": "g5",
        "domain": "finance",
        "text": text,
        "entities": [
            {"entity": "4111-1111-1111-1111", "types": ["credit_card_number"]},
            {"entity": "UID-PRWBO4TB", "types": ["unique_identifier"]},
        ],
    }

    record, dropped = convert_gretel_row(row, default_uid="row-0")

    assert record is not None
    assert dropped == Counter()
    _example_id, _text, spans = parse_labeled_record(record)
    assert [(span.text, span.label) for span in spans] == [
        ("4111-1111-1111-1111", "id_number"),
        ("UID-PRWBO4TB", "id_number"),
    ]


def test_convert_gretel_row_keeps_time_as_date_gretel_scoped() -> None:
    text = "Appointment confirmed at 3:00 PM with the nurse."
    row = {
        "uid": "g6",
        "domain": "healthcare",
        "text": text,
        "entities": [{"entity": "3:00 PM", "types": ["time"]}],
    }

    record, dropped = convert_gretel_row(row, default_uid="row-0")

    assert record is not None
    assert "time" not in dropped
    _example_id, _text, spans = parse_labeled_record(record)
    assert [(span.text, span.label) for span in spans] == [("3:00 PM", "date")]


def test_convert_nemotron_row_drops_bare_time() -> None:
    text = "Meeting scheduled at 3:00 PM today."
    row = {
        "uid": "n2",
        "locale": "us",
        "domain": "scheduling",
        "text": text,
        "spans": [{"start": 21, "end": 28, "text": "3:00 PM", "label": "time"}],
    }

    record, dropped = convert_nemotron_row(row, default_uid="row-0")

    assert record is None
    assert dropped["time"] == 1


def test_convert_gretel_row_drops_public_url_and_keeps_source_tag() -> None:
    text = "Contact Bob at bob@example.com or visit https://example.com."
    row = {
        "uid": "g3",
        "domain": "support",
        "text": text,
        "entities": [
            {"entity": "Bob", "types": ["name"]},
            {"entity": "bob@example.com", "types": ["email"]},
            {"entity": "https://example.com", "types": ["url"]},
        ],
    }

    record, dropped = convert_gretel_row(row, default_uid="row-0")

    assert record is not None
    assert dropped["url"] == 1
    _example_id, _text, spans = parse_labeled_record(record)
    assert [span.label for span in spans] == ["human_name", "email_address"]
    assert record["info"]["source"] == "support"
    assert record["info"]["source_dataset"] == "gretelai/gretel-pii-masking-en-v1"


def test_convert_gretel_row_returns_none_when_no_pii_spans_survive() -> None:
    row = {
        "uid": "g4",
        "domain": "support",
        "text": "Visit https://example.com for the public guideline.",
        "entities": [{"entity": "https://example.com", "types": ["url"]}],
    }

    record, dropped = convert_gretel_row(row, default_uid="row-0")

    assert record is None
    assert dropped["url"] == 1


def test_dedupe_collapses_whitespace_and_case_variants() -> None:
    from meddies_pii.training.bioes.data.mixed_build import _dedupe_records_by_text

    rows: list[dict[str, object]] = [
        {"text": "Bệnh viện Chợ Rẫy"},
        {"text": "bệnh viện  chợ rẫy "},
        {"text": "Bệnh viện Bạch Mai"},
    ]
    out, dropped = _dedupe_records_by_text(rows)

    assert dropped == 1
    assert len(out) == 2


def test_gretel_rows_route_through_language_gate() -> None:
    assert is_supported_meddies_language("en")
    assert not is_supported_meddies_language("sw")


def test_gretel_dispatch_yields_pii_label_spans() -> None:
    from meddies_pii.training.bioes.data.mixed_sources import (
        EXTERNAL_DATASETS,
        convert_external,
    )

    dataset_id = "gretelai/gretel-pii-masking-en-v1"
    assert dataset_id in EXTERNAL_DATASETS

    row: dict[str, object] = {
        "uid": "g1",
        "domain": "healthcare",
        "text": "Patient Urvashi Jaggi, MRN MRN-99812.",
        "entities": [
            {"entity": "Urvashi Jaggi", "types": ["name"]},
            {"entity": "MRN-99812", "types": ["medical_record_number"]},
        ],
    }

    record, dropped = convert_external(dataset_id, row, row_index=0)

    assert record is not None
    assert dropped == Counter()
    _example_id, _text, spans = parse_labeled_record(record)
    assert [(span.text, span.label) for span in spans] == [
        ("Urvashi Jaggi", "human_name"),
        ("MRN-99812", "id_number"),
    ]
    assert record["info"]["source_dataset"] == dataset_id


def test_local_external_rows_dispatch_converter_by_filename_stem(
    tmp_path: Path,
) -> None:
    from meddies_pii.training.bioes.data.mixed_sources import load_local_external_rows

    external_dir = tmp_path / "external"
    external_dir.mkdir()

    gretel_row = {
        "uid": "g1",
        "domain": "healthcare",
        "text": "Email urvashi@example.com.",
        "entities": [{"entity": "urvashi@example.com", "types": ["email"]}],
    }
    nemotron_row = {
        "uid": "n1",
        "locale": "us",
        "domain": "Identity Verification Services",
        "text": "Jason was born 1987-05-22.",
        "spans": [
            {"start": 0, "end": 5, "text": "Jason", "label": "first_name"},
            {"start": 15, "end": 25, "text": "1987-05-22", "label": "date_of_birth"},
        ],
    }
    (external_dir / "gretel.jsonl").write_text(json.dumps(gretel_row, ensure_ascii=False) + "\n", encoding="utf-8")
    (external_dir / "nvidia-health.jsonl").write_text(
        json.dumps(nemotron_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows, _dropped = load_local_external_rows(external_dir)

    by_source: dict[str, dict[str, object]] = {}
    for row in rows:
        info = row.get("info")
        assert isinstance(info, dict)
        source_dataset = info.get("source_dataset")
        assert isinstance(source_dataset, str)
        by_source[source_dataset] = row
    assert "gretelai/gretel-pii-masking-en-v1" in by_source
    assert "nvidia/Nemotron-PII" in by_source

    _id, _text, gretel_spans = parse_labeled_record(by_source["gretelai/gretel-pii-masking-en-v1"])
    assert [(s.text, s.label) for s in gretel_spans] == [
        ("urvashi@example.com", "email_address"),
    ]
    _id, _text, nemo_spans = parse_labeled_record(by_source["nvidia/Nemotron-PII"])
    assert [(s.text, s.label) for s in nemo_spans] == [
        ("Jason", "human_name"),
        ("1987-05-22", "date"),
    ]
