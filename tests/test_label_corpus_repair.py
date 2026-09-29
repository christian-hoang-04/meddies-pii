from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
from typing import TYPE_CHECKING

from meddies_pii.generation.label_corpus.catalog import SCENARIOS
from meddies_pii.generation.label_corpus.repair import (
    repair_inline_tag_format,
    repair_rejected_artifacts,
    split_rejected_records,
)
from meddies_pii.generation.label_corpus.synthetic import language_paths
from meddies_pii.generation.label_corpus.validate import validate_tagged_document
from meddies_pii.historical_artifacts import LEGACY_REPAIRED_REJECTS_DATASET_ID
from meddies_pii.jsonl import read_jsonl, write_jsonl

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.generation.label_corpus.catalog import Scenario


def _scenario(name: str) -> Scenario:
    return next(s for s in SCENARIOS if s.name == name)


def test_split_rejected_records_separates_api_errors_from_content_rejects() -> None:
    rows: list[dict[str, object]] = [
        {"errors": ["api_error"], "error": "429 Too Many Requests"},
        {"errors": ["leftover_label_marker"], "content": "Patient [A]<human_name>"},
        {"errors": ["api_error", "leftover_label_marker"], "error": "RetryError"},
    ]

    split = split_rejected_records(rows)

    assert split.api_errors == [rows[0], rows[2]]
    assert split.content_rejects == [rows[1]]


def test_repair_inline_tag_format_recovers_common_structured_value_suffixes() -> None:
    content = """MSH|^~\\&|Epic|AcmeHealth|Cerner|MetroHospital|20231026141500||ADT^A01|MSGID001234|P|2.5.1
PID|1||[MRN77429]<id_number>^^^AcmeHealth^MR||[Doe, Jane \
Elizabeth]<human_name>||19850315<date>|F|||742 Evergreen Terrace, Apt 3B, Springfield, IL \
62704<address>||[555-0199] <phone_number>|||jane.doe at example dot org<email_address>
PV1|1|I|MED^302^A||||[Dr. Anna Ortiz]<human_name>|||[Metro Hospital]<company_name>
NTE|1|L|Portal Deep Link: https://patient.example.test/results? token=abc123<private_url>
NTE|2|L|Session Token: Bearer abc.def.ghi<secret>
NTE|3|L|Signed on [2024-03-15]<date]
"""

    repaired = repair_inline_tag_format(content)
    result = validate_tagged_document(
        repaired,
        language="English",
        required_labels=("private_url", "secret"),
    )

    assert result.ok, result.errors
    assert "19850315<date>" not in repaired
    assert "[19850315]<date>" in repaired
    assert "[555-0199]<phone_number>" in repaired
    assert "[https://patient.example.test/results? token=abc123]<private_url>" in repaired
    assert "Session Token: [Bearer abc.def.ghi]<secret>" in repaired
    assert "[2024-03-15]<date>" in repaired


def test_repair_inline_tag_format_wraps_value_after_field_prefix_not_prefix_itself() -> None:
    content = "Patient Name: Dr. Marcus Chen<human_name>\n"

    repaired = repair_inline_tag_format(content)

    assert "Patient Name: [Dr. Marcus Chen]<human_name>" in repaired
    assert "[Patient Name: Dr. Marcus Chen]<human_name>" not in repaired


def test_repair_rejected_records_returns_repaired_candidates_and_keeps_api_errors() -> None:
    from meddies_pii.generation.label_corpus.repair import repair_rejected_records

    rejected_rows: list[dict[str, object]] = [
        {"errors": ["api_error"], "error": "429 Too Many Requests"},
        {
            "attempt_index": 42,
            "scenario": "private_portal_secret_focus",
            "document_type": "HL7-style admission/update message",
            "text_format": "HL7-like pipe-delimited text",
            "errors": ["leftover_label_marker"],
            "content": """PID|1||[MRN77429]<id_number>||Patient Name: Dr. Marcus \
Chen<human_name>||19850315<date>|F|||742 Evergreen Terrace, Springfield, IL<address>||[555-0199] \
<phone_number>|||jane.doe at example dot org<email_address>
PV1|1|I|||||[Metro Hospital]<company_name>
NTE|1|L|Portal Deep Link: https://patient.example.test/results? token=abc123<private_url>
NTE|2|L|Session Token: Bearer abc.def.ghi<secret>
NTE|3|L|Follow-up [2024-03-15]<date] with [Dr. Jane Smith]<human_name>
""",
        },
    ]

    result = repair_rejected_records(
        rejected_rows,
        language="English",
        existing_text_hashes=set(),
        model="mimo-v2.5-pro",
        provider="mimo",
    )

    assert result.api_errors == [rejected_rows[0]]
    assert len(result.content_rejects) == 1
    assert len(result.repaired_candidates) == 1
    assert result.unrepaired_content_rejects == []
    candidate = result.repaired_candidates[0]
    assert candidate["info"]["source_dataset"] == LEGACY_REPAIRED_REJECTS_DATASET_ID
    assert candidate["info"]["repair_applied"] is True
    assert candidate["info"]["original_errors"] == ["leftover_label_marker"]
    assert {span["category"] for span in candidate["label"]} >= {
        "private_url",
        "secret",
    }


def test_repair_rejected_artifacts_writes_sidecar_files(tmp_path: Path) -> None:
    paths = language_paths(tmp_path, "English")
    write_jsonl(
        paths.rejected,
        [
            {"errors": ["api_error"], "error": "429 Too Many Requests"},
            {
                "attempt_index": 42,
                "scenario": "private_portal_secret_focus",
                "document_type": "HL7-style admission/update message",
                "text_format": "HL7-like pipe-delimited text",
                "errors": ["leftover_label_marker"],
                "content": """PID|1||[MRN77429]<id_number>||Patient Name: Dr. Marcus \
Chen<human_name>||19850315<date>|F|||742 Evergreen Terrace, Springfield, IL<address>||[555-0199] \
<phone_number>|||jane.doe at example dot org<email_address>
PV1|1|I|||||[Metro Hospital]<company_name>
NTE|1|L|Portal Deep Link: https://patient.example.test/results? token=abc123<private_url>
NTE|2|L|Session Token: Bearer abc.def.ghi<secret>
NTE|3|L|Follow-up [2024-03-15]<date] with [Dr. Jane Smith]<human_name>
""",
            },
        ],
    )

    summary = repair_rejected_artifacts(
        tmp_path,
        "English",
        model="mimo-v2.5-pro",
        provider="mimo",
    )

    assert summary["api_error_count"] == 1
    assert summary["repaired_candidate_count"] == 1
    assert (tmp_path / "api_errors.en.jsonl").exists()
    assert (tmp_path / "content_rejected.en.jsonl").exists()
    assert (tmp_path / "unrepaired_content_rejected.en.jsonl").exists()
    assert (tmp_path / "repaired_candidates.en.jsonl").exists()
    assert (tmp_path / "repair_summary.en.json").exists()
    assert len(list(read_jsonl(tmp_path / "repaired_candidates.en.jsonl"))) == 1


def test_repair_inline_tag_format_combines_split_bracket_fragments_before_label() -> None:
    content = "Điện thoại        : [024] [3821] [5576]<phone_number>\n"

    repaired = repair_inline_tag_format(content)

    assert "[024 3821 5576]<phone_number>" in repaired
    assert "[5576]<phone_number>" not in repaired


def test_repair_inline_tag_format_combines_multiple_labeled_phone_fragments() -> None:
    content = "Số ĐT liên hệ: [0918] <phone_number> [345 678]<phone_number>\n"

    repaired = repair_inline_tag_format(content)

    assert "[0918 345 678]<phone_number>" in repaired
    assert repaired.count("<phone_number>") == 1


def test_repair_inline_tag_format_combines_hyphenated_bracket_fragments_before_phone_label() -> None:
    content = "ROW: Patient Phone ........... [503]-[819]-[4427]<phone_number>\n"

    repaired = repair_inline_tag_format(content)

    assert "[503-819-4427]<phone_number>" in repaired
    assert "[4427]<phone_number>" not in repaired


def test_repair_inline_tag_format_includes_immediate_phone_prefix_before_labeled_fragment() -> None:
    content = "PID|||[742 Evergreen Terrace]<address>^^^US||(555) [891-2345]<phone_number>||English\n"

    repaired = repair_inline_tag_format(content)

    assert "[(555) 891-2345]<phone_number>" in repaired
    assert "[891-2345]<phone_number>" not in repaired
