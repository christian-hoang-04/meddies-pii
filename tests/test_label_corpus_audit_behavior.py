from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from meddies_pii.generation.label_corpus import audit
from meddies_pii.jsonl import read_jsonl, write_jsonl

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _record(
    *,
    identifier: str,
    scenario: str = "portal",
    phone: str = "617-555-0199",
    include_required: bool = True,
) -> dict[str, object]:
    private_url = "https://portal.example.test/result?token=abc"
    # reason: this synthetic credential is the PII the labeller under test has to find, so it is the
    # reason: fixture's payload rather than a secret in use. Renaming it would stop describing the row.
    secret = "token-123"  # ruff: ignore[hardcoded-password-string]
    text = f"{identifier}: phone {phone}; portal {private_url}; session {secret}."
    labels = []
    for category, value in (("phone_number", phone), ("private_url", private_url)):
        start = text.index(value)
        labels.append({"category": category, "start": start, "end": start + len(value)})
    if include_required:
        start = text.index(secret)
        labels.append({"category": "secret", "start": start, "end": start + len(secret)})
    return {
        "text": text,
        "label": labels,
        "info": {"id": identifier, "scenario": scenario},
    }


def test_check_record_reports_required_label_and_phone_boundaries() -> None:
    valid = _record(identifier="valid")
    example_id, issues, phones = audit.check_record(valid)
    assert example_id == "valid"
    assert issues == []
    assert phones == ["617-555-0199"]

    _, missing_issues, _ = audit.check_record(_record(identifier="missing", include_required=False))
    assert missing_issues == ["missing private_url/secret"]

    _, short_issues, _ = audit.check_record(_record(identifier="short", phone="4417"))
    assert short_issues == ["short numeric phone span '4417'"]

    underbounded = _record(identifier="underbounded", phone="916.442.8031")
    underbounded_text = underbounded["text"]
    assert isinstance(underbounded_text, str)
    underbounded_labels = underbounded["label"]
    assert isinstance(underbounded_labels, list)
    phone_start = underbounded_text.index("916.442")
    underbounded_labels[0] = {
        "category": "phone_number",
        "start": phone_start,
        "end": phone_start + len("916.442"),
    }
    _, underbounded_issues, underbounded_phones = audit.check_record(underbounded)
    assert underbounded_phones == ["916.442"]
    assert underbounded_issues == ["underbounded phone span '916.442' vs window '916.442.8031'"]


def test_split_clean_and_flagged_quarantines_schema_and_label_failures() -> None:
    valid = _record(identifier="valid")
    clean, flagged = audit.split_clean_and_flagged([
        valid,
        _record(identifier="missing", include_required=False),
        {"text": 9},
    ])

    assert clean == [valid]
    assert flagged[0]["spot_check_issues"] == ["missing private_url/secret"]
    assert flagged[1]["spot_check_issues"] == ["parse error: record text must be a string"]
    assert audit.split_clean_and_flagged([]) == ([], [])


def test_markdown_table_cell_escapes_supported_table_breakers() -> None:
    assert audit.markdown_table_cell("a\\b\r\nc|d") == "a\\\\b\\nc/d"
    assert audit.markdown_table_cell("abcdef", limit=4) == "abc…"
    assert audit.record_info({"info": ["not", "a", "mapping"]}) == {}


def test_audit_main_writes_deterministic_sorted_report_and_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vi_rows = [
        _record(identifier="vi-1", scenario="zeta"),
        _record(identifier="vi-2", scenario="alpha"),
        _record(identifier="vi-3", scenario="alpha"),
        _record(identifier="vi-4", scenario="beta|line\nvalue"),
        {"text": 7},
    ]
    en_rows = [
        _record(identifier="en-1", scenario="gamma"),
        _record(identifier="en-2", scenario="gamma"),
        _record(identifier="en-3", scenario="delta"),
    ]
    write_jsonl(tmp_path / "repaired_candidates.vi.jsonl", vi_rows)
    write_jsonl(tmp_path / "repaired_candidates.en.jsonl", en_rows)

    first_report = tmp_path / "first.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit.py",
            "--output-dir",
            str(tmp_path),
            "--report",
            str(first_report),
            "--write-clean",
        ],
    )
    audit.main()
    first_stdout = capsys.readouterr().out
    first_text = first_report.read_text(encoding="utf-8")

    second_report = tmp_path / "second.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit.py", "--output-dir", str(tmp_path), "--report", str(second_report)],
    )
    audit.main()
    second_stdout = capsys.readouterr().out

    assert str(first_report) in first_stdout
    assert "flagged=1 clean_vi=4 clean_en=3 sampled_issues=0" in first_stdout
    assert str(second_report) in second_stdout
    assert first_text == second_report.read_text(encoding="utf-8")
    assert "- `vi` source=5, clean=4, flagged=1." in first_text
    assert "- `en` source=3, clean=3, flagged=0." in first_text
    assert "parse error: record text must be a string" in first_text
    assert "Duplicate clean texts: 0." in first_text
    assert "PASS: 5x5 clean spot check passed" in first_text
    assert "beta/line\\nvalue" in first_text
    label_section = first_text.split("### Clean label distribution\n", 1)[1].split("\n### Clean scenario distribution", 1)[
        0
    ]
    assert label_section.index("`phone_number`") < label_section.index("`private_url`")
    assert label_section.index("`private_url`") < label_section.index("`secret`")
    assert len(list(read_jsonl(tmp_path / "repaired_candidates.clean.vi.jsonl"))) == 4
    flagged = list(read_jsonl(tmp_path / "repaired_candidates.flagged.vi.jsonl"))
    assert flagged[0]["spot_check_issues"] == ["parse error: record text must be a string"]
