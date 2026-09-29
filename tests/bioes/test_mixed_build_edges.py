from __future__ import annotations

import json
import sys
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.data import mixed_build

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _mapping_field(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return a row field the caller reads by key, asserting it really is a nested mapping.

    `_read_jsonl` returns `list[dict[str, object]]`, so the `info` block each assertion
    reads through arrives as `object`.

    Returns:
        The named field, narrowed to a mapping the caller can read by string key.

    """
    value = payload[key]
    assert is_str_mapping(value), f"{key} must be a mapping, got {type(value).__name__}"
    return value


def _medical_row(
    row_id: str,
    language: str,
    text: str,
    *,
    labels: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "label": labels if labels is not None else [{"category": "human_name"}],
        "info": {"id": row_id, "language": language},
    }


def _external_row(row_id: str, language: str) -> dict[str, object]:
    email = f"person{row_id}@example.com"
    text = f"Contact {email}."
    start = text.index(email)
    return {
        "uid": row_id,
        "language": language,
        "source_text": text,
        "privacy_mask": [
            {
                "label": "EMAIL",
                "value": email,
                "start": start,
                "end": start + len(email),
            },
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_main(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["mixed-build", *args])
    mixed_build.main()


def test_mixed_build_cli_reconciles_external_language_overfill_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical = tmp_path / "medical.jsonl"
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    _write_jsonl(
        medical,
        [
            *[_medical_row(f"en-{index}", "en", f"English medical {index}") for index in range(3)],
            *[_medical_row(f"other-{index}", "fr", f"Other medical {index}") for index in range(2)],
            _medical_row("vi-unused", "vi", "Vietnamese medical"),
        ],
    )
    _write_jsonl(
        external_dir / "ai4privacy.jsonl",
        [_external_row(str(index), "vi") for index in range(5)],
    )

    output_one = tmp_path / "first.jsonl"
    summary_one = tmp_path / "first.summary.json"
    output_two = tmp_path / "second.jsonl"
    summary_two = tmp_path / "second.summary.json"
    base_args = [
        "--medical-jsonl",
        str(medical),
        "--target-rows",
        "10",
        "--medical-ratio",
        "0.5",
        "--vi-ratio",
        "0.3",
        "--en-ratio",
        "0.3",
        "--external-dir",
        str(external_dir),
        "--seed",
        "17",
    ]

    _run_main(
        monkeypatch,
        [*base_args, "--output", str(output_one), "--summary", str(summary_one)],
    )
    _run_main(
        monkeypatch,
        [*base_args, "--output", str(output_two), "--summary", str(summary_two)],
    )

    first_rows = _read_jsonl(output_one)
    assert output_one.read_bytes() == output_two.read_bytes()
    assert summary_one.read_bytes() == summary_two.read_bytes()
    assert Counter(str(_mapping_field(row, "info")["language_bucket"]) for row in first_rows) == {
        "vi": 5,
        "en": 3,
        "other": 2,
    }
    assert len({_mapping_field(row, "info")["id"] for row in first_rows}) == len(first_rows)
    assert all("original_id" in _mapping_field(row, "info") for row in first_rows)


def test_mixed_build_cli_makes_underfill_explicit_without_medical_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical = tmp_path / "medical.jsonl"
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    _write_jsonl(
        medical,
        [
            _medical_row("vi-1", "vi", "Vietnamese medical"),
            _medical_row("en-1", "en", "English medical 1"),
            _medical_row("en-2", "en", "English medical 2"),
        ],
    )
    _write_jsonl(
        external_dir / "ai4privacy.jsonl",
        [_external_row(str(index), "fr") for index in range(4)],
    )
    output = tmp_path / "output.jsonl"
    summary = tmp_path / "summary.json"
    args = [
        "--medical-jsonl",
        str(medical),
        "--target-rows",
        "8",
        "--medical-ratio",
        "0.5",
        "--vi-ratio",
        "0.25",
        "--en-ratio",
        "0.25",
        "--external-dir",
        str(external_dir),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ]

    with pytest.raises(ValueError, match="target mix requires duplicated/oversampled"):
        _run_main(monkeypatch, args)
    _run_main(monkeypatch, [*args, "--allow-underfill"])

    rows = _read_jsonl(output)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    medical_original_ids = [
        str(_mapping_field(row, "info")["original_id"])
        for row in rows
        if str(_mapping_field(row, "info")["id"]).startswith("mixed-medical:")
    ]
    assert payload["language_shortfalls"] == {"vi": 1}
    assert len(rows) == 7
    assert len(medical_original_ids) == len(set(medical_original_ids)) == 3


def test_mixed_build_cli_drops_invalid_text_duplicates_and_empty_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical = tmp_path / "medical.jsonl"
    output = tmp_path / "output.jsonl"
    summary = tmp_path / "summary.json"
    _write_jsonl(
        medical,
        [
            _medical_row("first", "vi", " Patient Lan "),
            _medical_row("duplicate", "vi", "patient   lan"),
            {"text": None, "label": [{"category": "human_name"}]},
            _medical_row("empty", "en", "No spans", labels=[]),
            {"text": "Wrong label shape", "label": "human_name"},
            _medical_row("distinct", "en", "Patient Minh"),
        ],
    )

    _run_main(
        monkeypatch,
        [
            "--medical-jsonl",
            str(medical),
            "--medical-limit",
            "6",
            "--external-count",
            "0",
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert [row["text"] for row in _read_jsonl(output)] == [
        " Patient Lan ",
        "Patient Minh",
    ]
    assert payload["duplicate_text_dropped"] == 2
    assert payload["empty_label_rows_dropped"] == 2


def test_mixed_build_cli_can_skip_finalization_and_enforce_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    medical = tmp_path / "medical.jsonl"
    output = tmp_path / "output.jsonl"
    summary = tmp_path / "summary.json"
    _write_jsonl(
        medical,
        [
            _medical_row("labeled", "vi", "Labeled"),
            _medical_row("empty", "en", "Empty", labels=[]),
        ],
    )
    args = [
        "--medical-jsonl",
        str(medical),
        "--external-count",
        "0",
        "--no-dedupe-text",
        "--keep-empty-label-rows",
        "--output",
        str(output),
        "--summary",
        str(summary),
    ]

    _run_main(monkeypatch, args)

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(_read_jsonl(output)) == 2
    assert payload["duplicate_text_dropped"] == 0
    assert payload["empty_label_rows_dropped"] == 0
    with pytest.raises(SystemExit, match="Assembly gate BLOCKED"):
        _run_main(monkeypatch, [*args, "--enforce-gate"])
