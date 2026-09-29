from __future__ import annotations

import json
from pathlib import Path

import pytest

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.training.bioes.assembly.mix import (
    AssemblyGateError,
    assemble_bioes_v2_corpus,
    discover_assembly_source_files,
    parse_waiver_cells,
)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_discovery_waivers_and_assembly_keep_ordered_deduped_clean_rows(
    tmp_path: Path,
) -> None:
    first = tmp_path / "base" / f"a.train.{LEGACY_ARTIFACT_TOKEN}.jsonl"
    second = tmp_path / "hf_configs" / "pii-bioes.train.jsonl"
    _write_jsonl(
        first,
        [
            {
                "text": "Alice",
                "label": [{"category": "human_name", "text": "Alice"}],
                "info": {"language": "en"},
            },
            {
                "text": "Patient   Alice",
                "label": [{"category": "human_name", "text": "Alice"}],
                "info": {"language": "en"},
            },
            {"label": []},
            "not-an-object",
        ],
    )
    _write_jsonl(
        second,
        [
            {
                "text": "Patient Alice",
                "label": [{"category": "human_name", "text": "Alice"}],
                "info": {"language": "en"},
            },
            {
                "text": "bad",
                "label": [{"category": "phone_number", "text": "not a phone"}],
                "info": {"language": "en"},
            },
            {
                "text": "fine",
                "label": [
                    {"category": "human_name", "text": "Fine"},
                    {"category": "email_address", "text": "fine@example.com"},
                    {"category": "phone_number", "text": "no phone"},
                ],
                "info": {"language": "en"},
            },
        ],
    )

    assert discover_assembly_source_files(tmp_path) == [first, second]
    assert parse_waiver_cells(["en:human_name", "bad", ":phone_number", "vi:"]) == {("en", "human_name")}

    result = assemble_bioes_v2_corpus(
        tmp_path,
        output_path=Path("mix/output.jsonl"),
        label_floors={},
        entropy_targets={},
        enforce_gate=False,
    )

    rows = [json.loads(line) for line in result.output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["text"] for row in rows] == ["Alice", "Patient   Alice", "fine"]
    assert result.total_before_dedup == 6
    assert result.duplicate_text_dropped == 1
    assert result.rows_missing_text_dropped == 1
    assert result.bad_spans_dropped == 1
    assert result.docs_quarantined == 1
    assert result.per_source_rows == {
        f"base/a.train.{LEGACY_ARTIFACT_TOKEN}.jsonl": 2,
        "hf_configs/pii-bioes.train.jsonl": 1,
    }
    assert result.manifest_path.exists()


def test_assembly_gate_refuses_output_and_external_output_path(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "synthetic" / "rows.jsonl",
        [{"text": "Alice", "label": [{"category": "human_name", "text": "Alice"}]}],
    )

    with pytest.raises(AssemblyGateError, match="Assembly gate BLOCKED"):
        assemble_bioes_v2_corpus(tmp_path, output_path=Path("mix/blocked.jsonl"))
    assert not (tmp_path / "mix/blocked.jsonl").exists()

    with pytest.raises(ValueError, match="inside the corpus root"):
        assemble_bioes_v2_corpus(
            tmp_path,
            output_path=tmp_path.parent / "outside.jsonl",
            enforce_gate=False,
        )
