from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.data import build_legacy_pii_label_corpus

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _run_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    config: str,
    limit: int | None,
) -> tuple[list[tuple[str, str, str]], dict[str, object]]:
    calls: list[tuple[str, str, str]] = []

    def fake_load_dataset(
        dataset_id: str,
        dataset_config: str,
        *,
        split: str,
    ) -> list[dict[str, object]]:
        calls.append((dataset_id, dataset_config, split))
        return [
            {"uid": f"{dataset_config}-0", "raw": "No PII"},
            {
                "uid": f"{dataset_config}-1",
                "raw": "Preserved metadata",
                "source": "preserved-source",
                "language": "Preserved Language",
            },
        ]

    output = tmp_path / "corpus.jsonl"
    audit = tmp_path / "audit.jsonl"
    summary = tmp_path / "summary.json"
    argv = [
        "build-legacy-pii-label-corpus",
        "--dataset-id",
        "fixture/dataset",
        "--config",
        config,
        "--split",
        "validation",
        "--output",
        str(output),
        "--audit",
        str(audit),
        "--summary",
        str(summary),
    ]
    if limit is not None:
        argv.extend(("--limit", str(limit)))
    monkeypatch.setattr(build_legacy_pii_label_corpus, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(sys, "argv", argv)

    build_legacy_pii_label_corpus.main()

    payload: object = json.loads(summary.read_text(encoding="utf-8"))
    assert is_str_mapping(payload)
    return calls, dict(payload)


def test_builder_combines_configs_limits_each_source_and_sets_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls, summary = _run_builder(
        monkeypatch,
        tmp_path,
        config="vietnamese-translated, english",
        limit=1,
    )
    records = [json.loads(line) for line in (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]

    assert calls == [
        ("fixture/dataset", "vietnamese-translated", "validation"),
        ("fixture/dataset", "english", "validation"),
    ]
    assert summary["rows"] == 2
    assert [record["info"]["source"] for record in records] == [
        "vietnamese-translated",
        "english",
    ]
    assert [record["info"]["language"] for record in records] == [
        "Vietnamese",
        "English",
    ]


def test_builder_expands_all_languages_and_preserves_existing_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        build_legacy_pii_label_corpus,
        "MEDDIES_PII_LANGUAGE_CONFIGS",
        ("english", "vietnamese-translated"),
    )

    calls, summary = _run_builder(
        monkeypatch,
        tmp_path,
        config="all-languages",
        limit=None,
    )
    records = [json.loads(line) for line in (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]

    assert [config for _, config, _ in calls] == [
        "english",
        "vietnamese-translated",
    ]
    assert summary["rows"] == 4
    assert records[1]["info"]["source"] == "preserved-source"
    assert records[1]["info"]["language"] == "Preserved Language"
