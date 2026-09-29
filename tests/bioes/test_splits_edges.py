from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from meddies_pii.json_types import is_str_mapping
from meddies_pii.taxonomy import PII_LABELS
from meddies_pii.training.bioes.data import splits

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def _row(
    row_id: str,
    language: str,
    text: str,
    *,
    domain: str = "medical",
    labels: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "label": labels if labels is not None else [{"category": "human_name"}],
        "info": {
            "id": row_id,
            "language": language,
            "domain_bucket": domain,
            "source_dataset": "fixture",
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_carve_heldout_rejects_invalid_sizes_and_unavailable_bucket() -> None:
    rows = [
        *[_row(f"vi-{index}", "vi", f"Vietnamese {index}") for index in range(3)],
        *[_row(f"en-{index}", "en", f"English {index}") for index in range(3)],
    ]

    with pytest.raises(ValueError, match="heldout_rows must be positive"):
        splits.carve_heldout(rows, heldout_rows=0, seed=1)
    with pytest.raises(ValueError, match="smaller than the input row count"):
        splits.carve_heldout(rows, heldout_rows=len(rows), seed=1)
    with pytest.raises(ValueError, match="bucket 'other' has 0 candidate rows"):
        splits.carve_heldout(rows, heldout_rows=3, seed=1)


def test_carve_heldout_rejects_a_small_sample_outside_tolerance() -> None:
    rows = [
        _row("other-1", "fr", "French one"),
        _row("other-2", "fr", "French two"),
    ]

    with pytest.raises(ValueError, match=r"bucket 'vi' balance 0\.000 exceeds tolerance"):
        splits.carve_heldout(rows, heldout_rows=1, seed=1, tolerance=0.05)


def _info_ids(rows: Sequence[Mapping[str, object]]) -> set[str]:
    """Return the ``info.id`` of every row whose ``info`` block is a readable mapping.

    `carve_heldout` hands back `dict[str, object]` rows, so each `info` block reads as
    `object`. Rows carrying a malformed `info` are skipped, which is exactly what the
    `isinstance(row.get("info"), dict)` filter this replaces did.

    Returns:
        The set of row ids, with malformed-metadata rows left out.

    """
    ids: set[str] = set()
    for row in rows:
        info = row.get("info")
        if is_str_mapping(info):
            ids.add(str(info.get("id", "")))
    return ids


def test_carve_heldout_handles_invalid_metadata_and_keeps_keys_disjoint() -> None:
    rows: list[dict[str, object]] = [
        *[_row(f"vi-{index}", "vi", f"Vietnamese {index}") for index in range(4)],
        *[_row(f"en-{index}", "en", f"English {index}") for index in range(4)],
        *[_row(f"other-{index}", "fr", f"Other {index}") for index in range(4)],
        {"text": "metadata is not a mapping", "label": [], "info": "malformed"},
    ]

    train, heldout = splits.carve_heldout(rows, heldout_rows=10, seed=4)

    assert _info_ids(train).isdisjoint(_info_ids(heldout))
    train_texts = {splits.normalize_text(str(row.get("text") or "")) for row in train}
    heldout_texts = {splits.normalize_text(str(row.get("text") or "")) for row in heldout}
    assert train_texts.isdisjoint(heldout_texts)


def test_splits_cli_rejects_unavailable_label_quota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _row(
            "all-but-secret",
            "vi",
            "Every label except secret",
            labels=[{"category": label} for label in PII_LABELS if label != "secret"],
        ),
        _row("other", "en", "Other candidate"),
    ]
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, rows)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "splits",
            "--input",
            str(input_path),
            "--train-output",
            str(tmp_path / "train.jsonl"),
            "--validation-output",
            str(tmp_path / "validation.jsonl"),
            "--summary",
            str(tmp_path / "summary.json"),
            "--validation-rows",
            "1",
            "--min-docs-per-label",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="label 'secret' only has 0 candidate docs"):
        splits.main()


def test_splits_cli_writes_balanced_policy_and_zero_overlap_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = [
        _row(
            "all-labels",
            "vi",
            "One row with each supported label",
            labels=[{"category": label} for label in PII_LABELS],
        ),
        _row("vi-medical", "vi", "Vietnamese medical"),
        _row("vi-general", "vi", "Vietnamese general", domain="general"),
        _row("en-medical", "en", "English medical"),
        _row("en-general", "en", "English general", domain="general"),
        _row("other-medical", "fr", "French medical"),
        _row("other-general", "fr", "French general", domain="general"),
    ]
    _write_jsonl(input_path, rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "splits",
            "--input",
            str(input_path),
            "--train-output",
            str(train_path),
            "--validation-output",
            str(validation_path),
            "--summary",
            str(summary_path),
            "--validation-rows",
            "4",
            "--min-docs-per-label",
            "1",
            "--seed",
            "31",
        ],
    )

    splits.main()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(train_path.read_text(encoding="utf-8").splitlines()) == 3
    assert len(validation_path.read_text(encoding="utf-8").splitlines()) == 4
    assert summary["input_rows"] == 7
    assert summary["validation_policy"]["seed"] == 31
    assert summary["validation"]["label_doc_counts"] == {label: 4 if label == "human_name" else 1 for label in PII_LABELS}
    assert summary["split_validation"] == {
        "train_validation_id_overlap": 0,
        "train_validation_text_overlap": 0,
        "duplicate_train_ids": 0,
        "duplicate_validation_ids": 0,
    }
