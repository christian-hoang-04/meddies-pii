"""Corpus manifest writer — round-trip, stable hashing, drift detection.

ADR 0008 §6: the bioes-v2 MANIFEST must fail *loud* on drift. A lean manifest
fails silent (a stable external file mutates, the build still "reproduces" from
a now-different source). The richer manifest records ``sha256_16`` on every
stable external file plus a ``mix_recipe`` block, so ``verify_manifest`` can
recompute the hashes and surface any mismatch.

These tests pin four properties:
- write -> read round-trips (the dict written equals the dict returned)
- ``sha256_16`` is stable across calls and is exactly 16 hex chars
- ``verify_manifest`` returns ``[]`` on an untouched corpus
- mutating a hashed file makes ``verify_manifest`` name that file (fail-loud)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from meddies_pii.training.bioes.data.corpus_manifest import (
    sha256_16,
    verify_manifest,
    write_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_dataset_file(corpus_root: Path, rel: str, body: str) -> Path:
    path = corpus_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_sha256_16_is_stable_and_16_hex_chars(tmp_path: Path) -> None:
    path = tmp_path / "external.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")

    first = sha256_16(path)
    second = sha256_16(path)

    assert first == second
    assert len(first) == 16
    assert all(c in "0123456789abcdef" for c in first)


def test_sha256_16_differs_on_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("alpha\n", encoding="utf-8")
    b.write_text("beta\n", encoding="utf-8")

    assert sha256_16(a) != sha256_16(b)


def test_write_manifest_round_trips(tmp_path: Path) -> None:
    _write_dataset_file(tmp_path, "external/nvidia-health.jsonl", '{"x": 1}\n')
    datasets = [
        {
            "id": "nvidia_health",
            "role": "external-source",
            "current_path": "external/nvidia-health.jsonl",
            "hash_file": True,
        },
    ]

    returned = write_manifest(tmp_path, datasets=datasets)

    on_disk = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert on_disk == returned
    assert on_disk["datasets"][0]["id"] == "nvidia_health"
    assert "schema" in on_disk
    assert "generated" in on_disk
    assert on_disk["labels_valid"] == [
        "address",
        "company_name",
        "date",
        "email_address",
        "human_name",
        "id_number",
        "phone_number",
        "private_url",
        "secret",
    ]


def test_write_manifest_hashes_stable_external_files(tmp_path: Path) -> None:
    target = _write_dataset_file(tmp_path, "external/nemotron-health.jsonl", '{"row": 1}\n')
    datasets = [
        {
            "id": "nemotron_health",
            "role": "external-source",
            "current_path": "external/nemotron-health.jsonl",
            "hash_file": True,
        },
    ]

    manifest = write_manifest(tmp_path, datasets=datasets)

    assert manifest["datasets"][0]["sha256_16"] == sha256_16(target)


def test_write_manifest_carries_generation_run_pointer_on_synthetic(
    tmp_path: Path,
) -> None:
    _write_dataset_file(tmp_path, "synthetic/opencode_zen/accepted.vi.jsonl", '{"row": 1}\n')
    datasets = [
        {
            "id": "opencode_zen_daily",
            "role": "synthetic-bulk",
            "current_path": "synthetic/opencode_zen/accepted.vi.jsonl",
            "generation_run": "daily_runs/2026-06-11.summary.json",
        },
    ]

    manifest = write_manifest(tmp_path, datasets=datasets)

    entry = manifest["datasets"][0]
    assert entry["generation_run"] == "daily_runs/2026-06-11.summary.json"
    assert entry["sha256_16"] is None


def test_write_manifest_records_mix_recipe_block(tmp_path: Path) -> None:
    _write_dataset_file(tmp_path, "external/a.jsonl", "x\n")
    recipe = {
        "ratios": {"vi": 0.30, "en": 0.30, "other": 0.40},
        "seed": 1234,
        "dedup_policy": "text_hash(normalize_text(text))",
        "audit_report": "reports/2026-06-11-assembly-audit.json",
    }

    manifest = write_manifest(
        tmp_path,
        datasets=[
            {
                "id": "a",
                "role": "external-source",
                "current_path": "external/a.jsonl",
                "hash_file": True,
            },
        ],
        mix_recipe=recipe,
    )

    assert manifest["mix_recipe"] == recipe


def test_verify_manifest_clean_on_untouched_corpus(tmp_path: Path) -> None:
    _write_dataset_file(tmp_path, "external/a.jsonl", '{"row": 1}\n')
    _write_dataset_file(tmp_path, "external/b.jsonl", '{"row": 2}\n')
    write_manifest(
        tmp_path,
        datasets=[
            {
                "id": "a",
                "role": "external-source",
                "current_path": "external/a.jsonl",
                "hash_file": True,
            },
            {
                "id": "b",
                "role": "external-source",
                "current_path": "external/b.jsonl",
                "hash_file": True,
            },
        ],
    )

    assert verify_manifest(tmp_path) == []


def test_verify_manifest_reports_mutated_file(tmp_path: Path) -> None:
    target = _write_dataset_file(tmp_path, "external/a.jsonl", '{"row": 1}\n')
    write_manifest(
        tmp_path,
        datasets=[
            {
                "id": "a",
                "role": "external-source",
                "current_path": "external/a.jsonl",
                "hash_file": True,
            },
        ],
    )

    target.write_text('{"row": 999}\n', encoding="utf-8")

    findings = verify_manifest(tmp_path)

    assert len(findings) == 1
    assert "external/a.jsonl" in findings[0]


def test_verify_manifest_reports_missing_hashed_file(tmp_path: Path) -> None:
    target = _write_dataset_file(tmp_path, "external/a.jsonl", '{"row": 1}\n')
    write_manifest(
        tmp_path,
        datasets=[
            {
                "id": "a",
                "role": "external-source",
                "current_path": "external/a.jsonl",
                "hash_file": True,
            },
        ],
    )

    target.unlink()

    findings = verify_manifest(tmp_path)

    assert len(findings) == 1
    assert "external/a.jsonl" in findings[0]


def test_verify_manifest_ignores_unhashed_entries(tmp_path: Path) -> None:
    _write_dataset_file(tmp_path, "synthetic/opencode_zen/accepted.vi.jsonl", '{"row": 1}\n')
    write_manifest(
        tmp_path,
        datasets=[
            {
                "id": "opencode_zen_daily",
                "role": "synthetic-bulk",
                "current_path": "synthetic/opencode_zen/accepted.vi.jsonl",
                "generation_run": "daily_runs/2026-06-11.summary.json",
            },
        ],
    )

    assert verify_manifest(tmp_path) == []
