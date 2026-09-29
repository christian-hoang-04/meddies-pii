"""Pin manifest shape after retiring the OPF-era ``label_space`` field.

The ``label_space`` parameter was dead config: write-only, hardcoded to
``"meddies"`` at every entry point, threaded through 6 modules without ever
being read. Refactor #6 PR 1 removes it end-to-end and bumps
``MANIFEST_SCHEMA_VERSION`` from 1 to 2 to mark the schema event.

These tests pin the post-removal contract so a future addition can't
silently re-introduce the field.
"""

from __future__ import annotations

from typing import Any

import pytest

from meddies_pii.training.bioes.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    WorkloadCandidate,
    build_workload_manifest,
)


def _candidate(uid: str = "row-1", token_length: int = 8) -> WorkloadCandidate:
    return WorkloadCandidate(
        uid=uid,
        row_hash="0" * 40,
        raw_chars=10,
        token_length=token_length,
    )


def _manifest_kwargs() -> dict[str, Any]:
    return {
        "name": "test-manifest",
        "dataset_id": "Meddies/meddies-pii",
        "dataset_config": "train",
        "dataset_split": "train",
        "dataset_revision": "abc",
        "model_id": "LiquidAI/LFM2.5-350M-Base",
        "model_revision": "def",
        "tokenizer_id": "LiquidAI/LFM2.5-350M-Base",
        "tokenizer_revision": "def",
        "max_length": 4096,
        "fixed_pad_length": 4096,
        "selection_policy": "tokenized_longest_first_full_scan",
        "required_examples": 1,
        "candidates": [_candidate()],
        "created_at_utc": "2026-01-01T00:00:00Z",
    }


def test_manifest_schema_version_is_two() -> None:
    assert MANIFEST_SCHEMA_VERSION == 2


def test_build_workload_manifest_rejects_label_space_kwarg() -> None:
    with pytest.raises(TypeError, match="label_space"):
        # reason: the unknown keyword is the behaviour under test — the manifest builder must refuse a
        # reason: `label_space` kwarg, so ty reporting it as unknown is the assertion agreeing.
        build_workload_manifest(label_space="meddies", **_manifest_kwargs())  # ty: ignore[unknown-argument]


def test_built_manifest_has_no_label_space_key() -> None:
    manifest = build_workload_manifest(**_manifest_kwargs())
    assert "label_space" not in manifest
    assert manifest["schema_version"] == 2
