from __future__ import annotations

import pytest

from meddies_pii.historical_artifacts import (
    LEGACY_ADVERSARIAL_SOURCE,
    LEGACY_LABEL_POLICY,
    LEGACY_RECORD_ID_PREFIX,
    LEGACY_REPAIRED_REJECTS_DATASET_ID,
    LEGACY_SYNTHETIC_DATASET_IDS,
    legacy_audit_jsonl_locator,
    legacy_jsonl_locator,
    legacy_summary_json_locator,
)


def test_historical_wire_values_remain_exact() -> None:
    assert LEGACY_LABEL_POLICY == "meddies9_bioes"
    assert LEGACY_RECORD_ID_PREFIX == "mimo_meddies9_"
    assert LEGACY_ADVERSARIAL_SOURCE == "meddies9.synthetic_adversarial"
    assert LEGACY_REPAIRED_REJECTS_DATASET_ID == "Meddies/mimo-meddies9-repaired-rejects"
    assert LEGACY_SYNTHETIC_DATASET_IDS == {
        "medical": "Meddies/mimo-meddies9-targeted-synthetic",
        "general": "Meddies/mimo-meddies9-general-synthetic",
        "code_logs": "Meddies/mimo-meddies9-code-log-synthetic",
        "mixed": "Meddies/mimo-meddies9-general-synthetic",
        "challenge": "Meddies/mimo-meddies9-adversarial-challenge",
    }


def test_historical_artifact_locators_remain_exact() -> None:
    assert legacy_jsonl_locator("train") == "train.meddies9.jsonl"
    assert legacy_audit_jsonl_locator("train") == "train.meddies9.audit.jsonl"
    assert legacy_summary_json_locator("train") == "train.meddies9.summary.json"


def test_historical_dataset_identifiers_are_read_only() -> None:
    with pytest.raises(TypeError):
        # reason: the rejected write is the behaviour under test — this must stay a type error, and an
        # reason: annotation admitting it would delete the very property the assertion proves.
        LEGACY_SYNTHETIC_DATASET_IDS["new"] = "Meddies/new"  # ty: ignore[invalid-assignment]
