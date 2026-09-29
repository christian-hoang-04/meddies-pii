from __future__ import annotations

from meddies_pii.historical_artifacts import (
    LEGACY_LABEL_POLICY,
    LEGACY_RECORD_ID_PREFIX,
    LEGACY_SYNTHETIC_DATASET_IDS,
)
from meddies_pii.training.bioes.data.augmentation import normalize_record


def test_normalize_record_preserves_generation_metadata() -> None:
    record = {
        "text": "Support case CASE-1 for Maya.",
        "label": [{"category": "human_name", "start": 24, "end": 28, "text": "Maya"}],
        "info": {
            "id": f"{LEGACY_RECORD_ID_PREFIX}en_abc",
            "source_dataset": LEGACY_SYNTHETIC_DATASET_IDS["general"],
            "source": "mimo",
            "language": "English",
            "language_bucket": "en",
            "domain_bucket": "general",
            "domain_profile": "general",
            "scenario": "general_consumer_admin_support",
            "split_purpose": "challenge",
            "text_format": "plain support note",
            "generation_model": "MiMo-V2.5-Pro",
            "label_policy": LEGACY_LABEL_POLICY,
        },
    }

    normalized = normalize_record(record, default_id="fallback")

    assert normalized["info"]["scenario"] == "general_consumer_admin_support"
    assert normalized["info"]["domain_profile"] == "general"
    assert normalized["info"]["split_purpose"] == "challenge"
    assert normalized["info"]["text_format"] == "plain support note"
    assert normalized["info"]["generation_model"] == "MiMo-V2.5-Pro"
    assert normalized["info"]["label_policy"] == LEGACY_LABEL_POLICY
