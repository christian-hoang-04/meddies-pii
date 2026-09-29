"""The 1.5m release ships vi + APAC languages that intersect the 17 Meddies-supported languages.

Quotas must reach beyond the en/de/es/fr/pt set.

"""

from __future__ import annotations

from meddies_pii.training.bioes.data.augmentation import text_hash
from meddies_pii.training.bioes.data.openpii_candidates import (
    DEFAULT_OPENPII_DATASET_ID,
    DEFAULT_OPENPII_LANGUAGE_QUOTAS,
    parse_language_quotas,
    select_openpii_candidates_from_rows,
)
from meddies_pii.training.bioes.data.splits import (
    normalize_text,
)


def _row(uid: int, language: str, name: str = "Alice") -> dict[str, object]:
    text = f"{name} email user{uid}@example.com phone 555-010{uid} lives at {uid} Main St."

    def mask(label: str, value: str) -> dict[str, object]:
        start = text.index(value)
        return {
            "label": label,
            "start": start,
            "end": start + len(value),
            "value": value,
            "label_index": 0,
        }

    return {
        "uid": uid,
        "language": language,
        "split": "train",
        "source_text": text,
        "privacy_mask": [
            mask("GIVENNAME", name),
            mask("EMAIL", f"user{uid}@example.com"),
            mask("TELEPHONENUM", f"555-010{uid}"),
            mask("STREET", f"{uid} Main St"),
        ],
    }


def test_select_openpii_candidates_respects_language_quotas_and_existing_texts() -> None:
    duplicate_text = str(_row(1, "en")["source_text"])
    rows = [
        _row(1, "en"),
        _row(6, "bg", "Filip"),
        _row(2, "en", "Bob"),
        _row(3, "en", "Cara"),
        _row(4, "de", "Dora"),
        _row(5, "de", "Erik"),
    ]

    selected, summary = select_openpii_candidates_from_rows(
        rows,
        quotas={"en": 2, "de": 1},
        existing_ids=set(),
        existing_text_hashes={text_hash(normalize_text(duplicate_text))},
    )

    assert [row["info"]["language"] for row in selected] == ["en", "en", "de"]
    assert summary["language_counts"] == {"de": 1, "en": 2}
    assert summary["duplicate_text_dropped"] == 1
    assert summary["unsupported_language_dropped"] == {"bg": 1}
    assert summary["quota_shortfalls"] == {}


def test_select_openpii_candidates_drops_rows_over_max_length() -> None:
    rows = [_row(1, "en"), _row(2, "en", "Bob")]

    selected, summary = select_openpii_candidates_from_rows(
        rows,
        quotas={"en": 2},
        existing_ids=set(),
        existing_text_hashes=set(),
        token_length_fn=lambda record: 5000 if record["info"]["id"].endswith(":1") else 128,
        max_length=4096,
    )

    assert len(selected) == 1
    assert selected[0]["info"]["id"].endswith(":2")
    assert summary["dropped_over_max_length"] == 1
    assert summary["quota_shortfalls"] == {"en": 1}


def test_parse_language_quotas_defaults_and_overrides() -> None:
    assert parse_language_quotas(None)["en"] == 4000
    assert parse_language_quotas(["en=2", "de=1"]) == {"de": 1, "en": 2}


def test_default_dataset_id_targets_openpii_1_5m() -> None:
    assert DEFAULT_OPENPII_DATASET_ID == "ai4privacy/pii-masking-openpii-1.5m"


def test_default_quotas_cover_vietnamese_and_apac_languages() -> None:
    for language in ("vi", "ko", "ms", "zh", "ja", "id", "tl"):
        assert DEFAULT_OPENPII_LANGUAGE_QUOTAS.get(language, 0) > 0, language
    assert DEFAULT_OPENPII_LANGUAGE_QUOTAS["vi"] >= DEFAULT_OPENPII_LANGUAGE_QUOTAS["de"]


def test_select_openpii_candidates_fills_vietnamese_quota() -> None:
    rows = [_row(1, "vi", "Lan"), _row(2, "vi", "Minh"), _row(3, "ko", "Jin")]

    selected, summary = select_openpii_candidates_from_rows(
        rows,
        quotas={"vi": 2, "ko": 1},
        existing_ids=set(),
        existing_text_hashes=set(),
    )

    assert [row["info"]["language"] for row in selected] == ["vi", "vi", "ko"]
    assert summary["language_counts"] == {"ko": 1, "vi": 2}
    assert summary["quota_shortfalls"] == {}
