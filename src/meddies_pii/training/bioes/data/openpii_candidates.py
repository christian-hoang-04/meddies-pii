"""OpenPII candidate selection for `pii-bioes` general-domain augmentation.

The mixed bundle builder samples multiple external datasets. This module is the
source-specific adapter for the `ai4privacy/pii-masking-openpii-1m` plan: fill
explicit language quotas, convert to the Meddies Labels label-list schema, and apply
the cheap candidate gates before the shared augmentation merge path runs the
full split validation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from meddies_pii.training.bioes.data.augmentation_artifacts import text_hash
from meddies_pii.training.bioes.data.mixed import convert_ai4privacy_row
from meddies_pii.training.bioes.data.splits import normalize_text

if TYPE_CHECKING:
    from meddies_pii.training.bioes.data.record_schema import NormalizedRecord

DEFAULT_OPENPII_DATASET_ID = "ai4privacy/pii-masking-openpii-1.5m"
DEFAULT_OPENPII_LANGUAGE_QUOTAS: dict[str, int] = {
    "en": 4_000,
    "vi": 4_000,
    "de": 1_500,
    "es": 1_500,
    "fr": 1_500,
    "pt": 1_500,
    "zh": 1_500,
    "ja": 1_500,
    "ko": 1_500,
    "id": 1_500,
    "ms": 1_500,
    "tl": 1_500,
}
"""The 1.5m release ships 30 languages.

Quotas cover only those that intersect the 17 Meddies-supported languages (constants.SUPPORTED_LANGUAGES): the original
en/de/es/fr/pt plus Vietnamese and the APAC set the 1.5m adds (~20k rows each for vi/ko/ms/zh/ja/id/tl).

"""

TokenLengthFn = Callable[[Mapping[str, Any]], int]


def _normalize_quotas(quotas: Mapping[str, int]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for raw_language, raw_count in quotas.items():
        language = raw_language.strip().lower()
        count = int(raw_count)
        if not language:
            msg = "OpenPII quota language cannot be empty"
            raise ValueError(msg)
        if count < 0:
            msg = f"OpenPII quota for {language!r} must be non-negative"
            raise ValueError(msg)
        normalized[language] = count
    if not any(normalized.values()):
        msg = "At least one OpenPII language quota must be positive"
        raise ValueError(msg)
    return normalized


def parse_language_quotas(raw_quotas: list[str] | None) -> dict[str, int]:
    if not raw_quotas:
        return dict(DEFAULT_OPENPII_LANGUAGE_QUOTAS)
    quotas: dict[str, int] = {}
    for raw_quota in raw_quotas:
        if "=" not in raw_quota:
            msg = f"OpenPII quota must use LANG=COUNT format: {raw_quota!r}"
            raise ValueError(msg)
        language, count_text = raw_quota.split("=", 1)
        quotas[language.strip().lower()] = int(count_text)
    return _normalize_quotas(quotas)


# reason: select openpii coordinates normalize with values; extra seams would desync retries and counters.
def select_openpii_candidates_from_rows(  # ruff: ignore[complex-structure,too-many-arguments,too-many-locals,too-many-statements]
    rows: Iterable[Mapping[str, Any]],
    *,
    quotas: Mapping[str, int] | None = None,
    existing_ids: set[str],
    existing_text_hashes: set[str],
    dataset_id: str = DEFAULT_OPENPII_DATASET_ID,
    token_length_fn: TokenLengthFn | None = None,
    max_length: int = 4096,
    max_scan: int | None = None,
) -> tuple[list[NormalizedRecord], dict[str, object]]:
    """Select converted OpenPII rows until exact per-language quotas are filled.

    The caller owns the stream. This function is deterministic over iteration
    order but quotas prevent order bias from turning a 10k candidate set into a
    mostly-German prefix sample.

    Normalize before hashing so case/whitespace variants collapse to one key — and so this matches the normalized leakage
    hashes callers seed from `splits.normalize_text` (otherwise existing-hash dedup silently never fires).

    Returns:
        The selected records and a report of what the scan saw: rows read, per-language quota
        fill, and drop reasons. Selection stops per language once its quota is exact, so the
        result is quota-shaped rather than stream-shaped -- which is the whole point, since the
        stream's own order would otherwise hand back a mostly-German prefix.

    """
    normalized_quotas = _normalize_quotas(quotas or DEFAULT_OPENPII_LANGUAGE_QUOTAS)
    selected: list[NormalizedRecord] = []
    selected_by_language: Counter[str] = Counter()
    raw_language_counts: Counter[str] = Counter()
    unsupported_language_dropped: Counter[str] = Counter()
    dropped_external_spans: Counter[str] = Counter()
    malformed_or_unmapped_rows = 0
    duplicate_ids = 0
    duplicate_texts = 0
    dropped_over_max_length = 0
    scanned = 0

    seen_ids = set(existing_ids)
    seen_text_hashes = set(existing_text_hashes)
    target_total = sum(normalized_quotas.values())

    for row_index, row in enumerate(rows):
        if max_scan is not None and scanned >= max_scan:
            break
        if len(selected) >= target_total:
            break
        scanned += 1

        language = str(row.get("language") or "UNKNOWN").strip().lower()
        raw_language_counts[language] += 1
        if language not in normalized_quotas:
            unsupported_language_dropped[language] += 1
            continue
        if selected_by_language[language] >= normalized_quotas[language]:
            continue

        converted, dropped = convert_ai4privacy_row(
            row,
            dataset_id=dataset_id,
            default_uid=f"{dataset_id}:{row_index}",
        )
        dropped_external_spans.update(dropped)
        if converted is None:
            malformed_or_unmapped_rows += 1
            continue

        info = converted.get("info")
        record_id = str(info.get("id") if isinstance(info, Mapping) else "")
        digest = text_hash(normalize_text(str(converted.get("text") or "")))
        if record_id in seen_ids:
            duplicate_ids += 1
            continue
        if digest in seen_text_hashes:
            duplicate_texts += 1
            continue

        if token_length_fn is not None:
            token_length = int(token_length_fn(converted))
            if token_length > max_length:
                dropped_over_max_length += 1
                continue

        seen_ids.add(record_id)
        seen_text_hashes.add(digest)
        selected_by_language[language] += 1
        selected.append(converted)

    quota_shortfalls = {
        language: quota - selected_by_language.get(language, 0)
        for language, quota in sorted(normalized_quotas.items())
        if selected_by_language.get(language, 0) < quota
    }
    summary: dict[str, object] = {
        "dataset_id": dataset_id,
        "target_quotas": dict(sorted(normalized_quotas.items())),
        "scanned_rows": scanned,
        "candidate_rows": len(selected),
        "language_counts": dict(sorted(selected_by_language.items())),
        "quota_shortfalls": quota_shortfalls,
        "raw_language_counts": dict(sorted(raw_language_counts.items())),
        "unsupported_language_dropped": dict(sorted(unsupported_language_dropped.items())),
        "dropped_external_spans": dict(sorted(dropped_external_spans.items())),
        "malformed_or_unmapped_rows": malformed_or_unmapped_rows,
        "duplicate_id_dropped": duplicate_ids,
        "duplicate_text_dropped": duplicate_texts,
        "dropped_over_max_length": dropped_over_max_length,
        "max_length": max_length,
    }
    return selected, summary
