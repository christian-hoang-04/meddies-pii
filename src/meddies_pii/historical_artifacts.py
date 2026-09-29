"""Immutable identifiers retained for published Meddies Labels artifacts.

These values are historical wire contracts. New code must import them instead of
spelling the former project shorthand at ordinary call sites.

Published files retain this locator token in their names.

"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

# reason: a dataset-naming version token, not a credential: it builds HuggingFace dataset ids such
# reason: as `Meddies/mimo-meddies9-targeted-synthetic` and filename stems such as `*.meddies9.jsonl`.
LEGACY_ARTIFACT_TOKEN: Final = "meddies9"  # ruff: ignore[hardcoded-password-string]
"""Published row metadata and row IDs.

Changing either breaks existing records.

"""
LEGACY_LABEL_POLICY: Final = f"{LEGACY_ARTIFACT_TOKEN}_bioes"
LEGACY_RECORD_ID_PREFIX: Final = f"mimo_{LEGACY_ARTIFACT_TOKEN}_"
LEGACY_ADVERSARIAL_SOURCE: Final = f"{LEGACY_ARTIFACT_TOKEN}.synthetic_adversarial"
LEGACY_REPAIRED_REJECTS_DATASET_ID: Final = f"Meddies/mimo-{LEGACY_ARTIFACT_TOKEN}-repaired-rejects"
"""Published dataset metadata keeps these repository IDs unchanged."""
LEGACY_SYNTHETIC_DATASET_IDS: Final[Mapping[str, str]] = MappingProxyType({
    "medical": f"Meddies/mimo-{LEGACY_ARTIFACT_TOKEN}-targeted-synthetic",
    "general": f"Meddies/mimo-{LEGACY_ARTIFACT_TOKEN}-general-synthetic",
    "code_logs": f"Meddies/mimo-{LEGACY_ARTIFACT_TOKEN}-code-log-synthetic",
    "mixed": f"Meddies/mimo-{LEGACY_ARTIFACT_TOKEN}-general-synthetic",
    "challenge": f"Meddies/mimo-{LEGACY_ARTIFACT_TOKEN}-adversarial-challenge",
})


def legacy_jsonl_locator(stem: str) -> str:
    """Return the immutable JSONL locator for a published PII-label artifact.

    Returns:
        The filename under the frozen legacy token. The token is fixed rather than derived from
        the current naming, so a locator for an already-published artifact keeps resolving
        after the project's naming moves on.

    """
    return f"{stem}.{LEGACY_ARTIFACT_TOKEN}.jsonl"


def legacy_audit_jsonl_locator(stem: str) -> str:
    """Return the immutable migration-audit locator for a published artifact.

    Returns:
        The audit filename beside the artifact's own, under the same frozen token. The audit
        sits next to the data it explains, so a published artifact and the record of how it was
        migrated cannot drift apart.

    """
    return f"{stem}.{LEGACY_ARTIFACT_TOKEN}.audit.jsonl"


def legacy_summary_json_locator(stem: str) -> str:
    """Return the immutable summary locator for a published artifact.

    Returns:
        The summary filename under the same frozen token, completing the three-file set every
        published artifact carries: records, per-row audit, and summary.

    """
    return f"{stem}.{LEGACY_ARTIFACT_TOKEN}.summary.json"
