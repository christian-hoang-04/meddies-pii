"""Code-owned trust registry for regex shipping evidence.

Descriptive reports may use any content-addressed corpus. Shipping eligibility is
stricter: only an exact entry in this reviewed registry can count as blinded
confirmation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from meddies_pii.evaluation.identity import canonical_sha256

MEDDIES_PII_V2_DATASET = "Meddies/meddies-pii-v2"
MEDDIES_PII_V2_EXPOSED_REVISION = "28aaef5dffd36aabead650c74658a6f814eb4db0"
MEDDIES_PII_V2_EXPOSED_SHARDS = ("eval", "eval-challenge")


@dataclass(frozen=True, slots=True)
class ApprovedBlindedShippingCorpus:
    """One exact corpus approved by review as blinded shipping evidence."""

    dataset_id: str
    revision: str
    shard: str
    fixture_sha256: str
    source_rows_sha256: str
    approval_audit_sha256: str


APPROVED_BLINDED_SHIPPING_CORPORA: tuple[ApprovedBlindedShippingCorpus, ...] = ()
"""Intentionally empty. Adding a corpus requires a reviewed source change and a corresponding update to
TRUSTED_CORPUS_REGISTRY_SHA256 below.
"""


def trusted_corpus_registry_payload() -> dict[str, object]:
    """Return the canonical payload whose digest is pinned in this module.

    Returns:
        The canonical trusted-corpus registry payload.

    """
    return {
        "schema_version": 1,
        "known_exposed": [
            {
                "dataset_id": MEDDIES_PII_V2_DATASET,
                "revision": MEDDIES_PII_V2_EXPOSED_REVISION,
                "shards": list(MEDDIES_PII_V2_EXPOSED_SHARDS),
                "status": "historically_exposed",
            },
        ],
        "approved_blinded_shipping": [
            {
                "dataset_id": corpus.dataset_id,
                "revision": corpus.revision,
                "shard": corpus.shard,
                "fixture_sha256": corpus.fixture_sha256,
                "source_rows_sha256": corpus.source_rows_sha256,
                "blinded": True,
                "approval_audit_sha256": corpus.approval_audit_sha256,
            }
            for corpus in APPROVED_BLINDED_SHIPPING_CORPORA
        ],
    }


TRUSTED_CORPUS_REGISTRY_SHA256 = "36a62ae9d535838807fac5234cc92dcc977381b1db5eb626b1da2e8540f33060"

if canonical_sha256(trusted_corpus_registry_payload()) != TRUSTED_CORPUS_REGISTRY_SHA256:
    msg = "trusted corpus registry changed without updating its pinned SHA-256"
    raise RuntimeError(msg)


def is_known_exposed(*, dataset_id: str, revision: str, shard: str) -> bool:
    """Return whether the exact dataset revision and shard are known exposed.

    Returns:
        `True` when the dataset revision and shard match a known exposed entry.

    """
    return (
        dataset_id == MEDDIES_PII_V2_DATASET
        and revision == MEDDIES_PII_V2_EXPOSED_REVISION
        and shard in MEDDIES_PII_V2_EXPOSED_SHARDS
    )


# reason: the shipping gate must show every reviewed corpus-identity atom; bundling would hide an omitted field.
def matches_approved_blinded_shipping_corpus(  # ruff: ignore[too-many-arguments]
    *,
    dataset_id: str,
    revision: str,
    shard: str,
    fixture_sha256: str,
    source_rows_sha256: str,
    approval_audit_sha256: str | None,
) -> bool:
    """Require an exact match to one reviewed blinded-corpus registry entry.

    Returns:
        `True` only for an exact approved blinded shipping-corpus identity.

    """
    if approval_audit_sha256 is None:
        return False
    return any(
        corpus.dataset_id == dataset_id
        and corpus.revision == revision
        and corpus.shard == shard
        and corpus.fixture_sha256 == fixture_sha256
        and corpus.source_rows_sha256 == source_rows_sha256
        and corpus.approval_audit_sha256 == approval_audit_sha256
        for corpus in APPROVED_BLINDED_SHIPPING_CORPORA
    )
