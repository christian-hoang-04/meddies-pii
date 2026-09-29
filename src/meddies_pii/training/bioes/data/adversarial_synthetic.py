"""Deterministic Meddies Labels adversarial examples for audit/smoke datasets."""

from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: this module's confusables ARE its subject matter — the adversarial PII corpus names the
# reason: evasion it must generate, so normalising them to ASCII would delete the technique itself.
# ruff: file-ignore[hardcoded-password-string]
# reason: hand-written synthetic training text whose POINT is to look like an obfuscated secret, so
# reason: the BIOES model learns to tag the `secret` label. Both strings are fabricated and
# reason: deliberately spaced; neither is live and no service would accept either.
from typing import TYPE_CHECKING

from meddies_pii.historical_artifacts import LEGACY_ADVERSARIAL_SOURCE

if TYPE_CHECKING:
    from collections.abc import Sequence


def _span(text: str, value: str, label: str) -> dict[str, object]:
    start = text.index(value)
    return {
        "category": label,
        "start": start,
        "end": start + len(value),
        "text": value,
    }


def _record(
    example_id: str,
    text: str,
    spans: Sequence[tuple[str, str]] = (),
) -> dict[str, object]:
    return {
        "text": text,
        "label": [_span(text, value, label) for value, label in spans],
        "info": {"id": example_id, "source": LEGACY_ADVERSARIAL_SOURCE},
    }


def build_adversarial_pii_label_examples() -> list[dict[str, object]]:
    """Return a small, hand-checkable set of Meddies Labels edge cases.

    These examples are not meant to replace the real data audit. They cover
    OPF-style weak slices that were sparse in the current Meddies Labels data and are
    suitable for smoke training or a held-out adversarial eval file.

    Returns:
        The hand-written examples as Meddies Labels records. They are constructed rather than
        sampled, so the set is stable across runs and small enough for a person to check --
        which is the point: it covers the obfuscation shapes the real corpus is thin on.

    """
    at_dot = "mai.nguyen [at] hospital [dot] vn"
    split_url = "https://portal.hospital.vn/patients/123/\nresults?token=abc"
    phonetic_secret = "alpha bravo charlie one two three"
    symbol_phone = "+84 912–345–678"
    emoji_address = "482 Maple 🛣️ Apt 3B"
    spaced_secret = "sk_live_ 7Qz . P9x . 2Kj"

    return [
        _record(
            "synthetic-at-dot-email",
            f"Contact the care team at {at_dot} for discharge planning.",
            ((at_dot, "email_address"),),
        ),
        _record(
            "synthetic-linebreak-private-url",
            f"Patient result link: {split_url}",
            ((split_url, "private_url"),),
        ),
        _record(
            "synthetic-phonetic-secret",
            f"Portal reset phrase is {phonetic_secret}.",
            ((phonetic_secret, "secret"),),
        ),
        _record(
            "synthetic-symbol-phone",
            f"Emergency contact phone: {symbol_phone}.",
            ((symbol_phone, "phone_number"),),
        ),
        _record(
            "synthetic-emoji-address",
            f"Home address: {emoji_address}.",
            ((emoji_address, "address"),),
        ),
        _record(
            "synthetic-spacing-secret",
            f"API key for portal integration: {spaced_secret}",
            ((spaced_secret, "secret"),),
        ),
        _record(
            "synthetic-public-url-date-negative",
            "Guideline published on 2024-01-15 at https://moh.gov.vn/guidelines/diabetes.",
        ),
    ]
