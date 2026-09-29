"""Stable public imports for domain primitives owned by focused modules."""

from __future__ import annotations

from meddies_pii.annotations.label_aliases import INVALID_LABELS, LABEL_MAP
from meddies_pii.generation.prompts import (
    EXTRACTION_PROMPT,
    REVIEW_PROMPT,
    SYSTEM_PROMPT,
)
from meddies_pii.generation.text_formats import (
    CANONICAL_TEXT_FORMATS,
    TEXT_FORMAT_ALIASES,
    canonical_text_format,
)
from meddies_pii.languages import SUPPORTED_LANGUAGES
from meddies_pii.tags import TAG_PATTERN
from meddies_pii.taxonomy import (
    PII_LABEL_SET,
    PII_LABELS,
    PII_LABELS_BRACKETED,
    VALID_LABELS,
    VALID_LABELS_BRACKETED,
)

__all__ = (
    "CANONICAL_TEXT_FORMATS",
    "EXTRACTION_PROMPT",
    "INVALID_LABELS",
    "LABEL_MAP",
    "PII_LABELS",
    "PII_LABELS_BRACKETED",
    "PII_LABEL_SET",
    "REVIEW_PROMPT",
    "SUPPORTED_LANGUAGES",
    "SYSTEM_PROMPT",
    "TAG_PATTERN",
    "TEXT_FORMAT_ALIASES",
    "VALID_LABELS",
    "VALID_LABELS_BRACKETED",
    "canonical_text_format",
)
