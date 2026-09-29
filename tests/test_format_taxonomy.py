"""text_format taxonomy normalizer: prose format aliases fold onto the canonical surface-form buckets.

Folding them is what makes corpus stats report one taxonomy rather than prose-vs-canonical
duplicates ("Markdown sections" alongside MARKDOWN).

The drift guard is load-bearing: every prose format the pii_label generator can
emit must fold onto a canonical bucket — an unmapped alias would silently leak as
its own bucket in the data-shape report.

a document_type fallback that isn't a format (e.g. "discharge summary") must not be silently rewritten — only known prose
aliases fold.

"""

from __future__ import annotations

from meddies_pii.generation.label_corpus.catalog import (
    CODE_LOG_TEXT_FORMATS,
    GENERAL_TEXT_FORMATS,
    TEXT_FORMATS,
)
from meddies_pii.generation.text_formats import (
    CANONICAL_TEXT_FORMATS,
    TEXT_FORMAT_ALIASES,
    canonical_text_format,
)


def test_every_alias_targets_a_canonical_bucket() -> None:
    assert set(TEXT_FORMAT_ALIASES.values()) <= set(CANONICAL_TEXT_FORMATS)


def test_every_prose_format_has_a_mapping() -> None:
    prose = set(TEXT_FORMATS) | set(GENERAL_TEXT_FORMATS) | set(CODE_LOG_TEXT_FORMATS)
    for fmt in prose:
        folded = canonical_text_format(fmt)
        assert folded in CANONICAL_TEXT_FORMATS, f"{fmt!r} -> {folded!r} not canonical"


def test_canonical_codes_pass_through_unchanged() -> None:
    for code in CANONICAL_TEXT_FORMATS:
        assert canonical_text_format(code) == code


def test_known_duplicates_merge_to_their_twin() -> None:
    assert canonical_text_format("Markdown sections") == "MARKDOWN"
    assert canonical_text_format("table-like rows") == "TABULAR"
    assert canonical_text_format("messy nurse/admin note") == "MESSY_NOTE"


def test_unknown_label_passes_through() -> None:
    assert canonical_text_format("discharge summary") == "discharge summary"
