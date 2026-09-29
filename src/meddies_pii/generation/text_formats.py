"""Canonical document surface formats for generated PII corpus records."""

from __future__ import annotations

CANONICAL_TEXT_FORMATS = (
    "MARKDOWN",
    "PLAIN_TEXT",
    "MESSY_NOTE",
    "TABULAR",
    "LIST",
    "JSON",
    "XML",
    "HL7_V2",
    "FHIR_JSON",
    "DICOM_METADATA",
)
"""The canonical surface forms a clinical document can take.

The single source of truth for the text_format taxonomy; label-corpus prose formats fold onto these via
TEXT_FORMAT_ALIASES. test_format_taxonomy guards that every prose alias maps here.

"""

TEXT_FORMAT_ALIASES = {
    "plain clinical note": "PLAIN_TEXT",
    "plain support note": "PLAIN_TEXT",
    "voice-scribe transcript": "PLAIN_TEXT",
    "clinical dialogue transcript": "PLAIN_TEXT",
    "chat transcript": "PLAIN_TEXT",
    "email thread": "PLAIN_TEXT",
    "patient portal audit log": "PLAIN_TEXT",
    "application log lines": "PLAIN_TEXT",
    "Markdown sections": "MARKDOWN",
    "support ticket with code block": "MARKDOWN",
    "messy nurse/admin note": "MESSY_NOTE",
    "OCR text with light noise": "MESSY_NOTE",
    "table-like rows": "TABULAR",
    "form fields": "LIST",
    ".env key-value block": "LIST",
    "HTTP headers": "LIST",
    "YAML-like config": "LIST",
    "JSON-like object": "JSON",
    "FHIR-like JSON": "FHIR_JSON",
    "HL7-like pipe-delimited text": "HL7_V2",
}
"""The label-corpus synthetic generator names formats in prose.

Synthetic.py TEXT_FORMATS / GENERAL_TEXT_FORMATS / CODE_LOG_TEXT_FORMATS). Fold each prose alias onto its canonical surface
form so corpus stats report one taxonomy instead of prose-vs-canonical duplicates ("Markdown sections" beside MARKDOWN).
Surface form only — document identity (email vs chat vs log) is carried by document_type, not text_format, so collapsing
the prose here loses nothing.

"""


def canonical_text_format(raw: str) -> str:
    """Normalize a text_format label to its canonical surface-form bucket.

    Canonical codes (MARKDOWN, TABULAR, …) pass through unchanged; prose aliases
    from the synthetic generator fold onto their canonical twin. Unknown strings
    (e.g. a document_type fallback) pass through so nothing is silently rewritten.

    Returns:
        The canonical bucket for a known alias, otherwise the input unchanged. Passing unknown
        strings through rather than mapping them to a default keeps an unrecognized label
        visible in the corpus, where it can be found and added, instead of being absorbed into
        a bucket it does not belong to.

    """
    return TEXT_FORMAT_ALIASES.get(raw, raw)
