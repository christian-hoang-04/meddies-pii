"""Parse the Meddies inline ``[value]<label>`` tag format into Meddies Labels records.

The per-language ``Meddies/meddies-pii`` configs encode PII inline in the
``text`` field (e.g. ``Facture N°: [FAC-2023-88472]<id_number>``) plus rich
metadata (``document_type``, ``text_format``, ``edge_cases``). This parser strips
the tags to clean text and emits exact-offset ``{category,start,end,text}``
spans — the bracket positions are ground truth, so no find-in-text re-matching
is needed. Only the 9 PII labels are kept.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypedDict

from meddies_pii.taxonomy import PII_LABELS

if TYPE_CHECKING:
    from collections.abc import Mapping

VALID_LABELS = frozenset(PII_LABELS)


class TaggedSpan(TypedDict):
    """One exact-offset PII span carved out of inline ``[value]<label>`` markup.

    ``start`` and ``end`` index the tag-stripped clean text, so ``clean[start:end]``
    is ``text`` by construction.
    """

    category: str
    start: int
    end: int
    text: str


class ConfigRecord(TypedDict):
    """One Meddies Labels record as ``convert_config_row`` emits it.

    ``text`` is the tag-stripped source, ``label`` the exact-offset
    ``{category,start,end,text}`` spans, and ``info`` the provenance block, which
    always carries ``source``, ``language`` and ``uid`` and optionally
    ``document_type``, ``text_format`` and ``edge_cases``.
    """

    text: str
    label: list[TaggedSpan]
    info: dict[str, object]


_TAG = re.compile(r"\[([^\[\]]+)\]<([a-z_]+)>")
"""A tagged span: [value]<label>.

value holds no brackets; label is snake_case.

"""


def parse_inline_tagged(text: str) -> tuple[str, list[TaggedSpan]]:
    """Return (clean_text, spans) from inline ``[value]<label>`` markup.

    Returns:
        The text with every marker removed, and the spans whose offsets index that cleaned
        text rather than the marked-up input. Offsets are accumulated as the markers are
        stripped, so no second pass has to re-find the values -- which would mis-locate a value
        that also appears untagged elsewhere in the document.

    """
    clean_parts: list[str] = []
    spans: list[TaggedSpan] = []
    clean_len = 0
    last = 0
    for match in _TAG.finditer(text):
        before = text[last : match.start()]
        clean_parts.append(before)
        clean_len += len(before)
        value, label = match.group(1), match.group(2)
        clean_parts.append(value)
        if label in VALID_LABELS:
            spans.append({
                "category": label,
                "start": clean_len,
                "end": clean_len + len(value),
                "text": value,
            })
        clean_len += len(value)
        last = match.end()
    clean_parts.append(text[last:])
    return "".join(clean_parts), spans


def convert_config_row(
    row: Mapping[str, object],
    *,
    uid: str,
    default_language: str | None = None,
) -> ConfigRecord | None:
    """Convert one per-language config row to a Meddies Labels record (or None).

    ``default_language`` is the per-config fallback used when the row carries no
    ``language`` field — the vietnamese-translated config emits rows with
    ``language=None``, so without this the gate resolver would bucket all 10k of
    them as "unknown" and drop them from the coverage grid and language entropy.

    Returns:
        The converted record, or ``None`` when the row carries no string ``text``. Returning
        ``None`` per row rather than raising lets one malformed row be dropped from a config of
        thousands without ending the conversion.

    """
    text = row.get("text")
    if not isinstance(text, str):
        return None
    clean, spans = parse_inline_tagged(text)
    if not spans:
        return None
    info: dict[str, object] = {
        "source": "meddies-pii-hf-config",
        "language": str(row.get("language") or default_language or "unknown"),
        "uid": uid,
    }
    for key in ("document_type", "text_format"):
        value = row.get(key)
        if isinstance(value, str) and value:
            info[key] = value
    edge_cases = row.get("edge_cases")
    if isinstance(edge_cases, list):
        cases = [case for case in edge_cases if isinstance(case, str) and case]
        if cases:
            info["edge_cases"] = cases
    return {"text": clean, "label": spans, "info": info}
