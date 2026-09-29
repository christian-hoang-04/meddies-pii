"""Acceptance-side span-quality filter.

It drops structure-capture and garbage spans
while preserving the messy-but-real ACCEPTABLE hard cases.

See `$CONTEXT/docs/data-quality-criteria.md`. The load-bearing distinction is
decoration-vs-containment: noise that *decorates* the entity is kept; scaffolding
that *contains* it (a FHIR/JSON/DICOM block with the PII as a nested field) is the
span and is dropped. Targets STRUCTURE + multiplicity, never raw length, so OCR
noise / concatenation / native scripts / JWT cookies survive.

Implements the false-reject-safe subset: V1 (struct-prefix), V2 (struct-body),
V3 (phone ceiling), V5 (email-shape), V7 (entity-presence: id/phone need a digit;
>60-char human_name spanning sentences is paragraph-capture). V8 (doc-quarantine)
lives in the assembler. Deliberately deferred as too false-reject-risky without
semantic checks: V4 date/address prose, V6 numeric-secret (would clip 6-digit
OTPs), company-name and date entity-presence (no safe surface pattern).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from meddies_pii.taxonomy import PII_LABELS, label_regex_alt

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MIN_STRUCTURE_MARKER_COUNT = 2
MIN_SENTENCE_MARKER_COUNT = 2

_STRUCT_PREFIX = re.compile(r'^\s*(?:\{|\[\s*[{"\[]|<[A-Za-z/]|"[A-Za-z_]+"\s*:|\(\d{4},\d{4}\))')
"""V1.

A span that, after left-stripping whitespace, BEGINS with a serialization token is the scaffolding, not the entity. (`{`
`[` JSON/array, `<tag` XML/HTML, `"key":` a quoted JSON key, `(dddd,dddd)` a DICOM tag tuple.).

"""

_STRUCT_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r'"(?:system|coding|sequence|code|valueString|resourceType|reference)"\s*:'),
    re.compile(r"https?://(?:terminology\.hl7\.org|hl7\.org|snomed\.info|www\.ama-assn\.org)"),
    re.compile(r"\(\d{4},\d{4}\)"),
    re.compile(r"</[A-Za-z]"),
)
"""V2 — structure embedded mid-span.

≥2 of these serialization markers means the span captured a record body even if a few prose chars precede the first brace.
Bare newlines are intentionally NOT a marker (a real multi-line address has them but none of these) — protects ACCEPTABLE
table-formatted addresses.

"""

_PHONE_MAX_CHARS = 30
"""V3 — phone is the most format-constrained label.

The longest legit international+extension is ~25 chars, so >30 is garbage (longtail bad phones were 146-735 chars). Short
extensions / redacted numbers are well under this.

"""

_EMAIL_SHAPE = re.compile(r"@|\uff20| at | dot ", re.IGNORECASE)
"""V5 — an email needs an `@` (incl.

fullwidth) or a spoken-obfuscation marker. A long span with none is prose the tagger captured, not an address.

"""
_EMAIL_MIN_PROSE = 40
_DIGIT = re.compile(r"\d")
"""V7 — entity-presence.

Id_number / phone_number MUST contain a digit (an id/phone with none is wrong-label prose). human_name has no digit rule.

"""
_SENTENCE_MARK = re.compile(r"[.!?。]")
"""V7 (name).

A >60-char human_name spanning ≥2 sentence marks is paragraph-capture or a multi-name list, not one name. The 60-char floor
protects titled names ("Dr. Smith Jr.") and genuinely long single names with no sentence punctuation.

"""
_NAME_PROSE_MIN = 60

LEADING_LABELS = PII_LABELS
"""A stray PII label marker (e.g.

`<id_number>`) glued onto the FRONT of a real value — an old-base generation artifact where the model emitted the tag
inside the value. Stripped before the structure check so V1's leading-`<` rule recovers the entity ("<id_number>MRN998231"
-> "MRN998231") instead of dropping the span.

"""
_LEADING_LABEL_MARKER = re.compile(r"^\s*</?(?:" + label_regex_alt() + r")>\s*", re.IGNORECASE)


def _strip_leading_markers(text: str) -> str:
    prev = ""
    while prev != text:
        prev = text
        text = _LEADING_LABEL_MARKER.sub("", text, count=1)
    return text


def is_structure_capture(text: str) -> bool:
    """Report whether the span is a serialization block rather than the PII entity.

    The block forms are FHIR/JSON/XML/DICOM, caught by
    V1 (leading token) or V2 (≥2 embedded markers).

    Returns:
        True when the text opens with a structure marker, or carries at least the minimum
        count of embedded ones. Either alone is enough; the two rules catch a block that
        starts as structure and one that is structure throughout.

    """
    if _STRUCT_PREFIX.match(text):
        return True
    return sum(1 for marker in _STRUCT_MARKERS if marker.search(text)) >= MIN_STRUCTURE_MARKER_COUNT


# reason: bad span reason's exits distinguish V1_struct_prefix; merging them would blur decision order.
def bad_span_reason(label: str, text: str) -> str | None:  # ruff: ignore[too-many-return-statements]
    """Name which validator drops this span, or None when none does.

    The false-reject-safe
    subset: V1/V2 structure-capture, V3 phone ceiling, V5 email-shape, V7
    entity-presence (id/phone need a digit) + name-paragraph. Deferred as too
    false-reject-risky without semantic checks: V4 date/address prose, V6
    numeric-secret (would clip 6-digit OTPs), company/date entity-presence.

    Returns:
        The identifier of the first rule that fires, or None when the span survives every
        one. The order is the contract: structure capture is tested before the per-label
        rules, so a serialization block is reported as V1 or V2 rather than as whatever
        label-specific rule it would also have tripped.

    """
    text = _strip_leading_markers(text)
    if _STRUCT_PREFIX.match(text):
        return "V1_struct_prefix"
    if sum(1 for marker in _STRUCT_MARKERS if marker.search(text)) >= MIN_STRUCTURE_MARKER_COUNT:
        return "V2_struct_body"
    if label == "phone_number":
        if len(text) > _PHONE_MAX_CHARS:
            return "V3_phone_too_long"
        if not _DIGIT.search(text):
            return "V7_no_digit"
    if label == "id_number" and not _DIGIT.search(text):
        return "V7_no_digit"
    if label == "email_address" and len(text) > _EMAIL_MIN_PROSE and not _EMAIL_SHAPE.search(text):
        return "V5_email_no_at"
    if (
        label == "human_name"
        and len(text) > _NAME_PROSE_MIN
        and len(_SENTENCE_MARK.findall(text)) >= MIN_SENTENCE_MARKER_COUNT
    ):
        return "V7_name_paragraph"
    return None


def is_bad_span(label: str, text: str) -> bool:
    """Report a span that must not be accepted into the corpus.

    False for every
    ACCEPTABLE hard case. See :func:`bad_span_reason` for the rule that fires.

    Returns:
        True when some rule drops the span. This is the boolean face of
        ``bad_span_reason``; call that one where the reason has to be recorded.

    """
    return bad_span_reason(label, text) is not None


def clean_spans(
    spans: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], int]:
    """Drop bad spans from a record's span list.

    Returns (kept, dropped_count).
    Accepts both span key shapes (``category`` and ``label``). A span with a stray
    leading ``<label>`` marker is CLEANED (marker stripped, ``start`` advanced) and
    kept, not dropped — the entity behind the marker is real.

    Returns:
        The kept spans and how many were dropped. A span whose label or text is not a string
        is kept unexamined rather than dropped, so this function never removes a span it
        could not read.

    """
    kept: list[Mapping[str, object]] = []
    dropped = 0
    for span in spans:
        label = span.get("category")
        if not isinstance(label, str):
            label = span.get("label")
        text = span.get("text")
        if not (isinstance(label, str) and isinstance(text, str)):
            kept.append(span)
            continue
        if is_bad_span(label, text):
            dropped += 1
            continue
        stripped = _strip_leading_markers(text)
        if stripped != text:
            cleaned = dict(span)
            cleaned["text"] = stripped
            start = span.get("start")
            if isinstance(start, int):
                cleaned["start"] = start + (len(text) - len(stripped))
            kept.append(cleaned)
        else:
            kept.append(span)
    return kept, dropped
