"""Accept/reject gate for the Meddies Labels synthetic generator.

Parses a generated document and validates tag well-formedness, span offsets,
length, phone shape, and native-script presence, then builds the accepted record.
The single definition of "is this document good enough to keep". The catalog of
what to generate lives in catalog.py; prompt phrasing in prompts.py.
"""

from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: these characters sit inside patterns that must MATCH them — the symbol-substitution and
# reason: separator detectors exist to catch confusable punctuation, so normalising one to ASCII
# reason: would silently stop this module detecting that evasion.
import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict

from meddies_pii.annotations.tagged_text import parse_tagged_text
from meddies_pii.historical_artifacts import (
    LEGACY_LABEL_POLICY,
    LEGACY_RECORD_ID_PREFIX,
    LEGACY_SYNTHETIC_DATASET_IDS,
)
from meddies_pii.languages import normalize_language
from meddies_pii.taxonomy import (
    PiiLabel,
    is_pii_label,
    label_regex_alt,
    require_pii_label,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.generation.label_corpus.catalog import Scenario
    from meddies_pii.spans import CharSpan

MIN_PHONE_DIGITS = 6
MAX_DOCUMENT_CHARACTERS = 16000

_TAG_FRAGMENT_RE = re.compile(r"\[[^\[\]]{1,500}\]<[a-zA-Z][a-zA-Z0-9_-]*>")
_LEFTOVER_LABEL_RE = re.compile(
    r"</?(?:" + label_regex_alt() + r")>",
    re.IGNORECASE,
)
_MALFORMED_LEFTOVER_LABEL_RE = re.compile(
    r"(?:\]<|</?(?:" + label_regex_alt() + r")(?:\]|\s|$))",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```", re.DOTALL)
_META_PREFIX_RE = re.compile(
    r"^(?:here is|below is|sure[,! ]|i will|i can|as an ai|document type:|technical requirements:)",
    re.IGNORECASE,
)


_VIETNAMESE_MARKER_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)

_SCRIPT_MARKER_RES: dict[str, re.Pattern[str]] = {
    "vi": _VIETNAMESE_MARKER_RE,
    "th": re.compile(r"[\u0e00-\u0e7f]"),
    "lo": re.compile(r"[\u0e80-\u0eff]"),
    "my": re.compile(r"[\u1000-\u109f]"),
    "ta": re.compile(r"[\u0b80-\u0bff]"),
    "zh": re.compile(r"[\u4e00-\u9fff]"),
    "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]"),
    "ko": re.compile(r"[\uac00-\ud7af]"),
    "ru": re.compile(r"[\u0400-\u04ff]"),
}
"""Per-language native-script markers.

Unicode ranges verified against actual code points: Thai U+0E00-0E7F, Lao U+0E80-0EFF, Myanmar U+1000-109F, Tamil
U+0B80-0BFF, CJK Han U+4E00-9FFF, Hiragana U+3040-309F + Katakana U+30A0-30FF, Hangul U+AC00-D7AF, Cyrillic U+0400-04FF.
Latin-script languages have no strict marker (None) since the document is already Latin.

"""


_DEFAULT_MIN_RAW_LENGTH = 240
"""CJK scripts pack far more meaning per character, so 240 chars is a long document; gate them lower.

Everything else keeps the default.

"""
_CJK_MIN_RAW_LENGTH = 120
_DENSE_SCRIPT_CODES: frozenset[str] = frozenset({"zh", "ja", "ko"})


PHONE_WINDOW_CHARS = set("0123456789 +().-–—\n\t")
_MIN_PHONE_DIGITS = 7
ACCEPT_MIN_SPANS = 2
"""Acceptance is correctness-only: keep any well-formed doc with at least this many valid spans.

Span COUNT is a prompt-time target (catalog.sample_span_target), not an acceptance filter.

"""
_NON_FATAL_ERRORS = frozenset({"short_phone_span", "underbounded_phone_span"})
"""Defects that should NOT reject an otherwise-rich doc.

They describe one imperfect span, not a broken document. A short phone is usually a real edge case (extension / partial /
redacted number) we WANT, and phone F1 is already the strongest label, so dropping a 19-span doc over one short phone is
pure loss. The span is kept; the issue is recorded for telemetry only.

"""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    raw_text: str = ""
    spans: tuple[CharSpan, ...] = ()
    errors: tuple[str, ...] = ()
    had_label_repairs: bool = False


class AcceptedLabel(TypedDict):
    category: PiiLabel
    start: int
    end: int
    text: str


class AcceptedRecordInfo(TypedDict):
    id: str
    source_dataset: str
    source: str
    generation_model: str
    language: str
    language_bucket: str
    domain_bucket: str
    domain_profile: str
    document_type: str
    text_format: str
    scenario: str
    split_purpose: str
    attempt_index: int
    label_policy: str
    had_label_repairs: bool
    edge_cases: list[str]
    repair_applied: NotRequired[bool]
    repair_changed: NotRequired[bool]
    original_errors: NotRequired[list[str]]


class AcceptedRecord(TypedDict):
    text: str
    label: list[AcceptedLabel]
    info: AcceptedRecordInfo


def _marker_re_for(code: str) -> re.Pattern[str] | None:
    """Native-script QA marker for a language code, or None for Latin scripts.

    Returns:
        The compiled marker pattern for a language written in a non-Latin script, or ``None``
        when the code has no entry. ``None`` means the script check does not apply rather than
        that it failed, so a Latin-script language passes without needing a marker.

    """
    return _SCRIPT_MARKER_RES.get(code.strip().lower())


def _min_raw_length_for(code: str) -> int:
    """Minimum accepted raw-text length; lower for dense CJK scripts.

    Returns:
        The lower CJK threshold for a dense script, otherwise the default. A CJK document
        carries far more content per character, so holding it to the Latin length would reject
        documents that are in fact substantial.

    """
    if code.strip().lower() in _DENSE_SCRIPT_CODES:
        return _CJK_MIN_RAW_LENGTH
    return _DEFAULT_MIN_RAW_LENGTH


def digit_count(value: str) -> int:
    return sum(ch.isdigit() for ch in value)


def expanded_phone_window(text: str, start: int, end: int) -> str:
    left = start
    while left > 0 and text[left - 1] in PHONE_WINDOW_CHARS:
        left -= 1
    right = end
    while right < len(text) and text[right] in PHONE_WINDOW_CHARS:
        right += 1
    return text[left:right].strip()


def _is_templated_phone(value: str) -> bool:
    """Report a lazy digit dump rather than a plausible number.

    An ordered run (0123456789) or a
    single repeated digit. Produced under span-count pressure; reject it.

    isdecimal(), not isdigit(): isdigit() admits subscript/superscript/circled glyphs ('₈', '⁹', '①') that int() then
    rejects with ValueError. The EDGE_CASES axis emits exactly those as obfuscation; they are not lazy templated dumps.

    Returns:
        ``True`` for a single repeated digit or a run ascending or descending by one with
        wraparound, ``False`` otherwise -- including for anything shorter than the phone-digit
        minimum, which is too short to judge. Both run tests use modulo 10, so ``8901`` counts
        as ascending; a real number rarely does that across its whole length.

    """
    digits = [int(c) for c in value if c.isdecimal()]
    if len(digits) < MIN_PHONE_DIGITS:
        return False
    if len(set(digits)) == 1:
        return True
    ascending = all((digits[i] + 1) % 10 == digits[i + 1] for i in range(len(digits) - 1))
    descending = all((digits[i] - 1) % 10 == digits[i + 1] for i in range(len(digits) - 1))
    return ascending or descending


def _strip_wrappers(content: str) -> str:
    cleaned = _THINK_RE.sub("", content).strip()
    fence = _CODE_FENCE_RE.search(cleaned)
    if fence and "]<" in fence.group(1):
        cleaned = fence.group(1).strip()
    lines = cleaned.splitlines()
    while lines and _META_PREFIX_RE.search(lines[0].strip()) and "]<" not in lines[0]:
        lines.pop(0)
    return "\n".join(lines).strip()


# reason: tagged text and strip share validate tagged's state; extraction would fragment diagnostics.
def validate_tagged_document(  # ruff: ignore[complex-structure,too-many-branches,too-many-statements]
    content: str,
    *,
    language: str,
    required_labels: Sequence[PiiLabel] = (),
    require_vietnamese_marker: bool = True,
) -> ValidationResult:
    profile = normalize_language(language)
    cleaned = _strip_wrappers(content)
    errors: list[str] = []
    if not cleaned:
        return ValidationResult(ok=False, errors=("empty_output",))
    if _META_PREFIX_RE.search(cleaned[:160]):
        errors.append("assistant_meta_text")
    if "]<" not in cleaned:
        errors.append("no_inline_tags")

    parsed = parse_tagged_text(cleaned, normalize_labels=True)
    spans = parsed.spans
    raw = parsed.raw.strip()
    labels: set[PiiLabel] = set()
    for span in spans:
        if is_pii_label(span.label):
            labels.add(span.label)

    leftover_tags = _TAG_FRAGMENT_RE.findall(raw)
    if leftover_tags:
        errors.append("leftover_or_unknown_tags")
    if _LEFTOVER_LABEL_RE.search(raw):
        errors.append("leftover_label_marker")
    if _MALFORMED_LEFTOVER_LABEL_RE.search(raw):
        errors.append("malformed_leftover_label_marker")
    if len(spans) < ACCEPT_MIN_SPANS:
        errors.append("too_few_spans")
    missing_required = sorted(set(required_labels) - labels)
    if missing_required:
        errors.append("missing_required_labels:" + ",".join(missing_required))
    if len(raw) < _min_raw_length_for(profile.code):
        errors.append("too_short")
    if len(raw) > MAX_DOCUMENT_CHARACTERS:
        errors.append("too_long")
    for span in spans:
        if raw[span.start : span.end] != span.text:
            errors.append("bad_span_offsets")
            break
        if not span.text.strip():
            errors.append("empty_span")
            break
        if span.label == "phone_number":
            if _is_templated_phone(span.text):
                errors.append("templated_phone_span")
                break
            span_digits = digit_count(span.text)
            expanded = expanded_phone_window(raw, span.start, span.end)
            expanded_digits = digit_count(expanded)
            if 0 < span_digits < _MIN_PHONE_DIGITS:
                errors.append("short_phone_span")
                break
            if expanded_digits >= _MIN_PHONE_DIGITS and expanded_digits > span_digits:
                errors.append("underbounded_phone_span")
                break
    if require_vietnamese_marker:
        marker = _marker_re_for(profile.code)
        if marker is not None and marker.search(raw) is None:
            error = "missing_vietnamese_diacritics" if profile.code == "vi" else "missing_native_script"
            errors.append(error)

    return ValidationResult(
        ok=not (set(errors) - _NON_FATAL_ERRORS),
        raw_text=raw,
        spans=spans,
        errors=tuple(errors),
        had_label_repairs=parsed.had_label_repairs,
    )


# reason: accepted record exposes validation/edge cases as its public contract; bundling would break callers.
def accepted_record(  # ruff: ignore[too-many-arguments]
    *,
    validation: ValidationResult,
    language: str,
    scenario: Scenario,
    document_type: str,
    text_format: str,
    model: str,
    provider: str,
    attempt_index: int,
    source_dataset: str = LEGACY_SYNTHETIC_DATASET_IDS["medical"],
    domain_bucket: str = "medical",
    domain_profile: str | None = None,
    split_purpose: str = "train",
    edge_cases: Sequence[str] = (),
) -> AcceptedRecord:
    profile = normalize_language(language)
    digest = hashlib.sha256(validation.raw_text.encode("utf-8")).hexdigest()[:16]
    return {
        "text": validation.raw_text,
        "label": [
            {
                "category": require_pii_label(span.label),
                "start": span.start,
                "end": span.end,
                "text": span.text,
            }
            for span in validation.spans
        ],
        "info": {
            "id": f"{LEGACY_RECORD_ID_PREFIX}{profile.code}_{digest}",
            "source_dataset": source_dataset,
            "source": provider,
            "generation_model": model,
            "language": profile.name,
            "language_bucket": profile.code,
            "domain_bucket": domain_bucket,
            "domain_profile": domain_profile or domain_bucket,
            "document_type": document_type,
            "text_format": text_format,
            "scenario": scenario.name,
            "split_purpose": split_purpose,
            "attempt_index": attempt_index,
            "label_policy": LEGACY_LABEL_POLICY,
            "had_label_repairs": validation.had_label_repairs,
            "edge_cases": list(edge_cases),
        },
    }
