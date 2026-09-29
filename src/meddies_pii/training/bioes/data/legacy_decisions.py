"""Conservative label decisions for legacy Meddies migration spans."""

from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: these characters sit inside patterns that must MATCH them — the symbol-substitution and
# reason: separator detectors exist to catch confusable punctuation, so normalising one to ASCII
# reason: would silently stop this module detecting that evasion.
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from meddies_pii.annotations.label_aliases import LABEL_MAP
from meddies_pii.annotations.source_mapping import coerce_pii_label
from meddies_pii.taxonomy import PII_LABELS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from meddies_pii.spans import CharSpan

SpanDecisionKind = Literal["keep", "drop", "quarantine"]

MAX_DATE_SURFACE_LENGTH = 80

_UNCHANGED_LABELS = frozenset({"address", "company_name", "email_address", "human_name", "phone_number"})
_ID_FINE_LABELS = frozenset({
    "account_number",
    "bank_routing_number",
    "biometric_identifier",
    "certificate_license_number",
    "credit_debit_card",
    "customer_id",
    "device_identifier",
    "employee_id",
    "health_plan_beneficiary_number",
    "id_number",
    "ipv4",
    "ipv6",
    "license_plate",
    "mac_address",
    "medical_record_number",
    "national_id",
    "passport",
    "serial_number",
    "ssn",
    "swift_bic",
    "swift_code",
    "tax_id",
    "unique_id",
    "vehicle_identifier",
})
_SECRET_FINE_LABELS = frozenset({
    "api_key",
    "access_token",
    "auth_token",
    "bearer_token",
    "csrf_token",
    "jwt",
    "password",
    "secret",
    "session_token",
    "token",
})
_URL_RE = re.compile(r"^[`'\"(]*(?:https?://|ftp://|www\.)", re.IGNORECASE)
_URL_ANYWHERE_RE = re.compile(r"(?:https?://|ftp://|www\.)", re.IGNORECASE)
_PRIVATE_URL_RE = re.compile(
    r"(?:/(?:patient|patients|record|records|lab-results|patient-results|prescription|prescriptions|claim|claims|appointment|appointments|message|messages|inbox|profile|portal)s?(?:/|$)|[?&](?:token|signature|sig|session|auth|key|access_key|patient|mrn|user|email)=)",
    re.IGNORECASE,
)
_PRIVATE_URL_CONTEXT_RE = re.compile(
    (
        r"(?:patient portal|patient record|patient result|cổng bệnh "
        r"nhân|\bsigned\b|\bprivate\b|medical record|electronic health record|ehr|access "
        r"token|session token|auth token|\blogin\b)"
    ),
    re.IGNORECASE,
)
_PUBLIC_URL_CONTEXT_RE = re.compile(
    (
        r"(?:public|guideline|guidelines|homepage|home page|website|web "
        r"site|article|journal|bibliography|documentation|docs|published|last "
        r"updated|resources?|about|contact|polic(?:y|ies)|procedures?|regulatory|strategy|training|manual|sop|further "
        r"reading)"
    ),
    re.IGNORECASE,
)
_PUBLIC_URL_VALUE_RE = re.compile(
    r"(?:\.gov(?:/|$)|cdc\.gov|news\.com|darkreading\.com|healthcare\.gov|/(?:guidelines?|docs?|documentation|articles?|resources?|polic(?:y|ies)|manuals?|surveys?)(?:/|\?|$)|patient-resources|/search\?)",
    re.IGNORECASE,
)
_SECRET_CONTEXT_RE = re.compile(
    (
        r"(?:\bpassword\b|\bpasscode\b|\bapi[_ -]?key\b|\baccess[_ -]?token\b|\bauth[_ "
        r"-]?token\b|\bbearer\b|\bjwt\b|\bsecret\b|\botp\b|\bpin\b|\bverification\b|"
        r"\breset\b|\bcookie\b|\bsession\b|\bcsrf\b|\bauth(?:entication)?\b|\blogin\b|mật "
        r"khẩu|mã xác thực|ma xac thuc)"
    ),
    re.IGNORECASE,
)
_AUTH_COOKIE_RE = re.compile(r"(?:session|jwt|csrf|xsrf|auth|token|bearer|login|secure)", re.IGNORECASE)
_PUBLIC_DATE_CONTEXT_RE = re.compile(
    (
        r"(?:guideline|guidelines|published|publication|copyright|©|article|journal|reference|bibliography|last "
        r"updated|revised|version)"
    ),
    re.IGNORECASE,
)
_PRIVATE_DATE_CONTEXT_RE = re.compile(
    (
        r"(?:patient|bệnh nhân|dob|date of birth|ngày "
        r"sinh|admitted|admission|discharged|discharge|appointment|visit|follow.?up|prescrib|"
        r"dispens|lab|result|claim|coverage|ngày khám|nhập viện|ra viện|tái khám|toa thuốc|xét nghiệm)"
    ),
    re.IGNORECASE,
)
_MALFORMED_DATE_SPAN_RE = re.compile(r"(?:https?://|ftp://|[{}\[\]\n\r])", re.IGNORECASE)
_TIME_RANGE_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?:\s?[AP]M)?\s*[-–—]\s*\d{1,2}:\d{2}(?:\s?[AP]M)?\b",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"\b\d+(?:\s*[-–—]\s*\d+)?\s+(?:business\s+days?|days?|weeks?|months?|years?|hours?|minutes?)\b",
    re.IGNORECASE,
)
_MALFORMED_STRUCTURED_SPAN_RE = re.compile(
    r"(?:https?://|ftp://|\"(?:url|system|coding|code)\"|[{}\[\]\n\r])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SpanDecision:
    text: str
    old_label: str
    start: int
    end: int
    decision: SpanDecisionKind
    rule: str
    new_label: str | None = None

    def asdict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "old_label": self.old_label,
            "new_label": self.new_label,
            "start": self.start,
            "end": self.end,
            "decision": self.decision,
            "rule": self.rule,
        }


def _context(text: str, start: int, end: int, *, window: int = 120) -> str:
    return text[max(0, start - window) : min(len(text), end + window)]


def _context_before(text: str, start: int, *, window: int = 80) -> str:
    return text[max(0, start - window) : start]


def _classify_url(value: str, context: str) -> tuple[SpanDecisionKind, str, str | None]:
    if _PRIVATE_URL_RE.search(value):
        return "keep", "private_url_context", "private_url"
    if _PUBLIC_URL_VALUE_RE.search(value) or _PUBLIC_URL_CONTEXT_RE.search(context):
        return "drop", "public_url_context", None
    return "quarantine", "ambiguous_url", None


def _classify_secretish(
    old_label: str,
    value: str,
    context: str,
    *,
    immediate_secret_context: bool,
) -> tuple[SpanDecisionKind, str, str | None]:
    if old_label in _SECRET_FINE_LABELS:
        return "keep", "secret_fine_label", "secret"
    if old_label in {"http_cookie", "session_cookie"}:
        if _AUTH_COOKIE_RE.search(value) or _AUTH_COOKIE_RE.search(context):
            return "keep", "secret_cookie_context", "secret"
        return "quarantine", "ambiguous_cookie", None
    if old_label == "pin" and not immediate_secret_context:
        return "keep", "id_fine_label", "id_number"
    if immediate_secret_context:
        return "keep", "secret_context", "secret"
    return "keep", "id_fine_label", "id_number"


def _classify_date(value: str, before_context: str) -> tuple[SpanDecisionKind, str, str | None]:
    if (
        len(value) > MAX_DATE_SURFACE_LENGTH
        or _MALFORMED_DATE_SPAN_RE.search(value)
        or _TIME_RANGE_RE.search(value)
        or _DURATION_RE.search(value)
    ):
        return "drop", "non_private_date_value", None
    if _PUBLIC_DATE_CONTEXT_RE.search(before_context) and not _PRIVATE_DATE_CONTEXT_RE.search(before_context):
        return "drop", "public_date_context", None
    return "keep", "same_label", "date"


def _is_malformed_structured_span(value: str) -> bool:
    return ("\n" in value or "{" in value or "[" in value) and _MALFORMED_STRUCTURED_SPAN_RE.search(value) is not None


# reason: decision for span orders context before classify url; helper seams would misattribute row errors.
def decision_for_span(span: CharSpan, raw: str, *, recover_collapsed: bool = False) -> SpanDecision:  # ruff: ignore[complex-structure,too-many-branches]
    old_label = span.label
    context = _context(raw, span.start, span.end)
    immediate_secret_context = _SECRET_CONTEXT_RE.search(_context_before(raw, span.start, window=48)) is not None
    canonical_label = old_label if old_label in PII_LABELS else coerce_pii_label(LABEL_MAP.get(old_label, ""))
    if old_label == "url" or _URL_RE.search(span.text):
        decision, rule, new_label = _classify_url(span.text, context)
    elif old_label in _SECRET_FINE_LABELS | {"pin", "http_cookie", "session_cookie"}:
        decision, rule, new_label = _classify_secretish(
            old_label,
            span.text,
            context,
            immediate_secret_context=immediate_secret_context,
        )
    elif old_label in _ID_FINE_LABELS:
        if _URL_RE.search(span.text):
            decision, rule, new_label = _classify_url(span.text, context)
        elif recover_collapsed and old_label == "id_number" and immediate_secret_context:
            decision, rule, new_label = "keep", "secret_context", "secret"
        else:
            decision, rule, new_label = "keep", "id_fine_label", "id_number"
    elif (canonical_label in _UNCHANGED_LABELS and _URL_ANYWHERE_RE.search(span.text)) or (
        canonical_label in PII_LABELS and _is_malformed_structured_span(span.text)
    ):
        decision, rule, new_label = "drop", "malformed_structured_span", None
    elif canonical_label in _UNCHANGED_LABELS:
        decision, rule, new_label = "keep", "same_label", canonical_label
    elif canonical_label == "date":
        decision, rule, new_label = _classify_date(span.text, _context_before(raw, span.start))
    elif canonical_label == "private_url":
        decision, rule, new_label = _classify_url(span.text, context)
    elif canonical_label == "secret":
        decision, rule, new_label = _classify_secretish(
            old_label,
            span.text,
            context,
            immediate_secret_context=immediate_secret_context,
        )
    elif canonical_label == "id_number":
        decision, rule, new_label = "keep", "id_fine_label", "id_number"
    else:
        decision, rule, new_label = "drop", "ignored_unsupported_source_label", None
    return SpanDecision(span.text, old_label, span.start, span.end, decision, rule, new_label)


def dedupe_and_validate_spans(spans: Iterable[CharSpan]) -> tuple[CharSpan, ...]:
    deduped = sorted(
        {
            (span.start, span.end, span.label, span.text): span
            for span in spans
            if span.start < span.end and span.label in PII_LABELS
        }.values(),
        key=lambda span: (span.start, span.end, span.label),
    )
    kept: list[CharSpan] = []
    for span in deduped:
        if kept and span.start < kept[-1].end:
            msg = f"overlapping Meddies Labels spans: {kept[-1]!r} overlaps {span!r}"
            raise ValueError(msg)
        kept.append(span)
    return tuple(kept)
