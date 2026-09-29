from __future__ import annotations

from .audit_models import CandidateSpan, SpanAuditRecord
from .audit_patterns import (
    AGE_EXPRESSION_RE,
    BARE_SMALL_INT_RE,
    DEPARTMENT_RE,
    DOSAGE_UNIT_RE,
    EMAIL_RE,
    FIELD_PREFIX_RE,
    ICD_CODE_RE,
    LAB_VALUE_UNIT_RE,
    NUMERIC_MEASUREMENT_RE,
    STRONG_ORG_RE,
    TIME_ONLY_RE,
    _has_date_surface,
    _is_valid_date_surface,
)
from .candidate_detection import _phone_candidate, _private_url_candidate

MIN_DATE_CHARACTERS = 4
MIN_MULTIWORD_ENTITY_WORDS = 2
MIN_ADDRESS_CHARACTERS = 6
MAX_DATE_SURFACE_LENGTH = 64


# reason: looks valid for keeps valid date beside private url; splitting would fork shared metric totals.
def _looks_valid_for_label(span: SpanAuditRecord) -> bool:  # ruff: ignore[complex-structure,too-many-return-statements]
    value = span.text.strip()
    if not value:
        return False
    if span.label == "date":
        return _is_valid_date_surface(value)
    if span.label == "email_address":
        return EMAIL_RE.search(value) is not None
    if span.label == "private_url":
        candidate = _private_url_candidate(
            CandidateSpan(
                label="private_url",
                start=span.start,
                end=span.end,
                text=value,
                reason="url_pattern",
                severity="medium",
            ),
        )
        return candidate is not None and candidate.severity == "high"
    if span.label == "phone_number":
        return (
            _phone_candidate(
                CandidateSpan(
                    label="phone_number",
                    start=span.start,
                    end=span.end,
                    text=value,
                    reason="phone_like_pattern",
                    severity="low",
                ),
                value,
            )
            is not None
        )
    if span.label == "id_number":
        if NUMERIC_MEASUREMENT_RE.fullmatch(value) or ICD_CODE_RE.fullmatch(value):
            return False
        return any(char.isdigit() for char in value) and len(value) >= MIN_DATE_CHARACTERS
    if span.label == "company_name":
        return (
            STRONG_ORG_RE.search(value) is not None
            or DEPARTMENT_RE.search(value) is not None
            or len(value.split()) >= MIN_MULTIWORD_ENTITY_WORDS
        )
    if span.label == "human_name":
        return ":" not in value and len(value.split()) >= MIN_MULTIWORD_ENTITY_WORDS
    if span.label == "address":
        return len(value) >= MIN_ADDRESS_CHARACTERS
    if span.label == "secret":
        return any(
            token in value.lower()
            for token in (
                "=",
                "token",
                "key",
                "secret",
                "password",
                "pwd",
                "pin",
                "otp",
                "cookie",
                "session",
                "csrf",
                "bearer",
            )
        )
    return True


def _non_pii_surface_reason(label: str, value: str) -> str | None:
    """Classify a span surface that is a medical/clinical artifact, never PII.

    The known internal contamination is lab reference-ranges tagged `id_number`,
    but the same surfaces (ranges, unit-bearing lab values, dosages, ages) leak
    under other labels too. Match on the surface, not the label, so every
    mislabel of the same non-PII text is quarantined.

    Returns:
        The reason this surface is a clinical artifact rather than PII, or ``None`` when the
        span looks like genuine PII and should be kept. ``None`` is the KEEP answer, so a
        caller reading it as "no reason found, therefore drop" inverts the filter.

    """
    if label == "id_number" and NUMERIC_MEASUREMENT_RE.fullmatch(value):
        return "numeric_reference_range_labeled_id_number"
    if LAB_VALUE_UNIT_RE.search(value):
        return "lab_value_with_unit_labeled_pii"
    if DOSAGE_UNIT_RE.search(value):
        return "dosage_or_unit_labeled_pii"
    if AGE_EXPRESSION_RE.search(value):
        return "age_expression_labeled_pii"
    return None


# reason: suspicious gold's exits distinguish span_contains_field_prefix; merging them would blur decision order.
def _suspicious_gold_reason(span: SpanAuditRecord) -> str | None:  # ruff: ignore[too-many-return-statements]
    value = span.text.strip()
    if FIELD_PREFIX_RE.search(value):
        return "span_contains_field_prefix"
    if span.label == "date" and TIME_ONLY_RE.fullmatch(value):
        return "date_gold_surface_is_time_only"
    non_pii_reason = _non_pii_surface_reason(span.label, value)
    if non_pii_reason is not None:
        return non_pii_reason
    if span.label == "date" and BARE_SMALL_INT_RE.fullmatch(value):
        return "bare_small_integer_labeled_date"
    if span.label == "date" and (len(value) > MAX_DATE_SURFACE_LENGTH or not _has_date_surface(value)):
        return "date_gold_surface_is_not_date_like"
    if span.label == "human_name" and ":" in value:
        return "human_name_contains_field_prefix_or_role"
    return None
