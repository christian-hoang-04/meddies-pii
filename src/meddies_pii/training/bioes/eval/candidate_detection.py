from __future__ import annotations

from typing import TYPE_CHECKING

from .audit_models import CandidateSpan, Severity, _context
from .audit_patterns import (
    COMPACT_DATE_RE,
    DATE_RE,
    DEPARTMENT_RE,
    DOT_DATE_RE,
    EMAIL_RE,
    IPV4_RE,
    ISO_DATETIME_RE,
    LOCALIZED_TEXTUAL_DATE_RE,
    ORG_STOP_RE,
    OTP_RE,
    PHONE_CUE_RE,
    PHONE_RE,
    PHONE_STRONG_SURFACE_RE,
    PRIVATE_URL_MARKERS,
    PUBLIC_URL_NEGATIVES,
    SECRET_RE,
    STRONG_ORG_RE,
    TEXTUAL_DATE_RE,
    TIME_CANDIDATE_RE,
    TRAILING_CHARS,
    URL_RE,
    _has_date_surface,
)

if TYPE_CHECKING:
    import re


def _normalize_candidate_bounds(text: str, start: int, end: int) -> tuple[int, int, str]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1] in TRAILING_CHARS:
        end -= 1
    return start, end, text[start:end]


def _trim_company_candidate(text: str, start: int, end: int) -> tuple[int, int, str]:
    surface = text[start:end]
    stop = ORG_STOP_RE.search(surface)
    if stop is not None:
        end = start + stop.start()
    return _normalize_candidate_bounds(text, start, end)


def _find_regex_candidates(
    text: str,
    pattern: re.Pattern[str],
    *,
    label: str,
    reason: str,
    severity: Severity,
) -> list[CandidateSpan]:
    candidates: list[CandidateSpan] = []
    for match in pattern.finditer(text):
        if label == "company_name":
            start, end, surface = _trim_company_candidate(text, match.start(), match.end())
        else:
            start, end, surface = _normalize_candidate_bounds(text, match.start(), match.end())
        if not surface:
            continue
        candidates.append(
            CandidateSpan(
                label=label,
                start=start,
                end=end,
                text=surface,
                reason=reason,
                severity=severity,
            ),
        )
    return candidates


def _private_url_candidate(url: CandidateSpan) -> CandidateSpan | None:
    lowered = url.text.lower().rstrip("/")
    if any(marker in lowered for marker in PUBLIC_URL_NEGATIVES):
        return None
    path_like = lowered.count("/") >= MIN_PATH_SEPARATOR_COUNT or "?" in lowered
    private_host = "localhost" in lowered or "127.0.0.1" in lowered or "192.168." in lowered or "10." in lowered
    marker_hit = any(marker in lowered for marker in PRIVATE_URL_MARKERS)
    if marker_hit or private_host:
        return CandidateSpan(
            label=url.label,
            start=url.start,
            end=url.end,
            text=url.text,
            reason="private_url_marker_candidate",
            severity="high",
        )
    if path_like:
        return CandidateSpan(
            label=url.label,
            start=url.start,
            end=url.end,
            text=url.text,
            reason="url_with_path_candidate",
            severity="medium",
        )
    return None


def find_label_candidates(text: str) -> list[CandidateSpan]:
    phone_candidates = [
        candidate
        for candidate in (
            _phone_candidate(candidate, text)
            for candidate in _find_regex_candidates(
                text,
                PHONE_RE,
                label="phone_number",
                reason="phone_like_pattern",
                severity="low",
            )
        )
        if candidate is not None
    ]
    url_candidates = [
        candidate
        for candidate in (
            _private_url_candidate(candidate)
            for candidate in _find_regex_candidates(
                text,
                URL_RE,
                label="private_url",
                reason="url_pattern",
                severity="medium",
            )
        )
        if candidate is not None
    ]
    candidates = [
        *_find_regex_candidates(
            text,
            STRONG_ORG_RE,
            label="company_name",
            reason="strong_organization_keyword_candidate",
            severity="high",
        ),
        *_find_regex_candidates(
            text,
            DEPARTMENT_RE,
            label="company_name",
            reason="department_keyword_candidate",
            severity="medium",
        ),
        *_find_regex_candidates(
            text,
            ISO_DATETIME_RE,
            label="date",
            reason="iso_datetime_pattern",
            severity="medium",
        ),
        *_find_regex_candidates(
            text,
            DATE_RE,
            label="date",
            reason="date_pattern",
            severity="medium",
        ),
        *_find_regex_candidates(
            text,
            DOT_DATE_RE,
            label="date",
            reason="dot_date_pattern",
            severity="medium",
        ),
        *_find_regex_candidates(
            text,
            COMPACT_DATE_RE,
            label="date",
            reason="compact_hl7_date_pattern",
            severity="medium",
        ),
        *_find_regex_candidates(
            text,
            TEXTUAL_DATE_RE,
            label="date",
            reason="textual_date_pattern",
            severity="medium",
        ),
        *_find_regex_candidates(
            text,
            LOCALIZED_TEXTUAL_DATE_RE,
            label="date",
            reason="localized_textual_date_pattern",
            severity="medium",
        ),
        *_find_regex_candidates(
            text,
            TIME_CANDIDATE_RE,
            label="date",
            reason="time_only_pattern",
            severity="low",
        ),
        *_find_regex_candidates(
            text,
            EMAIL_RE,
            label="email_address",
            reason="email_pattern",
            severity="high",
        ),
        *url_candidates,
        *_secret_candidates(text),
        *phone_candidates,
    ]
    deduped: dict[tuple[str, int, int, str], CandidateSpan] = {}
    for candidate in candidates:
        deduped[candidate.label, candidate.start, candidate.end, candidate.text] = candidate
    return sorted(deduped.values(), key=lambda item: (item.start, item.end, item.label))


MIN_PATH_SEPARATOR_COUNT = 3
MIN_PHONE_DIGITS = 8

_MY_MONTH = (
    r"တန်ခူး|ကဆုန်|နယုန်|ဝါဆို|ဝါခေါင်|တော်သလင်း|"
    r"သီတင်းကျွတ်|တန်ဆောင်မုန်း|နတ်တော်|ပြာသို|တပို့တွဲ|တပေါင်း|မတ်လ"
)
"""Spelled-out native dates (voice-scribe / OCR transcription) carry number-WORDS.

Not digits, so the digit-anchored NATIVE_DATE_RE above misses them (Korean, Japanese hiragana, Thai/Lao/Tamil/Burmese
spell-outs). Detect them by their unambiguous calendar markers instead: a month NAME alone, or >=2 distinct year/month/day
marker words. The Korean day-marker and Burmese year-marker are digit homographs (=1 and =2) but stay phone-safe under the
>=2-distinct rule, so a phone/ID spoken as number-words scores 0 markers and is never read as a date.

"""


def _phone_candidate(candidate: CandidateSpan, document_text: str) -> CandidateSpan | None:
    surface = candidate.text.strip()
    compact_digits = "".join(char for char in surface if char.isdigit())
    if len(compact_digits) < MIN_PHONE_DIGITS:
        return None
    if "/" in surface or _has_date_surface(surface) or IPV4_RE.fullmatch(surface):
        return None
    context = _context(document_text, candidate.start, candidate.end, width=36)
    has_cue = PHONE_CUE_RE.search(context) is not None
    strong_surface = PHONE_STRONG_SURFACE_RE.fullmatch(surface) is not None
    if not has_cue and not strong_surface:
        return None
    return CandidateSpan(
        label=candidate.label,
        start=candidate.start,
        end=candidate.end,
        text=candidate.text,
        reason="phone_cue_candidate" if has_cue else "strong_phone_surface_candidate",
        severity="medium" if has_cue else "low",
    )


def _secret_candidates(text: str) -> list[CandidateSpan]:
    candidates: list[CandidateSpan] = []
    for pattern, reason in (
        (SECRET_RE, "secret_assignment_candidate"),
        (OTP_RE, "otp_or_pin_candidate"),
    ):
        for match in pattern.finditer(text):
            start, end, surface = _normalize_candidate_bounds(text, match.start(), match.end())
            if surface:
                candidates.append(
                    CandidateSpan(
                        label="secret",
                        start=start,
                        end=end,
                        text=surface,
                        reason=reason,
                        severity="high",
                    ),
                )
    return candidates
