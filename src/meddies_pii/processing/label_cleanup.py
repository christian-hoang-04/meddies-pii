"""Quality filters for PII label dictionaries.

Two public entry points:

- ``clean_bracket_artifacts`` — strips ``[]{}<>`` characters from string
  values that leaked through templating; drops empty results.
- ``apply_quality_filters`` — per-label heuristic filters that reject
  obvious non-PII (FHIR resource IDs, IP addresses, MAC addresses,
  standalone times, role titles, honorifics) and deduplicates values.

Both operate on the project's PII label-dict format
(``{label: list[value]}``) and return a new dict; inputs are never
mutated. Used by ``hard_examples`` (gold/Gemini selection) and by
``merge_gemini_labels`` (post-Gemini cleanup).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

import re

from meddies_pii.json_types import is_str_list

MAX_HOUR = 23
CJK_UNIFIED_START = 0x4E00
CJK_UNIFIED_END = 0x9FFF
CJK_EXTENSION_A_START = 0x3400
CJK_EXTENSION_A_END = 0x4DBF
CJK_COMPATIBILITY_START = 0xF900
CJK_COMPATIBILITY_END = 0xFAFF

_ROLE_TITLES = frozenset({
    "it manager",
    "attending physician",
    "nurse practitioner",
    "registered nurse",
    "medical director",
    "chief of staff",
    "head nurse",
    "staff nurse",
    "charge nurse",
    "physician assistant",
})
_STANDALONE_HONORIFICS = frozenset({"dr.", "mr.", "mrs.", "ms.", "prof."})

_FHIR_PREFIXES = (
    "patient-",
    "practitioner-",
    "prac-",
    "encounter-",
    "organization-",
    "obs-",
    "referral-",
    "consent-",
    "condition-",
    "medication-",
    "allergy-",
    "appointment-",
    "schedule-",
    "sch-",
    "slot-",
    "location-",
    "device-",
)


def clean_bracket_artifacts(data: Mapping[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key, values in data.items():
        if not isinstance(values, list):
            cleaned[key] = values
            continue
        cleaned_values: list[object] = []
        for value in values:
            if isinstance(value, str):
                stripped = value.strip("[]{}<>")
                if stripped:
                    cleaned_values.append(stripped)
            else:
                cleaned_values.append(value)
        cleaned[key] = cleaned_values
    return cleaned


def _filter_id_numbers(values: list[str]) -> list[str]:
    result = []
    for v in values:
        if "://" in v:
            continue
        if v.strip().lower().startswith("www."):
            continue
        if re.match(r"\d+\.\d+\.\d+\.\d+", v):
            continue
        if re.search(r"([0-9A-Fa-f]{2}:){5}", v):
            continue
        if "Mozilla/" in v:
            continue
        if v.strip().upper() in {"COMPLETED", "DIGITALLY SIGNED"}:
            continue
        if re.match(r"^\d{1,3}$", v.strip()):
            continue
        if v.strip().lower().startswith(_FHIR_PREFIXES):
            continue
        result.append(v)
    return result


def _filter_dates(values: list[str]) -> list[str]:
    result = []
    for v in values:
        s = v.strip()
        if re.match(r"^\d{1,3}$", s):
            continue
        if s.upper() in {
            "COMPLETED",
            "PENDING",
            "CANCELLED",
            "APPROVED",
            "DIGITALLY SIGNED",
        }:
            continue
        if re.match(r"^\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?$", s):
            continue
        if re.match(r"^\d{6}$", s) and int(s[:2]) <= MAX_HOUR:
            continue
        result.append(v)
    return result


def _is_cjk(char: str) -> bool:
    """Report whether a character is CJK by any of the three blocks this module recognises.

    Returns:
        Whether the character falls in the CJK Unified, Extension A, or Compatibility block.

    """
    cp = ord(char)
    return (
        CJK_UNIFIED_START <= cp <= CJK_UNIFIED_END
        or CJK_EXTENSION_A_START <= cp <= CJK_EXTENSION_A_END
        or CJK_COMPATIBILITY_START <= cp <= CJK_COMPATIBILITY_END
    )


def _filter_names(values: list[str]) -> list[str]:
    result = []
    for v in values:
        s = v.strip()
        if not s:
            continue
        if s.lower() in _ROLE_TITLES:
            continue
        if s.lower() in _STANDALONE_HONORIFICS:
            continue
        if len(s) == 1 and not _is_cjk(s):
            continue
        if "\n" in s or "{" in s:
            continue
        result.append(v)
    return result


def _string_list(value: object) -> list[str] | None:
    return value if is_str_list(value) else None


def _deduplicate_values(data: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, values in data.items():
        if not isinstance(values, list):
            result[key] = values
            continue
        seen: list[object] = []
        for value in values:
            if value not in seen:
                seen.append(value)
        result[key] = seen
    return result


def apply_quality_filters(data: Mapping[str, object]) -> dict[str, object]:
    result = dict(data)
    id_numbers = _string_list(result.get("id_number"))
    if id_numbers is not None:
        result["id_number"] = _filter_id_numbers(id_numbers)
    dates = _string_list(result.get("date"))
    if dates is not None:
        result["date"] = _filter_dates(dates)
    names = _string_list(result.get("human_name"))
    if names is not None:
        result["human_name"] = _filter_names(names)
    return _deduplicate_values(result)
