from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RejectionVerdict(StrEnum):
    OVER_FILTER = "over-filter"
    PARSER_BUG = "parser-bug"
    BORDERLINE = "borderline"
    CORRECT = "correct"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True, slots=True)
class RejectionClassification:
    verdict: RejectionVerdict
    explanation: str

    @property
    def recoverable(self) -> bool:
        return self.verdict in {
            RejectionVerdict.OVER_FILTER,
            RejectionVerdict.PARSER_BUG,
        }


_CLASSIFICATIONS = {
    "short_phone_span": RejectionClassification(
        RejectionVerdict.OVER_FILTER,
        "Whole doc dropped for one short phone span (extension / partial / redacted number is a valid hard case).",
    ),
    "underbounded_phone_span": RejectionClassification(
        RejectionVerdict.OVER_FILTER,
        "Whole doc dropped because a phone span under-captured digits — a split/odd-format number we want.",
    ),
    "missing_required_labels": RejectionClassification(
        RejectionVerdict.BORDERLINE,
        "Doc is rich but missing one scenario-required label; the requirement may be too strict.",
    ),
    "leftover_label_marker": RejectionClassification(
        RejectionVerdict.PARSER_BUG,
        "A leftover `]<label>` marker — our inline-tag parser failed on a doc the model tagged fine.",
    ),
    "malformed_leftover_label_marker": RejectionClassification(
        RejectionVerdict.PARSER_BUG,
        "Malformed leftover marker — same parser failure class.",
    ),
    "leftover_or_unknown_tags": RejectionClassification(
        RejectionVerdict.PARSER_BUG,
        "Leftover/unknown tag fragment — parser confusion (often brackets inside JSON/FHIR).",
    ),
    "bad_span_offsets": RejectionClassification(
        RejectionVerdict.PARSER_BUG,
        "Span text doesn't match its offsets — parser boundary error.",
    ),
    "empty_output": RejectionClassification(
        RejectionVerdict.CORRECT,
        "Model returned nothing — correctly dropped.",
    ),
    "no_inline_tags": RejectionClassification(
        RejectionVerdict.CORRECT,
        "No inline tags at all — correctly dropped.",
    ),
    "too_few_spans": RejectionClassification(
        RejectionVerdict.CORRECT,
        "Genuinely sparse (too few spans) — correctly dropped.",
    ),
    "too_few_unique_labels": RejectionClassification(
        RejectionVerdict.CORRECT,
        "Genuinely sparse (too few distinct labels) — correctly dropped.",
    ),
    "too_short": RejectionClassification(
        RejectionVerdict.CORRECT,
        "Document too short — correctly dropped.",
    ),
    "api_error": RejectionClassification(
        RejectionVerdict.CORRECT,
        "Generation/API error, not a content issue.",
    ),
    "missing_vietnamese_diacritics": RejectionClassification(
        RejectionVerdict.CORRECT,
        "Vietnamese text without diacritics — correctly dropped.",
    ),
    "assistant_meta_text": RejectionClassification(
        RejectionVerdict.CORRECT,
        "Model emitted assistant chatter — correctly dropped.",
    ),
}

_UNCLASSIFIED = RejectionClassification(
    RejectionVerdict.UNCLASSIFIED,
    "No rejection policy is registered for this reason; manual classification required.",
)


def classify_rejection_reason(reason: str) -> RejectionClassification:
    """Return the explicit data-quality policy for one rejection reason.

    Returns:
        The recorded classification, or the shared ``_UNCLASSIFIED`` sentinel for a reason
        this policy table does not name. An unknown reason is never an error: a new rejection
        reason appearing upstream shows up in reports as unclassified rather than crashing
        the audit, which is what makes the table safe to extend after the fact.

    """
    return _CLASSIFICATIONS.get(reason, _UNCLASSIFIED)
