"""Merge tiered regex candidates into a model result without erasing model spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from meddies_pii.regex_runtime.rules import (
    RegexCandidate,
    regex_candidates,
    union_accepted,
)
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import PiiLabel, require_pii_label

if TYPE_CHECKING:
    from collections.abc import Sequence

_TRIMMABLE_STRUCTURED_LABELS = frozenset({"email_address", "phone_number", "private_url", "id_number", "secret"})
_MIN_TRIMMED_REMAINDER_CHARS = 3


@dataclass(frozen=True, slots=True)
class RegexConflict:
    reason: str
    model_span: CharSpan
    regex_span: CharSpan


@dataclass(frozen=True, slots=True)
class RegexPostprocessResult:
    spans: tuple[CharSpan, ...]
    conflicts: tuple[RegexConflict, ...]


def apply_regex_postprocess(
    text: str,
    model_spans: Sequence[CharSpan],
    *,
    language: str | None = None,
) -> RegexPostprocessResult:
    """Return the paired candidate view while preserving model precedence.

    Returns:
        The merged spans, accepted regex candidates, and overlap conflicts.

    """
    return merge_tiered_candidates(
        text,
        model_spans,
        regex_candidates(text, language=language),
        language=language,
    )


def merge_tiered_candidates(
    text: str,
    model_spans: Sequence[CharSpan],
    candidates: Sequence[RegexCandidate],
    *,
    language: str | None = None,
) -> RegexPostprocessResult:
    """Fold candidates into the model spans under their tier's model-overlap rule.

    Tier precedence between candidates is already settled before this call; what a
    tier decides here is how far it may act on a *model* span:

    * AUTH and CONTEXT may add a span where the model found nothing, and may widen
      a model span of the same label. The difference between them is upstream —
      CONTEXT candidates that collided with an AUTH region no longer exist.
    * SNAP never adds a whole candidate. It only widens a model span of the same
      label that it overlaps; a structured candidate that crosses a differently
      labelled model span may still retain its non-overlapping remainder.

    One invariant holds across every tier: a regex span never erases a model span of
    a different label. That collision is recorded as a conflict and the model wins.
    A same-label widening is applied only when a rule of that label accepts the
    unioned text, so a union can never manufacture a value no rule would match.

    Returns:
        The model-precedence merge result with accepted candidates and conflicts.

    """
    merged = list(model_spans)
    conflicts: list[RegexConflict] = []
    for candidate in candidates:
        span = candidate.span
        exact = next(
            (model for model in merged if (model.start, model.end, model.label) == (span.start, span.end, span.label)),
            None,
        )
        if exact is not None:
            continue
        overlaps = [model for model in merged if _overlaps(model, span)]
        if not overlaps:
            if candidate.tier != "SNAP":
                merged.append(span)
            continue
        if any(model.label != span.label for model in overlaps):
            different_label_overlaps = tuple(model for model in overlaps if model.label != span.label)
            if span.label in _TRIMMABLE_STRUCTURED_LABELS:
                conflicts.extend(
                    RegexConflict("different_label_trimmed_remainder", model, span) for model in different_label_overlaps
                )
                merged.extend(_non_overlapping_remainders(text, span, overlaps))
                continue
            conflicts.extend(
                RegexConflict("different_label_preserves_model", model, span) for model in different_label_overlaps
            )
            continue
        union_start = min(span.start, *(model.start for model in overlaps))
        union_end = max(span.end, *(model.end for model in overlaps))
        union_text = text[union_start:union_end]
        if not union_accepted(span.label, union_text, language=language):
            conflicts.extend(RegexConflict("same_label_union_failed_validation", model, span) for model in overlaps)
            continue
        overlap_ids = {id(model) for model in overlaps}
        merged = [model for model in merged if id(model) not in overlap_ids]
        merged.append(CharSpan(union_start, union_end, union_text, span.label))
    return RegexPostprocessResult(
        spans=tuple(sorted(merged, key=lambda span: (span.start, span.end, span.label))),
        conflicts=tuple(conflicts),
    )


def _overlaps(left: CharSpan, right: CharSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _non_overlapping_remainders(text: str, candidate: CharSpan, overlaps: Sequence[CharSpan]) -> tuple[CharSpan, ...]:
    label = require_pii_label(candidate.label)
    blocked = sorted(
        (
            max(candidate.start, model.start),
            min(candidate.end, model.end),
        )
        for model in overlaps
    )
    remainder_start = candidate.start
    remainders: list[CharSpan] = []
    for blocked_start, blocked_end in blocked:
        if blocked_start > remainder_start:
            _append_remainder(remainders, text, remainder_start, blocked_start, label)
        remainder_start = max(remainder_start, blocked_end)
    if remainder_start < candidate.end:
        _append_remainder(remainders, text, remainder_start, candidate.end, label)
    return tuple(remainders)


def _append_remainder(remainders: list[CharSpan], text: str, start: int, end: int, label: PiiLabel) -> None:
    if end - start >= _MIN_TRIMMED_REMAINDER_CHARS:
        remainders.append(CharSpan(start, end, text[start:end], label))
