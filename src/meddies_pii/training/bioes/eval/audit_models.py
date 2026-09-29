from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Severity = Literal["high", "medium", "low"]


@dataclass(frozen=True, slots=True)
class SpanAuditRecord:
    label: str
    start: int
    end: int
    text: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SpanAuditRecord:
        return cls(
            label=str(payload["label"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            text=str(payload["text"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class CandidateSpan:
    label: str
    start: int
    end: int
    text: str
    reason: str
    severity: Severity

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "reason": self.reason,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class EvalAuditIssue:
    uid: str
    issue_type: str
    severity: Severity
    label: str
    start: int
    end: int
    text: str
    reason: str
    recommended_action: str
    context: str
    evidence: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "uid": self.uid,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "context": self.context,
            "evidence": self.evidence,
        }


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _same_boundary(span: SpanAuditRecord, other: SpanAuditRecord) -> bool:
    return span.label == other.label and span.start == other.start and span.end == other.end


def _context(text: str, start: int, end: int, *, width: int = 180) -> str:
    window_start = max(0, start - width)
    window_end = min(len(text), end + width)
    prefix = "…" if window_start > 0 else ""
    suffix = "…" if window_end < len(text) else ""
    return prefix + text[window_start:window_end] + suffix


def _candidate_exact_match(
    candidate: CandidateSpan,
    spans: Sequence[SpanAuditRecord],
    *,
    label_required: bool = True,
) -> bool:
    for span in spans:
        if label_required and span.label != candidate.label:
            continue
        if candidate.start == span.start and candidate.end == span.end:
            return True
    return False


def _overlapping_spans(
    candidate: CandidateSpan,
    spans: Sequence[SpanAuditRecord],
    *,
    label_required: bool | None = None,
) -> list[SpanAuditRecord]:
    overlapping: list[SpanAuditRecord] = []
    for span in spans:
        if label_required is True and span.label != candidate.label:
            continue
        if label_required is False and span.label == candidate.label:
            continue
        if _overlap(candidate.start, candidate.end, span.start, span.end) > 0:
            overlapping.append(span)
    return overlapping


def _span_covered(
    span: SpanAuditRecord,
    others: Sequence[SpanAuditRecord],
    *,
    label_required: bool = True,
) -> bool:
    for other in others:
        if label_required and span.label != other.label:
            continue
        if _overlap(span.start, span.end, other.start, other.end) > 0:
            return True
    return False
