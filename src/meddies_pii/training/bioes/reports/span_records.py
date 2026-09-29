from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SpanKey = tuple[str, int, int]
SpanBoundaryKey = tuple[int, int, str]


@dataclass(frozen=True, slots=True)
class ReportSpan:
    label: str
    start: int
    end: int
    text: str

    @property
    def key(self) -> SpanKey:
        return self.label, self.start, self.end

    @property
    def boundary_key(self) -> SpanBoundaryKey:
        return self.start, self.end, self.text

    def to_record(self) -> dict[str, object]:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


def report_span_from_mapping(span: Mapping[str, object]) -> ReportSpan:
    return ReportSpan(
        label=str(span["label"]),
        start=_int_field(span["start"], field="start"),
        end=_int_field(span["end"], field="end"),
        text=str(span.get("text", "")),
    )


def report_span_key(span: Mapping[str, object]) -> SpanKey:
    return report_span_from_mapping(span).key


def report_span_record(span: Mapping[str, object]) -> dict[str, object]:
    return report_span_from_mapping(span).to_record()


def report_span_mappings(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    spans: list[Mapping[str, object]] = [
        {str(key): raw for key, raw in item.items()} for item in value if isinstance(item, Mapping)
    ]
    return spans


def _int_field(value: object, *, field: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        return int(value)
    msg = f"span {field} must be int-like, got {type(value).__name__}"
    raise TypeError(msg)
