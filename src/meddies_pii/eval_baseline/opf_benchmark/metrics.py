from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, cast

from meddies_pii.annotations.source_mapping import map_native_label_to_pii_label
from meddies_pii.eval_baseline.adapters.opf import OPF_LABEL_FOLD
from meddies_pii.evaluation.span_metrics import (
    containment_span_report_by_doc,
    exact_span_report_by_doc,
)
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


MIN_TIMING_REPEATS = 3


@dataclass(frozen=True, slots=True)
class TimingRepeat:
    timed_seconds: float
    warmup_seconds: float
    peak_gpu_memory_mb: int


@dataclass(frozen=True, slots=True)
class TimingSummary:
    median_seconds: float
    min_seconds: float
    max_seconds: float
    docs_per_sec: float
    tokens_per_sec: float
    peak_gpu_memory_mb: int


@dataclass(frozen=True, slots=True)
class CorrectnessSummary:
    exact_agree_f1: float
    containment_agree_f1: float
    exact_gold_f1: float
    containment_gold_f1: float
    exact_agree_precision: float
    exact_agree_recall: float
    containment_agree_precision: float
    containment_agree_recall: float


def summarize_timing(
    *,
    repeats: Sequence[TimingRepeat],
    docs: int,
    tokens: int,
) -> TimingSummary:
    if len(repeats) < MIN_TIMING_REPEATS:
        msg = "benchmark timing requires at least 3 repeats"
        raise ValueError(msg)
    timed = [repeat.timed_seconds for repeat in repeats]
    if any(value <= 0 for value in timed):
        msg = "timed_seconds must be positive"
        raise ValueError(msg)
    median_seconds = float(median(timed))
    return TimingSummary(
        median_seconds=median_seconds,
        min_seconds=float(min(timed)),
        max_seconds=float(max(timed)),
        docs_per_sec=float(docs) / median_seconds,
        tokens_per_sec=float(tokens) / median_seconds,
        peak_gpu_memory_mb=max(repeat.peak_gpu_memory_mb for repeat in repeats),
    )


def compute_correctness(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    crf_reference_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
) -> CorrectnessSummary:
    predicted = normalize_opf_spans_by_doc(predicted_by_doc)
    reference = normalize_opf_spans_by_doc(crf_reference_by_doc)
    gold = normalize_opf_spans_by_doc(gold_by_doc)
    exact_agree = cast("Mapping[str, float]", exact_span_report_by_doc(predicted, reference)["typed"])
    containment_agree = cast(
        "Mapping[str, float]",
        containment_span_report_by_doc(predicted, reference)["typed"],
    )
    exact_gold = cast("Mapping[str, float]", exact_span_report_by_doc(predicted, gold)["typed"])
    containment_gold = cast("Mapping[str, float]", containment_span_report_by_doc(predicted, gold)["typed"])
    return CorrectnessSummary(
        exact_agree_f1=float(exact_agree["f1"]),
        containment_agree_f1=float(containment_agree["f1"]),
        exact_gold_f1=float(exact_gold["f1"]),
        containment_gold_f1=float(containment_gold["f1"]),
        exact_agree_precision=float(exact_agree["precision"]),
        exact_agree_recall=float(exact_agree["recall"]),
        containment_agree_precision=float(containment_agree["precision"]),
        containment_agree_recall=float(containment_agree["recall"]),
    )


def normalize_opf_spans_by_doc(
    spans_by_doc: Mapping[str, Sequence[CharSpan]],
) -> dict[str, list[CharSpan]]:
    return {
        doc_id: [span for span in (_normalize_span(span) for span in spans) if span is not None]
        for doc_id, spans in spans_by_doc.items()
    }


def _normalize_span(span: CharSpan) -> CharSpan | None:
    mapped = OPF_LABEL_FOLD.get(span.label) or map_native_label_to_pii_label(span.label)
    if mapped is None:
        return None
    return CharSpan(start=span.start, end=span.end, text=span.text, label=mapped)
