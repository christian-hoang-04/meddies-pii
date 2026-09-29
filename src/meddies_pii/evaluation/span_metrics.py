"""Exact and containment metrics for character-span evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, SupportsFloat, TypedDict

from meddies_pii.taxonomy import PII_LABELS, PiiLabel, is_pii_label

if TYPE_CHECKING:
    from meddies_pii.spans import CharSpan

SpanKey = tuple[str, int, int]
UntypedSpanKey = tuple[int, int]
SpanMode = Literal["typed", "untyped"]
ContainmentSpanKey = tuple[str | None, int, int]


class ExactSpanMetricBlock(TypedDict):
    tp: float
    pred_total: float
    gold_total: float
    precision: float
    recall: float
    f1: float


class ContainmentSpanMetricBlock(TypedDict):
    precision_tp: float
    recall_tp: float
    pred_total: float
    gold_total: float
    precision: float
    recall: float
    f1: float


class MetricDelta(TypedDict):
    precision: float
    recall: float
    f1: float


MetricSliceReport = dict[str, object]
SpanMetricBlock = ExactSpanMetricBlock | ContainmentSpanMetricBlock
MetricBlockForDelta = SpanMetricBlock | Mapping[str, float]


class ExactSpanSliceReport(TypedDict):
    support_docs: int
    support_gold_spans: int
    support_predicted_spans: int
    typed: ExactSpanMetricBlock
    untyped: ExactSpanMetricBlock
    untyped_minus_typed: MetricDelta


class ContainmentSpanSliceReport(TypedDict):
    support_docs: int
    support_gold_spans: int
    support_predicted_spans: int
    typed: ContainmentSpanMetricBlock
    untyped: ContainmentSpanMetricBlock
    untyped_minus_typed: MetricDelta


class ContainmentSpanDocReport(ContainmentSpanMetricBlock):
    typed: ContainmentSpanMetricBlock
    untyped: ContainmentSpanMetricBlock
    untyped_minus_typed: MetricDelta


class ContainmentSpanReport(ContainmentSpanDocReport):
    slices: Mapping[str, object]


class ExactSpanDocReport(ExactSpanMetricBlock):
    typed: ExactSpanMetricBlock
    untyped: ExactSpanMetricBlock
    untyped_minus_typed: MetricDelta


class ExactSpanReport(ExactSpanDocReport):
    slices: Mapping[str, object]
    containment_span: ContainmentSpanReport


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _span_key(span: CharSpan, *, mode: SpanMode = "typed") -> SpanKey | UntypedSpanKey:
    if mode == "typed":
        return (span.label, span.start, span.end)
    if mode == "untyped":
        return (span.start, span.end)
    msg = f"Unsupported exact-span mode: {mode!r}"
    raise ValueError(msg)


def exact_span_counts(
    predicted: Sequence[CharSpan],
    gold: Sequence[CharSpan],
    *,
    mode: SpanMode = "typed",
) -> dict[str, int]:
    pred = Counter(_span_key(span, mode=mode) for span in predicted)
    target = Counter(_span_key(span, mode=mode) for span in gold)
    return {
        "tp": sum((pred & target).values()),
        "pred_total": sum(pred.values()),
        "gold_total": sum(target.values()),
    }


def exact_span_prf_from_counts(counts: Mapping[str, int | float]) -> ExactSpanMetricBlock:
    tp = float(counts["tp"])
    pred_total = float(counts["pred_total"])
    gold_total = float(counts["gold_total"])
    precision = _safe_div(tp, pred_total)
    recall = _safe_div(tp, gold_total)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return ExactSpanMetricBlock(
        tp=tp,
        pred_total=pred_total,
        gold_total=gold_total,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def exact_span_prf(
    predicted: Sequence[CharSpan],
    gold: Sequence[CharSpan],
    *,
    mode: SpanMode = "typed",
) -> ExactSpanMetricBlock:
    return exact_span_prf_from_counts(exact_span_counts(predicted, gold, mode=mode))


def exact_span_prf_by_doc(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
    *,
    mode: SpanMode = "typed",
) -> ExactSpanMetricBlock:
    totals = {"tp": 0, "pred_total": 0, "gold_total": 0}
    for doc_id in sorted(set(predicted_by_doc) | set(gold_by_doc)):
        counts = exact_span_counts(
            predicted_by_doc.get(doc_id, ()),
            gold_by_doc.get(doc_id, ()),
            mode=mode,
        )
        totals["tp"] += counts["tp"]
        totals["pred_total"] += counts["pred_total"]
        totals["gold_total"] += counts["gold_total"]
    return exact_span_prf_from_counts(totals)


def _spans_with_label(spans: Sequence[CharSpan], label: str) -> tuple[CharSpan, ...]:
    return tuple(span for span in spans if span.label == label)


def _counts_by_label(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
    *,
    count_fn: Callable[[Sequence[CharSpan], Sequence[CharSpan]], Mapping[str, int]],
    count_keys: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Micro-aggregate per-label counts across docs for one count family.

    Every label in ``PII_LABELS`` gets a totals bucket, so a label that
    is present in gold but never predicted (or absent entirely) yields zeroed
    counts instead of a missing key.

    Returns:
        A totals bucket for every label in ``PII_LABELS``, each holding that label's counts summed across the docs.

    """
    totals: dict[str, dict[str, int]] = {label: dict.fromkeys(count_keys, 0) for label in PII_LABELS}
    for doc_id in sorted(set(predicted_by_doc) | set(gold_by_doc)):
        predicted = predicted_by_doc.get(doc_id, ())
        gold = gold_by_doc.get(doc_id, ())
        for label in PII_LABELS:
            counts = count_fn(_spans_with_label(predicted, label), _spans_with_label(gold, label))
            for key in count_keys:
                totals[label][key] += counts[key]
    return totals


def exact_span_prf_by_label(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
) -> dict[str, ExactSpanMetricBlock]:
    """Per-label typed exact-span PRF, micro-averaged over docs.

    Counts (tp/pred_total/gold_total) are aggregated per label across all docs,
    then converted to PRF. Every allowed label is reported; a label seen only in
    gold yields precision/recall/f1 = 0 rather than being dropped.

    Returns:
        Precision, recall, and F1 for each label, computed from that label's counts summed across every doc.

    """
    totals = _counts_by_label(
        predicted_by_doc,
        gold_by_doc,
        count_fn=lambda predicted, gold: exact_span_counts(predicted, gold, mode="typed"),
        count_keys=("tp", "pred_total", "gold_total"),
    )
    return {label: exact_span_prf_from_counts(counts) for label, counts in totals.items()}


def exact_span_report_by_doc(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
) -> ExactSpanDocReport:
    """Return containment-style typed and untyped exact-span metrics.

    Existing top-level ``precision``/``recall``/``f1`` keys remain the typed
    metric for backward compatibility. The nested metrics make label-confusion
    visible: untyped can be high while typed is low when boundaries are right
    but entity classes are wrong.

    Returns:
        The typed metric at the top level, which older callers read directly, alongside the ``typed``, ``untyped``, and
        ``untyped_minus_typed`` sub-reports.

    """
    typed = exact_span_prf_by_doc(predicted_by_doc, gold_by_doc, mode="typed")
    untyped = exact_span_prf_by_doc(predicted_by_doc, gold_by_doc, mode="untyped")
    return {
        **typed,
        "typed": typed,
        "untyped": untyped,
        "untyped_minus_typed": {
            "precision": untyped["precision"] - typed["precision"],
            "recall": untyped["recall"] - typed["recall"],
            "f1": untyped["f1"] - typed["f1"],
        },
    }


def exact_span_slice_report_by_doc(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
    doc_slices: Mapping[str, Sequence[str]],
    *,
    slice_names: Sequence[str] | None = None,
) -> dict[str, ExactSpanSliceReport]:
    """Compute typed/untyped exact-span metrics for pre-declared doc slices.

    Slice membership must be derived from source text or metadata, never from
    model predictions. Empty slices are retained when ``slice_names`` is passed
    so reports cannot silently hide unsupported adversarial families.

    Returns:
        One exact-span report per declared slice, keyed by the slice name.

    """
    inferred_names = sorted({name for names in doc_slices.values() for name in names})
    ordered_names = list(slice_names) if slice_names is not None else inferred_names
    report: dict[str, ExactSpanSliceReport] = {}
    for slice_name in ordered_names:
        doc_ids = sorted(doc_id for doc_id, names in doc_slices.items() if slice_name in set(names))
        predicted_subset = {
            doc_id: predicted_by_doc.get(doc_id, ())
            for doc_id in doc_ids
            if doc_id in predicted_by_doc or doc_id in gold_by_doc
        }
        gold_subset = {
            doc_id: gold_by_doc.get(doc_id, ())
            for doc_id in doc_ids
            if doc_id in predicted_by_doc or doc_id in gold_by_doc
        }
        typed = exact_span_prf_by_doc(predicted_subset, gold_subset, mode="typed")
        untyped = exact_span_prf_by_doc(predicted_subset, gold_subset, mode="untyped")
        report[slice_name] = ExactSpanSliceReport(
            support_docs=len(doc_ids),
            support_gold_spans=int(typed["gold_total"]),
            support_predicted_spans=int(typed["pred_total"]),
            typed=typed,
            untyped=untyped,
            untyped_minus_typed=MetricDelta(
                precision=untyped["precision"] - typed["precision"],
                recall=untyped["recall"] - typed["recall"],
                f1=untyped["f1"] - typed["f1"],
            ),
        )
    return report


def _containment_span_key(span: CharSpan, *, mode: SpanMode) -> ContainmentSpanKey:
    if mode == "typed":
        return (span.label, span.start, span.end)
    if mode == "untyped":
        return (None, span.start, span.end)
    msg = f"unsupported span mode: {mode!r}"
    raise ValueError(msg)


def _span_matches_containment(
    predicted: ContainmentSpanKey,
    gold: ContainmentSpanKey,
    *,
    direction: Literal["pred_in_gold", "gold_in_pred"],
) -> bool:
    pred_label, pred_start, pred_end = predicted
    gold_label, gold_start, gold_end = gold
    if pred_label is not None and gold_label is not None and pred_label != gold_label:
        return False
    if direction == "pred_in_gold":
        return gold_start <= pred_start and gold_end >= pred_end
    if direction == "gold_in_pred":
        return pred_start <= gold_start and pred_end >= gold_end
    msg = f"unsupported containment direction: {direction!r}"
    raise ValueError(msg)


def containment_span_counts(
    predicted: Sequence[CharSpan],
    gold: Sequence[CharSpan],
    *,
    mode: SpanMode = "typed",
) -> dict[str, int]:
    """Return OpenAI Privacy Filter-style containment span counts.

    Containment span metrics do not use strict exact-boundary equality for
    ``detection.span``. Precision gives credit when a predicted span is fully
    contained in a gold span; recall gives credit when a gold span is fully
    contained in a predicted span. ``mode='untyped'`` applies the same geometry
    while ignoring category identity.

    Returns:
        The containment counts: ``precision_tp``, ``recall_tp``, ``pred_total``, and ``gold_total``.

    """
    predicted_keys = [_containment_span_key(span, mode=mode) for span in predicted]
    gold_keys = [_containment_span_key(span, mode=mode) for span in gold]
    matched_predicted = {
        pred_idx
        for pred_idx, pred_key in enumerate(predicted_keys)
        for gold_key in gold_keys
        if _span_matches_containment(pred_key, gold_key, direction="pred_in_gold")
    }
    matched_gold = {
        gold_idx
        for gold_idx, gold_key in enumerate(gold_keys)
        for pred_key in predicted_keys
        if _span_matches_containment(pred_key, gold_key, direction="gold_in_pred")
    }
    return {
        "precision_tp": len(matched_predicted),
        "recall_tp": len(matched_gold),
        "pred_total": len(predicted_keys),
        "gold_total": len(gold_keys),
    }


def typed_span_counts_by_label(
    predicted: Sequence[CharSpan],
    gold: Sequence[CharSpan],
) -> tuple[dict[PiiLabel, dict[str, int]], dict[PiiLabel, dict[str, int]]]:
    """Return exact and containment counts for every Meddies label in one pass.

    This is the count-level companion to the per-label PRF reports. It avoids
    recomputing each document's general metrics when several aggregate slices
    consume the same row observation.

    Returns:
        The exact counts and the containment counts, each keyed by Meddies label.

    """
    predicted_exact: dict[PiiLabel, Counter[tuple[int, int]]] = {label: Counter() for label in PII_LABELS}
    gold_exact: dict[PiiLabel, Counter[tuple[int, int]]] = {label: Counter() for label in PII_LABELS}
    predicted_by_label: dict[PiiLabel, list[CharSpan]] = {label: [] for label in PII_LABELS}
    gold_by_label: dict[PiiLabel, list[CharSpan]] = {label: [] for label in PII_LABELS}
    for span in predicted:
        if is_pii_label(span.label):
            predicted_exact[span.label][span.start, span.end] += 1
            predicted_by_label[span.label].append(span)
    for span in gold:
        if is_pii_label(span.label):
            gold_exact[span.label][span.start, span.end] += 1
            gold_by_label[span.label].append(span)
    exact: dict[PiiLabel, dict[str, int]] = {
        label: {
            "tp": sum((predicted_exact[label] & gold_exact[label]).values()),
            "pred_total": sum(predicted_exact[label].values()),
            "gold_total": sum(gold_exact[label].values()),
        }
        for label in PII_LABELS
    }
    containment: dict[PiiLabel, dict[str, int]] = {
        label: _same_label_containment_counts(predicted_by_label[label], gold_by_label[label]) for label in PII_LABELS
    }
    return exact, containment


def _same_label_containment_counts(predicted: Sequence[CharSpan], gold: Sequence[CharSpan]) -> dict[str, int]:
    return {
        "precision_tp": sum(
            any(gold_span.start <= predicted_span.start and gold_span.end >= predicted_span.end for gold_span in gold)
            for predicted_span in predicted
        ),
        "recall_tp": sum(
            any(
                predicted_span.start <= gold_span.start and predicted_span.end >= gold_span.end
                for predicted_span in predicted
            )
            for gold_span in gold
        ),
        "pred_total": len(predicted),
        "gold_total": len(gold),
    }


def containment_span_prf_from_counts(
    counts: Mapping[str, int | float],
) -> ContainmentSpanMetricBlock:
    precision_tp = float(counts["precision_tp"])
    recall_tp = float(counts["recall_tp"])
    pred_total = float(counts["pred_total"])
    gold_total = float(counts["gold_total"])
    precision = _safe_div(precision_tp, pred_total)
    recall = _safe_div(recall_tp, gold_total)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return ContainmentSpanMetricBlock(
        precision_tp=precision_tp,
        recall_tp=recall_tp,
        pred_total=pred_total,
        gold_total=gold_total,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def containment_span_prf(
    predicted: Sequence[CharSpan],
    gold: Sequence[CharSpan],
    *,
    mode: SpanMode = "typed",
) -> ContainmentSpanMetricBlock:
    return containment_span_prf_from_counts(containment_span_counts(predicted, gold, mode=mode))


def containment_span_prf_by_doc(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
    *,
    mode: SpanMode = "typed",
) -> ContainmentSpanMetricBlock:
    totals = {"precision_tp": 0, "recall_tp": 0, "pred_total": 0, "gold_total": 0}
    for doc_id in sorted(set(predicted_by_doc) | set(gold_by_doc)):
        counts = containment_span_counts(
            predicted_by_doc.get(doc_id, ()),
            gold_by_doc.get(doc_id, ()),
            mode=mode,
        )
        totals["precision_tp"] += counts["precision_tp"]
        totals["recall_tp"] += counts["recall_tp"]
        totals["pred_total"] += counts["pred_total"]
        totals["gold_total"] += counts["gold_total"]
    return containment_span_prf_from_counts(totals)


def containment_span_prf_by_label(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
) -> dict[str, ContainmentSpanMetricBlock]:
    """Per-label typed containment-span PRF, micro-averaged over docs.

    Mirrors ``exact_span_prf_by_label`` for the containment count family. Every
    allowed label is reported; a label seen only in gold yields zeroed PRF.

    Returns:
        Precision, recall, and F1 for each label, computed from that label's containment counts summed across every doc.

    """
    totals = _counts_by_label(
        predicted_by_doc,
        gold_by_doc,
        count_fn=lambda predicted, gold: containment_span_counts(predicted, gold, mode="typed"),
        count_keys=("precision_tp", "recall_tp", "pred_total", "gold_total"),
    )
    return {label: containment_span_prf_from_counts(counts) for label, counts in totals.items()}


def _containment_untyped_minus_typed(
    *,
    typed: MetricBlockForDelta,
    untyped: MetricBlockForDelta,
) -> MetricDelta:
    return MetricDelta(
        precision=untyped["precision"] - typed["precision"],
        recall=untyped["recall"] - typed["recall"],
        f1=untyped["f1"] - typed["f1"],
    )


def containment_span_report_by_doc(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
) -> ContainmentSpanDocReport:
    """Return containment typed/untyped containment span metrics.

    Returns:
        The typed metric at the top level, which older callers read directly, alongside the ``typed``, ``untyped``, and
        ``untyped_minus_typed`` sub-reports.

    """
    typed = containment_span_prf_by_doc(predicted_by_doc, gold_by_doc, mode="typed")
    untyped = containment_span_prf_by_doc(predicted_by_doc, gold_by_doc, mode="untyped")
    return {
        **typed,
        "typed": typed,
        "untyped": untyped,
        "untyped_minus_typed": _containment_untyped_minus_typed(typed=typed, untyped=untyped),
    }


def containment_span_slice_report_by_doc(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
    doc_slices: Mapping[str, Sequence[str]],
    *,
    slice_names: Sequence[str] | None = None,
) -> dict[str, ContainmentSpanSliceReport]:
    """Compute containment span metrics for pre-declared doc slices.

    Returns:
        One containment-span report per declared slice, keyed by the slice name.

    """
    inferred_names = sorted({name for names in doc_slices.values() for name in names})
    ordered_names = list(slice_names) if slice_names is not None else inferred_names
    report: dict[str, ContainmentSpanSliceReport] = {}
    for slice_name in ordered_names:
        doc_ids = sorted(doc_id for doc_id, names in doc_slices.items() if slice_name in set(names))
        predicted_subset = {
            doc_id: predicted_by_doc.get(doc_id, ())
            for doc_id in doc_ids
            if doc_id in predicted_by_doc or doc_id in gold_by_doc
        }
        gold_subset = {
            doc_id: gold_by_doc.get(doc_id, ())
            for doc_id in doc_ids
            if doc_id in predicted_by_doc or doc_id in gold_by_doc
        }
        typed = containment_span_prf_by_doc(predicted_subset, gold_subset, mode="typed")
        untyped = containment_span_prf_by_doc(predicted_subset, gold_subset, mode="untyped")
        report[slice_name] = ContainmentSpanSliceReport(
            support_docs=len(doc_ids),
            support_gold_spans=int(typed["gold_total"]),
            support_predicted_spans=int(typed["pred_total"]),
            typed=typed,
            untyped=untyped,
            untyped_minus_typed=_containment_untyped_minus_typed(typed=typed, untyped=untyped),
        )
    return report


_EXACT_SPAN_METRIC_FIELDS = (
    "tp",
    "pred_total",
    "gold_total",
    "precision",
    "recall",
    "f1",
)
_CONTAINMENT_SPAN_METRIC_FIELDS = (
    "precision_tp",
    "recall_tp",
    "pred_total",
    "gold_total",
    "precision",
    "recall",
    "f1",
)


def _float_value(value: object) -> float:
    if isinstance(value, str | bytes | bytearray):
        return float(value)
    if isinstance(value, SupportsFloat):
        return float(value)
    msg = f"metric value is not numeric: {value!r}"
    raise TypeError(msg)


def _coerce_metric_block(
    block: Mapping[str, object] | None,
    *,
    fallback: Mapping[str, object] | None = None,
) -> ExactSpanMetricBlock:
    source = block if block is not None else fallback
    source = source or {}
    return ExactSpanMetricBlock(
        tp=_float_value(source.get("tp", 0.0)),
        pred_total=_float_value(source.get("pred_total", 0.0)),
        gold_total=_float_value(source.get("gold_total", 0.0)),
        precision=_float_value(source.get("precision", 0.0)),
        recall=_float_value(source.get("recall", 0.0)),
        f1=_float_value(source.get("f1", 0.0)),
    )


def _exact_block_to_containment_span_block(
    block: ExactSpanMetricBlock,
) -> ContainmentSpanMetricBlock:
    return ContainmentSpanMetricBlock(
        precision_tp=block["tp"],
        recall_tp=block["tp"],
        pred_total=block["pred_total"],
        gold_total=block["gold_total"],
        precision=block["precision"],
        recall=block["recall"],
        f1=block["f1"],
    )


def _coerce_containment_span_metric_block(
    block: Mapping[str, object] | None,
    *,
    fallback_exact: ExactSpanMetricBlock,
) -> ContainmentSpanMetricBlock:
    if block is None:
        return _exact_block_to_containment_span_block(fallback_exact)
    fallback = _exact_block_to_containment_span_block(fallback_exact)
    return ContainmentSpanMetricBlock(
        precision_tp=_float_value(block.get("precision_tp", fallback["precision_tp"])),
        recall_tp=_float_value(block.get("recall_tp", fallback["recall_tp"])),
        pred_total=_float_value(block.get("pred_total", fallback["pred_total"])),
        gold_total=_float_value(block.get("gold_total", fallback["gold_total"])),
        precision=_float_value(block.get("precision", fallback["precision"])),
        recall=_float_value(block.get("recall", fallback["recall"])),
        f1=_float_value(block.get("f1", fallback["f1"])),
    )


def _metric_delta_from_sources(
    source: Mapping[str, object] | None,
    *,
    typed: MetricBlockForDelta,
    untyped: MetricBlockForDelta,
) -> MetricDelta:
    if source is None:
        return _containment_untyped_minus_typed(typed=typed, untyped=untyped)
    return MetricDelta(
        precision=_float_value(source.get("precision", untyped["precision"] - typed["precision"])),
        recall=_float_value(source.get("recall", untyped["recall"] - typed["recall"])),
        f1=_float_value(source.get("f1", untyped["f1"] - typed["f1"])),
    )


def _string_keyed_report(value: object) -> MetricSliceReport:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _string_keyed_block(value: object) -> Mapping[str, object] | None:
    """Read one metric block out of an untrusted payload, keying it by string.

    ``isinstance(value, Mapping)`` alone leaves the key type unknown, so the block is rebuilt with
    string keys rather than asserted to have them.

    Returns:
        The block when the value is a mapping, else ``None``.

    """
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _coerce_containment_span_report(
    metrics: Mapping[str, object],
    *,
    typed_fallback: ExactSpanMetricBlock,
    untyped_fallback: ExactSpanMetricBlock,
    exact_slices: Mapping[str, object],
) -> ContainmentSpanReport:
    source = metrics.get("containment_span")
    if not isinstance(source, Mapping):
        typed = _exact_block_to_containment_span_block(typed_fallback)
        untyped = _exact_block_to_containment_span_block(untyped_fallback)
        slices: MetricSliceReport = {}
        for slice_name, raw_bucket in exact_slices.items():
            if not isinstance(raw_bucket, Mapping):
                continue
            slice_typed = _coerce_metric_block(
                _string_keyed_block(raw_bucket.get("typed")),
                fallback=typed_fallback,
            )
            slice_untyped = _coerce_metric_block(
                _string_keyed_block(raw_bucket.get("untyped")),
                fallback=untyped_fallback,
            )
            containment_slice_typed = _exact_block_to_containment_span_block(slice_typed)
            containment_slice_untyped = _exact_block_to_containment_span_block(slice_untyped)
            slices[slice_name] = {
                **{
                    key: raw_bucket.get(key)
                    for key in (
                        "support_docs",
                        "support_gold_spans",
                        "support_predicted_spans",
                    )
                    if key in raw_bucket
                },
                "typed": containment_slice_typed,
                "untyped": containment_slice_untyped,
                "untyped_minus_typed": _containment_untyped_minus_typed(
                    typed=containment_slice_typed,
                    untyped=containment_slice_untyped,
                ),
            }
        return ContainmentSpanReport(
            precision_tp=typed["precision_tp"],
            recall_tp=typed["recall_tp"],
            pred_total=typed["pred_total"],
            gold_total=typed["gold_total"],
            precision=typed["precision"],
            recall=typed["recall"],
            f1=typed["f1"],
            typed=typed,
            untyped=untyped,
            untyped_minus_typed=_containment_untyped_minus_typed(typed=typed, untyped=untyped),
            slices=slices,
        )

    typed_source = source.get("typed")
    untyped_source = source.get("untyped")
    typed = _coerce_containment_span_metric_block(
        _string_keyed_block(typed_source),
        fallback_exact=typed_fallback,
    )
    untyped = _coerce_containment_span_metric_block(
        _string_keyed_block(untyped_source),
        fallback_exact=untyped_fallback,
    )
    delta_source = source.get("untyped_minus_typed")
    delta = _metric_delta_from_sources(
        _string_keyed_block(delta_source),
        typed=typed,
        untyped=untyped,
    )
    return ContainmentSpanReport(
        precision_tp=typed["precision_tp"],
        recall_tp=typed["recall_tp"],
        pred_total=typed["pred_total"],
        gold_total=typed["gold_total"],
        precision=typed["precision"],
        recall=typed["recall"],
        f1=typed["f1"],
        typed=typed,
        untyped=untyped,
        untyped_minus_typed=delta,
        slices=_string_keyed_report(source.get("slices")),
    )


def coerce_exact_span_report(metrics: Mapping[str, object]) -> ExactSpanReport:
    """Normalize legacy scalar PRF payloads into the typed/untyped schema.

    Returns:
        The same numbers in the typed/untyped schema, so a legacy scalar payload and a current one read identically.

    """
    typed_source = metrics.get("typed")
    typed = _coerce_metric_block(
        _string_keyed_block(typed_source),
        fallback=metrics,
    )
    untyped_source = metrics.get("untyped")
    untyped = _coerce_metric_block(
        _string_keyed_block(untyped_source),
        fallback=typed,
    )
    delta_source = metrics.get("untyped_minus_typed")
    delta = _metric_delta_from_sources(
        _string_keyed_block(delta_source),
        typed=typed,
        untyped=untyped,
    )
    exact_slices = _string_keyed_report(metrics.get("slices"))
    containment_span = _coerce_containment_span_report(
        metrics,
        typed_fallback=typed,
        untyped_fallback=untyped,
        exact_slices=exact_slices,
    )
    return ExactSpanReport(
        tp=typed["tp"],
        pred_total=typed["pred_total"],
        gold_total=typed["gold_total"],
        precision=typed["precision"],
        recall=typed["recall"],
        f1=typed["f1"],
        typed=typed,
        untyped=untyped,
        untyped_minus_typed=delta,
        slices=exact_slices,
        containment_span=containment_span,
    )
