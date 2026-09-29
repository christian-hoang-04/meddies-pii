from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from meddies_pii.annotations.bioes import (
    decode_bioes_from_offsets,
    viterbi_decode_logits,
)
from meddies_pii.annotations.span_records import validate_label
from meddies_pii.evaluation.span_metrics import (
    ContainmentSpanReport,
    ExactSpanReport,
    containment_span_report_by_doc,
    containment_span_slice_report_by_doc,
    exact_span_report_by_doc,
    exact_span_slice_report_by_doc,
)
from meddies_pii.training.bioes.data.manifest import row_hash_from_parts
from meddies_pii.training.bioes.eval.harness import ADVERSARIAL_SLICE_NAMES

from .batching import _collate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.spans import CharSpan
    from meddies_pii.training.bioes.data.artifacts import ProbeArtifacts
    from meddies_pii.training.bioes.data.preparation import PreparedRow
    from meddies_pii.training.bioes.eval.selection import TargetedSliceSelection


def _maybe_cudagraph_mark_step_begin(*, enabled: bool) -> None:
    if not enabled:
        return
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler else None
    if callable(marker):
        marker()


def _row_hash(row: PreparedRow) -> str:
    return row_hash_from_parts(uid=row.uid, raw=row.raw, spans=row.parsed.spans)


def _targeted_eval_quality_report(
    selection: TargetedSliceSelection,
    *,
    initial_metrics: ExactSpanReport,
    final_metrics: ExactSpanReport,
) -> dict[str, Any]:
    initial_containment_span = initial_metrics["containment_span"]
    final_containment_span = final_metrics["containment_span"]
    return {
        **selection.to_report(),
        "row_hashes": [_row_hash(row) for row in selection.prepared_rows],
        "initial_exact_span_typed": initial_metrics["typed"],
        "initial_exact_span_untyped": initial_metrics["untyped"],
        "initial_containment_span_typed": initial_containment_span["typed"],
        "initial_containment_span_untyped": initial_containment_span["untyped"],
        "final_exact_span_typed": final_metrics["typed"],
        "final_exact_span_untyped": final_metrics["untyped"],
        "final_containment_span_typed": final_containment_span["typed"],
        "final_containment_span_untyped": final_containment_span["untyped"],
        "exact_span_f1_delta": final_metrics["f1"] - initial_metrics["f1"],
        "containment_span_f1_delta": final_containment_span["f1"] - initial_containment_span["f1"],
        "final_adversarial_slice": final_metrics["slices"].get(selection.slice_name),
        "final_containment_adversarial_slice": final_containment_span["slices"].get(selection.slice_name),
    }


def _validate_eval_spans(spans: Sequence[CharSpan]) -> tuple[CharSpan, ...]:
    for span in spans:
        validate_label(span.label, field="eval span")
    return tuple(spans)


def _evaluate(
    artifacts: ProbeArtifacts,
    eval_rows: Sequence[PreparedRow],
    *,
    mark_step_begin: bool = False,
    transition_biases: dict[str, float] | None = None,
) -> ExactSpanReport:
    model = artifacts.tagger
    model.eval()
    id_to_label = {idx: label for label, idx in artifacts.label_to_id.items()}
    predicted_by_doc: dict[str, tuple[Any, ...]] = {}
    gold_by_doc: dict[str, tuple[Any, ...]] = {}
    doc_slices: dict[str, tuple[str, ...]] = {}
    device = str(next(model.parameters()).device)
    with torch.no_grad():
        for index, row in enumerate(eval_rows):
            batch = _collate([row], artifacts.tokenizer, device)
            _maybe_cudagraph_mark_step_begin(enabled=mark_step_begin)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
            pred_ids = viterbi_decode_logits(
                logits[0],
                id_to_label,
                row.tokenized.offset_mapping,
                transition_biases=transition_biases,
            )
            decoded = decode_bioes_from_offsets(row.raw, row.tokenized.offset_mapping, pred_ids, id_to_label)
            doc_key = f"{row.uid}:{index}"
            predicted_by_doc[doc_key] = decoded
            gold_by_doc[doc_key] = row.parsed.spans
            doc_slices[doc_key] = row.slices
    metrics = exact_span_report_by_doc(predicted_by_doc, gold_by_doc)
    containment_predicted_by_doc = {doc_id: _validate_eval_spans(spans) for doc_id, spans in predicted_by_doc.items()}
    containment_gold_by_doc = {doc_id: _validate_eval_spans(spans) for doc_id, spans in gold_by_doc.items()}
    containment_span_metrics = containment_span_report_by_doc(containment_predicted_by_doc, containment_gold_by_doc)
    return ExactSpanReport(
        **metrics,
        slices=exact_span_slice_report_by_doc(
            predicted_by_doc,
            gold_by_doc,
            doc_slices,
            slice_names=ADVERSARIAL_SLICE_NAMES,
        ),
        containment_span=ContainmentSpanReport(
            **containment_span_metrics,
            slices=containment_span_slice_report_by_doc(
                containment_predicted_by_doc,
                containment_gold_by_doc,
                doc_slices,
                slice_names=ADVERSARIAL_SLICE_NAMES,
            ),
        ),
    )
