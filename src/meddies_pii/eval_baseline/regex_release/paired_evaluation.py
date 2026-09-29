"""Paired model-versus-regex shard execution and score-time evidence."""

from __future__ import annotations

import re
import time
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, cast

from meddies_pii.eval_baseline.baseline import run as baseline_run
from meddies_pii.eval_baseline.baseline.aggregate import spans_from_values
from meddies_pii.eval_baseline.baseline.datasets import canonical_language
from meddies_pii.eval_baseline.baseline.views import (
    MODEL_CORE_VIEW,
    MODEL_PLUS_REGEX_VIEW,
    REGEX_EVALUATION_VIEWS,
)
from meddies_pii.eval_baseline.regex_release.regex_bootstrap import (
    PairedDocumentCounts,
    length_bucket,
    source_row_sha256,
)
from meddies_pii.eval_baseline.regex_release.regex_ignore_list import (
    audited_spans_for_row,
    count_ignored_false_positives,
)
from meddies_pii.evaluation.identity import dataset_shard_identity, file_sha256, is_sha256
from meddies_pii.evaluation.span_metrics import exact_span_counts
from meddies_pii.regex_runtime import regex_manifest
from meddies_pii.regex_runtime.postprocess import apply_regex_postprocess
from meddies_pii.spans import char_span_to_dict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from meddies_pii.eval_baseline.baseline.adapter import PiiAdapter
    from meddies_pii.eval_baseline.baseline.datasets import EvalRow
    from meddies_pii.eval_baseline.baseline.views import EvaluationView
    from meddies_pii.eval_baseline.regex_release.regex_bootstrap import LengthBucket
    from meddies_pii.eval_baseline.regex_release.regex_ignore_list import RegexIgnoreList
    from meddies_pii.eval_baseline.regex_release.regex_report import RegexCorpusManifest
    from meddies_pii.evaluation.identity import DatasetShardIdentity, EvaluationContract
    from meddies_pii.spans import CharSpan


# reason: the orchestration boundary keeps runtime dependencies separate from persisted evidence identity inputs.
# reason: both views must be validated, inferred, and persisted under one shared identity boundary.
def run_regex_view_shards(  # ruff: ignore[too-many-arguments,complex-structure,too-many-statements,too-many-locals]
    *,
    adapter: PiiAdapter,
    rows: Sequence[EvalRow],
    output_root: str | Path,
    specs: Mapping[EvaluationView, baseline_run.ShardSpec],
    evaluation_contracts: Mapping[EvaluationView, EvaluationContract],
    volume: baseline_run.VolumeLike,
    comparison_id: str,
    corpus_manifest: RegexCorpusManifest,
    force: bool = False,
) -> baseline_run.DualViewShardResult:
    """Persist model and regex views from one call to the model adapter.

    Returns:
        The paired shard results and their shared inference identity.

    Raises:
        ValueError: If comparison, view, dataset, shard, or corpus inputs violate the paired-view contract.
        RuntimeError: If existing evidence or persisted output cannot satisfy the paired-view identity contract.

    """
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", comparison_id) is None:
        msg = "comparison_id must be a filesystem-safe identifier"
        raise ValueError(msg)
    views = REGEX_EVALUATION_VIEWS
    if set(specs) != set(views) or set(evaluation_contracts) != set(views):
        msg = "regex evaluation requires model_core and model_plus_regex"
        raise ValueError(msg)
    first_spec = specs[MODEL_CORE_VIEW]
    if any(
        (spec.model, spec.dataset, spec.shard) != (first_spec.model, first_spec.dataset, first_spec.shard)
        for spec in specs.values()
    ):
        msg = "regex-view specs must score the same model dataset shard"
        raise ValueError(msg)
    if any(specs[view].view != view for view in views):
        msg = "regex-view specs must bind each named view explicitly"
        raise ValueError(msg)

    identities: dict[EvaluationView, DatasetShardIdentity] = {
        view: dataset_shard_identity(
            evaluation_contracts[view],
            dataset=specs[view].dataset,
            shard=specs[view].shard,
            rows=rows,
        )
        for view in views
    }
    environment_fingerprint = baseline_run.resolved_environment_fingerprint()
    environment_sha256 = str(environment_fingerprint["sha256"])
    manifest_sha256 = str(regex_manifest()["sha256"])
    source_row_sha256s = tuple(
        source_row_sha256(
            stable_id=row.stable_id,
            text_sha256=row.text_sha256,
            gold_spans=row.gold_spans,
            language=row.language,
            slices=tuple(sorted(row.slices)),
        )
        for row in rows
    )
    corpus_manifest.validate_persisted_fixture(
        dataset=first_spec.dataset,
        shard=first_spec.shard,
        fixture_sha256=identities[MODEL_CORE_VIEW].fixture_identity.digest,
        source_row_sha256s=source_row_sha256s,
    )
    paths: dict[EvaluationView, tuple[Path, Path, Path]] = {
        view: _regex_shard_paths(
            output_root,
            specs[view],
            comparison_id=comparison_id,
            manifest_sha256=manifest_sha256,
        )
        for view in views
    }
    existing_identity = (
        baseline_run.completed_paired_view_identity(
            paths,
            identities,
            views,
            expectation=baseline_run.PairedViewExpectation(
                regex_manifest_sha256=manifest_sha256,
                comparison_id=comparison_id,
                corpus_manifest_sha256=corpus_manifest.sha256,
                corpus_manifest_payload=corpus_manifest.to_payload(),
            ),
        )
        if not force
        else None
    )
    if existing_identity is not None:
        return baseline_run.DualViewShardResult(
            by_view={
                view: baseline_run.ShardResult(
                    spec=specs[view],
                    skipped=True,
                    rows_in=0,
                    spans_out=0,
                    labels_seen=frozenset(),
                    output_path=paths[view][0],
                    done_path=paths[view][1],
                    environment_fingerprint_sha256=environment_sha256,
                )
                for view in views
            },
            shared_inference_identity=existing_identity,
        )

    for view in views:
        paths[view][1].unlink(missing_ok=True)

    predict_start = time.perf_counter()
    model_predictions = adapter.predict([row.text for row in rows])
    elapsed_seconds = time.perf_counter() - predict_start
    if len(model_predictions) != len(rows):
        msg = f"{adapter.name} returned {len(model_predictions)} predictions for {len(rows)} rows"
        raise RuntimeError(msg)
    paired_predictions = [
        {
            MODEL_CORE_VIEW: tuple(model_spans),
            MODEL_PLUS_REGEX_VIEW: apply_regex_postprocess(row.text, model_spans, language=row.language).spans,
        }
        for row, model_spans in zip(rows, model_predictions, strict=True)
    ]
    shared_inference_identity = baseline_run.shared_inference_identity_from_rows(
        {
            "stable_id": row.stable_id,
            "text_sha256": row.text_sha256,
            "language": row.language,
            "slices": sorted(row.slices),
            "outputs": {
                view: [{"start": span.start, "end": span.end, "label": span.label} for span in prediction[view]]
                for view in views
            },
        }
        for row, prediction in zip(rows, paired_predictions, strict=True)
    )
    results: dict[EvaluationView, baseline_run.ShardResult] = {}
    for view in views:
        records: list[dict[str, object]] = []
        labels_seen: set[str] = set()
        samples: list[baseline_run.ShardSample] = []
        spans_out = 0
        for row, prediction, row_sha256 in zip(rows, paired_predictions, source_row_sha256s, strict=True):
            spans = prediction[view]
            baseline_run.validate_spans(row, spans, adapter.supported_labels)
            labels_seen.update(span.label for span in spans)
            spans_out += len(spans)
            if not samples:
                samples.extend(baseline_run.sample_spans(row, spans, limit=3))
            records.append({
                "id": row.text_sha256,
                "doc_id": row.stable_id,
                "pred_spans": [char_span_to_dict(span) for span in spans],
                "gold_spans": [char_span_to_dict(span) for span in row.gold_spans],
                "language": row.language,
                "slice": sorted(row.slices),
                "length_bucket": length_bucket(len(row.text)),
                "document_chars": len(row.text),
                "evaluation_view": view,
                "shared_inference_identity": shared_inference_identity,
                "regex_manifest_sha256": manifest_sha256,
                "comparison_id": comparison_id,
                "source_row_sha256": row_sha256,
                "corpus_manifest_sha256": corpus_manifest.sha256,
            })
        output_path, done_path, meta_path = paths[view]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_run.write_jsonl_atomically(output_path, records)
        result_sha256 = file_sha256(str(output_path))
        baseline_run.write_shard_meta(
            meta_path,
            elapsed_seconds=elapsed_seconds,
            rows_in=len(rows),
            spans_out=spans_out,
            fixture_sha256=identities[view].digest,
            result_sha256=result_sha256,
            environment_fingerprint=environment_fingerprint,
            evaluation_contract=evaluation_contracts[view],
            dataset_shard_identity=identities[view],
            provenance={
                "evaluation_view": view,
                "shared_inference_identity": shared_inference_identity,
                "regex_manifest_sha256": manifest_sha256,
                "comparison_id": comparison_id,
                "corpus_manifest": corpus_manifest.to_payload(),
                "corpus_manifest_sha256": corpus_manifest.sha256,
                "failure_state": None,
            },
        )
        results[view] = baseline_run.ShardResult(
            spec=specs[view],
            skipped=False,
            rows_in=len(rows),
            spans_out=spans_out,
            labels_seen=frozenset(labels_seen),
            output_path=output_path,
            done_path=done_path,
            samples=tuple(samples),
            elapsed_seconds=round(elapsed_seconds, 4),
            environment_fingerprint_sha256=environment_sha256,
        )
    for view in views:
        baseline_run.atomic_write_bytes(paths[view][1], b"done\n")
    volume.commit()
    return baseline_run.DualViewShardResult(
        by_view=results,
        shared_inference_identity=shared_inference_identity,
    )


def _regex_shard_paths(
    output_root: str | Path,
    spec: baseline_run.ShardSpec,
    *,
    comparison_id: str,
    manifest_sha256: str,
) -> tuple[Path, Path, Path]:
    if spec.view not in REGEX_EVALUATION_VIEWS or not is_sha256(manifest_sha256):
        msg = "regex shard path requires a named view and manifest SHA-256"
        raise ValueError(msg)
    root = (
        Path(output_root)
        / "results"
        / spec.model
        / "regex_ablation"
        / comparison_id
        / manifest_sha256
        / spec.view
        / spec.dataset
    )
    return (
        root / f"{spec.shard}.jsonl",
        root / f"{spec.shard}.done",
        root / f"{spec.shard}.meta.json",
    )


# reason: paired rows must retain span identity through every provenance check before counts collapse to integers.
def paired_regex_document_counts(  # ruff: ignore[complex-structure,too-many-statements,too-many-locals]
    model_rows: Sequence[Mapping[str, object]],
    regex_rows: Sequence[Mapping[str, object]],
    *,
    ignore_list: RegexIgnoreList | None = None,
    split: str | None = None,
) -> tuple[PairedDocumentCounts, ...]:
    """Validate paired persisted views and derive document-level exact counts.

    `ignore_list` nets audited false positives out of `model_fp`/`regex_fp` at
    the source, before the counts collapse into per-document integers — this is
    the only point downstream consumers can no longer recover span identity, so
    the exclusion must happen here or not at all. `split` (the dataset config,
    e.g. "eval" or "eval-challenge") is required whenever `ignore_list` is given,
    since ignore-list entries are keyed to a specific split and row.

    Returns:
        One exact-count record for each paired document, in input order.

    Raises:
        ValueError: If ignore-list context is incomplete or the paired rows violate identity or scoring invariants.

    """
    if ignore_list is not None and split is None:
        msg = "paired document counting needs split when ignore_list is set"
        raise ValueError(msg)
    observations: list[PairedDocumentCounts] = []
    for model_row, regex_row in zip_longest(model_rows, regex_rows):
        if model_row is None or regex_row is None:
            msg = "paired regex result artifacts have different row counts"
            raise ValueError(msg)
        identity_keys = ("id", "doc_id", "language", "slice", "gold_spans")
        if any(model_row.get(key) != regex_row.get(key) for key in identity_keys):
            msg = "paired regex result artifacts have mismatched row identity"
            raise ValueError(msg)
        shared_identity = model_row.get("shared_inference_identity")
        if (
            not is_sha256(shared_identity)
            or regex_row.get("shared_inference_identity") != shared_identity
            or model_row.get("evaluation_view") != MODEL_CORE_VIEW
            or regex_row.get("evaluation_view") != MODEL_PLUS_REGEX_VIEW
        ):
            msg = "paired regex result artifacts lack same-forward provenance"
            raise ValueError(msg)
        manifest_sha256 = model_row.get("regex_manifest_sha256")
        if not is_sha256(manifest_sha256) or regex_row.get("regex_manifest_sha256") != manifest_sha256:
            msg = "paired regex result artifacts have mismatched regex manifests"
            raise ValueError(msg)
        comparison_id = model_row.get("comparison_id")
        if not isinstance(comparison_id, str) or not comparison_id or regex_row.get("comparison_id") != comparison_id:
            msg = "paired regex result artifacts have mismatched comparison IDs"
            raise ValueError(msg)
        corpus_manifest_sha256 = model_row.get("corpus_manifest_sha256")
        persisted_source_row_sha256 = model_row.get("source_row_sha256")
        if (
            not is_sha256(corpus_manifest_sha256)
            or regex_row.get("corpus_manifest_sha256") != corpus_manifest_sha256
            or not is_sha256(persisted_source_row_sha256)
            or regex_row.get("source_row_sha256") != persisted_source_row_sha256
        ):
            msg = "paired regex result artifacts have mismatched corpus evidence"
            raise ValueError(msg)
        document_id = model_row.get("id")
        stable_id = model_row.get("doc_id")
        language = model_row.get("language")
        bucket = model_row.get("length_bucket")
        slices = model_row.get("slice")
        # reason: every field is part of one persisted row-schema predicate and shares one malformed-metadata failure.
        if (
            not isinstance(document_id, str)  # ruff: ignore[too-many-boolean-expressions]
            or not isinstance(stable_id, str)
            or not isinstance(language, str)
            or bucket not in {"short", "medium", "long"}
            or regex_row.get("length_bucket") != bucket
            or not isinstance(slices, list)
            or not all(isinstance(item, str) for item in slices)
        ):
            msg = "paired regex result row metadata is malformed"
            raise ValueError(msg)
        gold = spans_from_values(model_row.get("gold_spans"))
        computed_source_row_sha256 = source_row_sha256(
            stable_id=stable_id,
            text_sha256=document_id,
            gold_spans=gold,
            language=language,
            slices=cast("list[str]", slices),
        )
        if computed_source_row_sha256 != persisted_source_row_sha256:
            msg = "paired regex source-row identity does not match row fields"
            raise ValueError(msg)
        safety_reference = (
            *gold,
            *audited_spans_for_row(ignore_list, split=split or "", uid=stable_id),
        )
        gold_pii_chars = len(_covered_char_offsets(safety_reference))
        document_chars = _paired_document_chars(model_row, regex_row)
        model_predicted = spans_from_values(model_row.get("pred_spans"))
        regex_predicted = spans_from_values(regex_row.get("pred_spans"))
        model_counts = exact_span_counts(model_predicted, gold)
        regex_counts = exact_span_counts(regex_predicted, gold)
        model_fp_ignored = count_ignored_false_positives(
            model_predicted,
            gold,
            ignore_list,
            split=split or "",
            uid=stable_id,
        )
        regex_fp_ignored = count_ignored_false_positives(
            regex_predicted,
            gold,
            ignore_list,
            split=split or "",
            uid=stable_id,
        )
        observations.append(
            PairedDocumentCounts(
                document_id=document_id,
                language=canonical_language(language),
                length_bucket=cast("LengthBucket", bucket),
                source_row_sha256=persisted_source_row_sha256,
                corpus_manifest_sha256=corpus_manifest_sha256,
                model_tp=model_counts["tp"],
                model_fp=model_counts["pred_total"] - model_counts["tp"] - model_fp_ignored,
                model_fn=model_counts["gold_total"] - model_counts["tp"],
                regex_tp=regex_counts["tp"],
                regex_fp=regex_counts["pred_total"] - regex_counts["tp"] - regex_fp_ignored,
                regex_fn=regex_counts["gold_total"] - regex_counts["tp"],
                model_fp_ignored=model_fp_ignored,
                regex_fp_ignored=regex_fp_ignored,
                gold_pii_chars=gold_pii_chars,
                model_leaked_chars=gold_pii_chars - _covered_safety_reference_chars(safety_reference, model_predicted),
                regex_leaked_chars=gold_pii_chars - _covered_safety_reference_chars(safety_reference, regex_predicted),
                document_chars=document_chars,
                model_overredacted_chars=(
                    _overredacted_chars(safety_reference, model_predicted) if document_chars is not None else 0
                ),
                regex_overredacted_chars=(
                    _overredacted_chars(safety_reference, regex_predicted) if document_chars is not None else 0
                ),
            ),
        )
    return tuple(observations)


def _covered_char_offsets(spans: Sequence[CharSpan]) -> set[int]:
    return {offset for span in spans for offset in range(span.start, span.end)}


def _covered_safety_reference_chars(safety_reference: Sequence[CharSpan], predicted: Sequence[CharSpan]) -> int:
    """Count covered safety-reference characters, ignoring predicted labels.

    The safety reference is gold spans union exact audited ignore-list spans.

    Returns:
        The number of safety-reference character offsets covered by any prediction.

    """
    return len(_covered_char_offsets(safety_reference) & _covered_char_offsets(predicted))


def _overredacted_chars(safety_reference: Sequence[CharSpan], predicted: Sequence[CharSpan]) -> int:
    """Count predicted characters outside the character-safety reference.

    The safety reference is gold spans union exact audited ignore-list spans.

    Returns:
        The number of predicted character offsets outside the safety reference.

    """
    return len(_covered_char_offsets(predicted) - _covered_char_offsets(safety_reference))


def _paired_document_chars(model_row: Mapping[str, object], regex_row: Mapping[str, object]) -> int | None:
    """Read the document's total character count, if both paired rows carry it.

    Rows persisted before the `document_chars` schema addition (2026-08-11)
    carry neither view's field; both are read here so a mismatch between the
    two paired views is refused rather than silently resolved by picking one.

    Returns:
        The shared character count, or `None` when both legacy rows omit it.

    Raises:
        ValueError: If the paired rows disagree on the presence or value of the character count.

    """
    model_has_value = "document_chars" in model_row
    regex_has_value = "document_chars" in regex_row
    if not model_has_value and not regex_has_value:
        return None
    model_value = model_row.get("document_chars")
    regex_value = regex_row.get("document_chars")
    # reason: presence, integer shape, and cross-view equality form one document-length evidence predicate.
    if (
        not model_has_value  # ruff: ignore[too-many-boolean-expressions]
        or not regex_has_value
        or not isinstance(model_value, int)
        or isinstance(model_value, bool)
        or not isinstance(regex_value, int)
        or isinstance(regex_value, bool)
        or model_value != regex_value
    ):
        msg = "paired regex result artifacts have mismatched document_chars"
        raise ValueError(msg)
    return model_value
