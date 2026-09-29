from __future__ import annotations

# ruff: file-ignore[print]
# reason: results and progress travel back through the streamed run log, because Modal's large-result blob path is
# reason: unimplemented in this workspace.
# ruff: file-ignore[import-private-name]
# reason: this orchestrator composes the BIOES training internals `_prepare_rows`, `_select_source_rows`, `_batches`,
# reason: `_collate`, checkpoint helpers, data loaders, evaluation helpers, and runtime builders in their required
# reason: order. Each private import remains inside the training Module; a broad public facade would hide the
# reason: lifecycle invariants that `run_smoke_training` owns.
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from meddies_pii.annotations.bioes import ENTITY_LABELS
from meddies_pii.evaluation.span_metrics import coerce_exact_span_report
from meddies_pii.json_types import JsonObject, as_json_object
from meddies_pii.training.bioes.data.artifacts import _save_backbone_adapter, _save_classifier_state
from meddies_pii.training.bioes.data.preparation import _prepare_rows, _select_source_rows
from meddies_pii.training.bioes.eval.harness import new_slice_filter_report

from .batching import _batches, _collate
from .checkpointing import (
    _checkpoint_path,
    _load_training_checkpoint,
    _resolve_training_loop_plan,
    _save_training_checkpoint,
)
from .config import (
    SmokeTrainingConfig,
    SmokeTrainingPayload,
    SmokeTrainingResult,
    validate_smoke_training_config,
)
from .data_loading import _load_rows, _load_targeted_eval_selection
from .evaluation import _evaluate, _targeted_eval_quality_report
from .packing import collate_packed_units
from .packing_runtime import prepare_training_units
from .runtime import (
    _build_artifacts,
    _cuda_memory_report,
    _optimizer_for_config,
    _package_versions,
    _resolve_viterbi_transition_biases,
    _row_hash,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _require_json_object(value: object, *, source: str) -> JsonObject:
    json_object = as_json_object(value)
    if json_object is None:
        msg = f"{source} produced a non-JSON object"
        raise TypeError(msg)
    return json_object


def _log_smoke(message: str) -> None:
    print(f"bioes_smoke: {message}", flush=True)


# reason: run smoke owns resolve and build together; splitting would fragment diagnostics.
def run_smoke_training(  # ruff: ignore[too-many-locals,too-many-statements]
    config: SmokeTrainingConfig,
    *,
    # reason: a named scratch root inside the Modal container, not a shared-tmp security hazard. It is a
    # reason: default the caller overrides, and it must be stable rather than randomized so a resumed smoke
    # reason: run in the same container finds the checkpoints the previous one wrote.
    artifact_root: str = "/tmp/meddies-pii-bioes-smoke",  # ruff: ignore[hardcoded-temp-file]
    artifact_commit: Callable[[], None] | None = None,
) -> SmokeTrainingPayload:
    validate_smoke_training_config(config)
    entity_labels = ENTITY_LABELS
    viterbi_transition_biases = _resolve_viterbi_transition_biases(
        transition_biases=config.viterbi_transition_biases,
        calibration_path=config.viterbi_calibration_path,
    )
    _log_smoke(
        "build_artifacts start "
        f"backend={config.backend} model_id={config.model_id} "
        f"max_length={config.max_length} lora_rank={config.lora_rank} "
        f"lora_alpha={config.lora_alpha if config.lora_alpha is not None else max(8, config.lora_rank * 2)}",
    )
    artifacts = _build_artifacts(config)
    id_to_label = {idx: label for label, idx in artifacts.label_to_id.items()}
    _log_smoke(
        f"build_artifacts done device={next(artifacts.tagger.parameters()).device} labels={len(artifacts.label_vocab)}",
    )

    train_slice_filter_report = new_slice_filter_report()
    eval_slice_filter_report = new_slice_filter_report()

    _log_smoke(
        "load_train_source start "
        f"dataset={config.dataset_id}/{config.train_config} split={config.dataset_split} "
        f"limit={config.train_limit}",
    )
    train_source_rows, train_source_stats = _select_source_rows(
        _load_rows(
            config.train_config,
            config.train_limit,
            dataset_id=config.dataset_id,
            split=config.dataset_split,
            revision=config.dataset_revision,
        ),
        limit=None,
        allow_label_repairs=config.allow_label_repairs,
        require_label_json=config.require_label_json,
        sort_by_length=False,
        slice_filter_report=train_slice_filter_report,
    )
    _log_smoke(f"load_train_source done selected={len(train_source_rows)} candidates={train_source_stats.candidates}")
    _log_smoke(
        "load_eval_source start "
        f"dataset={config.resolved_eval_dataset_id()}/{config.eval_config} "
        f"split={config.eval_dataset_split or config.dataset_split} "
        f"limit={config.eval_limit}",
    )
    eval_source_rows, eval_source_stats = _select_source_rows(
        _load_rows(
            config.eval_config,
            config.eval_limit,
            dataset_id=config.resolved_eval_dataset_id(),
            split=config.eval_dataset_split or config.dataset_split,
            revision=config.resolved_eval_revision(),
        ),
        limit=None,
        allow_label_repairs=config.allow_label_repairs,
        require_label_json=config.require_label_json,
        sort_by_length=False,
        slice_filter_report=eval_slice_filter_report,
    )
    _log_smoke(f"load_eval_source done selected={len(eval_source_rows)} candidates={eval_source_stats.candidates}")
    _log_smoke("prepare_train_rows start")
    train_rows, train_stats = _prepare_rows(
        train_source_rows,
        artifacts.tokenizer,
        max_length=config.max_length,
        limit=config.train_limit,
        id_to_label=id_to_label,
        entity_labels=entity_labels,
        slice_filter_report=train_slice_filter_report,
    )
    if not train_rows:
        msg = f"Prepared 0 train rows for limit={config.train_limit}; stats={asdict(train_stats)}"
        raise RuntimeError(msg)
    _log_smoke(
        "prepare_train_rows done "
        f"accepted={len(train_rows)} candidates={train_stats.candidates} "
        f"skipped_round_trip={train_stats.skipped_round_trip} "
        f"skipped_truncated={train_stats.skipped_truncated} "
        f"skipped_alignment={train_stats.skipped_alignment}",
    )
    _log_smoke("prepare_eval_rows start")
    eval_rows, eval_stats = _prepare_rows(
        eval_source_rows,
        artifacts.tokenizer,
        max_length=config.max_length,
        limit=config.eval_limit,
        id_to_label=id_to_label,
        entity_labels=entity_labels,
        slice_filter_report=eval_slice_filter_report,
    )
    if not eval_rows:
        msg = f"Prepared 0 eval rows for limit={config.eval_limit}; stats={asdict(eval_stats)}"
        raise RuntimeError(msg)
    _log_smoke(
        "prepare_eval_rows done "
        f"accepted={len(eval_rows)} candidates={eval_stats.candidates} "
        f"skipped_round_trip={eval_stats.skipped_round_trip} "
        f"skipped_truncated={eval_stats.skipped_truncated} "
        f"skipped_alignment={eval_stats.skipped_alignment}",
    )
    targeted_eval_selection = _load_targeted_eval_selection(
        config,
        tokenizer=artifacts.tokenizer,
        id_to_label=id_to_label,
        entity_labels=entity_labels,
    )
    targeted_eval_rows = list(targeted_eval_selection.prepared_rows) if targeted_eval_selection is not None else []

    model = artifacts.tagger
    device = str(next(model.parameters()).device)
    optimizer = _optimizer_for_config(model, config, device=torch.device(device))
    loaded_checkpoint = None
    if config.resume_from_checkpoint is not None:
        loaded_checkpoint = _load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_dir=Path(config.resume_from_checkpoint),
            device=device,
        )
        _log_smoke(
            f"resume_checkpoint loaded path={loaded_checkpoint.path} completed_steps={loaded_checkpoint.completed_steps}",
        )
    _log_smoke(f"initial_eval start eval_examples={len(eval_rows)}")
    initial_metrics = coerce_exact_span_report(
        _evaluate(
            artifacts,
            eval_rows,
            transition_biases=viterbi_transition_biases,
        ),
    )
    _log_smoke(
        "initial_eval done "
        f"exact_f1={initial_metrics['f1']:.6f} "
        f"precision={initial_metrics['precision']:.6f} "
        f"recall={initial_metrics['recall']:.6f}",
    )
    initial_targeted_metrics = (
        coerce_exact_span_report(
            _evaluate(
                artifacts,
                targeted_eval_rows,
                transition_biases=viterbi_transition_biases,
            ),
        )
        if targeted_eval_selection is not None
        else None
    )
    train_losses: list[float] = list(loaded_checkpoint.train_loss_values) if loaded_checkpoint is not None else []

    prepared_training_units = prepare_training_units(
        config,
        model=model,
        tokenizer=artifacts.tokenizer,
        train_rows=train_rows,
        device=device,
        log=_log_smoke,
    )
    training_units = prepared_training_units.units
    train_packed_rows = prepared_training_units.packed_rows
    train_packing_utilization = prepared_training_units.utilization
    train_packing_boundary_token_count = prepared_training_units.boundary_token_count
    packing_attention_probe = prepared_training_units.attention_probe

    loop_plan = _resolve_training_loop_plan(config, training_unit_count=len(training_units))
    batch_sequence = list(_batches(training_units, config.batch_size))
    resume_completed_steps = loaded_checkpoint.completed_steps if loaded_checkpoint is not None else 0
    batch_iter = itertools.cycle(batch_sequence)
    batches_to_skip = (resume_completed_steps * config.gradient_accumulation_steps) % len(batch_sequence)
    for _ in range(batches_to_skip):
        next(batch_iter)
    _log_smoke(
        "train_loop start "
        f"steps={loop_plan.resolved_steps} epochs={loop_plan.epochs} "
        f"steps_per_epoch={loop_plan.steps_per_epoch} batch_size={config.batch_size} "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps} "
        f"train_examples={len(train_rows)} logging_steps={config.logging_steps} "
        f"checkpoint_every_steps={config.checkpoint_every_steps} "
        f"resume_completed_steps={resume_completed_steps}",
    )
    train_start_time = time.monotonic()
    latest_checkpoint = loaded_checkpoint.path if loaded_checkpoint is not None else None
    for step_index in range(loop_plan.resolved_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _accum_step in range(config.gradient_accumulation_steps):
            batch_rows = next(batch_iter)
            batch = (
                collate_packed_units(batch_rows, device=device)
                if config.packing
                else _collate(batch_rows, artifacts.tokenizer, device)
            )
            out = model(**batch)
            loss = out["loss"]
            # reason: narrowing a model output. The model is called with labels, so the branch that omits `loss`
            # reason: cannot be taken here, though the output mapping is typed loosely enough to admit None. The
            # reason: real training guard is the non-finite check on the next line, which does raise.
            assert loss is not None  # ruff: ignore[assert]
            if not torch.isfinite(loss):
                msg = "Encountered non-finite loss during smoke training"
                raise RuntimeError(msg)
            accumulated_loss += float(loss.detach().cpu().item())
            (loss / config.gradient_accumulation_steps).backward()
        optimizer.step()
        step_loss = accumulated_loss / config.gradient_accumulation_steps
        train_losses.append(step_loss)
        local_step = step_index + 1
        global_step = resume_completed_steps + local_step
        elapsed_s = time.monotonic() - train_start_time
        sec_per_step = elapsed_s / local_step
        eta_s = sec_per_step * (loop_plan.resolved_steps - local_step)
        if config.logging_steps and (local_step % config.logging_steps == 0 or local_step == loop_plan.resolved_steps):
            _log_smoke(
                "train_step "
                f"step={local_step}/{loop_plan.resolved_steps} "
                f"global_step={global_step} loss={step_loss:.6f} "
                f"elapsed_s={elapsed_s:.1f} sec_per_step={sec_per_step:.2f} "
                f"eta_s={eta_s:.1f}",
            )
        if config.checkpoint_every_steps and (
            global_step % config.checkpoint_every_steps == 0 or local_step == loop_plan.resolved_steps
        ):
            latest_checkpoint = _save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_dir=_checkpoint_path(artifact_root, config.backend, global_step),
                config=config,
                completed_steps=global_step,
                train_loss_values=train_losses,
                commit_callback=artifact_commit,
            )
            _log_smoke(f"checkpoint saved path={latest_checkpoint} completed_steps={global_step}")

    _log_smoke(f"final_eval start eval_examples={len(eval_rows)}")
    metrics = coerce_exact_span_report(
        _evaluate(
            artifacts,
            eval_rows,
            transition_biases=viterbi_transition_biases,
        ),
    )
    _log_smoke(
        f"final_eval done exact_f1={metrics['f1']:.6f} precision={metrics['precision']:.6f} \
recall={metrics['recall']:.6f}",
    )
    targeted_metrics = (
        coerce_exact_span_report(
            _evaluate(
                artifacts,
                targeted_eval_rows,
                transition_biases=viterbi_transition_biases,
            ),
        )
        if targeted_eval_selection is not None
        else None
    )
    artifact_dir = Path(artifact_root) / config.backend
    classifier_artifact = _save_classifier_state(model, artifact_dir)
    model_artifact = _save_backbone_adapter(model.backbone, artifact_dir)
    initial_untyped_metrics = initial_metrics["untyped"]
    initial_containment_span_metrics = initial_metrics["containment_span"]
    initial_containment_span_untyped_metrics = initial_containment_span_metrics["untyped"]
    typed_metrics = metrics["typed"]
    untyped_metrics = metrics["untyped"]
    untyped_minus_typed = metrics["untyped_minus_typed"]
    containment_span_metrics = metrics["containment_span"]
    containment_span_typed_metrics = containment_span_metrics["typed"]
    containment_span_untyped_metrics = containment_span_metrics["untyped"]
    containment_span_untyped_minus_typed = containment_span_metrics["untyped_minus_typed"]

    result = SmokeTrainingResult(
        backend=config.backend,
        steps=loop_plan.resolved_steps,
        epochs=config.epochs,
        steps_per_epoch=loop_plan.steps_per_epoch,
        fused_adamw=config.fused_adamw,
        config=asdict(config),
        viterbi_transition_biases=viterbi_transition_biases,
        mean_train_loss=sum(train_losses) / len(train_losses),
        first_train_loss=train_losses[0],
        final_train_loss=train_losses[-1],
        train_loss_delta=train_losses[-1] - train_losses[0],
        train_loss_values=[float(value) for value in train_losses],
        initial_eval_exact_span_f1=initial_metrics["f1"],
        initial_eval_exact_span_precision=initial_metrics["precision"],
        initial_eval_exact_span_recall=initial_metrics["recall"],
        initial_eval_exact_span_untyped_f1=initial_untyped_metrics["f1"],
        initial_eval_exact_span_untyped_precision=initial_untyped_metrics["precision"],
        initial_eval_exact_span_untyped_recall=initial_untyped_metrics["recall"],
        initial_eval_containment_span_f1=initial_containment_span_metrics["f1"],
        initial_eval_containment_span_precision=initial_containment_span_metrics["precision"],
        initial_eval_containment_span_recall=initial_containment_span_metrics["recall"],
        initial_eval_containment_span_untyped_f1=initial_containment_span_untyped_metrics["f1"],
        initial_eval_containment_span_untyped_precision=initial_containment_span_untyped_metrics["precision"],
        initial_eval_containment_span_untyped_recall=initial_containment_span_untyped_metrics["recall"],
        eval_exact_span_f1=metrics["f1"],
        eval_exact_span_precision=metrics["precision"],
        eval_exact_span_recall=metrics["recall"],
        eval_exact_span_untyped_f1=untyped_metrics["f1"],
        eval_exact_span_untyped_precision=untyped_metrics["precision"],
        eval_exact_span_untyped_recall=untyped_metrics["recall"],
        eval_containment_span_f1=containment_span_metrics["f1"],
        eval_containment_span_precision=containment_span_metrics["precision"],
        eval_containment_span_recall=containment_span_metrics["recall"],
        eval_containment_span_untyped_f1=containment_span_untyped_metrics["f1"],
        eval_containment_span_untyped_precision=containment_span_untyped_metrics["precision"],
        eval_containment_span_untyped_recall=containment_span_untyped_metrics["recall"],
        eval_exact_span_f1_delta=metrics["f1"] - initial_metrics["f1"],
        eval_containment_span_f1_delta=containment_span_metrics["f1"] - initial_containment_span_metrics["f1"],
        eval_exact_span_typed=typed_metrics,
        eval_exact_span_untyped=untyped_metrics,
        eval_exact_span_untyped_minus_typed=untyped_minus_typed,
        eval_containment_span_typed=containment_span_typed_metrics,
        eval_containment_span_untyped=containment_span_untyped_metrics,
        eval_containment_span_untyped_minus_typed=containment_span_untyped_minus_typed,
        eval_adversarial_slices=_require_json_object(metrics["slices"], source="evaluation slices"),
        eval_containment_adversarial_slices=_require_json_object(
            containment_span_metrics["slices"],
            source="containment evaluation slices",
        ),
        targeted_eval=(
            _targeted_eval_quality_report(
                targeted_eval_selection,
                initial_metrics=initial_targeted_metrics,
                final_metrics=targeted_metrics,
            )
            if targeted_eval_selection is not None
            and initial_targeted_metrics is not None
            and targeted_metrics is not None
            else None
        ),
        train_slice_filter_report=_require_json_object(train_slice_filter_report, source="train slice filter"),
        eval_slice_filter_report=_require_json_object(eval_slice_filter_report, source="evaluation slice filter"),
        train_examples=len(train_rows),
        eval_examples=len(eval_rows),
        train_packed_examples=len(train_packed_rows) if train_packed_rows is not None else None,
        train_packing_utilization=train_packing_utilization,
        train_packing_boundary_token_count=train_packing_boundary_token_count,
        packing_attention_probe=packing_attention_probe,
        train_row_uids=[row.uid for row in train_rows],
        eval_row_uids=[row.uid for row in eval_rows],
        train_row_hashes=[_row_hash(row) for row in train_rows],
        eval_row_hashes=[_row_hash(row) for row in eval_rows],
        package_versions=_package_versions(),
        cuda_memory=_cuda_memory_report(device),
        model_artifact=model_artifact,
        classifier_artifact=classifier_artifact,
        latest_checkpoint=latest_checkpoint,
        resume_from_checkpoint=(loaded_checkpoint.path if loaded_checkpoint is not None else None),
        completed_steps=resume_completed_steps + loop_plan.resolved_steps,
        notes=[
            (
                "Smoke phase intentionally avoids chunking; rows that cannot round-trip raw/tag structure "
                "or token alignment are excluded."
            ),
            "Tagged text / spans are treated as the canonical supervision source; grouped label JSON is metadata only.",
            (
                "Headline metric is typed exact-span precision/recall/F1 on held-out real rows after "
                "constrained BIOES Viterbi decoding."
            ),
            "Untyped exact-span metrics ignore labels but still require document and character-boundary matches.",
            (
                "containment eval_containment_span_* metrics use containment span matching (predicted "
                "span contains gold value, or gold contains predicted); exact-span metrics remain the "
                "stricter regression gate."
            ),
            (
                "Adversarial slice membership is assigned from source text/gold spans before row "
                "preparation; slice filter reports expose attempted, filtered, and prepared counts."
            ),
            f"fused_adamw={json.dumps(config.fused_adamw)}",
            f"lora_rank={json.dumps(config.lora_rank)}",
            f"lora_alpha={
                json.dumps(config.lora_alpha if config.lora_alpha is not None else max(8, config.lora_rank * 2))
            }",
            f"logging_steps={json.dumps(config.logging_steps)}",
            f"checkpoint_every_steps={json.dumps(config.checkpoint_every_steps)}",
            f"resume_from_checkpoint={json.dumps(config.resume_from_checkpoint)}",
            f"packing={json.dumps(config.packing)}",
            f"length_bucketing={json.dumps(config.length_bucketing)}",
            f"train_packed_examples={json.dumps(len(train_packed_rows) if train_packed_rows is not None else None)}",
            f"train_packing_utilization={json.dumps(train_packing_utilization)}",
            f"train_packing_boundary_token_count={json.dumps(train_packing_boundary_token_count)}",
            f"packing_attention_probe={json.dumps(packing_attention_probe, sort_keys=True)}",
            f"viterbi_transition_biases={json.dumps(viterbi_transition_biases, sort_keys=True)}",
            f"train_source_selection={json.dumps(asdict(train_source_stats), sort_keys=True)}",
            f"train_preparation={json.dumps(asdict(train_stats), sort_keys=True)}",
            f"eval_source_selection={json.dumps(asdict(eval_source_stats), sort_keys=True)}",
            f"eval_preparation={json.dumps(asdict(eval_stats), sort_keys=True)}",
            f"target_eval_slice={json.dumps(config.target_eval_slice)}",
            f"target_eval_limit={json.dumps(config.target_eval_limit)}",
        ],
    )
    return _require_json_object(asdict(result), source="smoke training result")


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_smoke_training(SmokeTrainingConfig()), indent=2))
