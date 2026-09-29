from __future__ import annotations

# ruff: file-ignore[import-private-name]
# reason: this script reuses the training path's OWN select, prepare, collate and evaluate helpers so the number it
# reason: reports is the number training produces; re-implementing them here would let the baseline and the trainer
# reason: drift apart silently. The fix that would satisfy the rule is a public re-export inside `src/`, which this
# reason: lane does not own, so it is declared here and flagged rather than worked around.
# ruff: file-ignore[implicit-namespace-package]
# reason: this module is launched as `uv run modal run <this path>` and is never imported, so it is a script
# reason: rather than a package member. An `__init__.py` would declare this directory a package it is not, and
# reason: the sibling scripts here that run under `python` say so with a shebang instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import modal

from meddies_pii.modal_runtime import add_source_pythonpath
from meddies_pii.training.bioes.modal.train import (
    ARTIFACT_VOLUME_MOUNT,
    ARTIFACT_VOLUME_NAME,
    DEFAULT_GPU,
    H100_GPU,
    artifact_volume,
)
from meddies_pii.training.bioes.modal.train import (
    image as train_image,
)
from meddies_pii.training.bioes.reports.inference_preview import (
    DEFAULT_INFERENCE_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULT_JSON,
    compare_spans,
    preview_metrics,
    write_inference_preview_report,
)

if TYPE_CHECKING:
    from meddies_pii.spans import CharSpan

APP_NAME = "meddies-lfm25-native-bioes-inference-preview"
DATASET_REVISION = "04be20f2c42d3f92b022edefbfb4d343fef78b2c"
DEFAULT_CHECKPOINT = "/artifacts/bioes/20260604_h100_8192_r128a256_pack_bs128_150step_ckpt10/unsloth"

image = add_source_pythonpath(train_image)
app = modal.App(APP_NAME, image=image)


def _span_to_dict(span: CharSpan) -> dict[str, object]:
    return {
        "label": span.label,
        "start": int(span.start),
        "end": int(span.end),
        "text": span.text,
    }


class _PreviewRequest(TypedDict):
    """Everything the preview needs except the GPU, which each entrypoint below supplies itself."""

    checkpoint: str
    preview_limit: int
    scan_multiplier: int
    max_length: int
    dataset_id: str
    dataset_revision: str | None
    eval_config: str
    eval_dataset_split: str


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def infer_preview_a10(**kwargs: Unpack[_PreviewRequest]) -> dict[str, Any]:
    return _infer_preview_impl(gpu=DEFAULT_GPU, **kwargs)


@app.function(
    gpu=H100_GPU,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def infer_preview_h100(**kwargs: Unpack[_PreviewRequest]) -> dict[str, Any]:
    return _infer_preview_impl(gpu=H100_GPU, **kwargs)


# reason: infer preview keeps gpu/eval dataset at its adapter seam; bundling would hide required inputs.
def _infer_preview_impl(  # ruff: ignore[too-many-arguments]
    *,
    gpu: str,
    checkpoint: str,
    preview_limit: int,
    scan_multiplier: int,
    max_length: int,
    dataset_id: str,
    dataset_revision: str | None,
    eval_config: str,
    eval_dataset_split: str,
) -> dict[str, Any]:
    from meddies_pii.annotations.bioes import ENTITY_LABELS
    from meddies_pii.eval_baseline.adapters.lfm_bioes import LfmBioesAdapter
    from meddies_pii.training.bioes.data.preparation import (
        _prepare_rows,
        _select_source_rows,
    )
    from meddies_pii.training.bioes.eval.harness import new_slice_filter_report
    from meddies_pii.training.bioes.trainers.trainer import _load_rows

    print(
        f"bioes_preview: build_artifacts start checkpoint={checkpoint} preview_limit={preview_limit}",
        flush=True,
    )
    adapter = LfmBioesAdapter(checkpoint_dir=checkpoint, max_seq_length=max_length)
    adapter.load()
    # reason: the preview needs the loaded tagger and label map, which the adapter exposes only here;
    # reason: the public fix is a re-export inside `src/`, which this lane does not own, so it is
    # reason: declared and flagged rather than worked around. The None guard below is the contract.
    artifacts = adapter._artifacts  # ruff: ignore[private-member-access]
    if artifacts is None:
        msg = "native BIOES adapter did not load checkpoint artifacts"
        raise RuntimeError(msg)
    id_to_label = {idx: label for label, idx in artifacts.label_to_id.items()}
    print(
        f"bioes_preview: checkpoint loaded device={next(artifacts.tagger.parameters()).device}",
        flush=True,
    )

    slice_filter_report = new_slice_filter_report()
    source_rows, source_stats = _select_source_rows(
        _load_rows(
            eval_config,
            preview_limit,
            scan_multiplier=scan_multiplier,
            dataset_id=dataset_id,
            split=eval_dataset_split,
            revision=dataset_revision,
        ),
        limit=None,
        allow_label_repairs=False,
        require_label_json=True,
        sort_by_length=False,
        slice_filter_report=slice_filter_report,
    )
    prepared_rows, prepare_stats = _prepare_rows(
        source_rows,
        artifacts.tokenizer,
        max_length=max_length,
        limit=preview_limit,
        id_to_label=id_to_label,
        entity_labels=ENTITY_LABELS,
        slice_filter_report=slice_filter_report,
    )
    print(
        f"bioes_preview: rows prepared source={len(source_rows)} prepared={len(prepared_rows)}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    predictions = adapter.predict([row.raw for row in prepared_rows])
    for row_index, (row, spans) in enumerate(zip(prepared_rows, predictions, strict=True)):
        predicted_spans = [_span_to_dict(span) for span in spans]
        gold_spans = [_span_to_dict(span) for span in row.parsed.spans]
        records.append({
            "row_index": row_index,
            "uid": row.uid,
            "text": row.raw,
            "text_chars": len(row.raw),
            "slices": list(row.slices),
            "gold_spans": gold_spans,
            "predicted_spans": predicted_spans,
            "exact": compare_spans(gold_spans, predicted_spans),
        })
    metrics = preview_metrics(records)
    print(
        f"bioes_preview: inference done rows={len(records)} f1={metrics['f1']:.6f}",
        flush=True,
    )
    return {
        "artifact_type": "bioes_inference_preview",
        "schema_version": 1,
        "provenance": {
            "modal_app": APP_NAME,
            "modal_profile": os.environ.get("MODAL_PROFILE"),
            "artifact_volume": ARTIFACT_VOLUME_NAME,
            "checkpoint": checkpoint,
            "gpu": gpu,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "eval_config": eval_config,
            "eval_dataset_split": eval_dataset_split,
            "max_length": max_length,
            "preview_limit": preview_limit,
            "scan_multiplier": scan_multiplier,
            "source_candidates": source_stats.candidates,
            "source_rows": len(source_rows),
            "prepared_candidates": prepare_stats.candidates,
            "prepared_rows": len(prepared_rows),
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "preview_metrics": metrics,
        "slice_filter_report": slice_filter_report,
        "records": records,
    }


# reason: run bioes exposes gpu/render only as its CLI contract; bundling would break callers.
@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    gpu: str = DEFAULT_GPU,
    checkpoint: str = DEFAULT_CHECKPOINT,
    preview_limit: int = 12,
    scan_multiplier: int = 8,
    max_length: int = 8192,
    dataset_id: str = "Meddies/meddies-pii",
    dataset_revision: str | None = DATASET_REVISION,
    eval_config: str = "pii-bioes",
    eval_dataset_split: str = "validation",
    result_json: str = str(DEFAULT_RESULT_JSON),
    inference_json: str = str(DEFAULT_INFERENCE_JSON),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    render_only: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
) -> None:
    inference_path = Path(inference_json)
    result_path = Path(result_json)
    output_path = Path(output_dir)
    selected_gpu = gpu.upper()
    if not render_only:
        kwargs: _PreviewRequest = {
            "checkpoint": checkpoint,
            "preview_limit": preview_limit,
            "scan_multiplier": scan_multiplier,
            "max_length": max_length,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "eval_config": eval_config,
            "eval_dataset_split": eval_dataset_split,
        }
        if selected_gpu in {"H100", "H100-80GB"}:
            preview_payload = infer_preview_h100.remote(**kwargs)
        elif selected_gpu in {"A10", "A10G"}:
            preview_payload = infer_preview_a10.remote(**kwargs)
        else:
            msg = f"Unsupported GPU {gpu!r}; expected A10 or H100"
            raise ValueError(msg)
        inference_path.parent.mkdir(parents=True, exist_ok=True)
        inference_path.write_text(
            json.dumps(preview_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    html_path = write_inference_preview_report(
        preview_json=inference_path,
        result_json=result_path,
        output_dir=output_path,
    )
    print(f"HTML preview: file://{html_path.resolve()}")
