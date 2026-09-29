#!/usr/bin/env python
"""Modal H100 benchmark harness for OpenAI Privacy Filter inference paths.

Dry-run locally without Modal:

    uv run --no-sync python scripts/ops/run_opf_inference_benchmark.py --dry-run

Run A/A2/B on Modal after review:

    MODAL_PROFILE=retraction uv run --no-sync modal run scripts/ops/run_opf_inference_benchmark.py --per-bucket 32

Run raw ONNX Path C after the A/A2/B run prints OPF_BENCHMARK_OUTPUT:

    MODAL_PROFILE=retraction uv run --no-sync modal run \
scripts/ops/run_opf_inference_benchmark.py::run_onnx --per-bucket 32 --reference-output-root \
/benchmark/opf-inference-...
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import argparse
import dataclasses
import importlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import modal

from meddies_pii.annotations.source_mapping import normalize_native_label
from meddies_pii.eval_baseline.adapters.opf import OPF_LABEL_FOLD
from meddies_pii.eval_baseline.adapters.opf_backend import (
    DEFAULT_NATIVE_CHECKPOINT_DIR,
    MODEL_ID,
)
from meddies_pii.eval_baseline.adapters.opf_backend import (
    apply_native_env as _apply_native_env,
)
from meddies_pii.eval_baseline.adapters.opf_backend import (
    build_native_batched_predictor as _build_native_batched_predictor,
)
from meddies_pii.eval_baseline.adapters.opf_backend import (
    detected_span_to_char_span as _detected_span_to_char_span,
)
from meddies_pii.eval_baseline.adapters.opf_backend import (
    prepare_native_checkpoint as _prepare_native_checkpoint,
)
from meddies_pii.eval_baseline.baseline.datasets import (
    EXTERNAL_CONFIGS,
    EvalRow,
    load_external_rows,
    load_v2_eval_rows,
)
from meddies_pii.eval_baseline.opf_benchmark.dataset import (
    BenchmarkDoc,
    PinnedBenchmarkDocSet,
    build_pinned_benchmark_doc_set,
)
from meddies_pii.eval_baseline.opf_benchmark.decoding import (
    bioes_token_entities_to_spans,
    token_logits_to_bioes_spans,
)
from meddies_pii.eval_baseline.opf_benchmark.harness import run_mock_benchmark
from meddies_pii.eval_baseline.opf_benchmark.metrics import (
    TimingRepeat,
    compute_correctness,
    summarize_timing,
)
from meddies_pii.eval_baseline.opf_benchmark.registry import (
    BenchmarkConfig,
    build_config_registry,
    select_representative_configs,
)
from meddies_pii.eval_baseline.opf_benchmark.results import (
    BenchmarkResultRow,
    format_results_grid,
)
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    import numpy as np
    import tokenizers

A2_BATCH_SIZE = 32

_RESOLVED_SCRIPT = Path(__file__).resolve()
REPO_ROOT = _RESOLVED_SCRIPT.parents[2] if len(_RESOLVED_SCRIPT.parents) > 2 else _RESOLVED_SCRIPT.parent  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
"""parents[2] is the repo root on the client (scripts/ops/<file> -> repo).

Modal re-imports this module inside the container at a shallow path (/root/<file>), where parents[2] would IndexError and
crash-loop every container at import. These local-source constants are only read at image-BUILD time on the client
(add_local_dir below), never in-container, so fall back safely when the repo layout isn't present.

"""
CONTEXT_ROOT = REPO_ROOT.parent / "meddies-pii-context"
OPF_REFERENCE_ROOT = CONTEXT_ROOT / "references/openai-privacy-filter"
OPF_REMOTE_ROOT = "/root/openai-privacy-filter"

APP_NAME = "meddies-opf-inference-benchmark"
OUTPUT_VOLUME_NAME = "meddies-pii-opf-benchmark"
OUTPUT_MOUNT = "/benchmark"
CACHE_VOLUME_NAME = "hf-cache"
CACHE_MOUNT = "/cache"
NATIVE_CHECKPOINT_DIR = DEFAULT_NATIVE_CHECKPOINT_DIR
ONNX_ASSET_DIR = f"{CACHE_MOUNT}/opf-onnx-assets"
H100_GPU = "H100"
TIMEOUT_SECONDS = 60 * 60
REPEAT_COUNT = 3
WARMUP_BATCHES = 3
ONNX_FILES = {
    "fp32": "onnx/model.onnx",
    "fp16": "onnx/model_fp16.onnx",
    "q4": "onnx/model_q4.onnx",
    "q4f16": "onnx/model_q4f16.onnx",
    "quantized": "onnx/model_quantized.onnx",
}

output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

benchmark_image = (
    add_source_pythonpath(
        modal.Image
        .from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d")
        .pip_install(
            "torch==2.13.0",
            "triton==3.7.1",
            "transformers==5.11.0",
            "datasets==4.5.0",
            "huggingface_hub==1.19.0",
            "safetensors==0.8.0",
            "tiktoken==0.12.0",
            "numpy==2.5.1",
        )
        .env({
            "HF_HOME": f"{CACHE_MOUNT}/huggingface",
            "OPF_CHECKPOINT": NATIVE_CHECKPOINT_DIR,
        }),
    )
    .env({"PYTHONPATH": f"{MODAL_SOURCE_ROOT}:{OPF_REMOTE_ROOT}"})
    .add_local_dir(str(OPF_REFERENCE_ROOT), remote_path=OPF_REMOTE_ROOT)
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
)
"""Torch 2.9.x has disclosed vulnerabilities.

Torch 2.13.0 requires the matching Triton 3.7 line; A10G timing parity is validated separately.

ONNX path (C) runs in its own image: optimum-onnx pins transformers below the >=5.6 the native/HF paths need.

"""

onnx_image = (
    add_source_pythonpath(
        modal.Image
        .from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d")
        .pip_install(
            "onnxruntime-gpu[cuda,cudnn]==1.23.0",
            "datasets==4.5.0",
            "huggingface_hub==1.19.0",
            "tokenizers==0.22.2",
            "numpy==2.5.1",
            "nvidia-ml-py==13.610.43",
        )
        .env({"HF_HOME": f"{CACHE_MOUNT}/huggingface"}),
    )
    .env({"PYTHONPATH": MODAL_SOURCE_ROOT})
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
)
"""The [cuda,cudnn] extra pulls the matching nvidia CUDA/cuDNN wheels.

The 1.23 GPU build needs libcudart.so.13); debian_slim ships no system CUDA, and unlike torch, onnxruntime does not bundle
it.

"""

app = modal.App(APP_NAME)
PredictFn = Callable[[Sequence[BenchmarkDoc]], dict[str, list[CharSpan]]]


@dataclasses.dataclass(frozen=True, slots=True)
class RealBenchmarkResult:
    rows: tuple[BenchmarkResultRow, ...]
    reference_by_bucket: dict[str, dict[str, list[CharSpan]]]
    doc_set_sha256: str


@app.function(
    image=benchmark_image,
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={OUTPUT_MOUNT: output_volume, CACHE_MOUNT: cache_volume},
)
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def run_h100(
    per_bucket: int = 32,
    max_configs: int = 0,
    representative: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    probe: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
) -> None:
    result = _run_h100_impl(
        per_bucket=per_bucket,
        max_configs=max_configs,
        representative=representative,
        probe=probe,
    )
    output_root = Path(OUTPUT_MOUNT) / f"opf-inference-{int(time.time())}"
    _write_outputs(output_root, result.rows)
    _write_reference_predictions(output_root, result)
    output_volume.commit()
    print(f"OPF_BENCHMARK_OUTPUT::{output_root}", flush=True)
    print("OPF_BENCHMARK_GRID_BEGIN", flush=True)
    print(format_results_grid(result.rows, max_rows=80), flush=True)
    print("OPF_BENCHMARK_GRID_END", flush=True)


@app.function(
    image=onnx_image,
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={OUTPUT_MOUNT: output_volume, CACHE_MOUNT: cache_volume},
)
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def run_onnx_h100(
    per_bucket: int = 32,
    max_configs: int = 0,
    representative: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    reference_output_root: str = "",
) -> None:
    result = _run_onnx_h100_impl(
        per_bucket=per_bucket,
        max_configs=max_configs,
        representative=representative,
        reference_output_root=reference_output_root,
    )
    output_root = Path(OUTPUT_MOUNT) / f"opf-inference-onnx-{int(time.time())}"
    _write_outputs(output_root, result.rows)
    output_volume.commit()
    print(f"OPF_ONNX_BENCHMARK_OUTPUT::{output_root}", flush=True)
    print("OPF_ONNX_BENCHMARK_GRID_BEGIN", flush=True)
    print(format_results_grid(result.rows, max_rows=80), flush=True)
    print("OPF_ONNX_BENCHMARK_GRID_END", flush=True)


@app.function(image=benchmark_image, timeout=600)
def dry_run() -> None:
    """Prove the wiring on Modal's CPU by running the full registry through the mock backend.

    No GPU and no model load: this validates registry build, doc-set sha256, ranking and grid
    formatting before any H100 spend. It mirrors the local ``--dry-run`` path but runs on Modal,
    honoring the no-local-compute rule.
    """
    _run_local_dry_run()


@app.local_entrypoint()
def dispatch_dry_run() -> None:
    dry_run.remote()
    print(json.dumps({"dispatched": "dry_run", "backend": "modal-cpu"}))


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(
    per_bucket: int = 32,
    max_configs: int = 0,
    representative: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    probe: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
) -> None:
    run_h100.remote(
        per_bucket=per_bucket,
        max_configs=max_configs,
        representative=representative,
        probe=probe,
    )
    print(
        json.dumps({
            "dispatched": "run_h100",
            "per_bucket": per_bucket,
            "max_configs": max_configs,
            "representative": representative,
            "probe": probe,
            "volume": OUTPUT_VOLUME_NAME,
        }),
    )


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def run_onnx(
    per_bucket: int = 32,
    max_configs: int = 0,
    representative: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    reference_output_root: str = "",
) -> None:
    run_onnx_h100.remote(
        per_bucket=per_bucket,
        max_configs=max_configs,
        representative=representative,
        reference_output_root=reference_output_root,
    )
    print(
        json.dumps({
            "dispatched": "run_onnx_h100",
            "per_bucket": per_bucket,
            "max_configs": max_configs,
            "representative": representative,
            "reference_output_root": reference_output_root,
            "volume": OUTPUT_VOLUME_NAME,
        }),
    )


def _run_h100_impl(
    per_bucket: int,
    max_configs: int,
    *,
    representative: bool = False,
    probe: bool = False,
) -> RealBenchmarkResult:
    """Targeted compile probe.

    A2 (the winner) at bs32, medium bucket, across all compile modes — answers "does torch.compile beat the compile=none
    164 docs/s baseline" without the full-sweep max-autotune thrash.

    """
    full_registry = build_config_registry()
    registry = tuple(config for config in full_registry if config.path_id != "C")
    if probe:
        registry = tuple(
            config
            for config in full_registry
            if config.path_id == "A2"
            and config.batch_size == A2_BATCH_SIZE
            and config.length_bucket == "medium"
            and config.skip_reason is None
        )
    elif representative:
        registry = tuple(config for config in select_representative_configs(full_registry) if config.path_id != "C")
    elif max_configs > 0:
        registry = registry[:max_configs]
    doc_set = _load_modal_doc_set(per_bucket=per_bucket)
    return _run_real_benchmark(
        doc_set=doc_set,
        registry=registry,
        reference_registry=full_registry,
    )


def _run_onnx_h100_impl(
    per_bucket: int,
    max_configs: int,
    *,
    representative: bool,
    reference_output_root: str,
) -> RealBenchmarkResult:
    if not reference_output_root:
        msg = (
            "Path C needs CRF reference predictions from run_h100; pass "
            "--reference-output-root /benchmark/opf-inference-..."
        )
        raise ValueError(
            msg,
        )
    full_registry = build_config_registry()
    registry = tuple(config for config in full_registry if config.path_id == "C")
    if representative:
        registry = tuple(config for config in select_representative_configs(full_registry) if config.path_id == "C")
    elif max_configs > 0:
        registry = registry[:max_configs]

    tokenizer = _load_onnx_tokenizer()

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text).ids)

    doc_set = _load_modal_doc_set(per_bucket=per_bucket, token_counter=count_tokens)
    reference_by_bucket = _read_reference_predictions(Path(reference_output_root), expected_doc_set_sha256=doc_set.sha256)
    return _run_real_benchmark(
        doc_set=doc_set,
        registry=registry,
        reference_registry=full_registry,
        reference_by_bucket=reference_by_bucket,
    )


def _load_modal_doc_set(*, per_bucket: int, token_counter: Callable[[str], int] | None = None) -> PinnedBenchmarkDocSet:
    rows: list[EvalRow] = []
    rows.extend(load_v2_eval_rows("eval", limit=max(256, per_bucket * 8)))
    rows.extend(load_v2_eval_rows("eval-challenge", limit=max(256, per_bucket * 8)))
    for config in EXTERNAL_CONFIGS:
        rows.extend(load_external_rows(config, limit=max(128, per_bucket * 4)))

    if token_counter is None:
        tokenizer = _load_tokenizer()

        def count_tokens(text: str) -> int:
            encoded = tokenizer(text, add_special_tokens=False)
            input_ids = encoded.get("input_ids", [])
            return len(input_ids) if isinstance(input_ids, list) else 0

    else:

        def count_tokens(text: str) -> int:
            return token_counter(text)

    return build_pinned_benchmark_doc_set(
        rows,
        per_bucket=per_bucket,
        token_counter=count_tokens,
    )


def _run_real_benchmark(
    *,
    doc_set: PinnedBenchmarkDocSet,
    registry: Sequence[BenchmarkConfig],
    reference_registry: Sequence[BenchmarkConfig] | None = None,
    reference_by_bucket: dict[str, dict[str, list[CharSpan]]] | None = None,
) -> RealBenchmarkResult:
    reference_by_bucket = (
        _build_reference_predictions(doc_set, reference_registry or registry)
        if reference_by_bucket is None
        else reference_by_bucket
    )

    rows: list[BenchmarkResultRow] = []
    for config in registry:
        if config.skip_reason:
            rows.append(config.to_skipped())
            continue
        docs = doc_set.docs_for_bucket(config.length_bucket)
        if not docs:
            rows.append(config.to_skipped(reason="no_docs_for_length_bucket"))
            continue
        # reason: run real's try keeps build with warmup; splitting would let sample setup drift.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            predictor = _build_predictor(config)
            _warmup(predictor, docs, config.batch_size)
            predictions, repeats = _timed_repeats(predictor, docs, config.batch_size)
            timing = summarize_timing(
                repeats=repeats,
                docs=len(docs),
                tokens=sum(doc.token_count for doc in docs),
            )
            gold = {doc.doc_id: list(doc.gold_spans) for doc in docs}
            correctness = compute_correctness(
                predictions,
                reference_by_bucket[config.length_bucket],
                gold,
            )
            rows.append(
                config.to_result(
                    docs_per_sec=timing.docs_per_sec,
                    tokens_per_sec=timing.tokens_per_sec,
                    median_seconds=timing.median_seconds,
                    min_seconds=timing.min_seconds,
                    max_seconds=timing.max_seconds,
                    peak_gpu_memory_mb=timing.peak_gpu_memory_mb,
                    exact_agree_f1=correctness.exact_agree_f1,
                    containment_agree_f1=correctness.containment_agree_f1,
                    exact_gold_f1=correctness.exact_gold_f1,
                    containment_gold_f1=correctness.containment_gold_f1,
                ),
            )
        except RuntimeError as exc:
            rows.append(config.to_error(str(exc)))
        # reason: `RuntimeError` is handled above with its own row. This arm exists so an unanticipated failure becomes
        # reason: a row in the result table, carrying its type name, rather than ending the benchmark.
        except Exception as exc:  # ruff: ignore[blind-except]
            rows.append(config.to_error(f"{type(exc).__name__}: {exc}"))
    return RealBenchmarkResult(
        rows=tuple(rows),
        reference_by_bucket=reference_by_bucket,
        doc_set_sha256=doc_set.sha256,
    )


def _build_reference_predictions(
    doc_set: PinnedBenchmarkDocSet,
    registry: Sequence[BenchmarkConfig],
) -> dict[str, dict[str, list[CharSpan]]]:
    canonical = _canonical_reference_configs(registry)
    reference_by_bucket: dict[str, dict[str, list[CharSpan]]] = {}
    for bucket, config in canonical.items():
        docs = doc_set.docs_for_bucket(bucket)
        predictor = _build_predictor(config)
        _warmup(predictor, docs, config.batch_size)
        reference_by_bucket[bucket] = predictor(docs)
    return reference_by_bucket


def _canonical_reference_configs(
    registry: Sequence[BenchmarkConfig],
) -> dict[str, BenchmarkConfig]:
    selected: dict[str, BenchmarkConfig] = {}
    for config in registry:
        if (
            config.path_id == "A"
            and config.batch_size == 1
            and config.compile_mode == "none"
            and config.triton == "on"
            and config.skip_reason is None
        ):
            selected[config.length_bucket] = config
    missing = sorted({"short", "medium", "long"} - set(selected))
    if missing:
        msg = f"registry missing canonical reference buckets: {missing}"
        raise RuntimeError(msg)
    return selected


def _build_predictor(config: BenchmarkConfig) -> PredictFn:
    if config.path_id == "A":
        return _build_native_reference_predictor(config)
    if config.path_id == "A2":
        return _build_native_batched_predictor(config)
    if config.path_id == "B":
        return _build_hf_pipeline_predictor(config)
    if config.path_id == "C":
        return _build_onnx_predictor(config)
    msg = f"unknown benchmark path: {config.path_id}"
    raise RuntimeError(msg)


def _build_native_reference_predictor(config: BenchmarkConfig) -> PredictFn:
    _apply_native_env(config)
    checkpoint = _prepare_native_checkpoint()
    opf_api = importlib.import_module("opf._api")
    redactor = opf_api.OPF(
        model=str(checkpoint),
        device="cuda",
        output_mode="typed",
        decode_mode="viterbi",
    )

    def predict(docs: Sequence[BenchmarkDoc]) -> dict[str, list[CharSpan]]:
        output: dict[str, list[CharSpan]] = {}
        for doc in docs:
            result = redactor.redact(doc.text)
            spans = getattr(result, "detected_spans", ())
            output[doc.doc_id] = [_detected_span_to_char_span(span) for span in spans]
        return output

    return predict


def _build_hf_pipeline_predictor(config: BenchmarkConfig) -> PredictFn:
    import torch
    from transformers import AutoModelForTokenClassification, pipeline

    tokenizer = _load_tokenizer()
    dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if config.attention_kernel != "default":
        model_kwargs["attn_implementation"] = config.attention_kernel
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID, **model_kwargs).to("cuda")
    model.eval()
    model_for_pipeline: Any = model
    if config.compile_mode != "none":
        model_for_pipeline = torch.compile(model, mode=config.compile_mode)
    pipeline_factory = cast("Any", pipeline)
    token_pipeline = pipeline_factory(
        "token-classification",
        model=model_for_pipeline,
        tokenizer=tokenizer,
        aggregation_strategy="none",
        batch_size=config.batch_size,
        device=0,
    )

    def predict(docs: Sequence[BenchmarkDoc]) -> dict[str, list[CharSpan]]:
        texts = [doc.text for doc in docs]
        raw = token_pipeline(texts)
        return _pipeline_token_outputs_to_spans(docs, raw)

    return predict


def _build_onnx_predictor(config: BenchmarkConfig) -> PredictFn:
    import numpy as np

    file_name = ONNX_FILES.get(config.onnx_variant)
    if file_name is None:
        msg = f"unsupported ONNX variant: {config.onnx_variant}"
        raise RuntimeError(msg)

    asset_root = _prepare_onnx_assets(file_name)
    model_path = asset_root / file_name
    if not model_path.exists():
        msg = f"ONNX model file not found after download: {model_path}"
        raise RuntimeError(msg)

    ort = importlib.import_module("onnxruntime")
    session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in session.get_providers():
        msg = f"ONNX Runtime did not attach CUDAExecutionProvider; providers={session.get_providers()}"
        raise RuntimeError(msg)

    tokenizer = _load_onnx_tokenizer()
    pad_token_id = _onnx_pad_token_id(tokenizer, asset_root / "tokenizer_config.json")
    id2label = _load_id2label(asset_root / "config.json")
    input_meta = {meta.name: meta for meta in session.get_inputs()}

    def predict(docs: Sequence[BenchmarkDoc]) -> dict[str, list[CharSpan]]:
        output: dict[str, list[CharSpan]] = {}
        for batch in _chunks(docs, config.batch_size):
            encoded = [tokenizer.encode(doc.text) for doc in batch]
            max_len = max((len(item.ids) for item in encoded), default=0)
            if max_len == 0:
                output.update({doc.doc_id: [] for doc in batch})
                continue

            input_ids = np.full((len(batch), max_len), pad_token_id, dtype=np.int64)
            attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
            offsets_by_row: list[list[tuple[int, int]]] = []
            for row_idx, item in enumerate(encoded):
                ids = [int(token_id) for token_id in item.ids]
                token_count = len(ids)
                if token_count == 0:
                    offsets_by_row.append([])
                    continue
                input_ids[row_idx, :token_count] = ids
                attention_mask[row_idx, :token_count] = 1
                offsets_by_row.append([(int(start), int(end)) for start, end in item.offsets])

            feed = _onnx_feed(input_meta, input_ids, attention_mask)
            raw_outputs = session.run(None, feed)
            logits = _first_onnx_logits(raw_outputs)
            for row_idx, doc in enumerate(batch):
                token_count = len(offsets_by_row[row_idx])
                row_logits = logits[row_idx, :token_count, :].tolist()
                output[doc.doc_id] = token_logits_to_bioes_spans(
                    doc.text,
                    row_logits,
                    offsets_by_row[row_idx],
                    id2label,
                )
        return output

    return predict


# reason: `AutoTokenizer.from_pretrained` is declared as `Unknown | TokenizersBackend | None |
# reason: SentencePieceBackend` in the pinned transformers, so there is no single honest static type to
# reason: write here — spelling the union out would add an `Unknown` arm and force every call site to
# reason: handle a `None` the pinned version does not produce. Its ONNX sibling below IS narrowed.
def _load_tokenizer() -> Any:  # ruff: ignore[any-type]
    # reason: transformers declares its public names only under `TYPE_CHECKING` and serves them at
    # reason: runtime through `_LazyModule`, so a static reader cannot prove the symbol is present.
    # reason: Verified against the pinned 5.14.1: `hasattr(transformers, "AutoTokenizer")` is True.
    from transformers import AutoTokenizer  # ty: ignore[possibly-missing-import]

    return AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)


def _load_onnx_tokenizer() -> tokenizers.Tokenizer:
    asset_root = _prepare_onnx_assets()
    # reason: the module is fetched through importlib because the client that launches this script has
    # reason: no ML stack, so the cast restores the type the container will actually produce.
    tokenizers_mod = importlib.import_module("tokenizers")
    return cast("tokenizers.Tokenizer", tokenizers_mod.Tokenizer.from_file(str(asset_root / "tokenizer.json")))


def _prepare_onnx_assets(model_file: str | None = None) -> Path:
    target = Path(ONNX_ASSET_DIR)
    target.mkdir(parents=True, exist_ok=True)
    allow_patterns = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    if model_file is not None:
        allow_patterns.append(model_file)
    hub = importlib.import_module("huggingface_hub")
    hub.snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target),
        allow_patterns=allow_patterns,
    )
    missing = [pattern for pattern in allow_patterns if not (target / pattern).exists()]
    if missing:
        msg = f"ONNX assets missing after download: {missing}"
        raise RuntimeError(msg)
    return target


def _onnx_pad_token_id(tokenizer: tokenizers.Tokenizer, tokenizer_config_path: Path) -> int:
    payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    pad_token = payload.get("pad_token") if isinstance(payload, Mapping) else None
    if isinstance(pad_token, str):
        token_id = tokenizer.token_to_id(pad_token)
        if isinstance(token_id, int):
            return token_id
    token_id = tokenizer.token_to_id("<|endoftext|>")
    return int(token_id) if isinstance(token_id, int) else 0


def _load_id2label(config_path: Path) -> dict[int, str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw = payload.get("id2label") if isinstance(payload, Mapping) else None
    if not isinstance(raw, Mapping):
        msg = f"config missing id2label mapping: {config_path}"
        raise RuntimeError(msg)
    id2label: dict[int, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and key.isdigit() and isinstance(value, str):
            id2label[int(key)] = value
    if not id2label:
        msg = f"config id2label mapping is empty: {config_path}"
        raise RuntimeError(msg)
    return id2label


def _chunks(docs: Sequence[BenchmarkDoc], batch_size: int) -> list[Sequence[BenchmarkDoc]]:
    size = max(1, batch_size)
    return [docs[start : start + size] for start in range(0, len(docs), size)]


def _onnx_feed(
    input_meta: Mapping[str, Any],
    input_ids: np.ndarray[Any, np.dtype[np.integer]],
    attention_mask: np.ndarray[Any, np.dtype[np.integer]],
) -> dict[str, Any]:
    feed: dict[str, Any] = {}
    for name, meta in input_meta.items():
        if name == "input_ids":
            feed[name] = input_ids.astype(_onnx_numpy_dtype(str(meta.type)))
        elif name == "attention_mask":
            feed[name] = attention_mask.astype(_onnx_numpy_dtype(str(meta.type)))
        else:
            msg = f"unsupported ONNX model input: {name}"
            raise RuntimeError(msg)
    return feed


def _onnx_numpy_dtype(type_name: str) -> type[np.generic]:
    import numpy as np

    if "int32" in type_name:
        return np.int32
    if "bool" in type_name:
        return np.bool_
    return np.int64


def _first_onnx_logits(outputs: Sequence[object]) -> np.ndarray[Any, np.dtype[np.floating]]:
    for output in outputs:
        if getattr(output, "ndim", None) == 3:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            # reason: the ONNX session hands back untyped outputs, and the rank-3 test above is the
            # reason: only thing that identifies the logits among them; the raise below is the refusal
            # reason: when nothing matches, so the cast never runs on an unchecked value.
            return cast("np.ndarray[Any, np.dtype[np.floating]]", output)
    msg = "ONNX model did not return a [batch, tokens, labels] logits"
    raise RuntimeError(msg)


def _warmup(predictor: PredictFn, docs: Sequence[BenchmarkDoc], batch_size: int) -> None:
    warmup_count = max(1, min(len(docs), max(1, batch_size) * WARMUP_BATCHES))
    for _ in range(WARMUP_BATCHES):
        predictor(docs[:warmup_count])
    _cuda_synchronize()


def _timed_repeats(
    predictor: PredictFn,
    docs: Sequence[BenchmarkDoc],
    batch_size: int,
) -> tuple[dict[str, list[CharSpan]], tuple[TimingRepeat, ...]]:
    repeats: list[TimingRepeat] = []
    last_predictions: dict[str, list[CharSpan]] = {}
    for _ in range(REPEAT_COUNT):
        _cuda_reset_peak_memory()
        warmup_start = time.perf_counter()
        predictor(docs[: max(1, min(len(docs), batch_size))])
        _cuda_synchronize()
        warmup_seconds = time.perf_counter() - warmup_start
        start = time.perf_counter()
        last_predictions = predictor(docs)
        _cuda_synchronize()
        timed_seconds = time.perf_counter() - start
        repeats.append(
            TimingRepeat(
                timed_seconds=timed_seconds,
                warmup_seconds=warmup_seconds,
                peak_gpu_memory_mb=_cuda_peak_memory_mb(),
            ),
        )
    return last_predictions, tuple(repeats)


def _cuda_synchronize() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_reset_peak_memory() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_memory_mb() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        return _nvml_used_memory_mb()

    if not torch.cuda.is_available():
        return _nvml_used_memory_mb()
    return int(torch.cuda.max_memory_allocated() / (1024 * 1024))


def _nvml_used_memory_mb() -> int:
    try:
        pynvml = importlib.import_module("pynvml")
    except ModuleNotFoundError:
        return 0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used / (1024 * 1024))
    # reason: the NVIDIA management library is absent outside the driver container, and raises its own error types
    # reason: when present but unreadable. The caller reads missing telemetry as zero, so nothing here should reach it.
    except Exception:  # ruff: ignore[blind-except]
        return 0


def _pipeline_outputs_to_spans(
    docs: Sequence[BenchmarkDoc],
    raw: object,
) -> dict[str, list[CharSpan]]:
    grouped = _as_batched_entities(raw)
    if len(grouped) != len(docs):
        msg = f"pipeline returned {len(grouped)} outputs for {len(docs)} docs"
        raise RuntimeError(msg)
    return {doc.doc_id: _entities_to_spans(doc.text, entities) for doc, entities in zip(docs, grouped, strict=True)}


def _pipeline_token_outputs_to_spans(
    docs: Sequence[BenchmarkDoc],
    raw: object,
) -> dict[str, list[CharSpan]]:
    grouped = _as_batched_entities(raw)
    if len(grouped) != len(docs):
        msg = f"pipeline returned {len(grouped)} outputs for {len(docs)} docs"
        raise RuntimeError(msg)
    return {
        doc.doc_id: bioes_token_entities_to_spans(doc.text, entities) for doc, entities in zip(docs, grouped, strict=True)
    }


def _as_batched_entities(raw: object) -> list[list[Mapping[str, object]]]:
    if not isinstance(raw, list):
        msg = f"unexpected pipeline output type: {type(raw).__name__}"
        raise RuntimeError(msg)
    if not raw:
        return []
    first = raw[0]
    if isinstance(first, Mapping):
        return [list(_only_mappings(raw))]
    return [list(_only_mappings(batch)) if isinstance(batch, list) else [] for batch in raw]


def _only_mappings(values: Sequence[object]) -> list[Mapping[str, object]]:
    return [cast("Mapping[str, object]", value) for value in values if isinstance(value, Mapping)]


def _entities_to_spans(text: str, entities: Sequence[Mapping[str, object]]) -> list[CharSpan]:
    spans: list[CharSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for entity in entities:
        raw_label = entity.get("entity_group") or entity.get("entity")
        if not isinstance(raw_label, str):
            continue
        label = normalize_native_label(raw_label)
        if label not in OPF_LABEL_FOLD:
            continue
        start = entity.get("start")
        end = entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        spans.append(CharSpan(start=start, end=end, text=text[start:end], label=label))
    return spans


def _write_outputs(output_root: Path, rows: Sequence[BenchmarkResultRow]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "results.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dataclasses.asdict(row), ensure_ascii=False) + "\n")
    grid = format_results_grid(tuple(rows))
    (output_root / "grid.tsv").write_text(grid + "\n", encoding="utf-8")
    ok_rows = [row for row in rows if row.status == "ok"]
    skipped = [row for row in rows if row.status == "skipped"]
    errored = [row for row in rows if row.status == "error"]
    summary = {
        "total_rows": len(rows),
        "ok_rows": len(ok_rows),
        "skipped_rows": len(skipped),
        "error_rows": len(errored),
        "recommendation": _recommendation_payload(rows),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_reference_predictions(output_root: Path, result: RealBenchmarkResult) -> None:
    payload = {
        "doc_set_sha256": result.doc_set_sha256,
        "reference_by_bucket": {
            bucket: {doc_id: [dataclasses.asdict(span) for span in spans] for doc_id, spans in predictions.items()}
            for bucket, predictions in result.reference_by_bucket.items()
        },
    }
    (output_root / "reference_predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_reference_predictions(
    output_root: Path,
    *,
    expected_doc_set_sha256: str,
) -> dict[str, dict[str, list[CharSpan]]]:
    path = output_root / "reference_predictions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        msg = f"reference predictions payload is not an object: {path}"
        raise RuntimeError(msg)
    actual_sha = payload.get("doc_set_sha256")
    if actual_sha != expected_doc_set_sha256:
        msg = f"reference predictions doc-set sha mismatch: expected {expected_doc_set_sha256}, got {actual_sha}"
        raise RuntimeError(
            msg,
        )
    raw_by_bucket = payload.get("reference_by_bucket")
    if not isinstance(raw_by_bucket, Mapping):
        msg = f"reference predictions missing buckets: {path}"
        raise RuntimeError(msg)

    parsed: dict[str, dict[str, list[CharSpan]]] = {}
    for bucket, raw_docs in raw_by_bucket.items():
        if not isinstance(bucket, str) or not isinstance(raw_docs, Mapping):
            continue
        parsed[bucket] = {}
        for doc_id, raw_spans in raw_docs.items():
            if not isinstance(doc_id, str) or not isinstance(raw_spans, list):
                continue
            spans = [span for raw_span in raw_spans if (span := _parse_reference_span(raw_span)) is not None]
            parsed[bucket][doc_id] = spans
    return parsed


def _parse_reference_span(raw: object) -> CharSpan | None:
    if not isinstance(raw, Mapping):
        return None
    start = raw.get("start")
    end = raw.get("end")
    text = raw.get("text")
    label = raw.get("label")
    if isinstance(start, int) and isinstance(end, int) and isinstance(text, str) and isinstance(label, str):
        return CharSpan(start=start, end=end, text=text, label=label)
    return None


def _recommendation_payload(rows: Sequence[BenchmarkResultRow]) -> dict[str, object]:
    from meddies_pii.eval_baseline.opf_benchmark.ranking import rank_benchmark_results

    recommendation = rank_benchmark_results(tuple(rows))
    return {
        "winner": dataclasses.asdict(recommendation.winner) if recommendation.winner else None,
        "reason": recommendation.reason,
        "best_by_path": {path: row.config_id for path, row in recommendation.best_by_path.items()},
        "acceptable_by_path": {path: row.config_id for path, row in recommendation.acceptable_by_path.items()},
    }


def _synthetic_dry_rows() -> list[EvalRow]:
    return [
        _synthetic_row("short", 70),
        _synthetic_row("medium", 260),
        _synthetic_row("long", 540),
    ]


def _synthetic_row(doc_id: str, token_count: int) -> EvalRow:
    words = [f"w{i}" for i in range(token_count)]
    words[1] = "John"
    words[2] = "Smith"
    text = " ".join(words)
    start = text.index("John")
    end = start + len("John Smith")
    return EvalRow(
        doc_id=doc_id,
        dataset="dry-run",
        shard="stub",
        text=text,
        gold_spans=(CharSpan(start=start, end=end, text=text[start:end], label="human_name"),),
        language="en",
        slices=frozenset({"dry-run"}),
    )


def _run_local_dry_run() -> None:
    doc_set = build_pinned_benchmark_doc_set(_synthetic_dry_rows(), per_bucket=1)
    result = run_mock_benchmark(doc_set=doc_set, registry=build_config_registry())
    print(f"DOC_SET_SHA256::{result.doc_set_sha256}")
    print("STUB_GRID_BEGIN")
    print(format_results_grid(result.rows, max_rows=80))
    print("STUB_GRID_END")
    print(
        json.dumps(
            {
                "rows": len(result.rows),
                "winner": result.recommendation.winner.config_id if result.recommendation.winner else None,
                "stub": True,
            },
            ensure_ascii=False,
        ),
    )


def _direct_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        msg = (
            "Use Modal for real runs: MODAL_PROFILE=retraction uv run --no-sync modal run "
            "scripts/ops/run_opf_inference_benchmark.py --per-bucket 32"
        )
        raise SystemExit(
            msg,
        )
    _run_local_dry_run()


if __name__ == "__main__":
    _direct_cli()
