"""Evaluate Liquid's exact public Space runtime through the Meddies harness.

All dataset access and model inference run remotely on Modal.

    MODAL_PROFILE=meddies-run uv run modal run \
      scripts/ops/run_lfm25_pii_baseline.py::run_matrix --full --force
    MODAL_PROFILE=meddies-run uv run modal run \
      scripts/ops/run_lfm25_pii_baseline.py::run_aggregate
"""

from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: this module is launched as `uv run modal run <this path>` and is never imported, so it is a script
# reason: rather than a package member. An `__init__.py` would declare this directory a package it is not, and
# reason: the sibling scripts here that run under `python` say so with a shebang instead.
# ruff: file-ignore[import-outside-top-level]
# reason: the heavy dependency loads in the subcommand that needs it, so starting the script does not pay for it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import json
import os
import time
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal

from meddies_pii.eval_baseline.adapters.lfm25_pii import (
    MODEL_ID,
    MODEL_REVISION,
    load_pinned_space_detector,
)
from meddies_pii.eval_baseline.baseline.datasets import (
    EVAL_DATASETS,
    EVAL_EXPECTED_ROWS,
    EXTERNAL_DATASET_REVISION,
    V2_DATASET_REVISION,
)
from meddies_pii.eval_baseline.baseline.views import (
    MODEL_CORE_VIEW,
    VENDOR_HYBRID_VIEW,
    EvaluationView,
)
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
    use_pinned_debian_snapshot,
)

if TYPE_CHECKING:
    from meddies_pii.evaluation.identity import EvaluationContract


def _evaluation_contract(view: EvaluationView) -> EvaluationContract:
    from meddies_pii import __file__ as package_file
    from meddies_pii.eval_baseline.adapters import lfm25_pii
    from meddies_pii.eval_baseline.baseline.run import resolved_environment_fingerprint
    from meddies_pii.evaluation.identity import (
        evaluation_contract,
        manifest_artifact,
        payload_artifact,
    )
    from meddies_pii.evaluation.runtime_provenance import (
        evaluation_runtime_source_artifact,
    )

    if package_file is None:
        msg = "cannot observe the Meddies evaluation runtime source"
        raise RuntimeError(msg)
    runtime_source = evaluation_runtime_source_artifact(Path(package_file).parent, __file__)
    return evaluation_contract(
        model=manifest_artifact(f"hf://{MODEL_ID}", MODEL_REVISION),
        vendor_inference_source=manifest_artifact("hf-space://LiquidAI/pii-detection", SPACE_REVISION),
        local_adapter_source=runtime_source,
        applied_label_prediction_contract=payload_artifact(
            "liquid-label-prediction",
            {
                "view": view,
                "label_fold": lfm25_pii.LIQUID_LABEL_FOLD,
            },
        ),
        decoder_contract=payload_artifact(
            "liquid-decoder",
            {
                "model_core": "PiiDetector.hd.model_spans",
                "vendor_hybrid": "PiiDetector.hd.hybrid_spans+detect-overlap",
                "selected_view": view,
            },
        ),
        resolved_runtime_environment=payload_artifact("resolved-runtime", resolved_environment_fingerprint()),
        scorer_contract=runtime_source,
        supported_labels=tuple(sorted(lfm25_pii.LFM25_PII_SUPPORTED_LABELS)),
        result_schema=(
            "id",
            "doc_id",
            "pred_spans",
            "gold_spans",
            "language",
            "slice",
            "evaluation_view",
            "shared_inference_identity",
        ),
    )


APP_NAME = "meddies-pii-lfm25-pii-baseline"
MODEL_NAME = "lfm25-pii"
SPACE_REVISION = "f645508a233038955a4e20bc1ccfcfb771bee1da"
SPACE_SOURCE_SHA256 = "00b4e681fdecc70e33e58588f1f7db63302a370b7cc4a72604fc5cfefba7f1af"
BASELINE_MOUNT = "/baseline"
CACHE_MOUNT = "/cache"
SPACE_REFERENCE_MOUNT = "/reference/liquidai-pii-detection-space"
EXPECTED_TOTAL_ROWS = sum(EVAL_EXPECTED_ROWS.values())
EXPECTED_TOTAL_GOLD_SPANS = 1_601_262
PROGRESS_EVERY = 2_000
SMOKE_MAX_ROWS = 8
PARITY_SMOKE_TEXTS = (
    "Contact alice@example.com for scheduling.",
    "Clinical note contains no identifier.",
    "Credential sk-abcdefghijklmnopqrstuv was rotated.",
    "Passport No: A12345678 was verified.",
)
BASELINE_VOLUME_NAME = os.environ.get("MEDDIES_PII_EVAL_RESULTS_VOLUME", "meddies-pii-baseline-eval-lfm25-pii")
CACHE_VOLUME_NAME = os.environ.get("MEDDIES_PII_EVAL_CACHE_VOLUME", "lfm25-pii-eval-hf-cache")
COMPILER_APT_PACKAGES = ("gcc=4:12.2.0-3",)


def _space_reference_local_or_none() -> Path | None:
    if not modal.is_local():
        return Path(SPACE_REFERENCE_MOUNT)
    source = Path(__file__).resolve()
    for ancestor in source.parents:
        candidate = ancestor / "meddies-pii-context" / "references" / "liquidai-pii-detection-space"
        if candidate.is_dir():
            return candidate
    return None


def _space_reference_local() -> Path:
    reference = _space_reference_local_or_none()
    if reference is None:
        msg = (
            "local pinned Liquid Space reference is missing; expected an ancestor "
            "containing meddies-pii-context/references/liquidai-pii-detection-space"
        )
        raise FileNotFoundError(msg)
    return reference


baseline_volume = modal.Volume.from_name(
    BASELINE_VOLUME_NAME,
    create_if_missing=True,
)
cache_volume = modal.Volume.from_name(
    CACHE_VOLUME_NAME,
    create_if_missing=True,
)
image = (
    add_source_pythonpath(
        use_pinned_debian_snapshot(
            modal.Image.from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d"),
        )
        .apt_install(*COMPILER_APT_PACKAGES)
        .pip_install(
            "torch==2.13.0",
            "transformers==5.11.0",
            "tokenizers==0.22.2",
            "safetensors==0.8.0",
            "huggingface-hub==1.19.0",
            "datasets==4.5.0",
            "pyarrow==23.0.1",
        ),
    )
    .env({
        "HF_HOME": f"{CACHE_MOUNT}/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
)
_space_reference = _space_reference_local_or_none()
"""Importing this module must not require the reference clone (entrypoint-import tests run without the context repo);
attach the mount only when it exists and enforce its presence at the entrypoints instead.
"""
if _space_reference is not None:
    image = image.add_local_dir(
        str(_space_reference),
        remote_path=SPACE_REFERENCE_MOUNT,
    )
app = modal.App(APP_NAME, image=image)


def _write_json(relative_path: str, payload: dict[str, Any]) -> Path:
    path = Path(BASELINE_MOUNT) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline_volume.commit()
    return path


def _download_model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=f"{CACHE_MOUNT}/huggingface",
        ),
    )


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32_768,
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BASELINE_MOUNT: baseline_volume, CACHE_MOUNT: cache_volume},
)
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def eval_cell(
    dataset: str,
    max_rows: int = 0,
    full: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    force: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
) -> str:
    from meddies_pii.eval_baseline.adapters.lfm25_pii import Lfm25PiiSpaceAdapter
    from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell
    from meddies_pii.eval_baseline.baseline.run import (
        ShardSpec,
        run_dual_view_shards,
        shard_result_json,
    )

    limit = None if full else (max_rows or 2_000)
    rows = load_eval_cell(
        dataset,
        v2_limit=limit,
        external_limit=limit,
        v2_revision=V2_DATASET_REVISION,
        external_revision=EXTERNAL_DATASET_REVISION,
    )
    if full and len(rows) != EVAL_EXPECTED_ROWS[dataset]:
        msg = f"{dataset} row count changed: expected {EVAL_EXPECTED_ROWS[dataset]}, got {len(rows)}"
        raise RuntimeError(msg)
    space_root = Path(SPACE_REFERENCE_MOUNT)
    detector = load_pinned_space_detector(
        snapshot_path=_download_model_snapshot(),
        source_path=space_root / "pii.py",
        space_root=space_root,
        space_revision=SPACE_REVISION,
        expected_source_sha256=SPACE_SOURCE_SHA256,
    )
    started = time.perf_counter()

    def write_progress(index: int, total: int) -> None:
        if index % PROGRESS_EVERY != 0 and index != total:
            return
        elapsed = time.perf_counter() - started
        state = {
            "status": "running" if index < total else "inference_complete",
            "dataset": dataset,
            "rows_done": index,
            "rows_total": total,
            "percent": round(index / total * 100, 2),
            "elapsed_seconds": round(elapsed, 2),
            "rows_per_second": round(index / elapsed, 2),
            "space_revision": SPACE_REVISION,
            "model_revision": MODEL_REVISION,
            "dtype": "float32",
            "tf32": True,
        }
        _write_json(f"state/{dataset}.json", state)
        print(f"LFM25_PROGRESS::{json.dumps(state)}", flush=True)

    adapter = Lfm25PiiSpaceAdapter(detector, on_progress=write_progress)

    _write_json(
        f"state/{dataset}.json",
        {
            "status": "loading_complete",
            "dataset": dataset,
            "rows_done": 0,
            "rows_total": len(rows),
        },
    )
    try:
        result = run_dual_view_shards(
            adapter=adapter,
            rows=rows,
            output_root=Path(BASELINE_MOUNT),
            specs={
                MODEL_CORE_VIEW: ShardSpec(
                    model=MODEL_NAME,
                    dataset=dataset,
                    shard="full",
                    view=MODEL_CORE_VIEW,
                ),
                VENDOR_HYBRID_VIEW: ShardSpec(
                    model=MODEL_NAME,
                    dataset=dataset,
                    shard="full",
                    view=VENDOR_HYBRID_VIEW,
                ),
            },
            evaluation_contracts={
                MODEL_CORE_VIEW: _evaluation_contract(MODEL_CORE_VIEW),
                VENDOR_HYBRID_VIEW: _evaluation_contract(VENDOR_HYBRID_VIEW),
            },
            volume=baseline_volume,
            force=force,
        )
    except Exception as exc:
        _write_json(
            f"state/{dataset}.json",
            {
                "status": "failed",
                "failure_state": True,
                "dataset": dataset,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    result_payload = {
        "shared_inference_identity": result.shared_inference_identity,
        "by_view": {view: json.loads(shard_result_json(shard_result)) for view, shard_result in result.by_view.items()},
    }
    line = f"LFM25_CELL::{json.dumps(result_payload, ensure_ascii=False)}"
    _write_json(
        f"state/{dataset}.json",
        {
            "status": "complete",
            "failure_state": False,
            "dataset": dataset,
            "rows_done": result.by_view[MODEL_CORE_VIEW].rows_in,
            "rows_total": len(rows),
            "result": result_payload,
        },
    )
    print(line, flush=True)
    return line


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32_768,
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BASELINE_MOUNT: baseline_volume, CACHE_MOUNT: cache_volume},
)
def parity_smoke() -> str:
    """Compare one-forward hybrid output to pinned ``detect`` on fixed smoke text."""
    from meddies_pii.eval_baseline.adapters.lfm25_pii import (
        Lfm25PiiSpaceAdapter,
        map_hybrid_spans,
    )

    detector = load_pinned_space_detector(
        snapshot_path=_download_model_snapshot(),
        source_path=Path(SPACE_REFERENCE_MOUNT) / "pii.py",
        space_root=Path(SPACE_REFERENCE_MOUNT),
        space_revision=SPACE_REVISION,
        expected_source_sha256=SPACE_SOURCE_SHA256,
    )
    adapter = Lfm25PiiSpaceAdapter(detector)
    dual = adapter.predict_views(list(PARITY_SMOKE_TEXTS))
    expected = []
    for text in PARITY_SMOKE_TEXTS:
        output = detector.detect(text)
        if not isinstance(output, dict):
            msg = "Liquid Space detector returned a malformed response"
            raise RuntimeError(msg)
        raw_spans = output.get("spans")
        if not isinstance(raw_spans, list):
            msg = "Liquid Space detector returned malformed spans"
            raise RuntimeError(msg)
        expected.append(map_hybrid_spans(text, raw_spans))
    mismatches = [
        index
        for index, (views, spans) in enumerate(zip(dual, expected, strict=True))
        if list(views.vendor_hybrid) != spans
    ]
    if mismatches:
        msg = f"dual-view hybrid parity failed for smoke rows {mismatches}"
        raise RuntimeError(msg)
    view_differences = sum(views.model_core != views.vendor_hybrid for views in dual)
    if view_differences == 0:
        msg = "parity smoke did not exercise a hybrid-only difference"
        raise RuntimeError(msg)
    payload = {
        "status": "ok",
        "rows": len(PARITY_SMOKE_TEXTS),
        "mismatch_count": 0,
        "view_difference_rows": view_differences,
        "views": [MODEL_CORE_VIEW, VENDOR_HYBRID_VIEW],
        "reference_forwards": len(PARITY_SMOKE_TEXTS),
        "fixture_sha256": sha256("\n".join(PARITY_SMOKE_TEXTS).encode("utf-8")).hexdigest(),
    }
    line = f"LFM25_PARITY_SMOKE::{json.dumps(payload, sort_keys=True)}"
    print(line, flush=True)
    return line


@app.function(
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BASELINE_MOUNT: baseline_volume},
)
def aggregate(view: EvaluationView = MODEL_CORE_VIEW) -> None:
    from meddies_pii.eval_baseline.adapters.lfm25_pii import (
        LFM25_PII_SUPPORTED_LABELS,
    )
    from meddies_pii.eval_baseline.baseline.aggregate import (
        aggregate_results,
        format_aggregate_report,
    )
    from meddies_pii.eval_baseline.baseline.run import (
        assert_frozen_fixture,
        expected_matrix_shard_identities,
        read_matrix_results,
    )

    contract = _evaluation_contract(view)

    try:
        matrix = read_matrix_results(
            BASELINE_MOUNT,
            MODEL_NAME,
            EVAL_DATASETS,
            require_done=True,
            expected_rows=EVAL_EXPECTED_ROWS,
            expected_evaluation_contract=contract,
            expected_dataset_shard_identities=expected_matrix_shard_identities(contract, EVAL_DATASETS),
            view=view,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if matrix.missing_datasets:
        msg = f"cannot aggregate; missing cells: {list(matrix.missing_datasets)}"
        raise RuntimeError(msg)
    rows_by_config = matrix.rows_by_dataset
    report = aggregate_results(
        rows_by_config,
        supported_labels=LFM25_PII_SUPPORTED_LABELS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
        result_sha256_by_config=matrix.result_sha256_by_dataset,
        timing_by_config=matrix.timing_by_dataset,
    )
    assert_frozen_fixture(report)
    if int(report["overall"]["rows"]) != EXPECTED_TOTAL_ROWS:
        msg = f"expected {EXPECTED_TOTAL_ROWS} rows, got {report['overall']['rows']}"
        raise RuntimeError(msg)
    if int(report["fixture"]["full9_gold_spans"]) != EXPECTED_TOTAL_GOLD_SPANS:
        msg = (
            "frozen fixture gold count changed: "
            f"expected {EXPECTED_TOTAL_GOLD_SPANS}, "
            f"got {report['fixture']['full9_gold_spans']}"
        )
        raise RuntimeError(msg)
    payload = {
        "manifest": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "space_revision": SPACE_REVISION,
            "inference": (
                "pinned Space PiiDetector.hd.model_spans"
                if view == MODEL_CORE_VIEW
                else "pinned Space model_spans+hybrid_spans+detect-overlap"
            ),
            "evaluation_view": view,
            "dtype": "float32",
            "tf32": True,
            "accelerator": "A10G",
            "scorer": "Meddies fixed-nine exact and containment span F1",
            "external_revision": EXTERNAL_DATASET_REVISION,
            "v2_revision": V2_DATASET_REVISION,
            "rows_by_cell": EVAL_EXPECTED_ROWS,
            "total_rows": EXPECTED_TOTAL_ROWS,
        },
        **report,
    }
    path = _write_json(f"reports/lfm25-pii-{view}-aggregate.json", payload)
    print("LFM25_AGGREGATE_BEGIN", flush=True)
    print(format_aggregate_report(report, model=MODEL_NAME), flush=True)
    print(f"LFM25_AGGREGATE_PATH::{path}", flush=True)
    print("LFM25_AGGREGATE_END", flush=True)


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def run_matrix(
    max_rows: int = 0,
    datasets: str = "",
    full: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    force: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
) -> None:
    from meddies_pii.eval_baseline.baseline.datasets import select_eval_datasets

    _space_reference_local()
    selected = select_eval_datasets(datasets)
    handles = [(dataset, eval_cell.spawn(dataset, max_rows, full, force)) for dataset in selected]
    failures: list[str] = []
    print(json.dumps({"dispatched": len(handles), "full": full}), flush=True)
    for dataset, handle in handles:
        try:
            print(handle.get(), flush=True)
        # reason: a Modal cell fails inside a remote container, so its exception reaches this process as a type this
        # reason: process never imported; the loop records the cell that failed and the rest of the fan-in still prints,
        # reason: which is exactly what the per-iteration `try` buys.
        except Exception as exc:  # ruff: ignore[blind-except,try-except-in-loop]
            failures.append(f"{dataset}: {type(exc).__name__}: {exc}")
            print(f"CELL_FAILED::{failures[-1]}", flush=True)
    if failures:
        msg = f"{len(failures)} LFM25 cells failed: {failures}"
        raise RuntimeError(msg)


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def smoke(
    dataset: str = "v2-eval",
    max_rows: int = SMOKE_MAX_ROWS,
    force: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
) -> None:
    """Run a bounded A10G smoke that emits both named evaluation views."""
    if not 1 <= max_rows <= SMOKE_MAX_ROWS:
        msg = f"smoke max_rows must be between 1 and {SMOKE_MAX_ROWS}"
        raise ValueError(msg)
    _space_reference_local()
    print(parity_smoke.remote())
    print(eval_cell.remote(dataset, max_rows=max_rows, full=False, force=force))


@app.local_entrypoint()
def run_parity_smoke() -> None:
    _space_reference_local()
    print(parity_smoke.remote())


@app.local_entrypoint()
def run_aggregate(view: EvaluationView = MODEL_CORE_VIEW) -> None:
    aggregate.remote(view)
