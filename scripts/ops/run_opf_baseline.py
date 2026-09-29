"""Modal eval harness for openai/privacy-filter on the Meddies eval matrix.

The adapter uses the native OPF A2 path: Triton MoE, batched window-pack, and
Viterbi-CRF decode. One H100 cell scores one dataset/config and writes to a
dedicated Volume, matching the OpenMed and GLiNER2 baseline harnesses.

    MODAL_PROFILE=retraction modal run scripts/ops/run_opf_baseline.py --max-rows 8
    MODAL_PROFILE=retraction modal run scripts/ops/run_opf_baseline.py::run_opf_matrix --full
    MODAL_PROFILE=retraction modal run scripts/ops/run_opf_baseline.py::run_aggregate_opf
"""

from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: this module is launched as `uv run modal run <this path>` and is never imported, so it is a script
# reason: rather than a package member. An `__init__.py` would declare this directory a package it is not, and
# reason: the sibling scripts here that run under `python` say so with a shebang instead.
# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import json
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from meddies_pii.eval_baseline.adapters.opf_backend import (
    DEFAULT_NATIVE_CHECKPOINT_DIR,
    MODEL_ID,
    MODEL_REVISION,
)
from meddies_pii.eval_baseline.baseline.datasets import EVAL_DATASETS, EVAL_EXPECTED_ROWS
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)

if TYPE_CHECKING:
    from meddies_pii.evaluation.identity import EvaluationContract


def _evaluation_contract() -> EvaluationContract:
    from meddies_pii import __file__ as package_file
    from meddies_pii.eval_baseline.adapters import opf
    from meddies_pii.eval_baseline.baseline.run import resolved_environment_fingerprint
    from meddies_pii.evaluation.identity import (
        evaluation_contract,
        manifest_artifact,
        payload_artifact,
    )
    from meddies_pii.evaluation.runtime_provenance import (
        evaluation_runtime_source_artifact,
        source_tree_artifact,
    )

    if package_file is None:
        msg = "cannot observe the Meddies evaluation runtime source"
        raise RuntimeError(msg)
    runtime_source = evaluation_runtime_source_artifact(Path(package_file).parent, __file__)
    vendor_source = source_tree_artifact("mounted-vendor://openai/privacy-filter", OPF_REMOTE_ROOT)
    return evaluation_contract(
        model=manifest_artifact(f"hf://{MODEL_ID}", MODEL_REVISION),
        vendor_inference_source=vendor_source,
        local_adapter_source=runtime_source,
        applied_label_prediction_contract=payload_artifact("opf-label-prediction", "native-fold"),
        decoder_contract=payload_artifact("opf-decoder", "native-viterbi-window-pack"),
        resolved_runtime_environment=payload_artifact("resolved-runtime", resolved_environment_fingerprint()),
        scorer_contract=runtime_source,
        supported_labels=tuple(sorted(opf.OPF_SUPPORTED_LABELS)),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )


FIRST_LIGHT_MAX_ROWS = 32

_RESOLVED_SCRIPT = Path(__file__).resolve()
REPO_ROOT = _RESOLVED_SCRIPT.parents[2] if len(_RESOLVED_SCRIPT.parents) > 2 else _RESOLVED_SCRIPT.parent  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
CONTEXT_ROOT = REPO_ROOT.parent / "meddies-pii-context"
OPF_REFERENCE_ROOT = CONTEXT_ROOT / "references/openai-privacy-filter"
OPF_REMOTE_ROOT = "/root/openai-privacy-filter"

BASELINE_VOLUME_NAME = "meddies-pii-baseline-eval-opf"
BASELINE_VOLUME_MOUNT = "/baseline"
CACHE_VOLUME_NAME = "hf-cache"
CACHE_MOUNT = "/cache"
H100_GPU = "H100"
TIMEOUT_SECONDS = 3 * 60 * 60

baseline_volume = modal.Volume.from_name(BASELINE_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

opf_image = (
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
            "OPF_CHECKPOINT": DEFAULT_NATIVE_CHECKPOINT_DIR,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }),
    )
    .env({"PYTHONPATH": f"{MODAL_SOURCE_ROOT}:{OPF_REMOTE_ROOT}"})
    .add_local_dir(str(OPF_REFERENCE_ROOT), remote_path=OPF_REMOTE_ROOT)
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
)
"""add_local_dir MUST be the last image step (Modal forbids build steps after local mounts).

Set env first, then mount the OPF package snapshot and local src.

"""

app = modal.App("meddies-pii-opf-baseline-eval")

EXTERNAL_CELL_LIMIT = 2000


@app.function(
    image=opf_image,
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BASELINE_VOLUME_MOUNT: baseline_volume, CACHE_MOUNT: cache_volume},
)
def opf_first_light(max_rows: int = 8) -> None:
    from meddies_pii.eval_baseline.adapters.opf import OpfAdapter
    from meddies_pii.eval_baseline.baseline.datasets import load_first_light_subset
    from meddies_pii.eval_baseline.baseline.run import ShardSpec, run_shard, shard_result_json
    from meddies_pii.eval_baseline.baseline.subset import write_subset_manifest

    if max_rows > FIRST_LIGHT_MAX_ROWS:
        msg = "OPF first-light is capped at 32 rows"
        raise ValueError(msg)

    subset = load_first_light_subset(limit=max_rows)
    rows = list(subset.rows)
    write_subset_manifest(
        Path(BASELINE_VOLUME_MOUNT) / "subsets" / f"{subset.name}-{subset.sha256[:12]}.json",
        subset,
    )
    adapter = OpfAdapter()
    adapter.load()
    result = run_shard(
        adapter=adapter,
        rows=rows,
        output_root=Path(BASELINE_VOLUME_MOUNT),
        spec=ShardSpec(
            model=adapter.name,
            dataset="smoke",
            shard=f"rows-{len(rows)}-{subset.sha256[:12]}",
        ),
        evaluation_contract=_evaluation_contract(),
        volume=baseline_volume,
    )
    print(f"OPF_SMOKE_RESULT::{shard_result_json(result)}", flush=True)


@app.local_entrypoint()
def main(max_rows: int = 8) -> None:
    if max_rows > FIRST_LIGHT_MAX_ROWS:
        msg = "OPF first-light is capped at 32 rows"
        raise ValueError(msg)
    opf_first_light.remote(max_rows=max_rows)
    print(json.dumps({"dispatched": "opf_first_light", "max_rows": max_rows}))


@app.function(
    image=opf_image,
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BASELINE_VOLUME_MOUNT: baseline_volume, CACHE_MOUNT: cache_volume},
)
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def opf_eval_cell(dataset: str, max_rows: int = 0, full: bool = False, force: bool = False) -> str:  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    """One (opf, dataset) eval cell on its own H100: predict + write to Volume.

    ``full`` drops the per-config 2k external cap to score the WHOLE split;
    ``force`` re-runs past an existing .done. Resume-safe when the persisted fixture
    identity matches.
    """
    from meddies_pii.eval_baseline.adapters.opf import OpfAdapter
    from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell
    from meddies_pii.eval_baseline.baseline.run import ShardSpec, run_shard, shard_result_json

    rows = load_eval_cell(
        dataset,
        v2_limit=max_rows or None,
        external_limit=None if full else (max_rows or EXTERNAL_CELL_LIMIT),
    )

    adapter = OpfAdapter()
    adapter.load()
    result = run_shard(
        adapter=adapter,
        rows=rows,
        output_root=Path(BASELINE_VOLUME_MOUNT),
        spec=ShardSpec(model="opf", dataset=dataset, shard="full"),
        evaluation_contract=_evaluation_contract(),
        volume=baseline_volume,
        force=force,
    )
    line = f"OPF_EVAL_RESULT::{shard_result_json(result)}"
    print(line, flush=True)
    return line


@app.function(image=opf_image, timeout=TIMEOUT_SECONDS)
def inspect_opf() -> None:
    """CPU preflight: verify label fold + mounted OPF package path, no model load."""
    from meddies_pii.eval_baseline.adapters.opf import (
        OPF_LABEL_FOLD,
        OPF_SUPPORTED_LABELS,
        OpfAdapter,
    )

    adapter = OpfAdapter()
    opf_package = Path(OPF_REMOTE_ROOT) / "opf"
    print(
        f"OPF_INSPECT:: name={adapter.name} "
        f"supported={sorted(adapter.supported_labels)} "
        f"fold_targets={sorted(set(OPF_LABEL_FOLD.values()))} "
        f"opf_package_exists={opf_package.exists()}",
        flush=True,
    )
    for native, meddies in sorted(OPF_LABEL_FOLD.items()):
        print(f"OPF_FOLD:: {native:<18} -> {meddies}", flush=True)
    print(f"OPF_SUPPORTED_LABELS::{sorted(OPF_SUPPORTED_LABELS)}", flush=True)


@app.local_entrypoint()
def run_inspect_opf() -> None:
    inspect_opf.remote()


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def run_opf_matrix(max_rows: int = 0, datasets: str = "", full: bool = False, force: bool = False) -> None:  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    """Fan out OPF across v2 + 15 external configs — one concurrent H100 per cell.

    Pass --datasets "a,b,c" to target cells, --full to score whole splits, and
    --force to overwrite an existing .done.
    """
    from meddies_pii.eval_baseline.baseline.datasets import select_eval_datasets

    selected = select_eval_datasets(datasets)
    handles = [(dataset, opf_eval_cell.spawn(dataset, max_rows, full, force)) for dataset in selected]
    print(json.dumps({"dispatched": "opf_matrix", "cells": len(handles), "full": full}))
    for dataset, handle in handles:
        try:
            print(handle.get())
        # reason: a Modal cell fails inside a remote container, so its exception reaches this process as a type this
        # reason: process never imported; the loop records the cell that failed and the rest of the fan-in still prints,
        # reason: which is exactly what the per-iteration `try` buys.
        except Exception as exc:  # ruff: ignore[blind-except,try-except-in-loop]
            print(f"CELL_FAILED::{dataset}::{type(exc).__name__}: {str(exc)[:90]}")


@app.function(
    image=opf_image,
    timeout=TIMEOUT_SECONDS,
    volumes={BASELINE_VOLUME_MOUNT: baseline_volume},
)
def aggregate_opf() -> None:
    """CPU: score every results/opf/<config>/full.jsonl and write aggregate JSON."""
    from meddies_pii.eval_baseline.adapters.opf import OPF_SUPPORTED_LABELS
    from meddies_pii.eval_baseline.baseline.aggregate import (
        aggregate_results,
        format_aggregate_report,
    )
    from meddies_pii.eval_baseline.baseline.run import (
        assert_frozen_fixture,
        expected_matrix_shard_identities,
        read_matrix_results,
    )

    contract = _evaluation_contract()

    matrix = read_matrix_results(
        Path(BASELINE_VOLUME_MOUNT),
        "opf",
        EVAL_DATASETS,
        require_done=True,
        expected_rows=EVAL_EXPECTED_ROWS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=expected_matrix_shard_identities(contract, EVAL_DATASETS),
    )
    for dataset in matrix.missing_datasets:
        print(f"OPF_AGG_MISSING::{dataset}", flush=True)

    rows_by_config = matrix.rows_by_dataset
    timing = matrix.timing_by_dataset
    report = aggregate_results(
        rows_by_config,
        supported_labels=OPF_SUPPORTED_LABELS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
        result_sha256_by_config=matrix.result_sha256_by_dataset,
        timing_by_config=timing,
    )
    assert_frozen_fixture(report)
    report_path = Path(BASELINE_VOLUME_MOUNT) / "reports" / "opf-aggregate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    baseline_volume.commit()
    print("OPF_AGG_BEGIN", flush=True)
    print(format_aggregate_report(report, model="opf"), flush=True)
    print("OPF_AGG_END", flush=True)


@app.local_entrypoint()
def run_aggregate_opf() -> None:
    aggregate_opf.remote()
