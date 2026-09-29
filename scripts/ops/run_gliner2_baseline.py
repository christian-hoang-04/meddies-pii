"""Modal eval harness for the GLiNER2 privacy-filter PII baseline.

Mirrors ``run_model_baseline.py``: one (model, dataset) eval cell per config,
each on its own H100, fanned out with resilient ``.get()`` and written to a Modal
Volume via the shared ``run_shard``. Results go to the gliner2-only Volume so this
baseline never collides with the openmed run.

    MODAL_PROFILE=retraction modal run scripts/ops/run_gliner2_baseline.py
    MODAL_PROFILE=retraction modal run \
        scripts/ops/run_gliner2_baseline.py::run_aggregate_gliner2

The image installs ``gliner2[local]`` (the base wheel omits torch) and sets
``FLASH_DEBERTA=1`` before any import. ``add_local_dir`` is the LAST image step —
every ``.env`` must precede it or Modal raises InvalidError.
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

from meddies_pii.eval_baseline.baseline.datasets import EVAL_DATASETS, EVAL_EXPECTED_ROWS
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)

if TYPE_CHECKING:
    from meddies_pii.evaluation.identity import EvaluationContract


def _evaluation_contract() -> EvaluationContract:
    from meddies_pii import __file__ as package_file
    from meddies_pii.eval_baseline.adapters import gliner2
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
        model=manifest_artifact(f"hf://{gliner2.MODEL_ID}", gliner2.MODEL_REVISION),
        vendor_inference_source=payload_artifact("python-package:gliner2", "1.3.2"),
        local_adapter_source=runtime_source,
        applied_label_prediction_contract=payload_artifact("gliner2-label-prediction", "schema-fold"),
        decoder_contract=payload_artifact("gliner2-decoder", "subword-window-512"),
        resolved_runtime_environment=payload_artifact("resolved-runtime", resolved_environment_fingerprint()),
        scorer_contract=runtime_source,
        supported_labels=tuple(sorted(gliner2.GLINER2_SUPPORTED_LABELS)),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )


BASELINE_VOLUME_NAME = "meddies-pii-baseline-eval-gliner2"
BASELINE_VOLUME_MOUNT = "/baseline"
H100_GPU = "H100"
TIMEOUT_SECONDS = 3 * 60 * 60
"""3h: full-scale cells (nemotron ~100k rows) can run long on a single H100."""

baseline_volume = modal.Volume.from_name(BASELINE_VOLUME_NAME, create_if_missing=True)

gliner2_image = add_source_pythonpath(
    modal.Image
    .from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d")
    .pip_install(
        "gliner2[local]==1.3.2",
        "datasets==4.5.0",
        "huggingface_hub==1.19.0",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "FLASH_DEBERTA": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }),
).add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
"""add_local_dir MUST be the last image step (Modal forbids any build step — incl.

.env — after add_local_*). FLASH_DEBERTA is set here so it is in the process environment before gliner2 imports DeBERTa;
the adapter also sets it defensively.

"""

app = modal.App("meddies-pii-gliner2-baseline-eval")

EXTERNAL_CELL_LIMIT = 2000
"""Same per-config matrix as the openmed run.

V2 splits + each external config is its own cell, so the report breaks down by language + source.

"""


@app.function(
    image=gliner2_image,
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BASELINE_VOLUME_MOUNT: baseline_volume},
)
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def gliner2_eval_cell(dataset: str, max_rows: int = 0, full: bool = False, force: bool = False) -> str:  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    """One (gliner2, dataset) eval cell on its own H100: predict + write to Volume.

    ``full`` drops the per-config 2k external cap to score the WHOLE split;
    ``force`` re-runs past an existing .done (e.g. overwrite a smoke/subset run).
    Resume-safe when the persisted fixture identity matches otherwise.
    """
    from meddies_pii.eval_baseline.adapters.gliner2 import Gliner2Adapter
    from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell
    from meddies_pii.eval_baseline.baseline.run import ShardSpec, run_shard, shard_result_json

    rows = load_eval_cell(
        dataset,
        v2_limit=max_rows or None,
        external_limit=None if full else (max_rows or EXTERNAL_CELL_LIMIT),
    )

    adapter = Gliner2Adapter()
    adapter.load()
    result = run_shard(
        adapter=adapter,
        rows=rows,
        output_root=Path(BASELINE_VOLUME_MOUNT),
        spec=ShardSpec(model="gliner2", dataset=dataset, shard="full"),
        evaluation_contract=_evaluation_contract(),
        volume=baseline_volume,
        force=force,
    )
    line = f"GLINER2_EVAL_RESULT::{shard_result_json(result)}"
    print(line, flush=True)
    return line


@app.function(image=gliner2_image, timeout=TIMEOUT_SECONDS)
def inspect_gliner2() -> None:
    """Print the adapter's schema and label fold on CPU, without loading the model.

    The preflight makes the wiring verifiable before any H100 cell cold-starts.
    """
    from meddies_pii.eval_baseline.adapters.gliner2 import (
        GLINER2_LABEL_FOLD,
        GLINER2_SOURCE_LABELS,
        Gliner2Adapter,
    )

    adapter = Gliner2Adapter()
    print(
        f"GLINER2_INSPECT:: name={adapter.name} "
        f"supported={sorted(adapter.supported_labels)} "
        f"source_labels={len(GLINER2_SOURCE_LABELS)} "
        f"fold_targets={sorted(set(GLINER2_LABEL_FOLD.values()))}",
        flush=True,
    )
    for native, meddies in sorted(GLINER2_LABEL_FOLD.items()):
        print(f"GLINER2_FOLD:: {native:<24} -> {meddies}", flush=True)


@app.local_entrypoint()
def run_inspect_gliner2() -> None:
    inspect_gliner2.remote()


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def run_gliner2_matrix(max_rows: int = 0, datasets: str = "", full: bool = False, force: bool = False) -> None:  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    """Fan out gliner2 across v2 + 15 external configs — one concurrent H100 per cell.

    Pass --datasets "a,b,c" to target specific cells, --full to score the whole
    split (drops the 2k external cap), --force to overwrite an existing .done.
    """
    from meddies_pii.eval_baseline.baseline.datasets import select_eval_datasets

    selected = select_eval_datasets(datasets)
    handles = [(dataset, gliner2_eval_cell.spawn(dataset, max_rows, full, force)) for dataset in selected]
    print(json.dumps({"dispatched": "gliner2_matrix", "cells": len(handles), "full": full}))
    for dataset, handle in handles:
        try:
            print(handle.get())
        # reason: a Modal cell fails inside a remote container, so its exception reaches this process as a type this
        # reason: process never imported; the loop records the cell that failed and the rest of the fan-in still prints,
        # reason: which is exactly what the per-iteration `try` buys.
        except Exception as exc:  # ruff: ignore[blind-except,try-except-in-loop]
            print(f"CELL_FAILED::{dataset}::{type(exc).__name__}: {str(exc)[:90]}")


@app.function(
    image=gliner2_image,
    timeout=TIMEOUT_SECONDS,
    volumes={BASELINE_VOLUME_MOUNT: baseline_volume},
)
def aggregate_gliner2() -> None:
    """Score every results/gliner2/<config>/full.jsonl on the Volume, on CPU and with no GPU.

    Scoring runs through the shared aggregator, prints the per-config, per-language and
    per-label F1 grid, and persists the full nested report to reports/gliner2-aggregate.json.
    """
    from meddies_pii.eval_baseline.adapters.gliner2 import GLINER2_SUPPORTED_LABELS
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
        "gliner2",
        EVAL_DATASETS,
        require_done=True,
        expected_rows=EVAL_EXPECTED_ROWS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=expected_matrix_shard_identities(contract, EVAL_DATASETS),
    )
    for dataset in matrix.missing_datasets:
        print(f"GLINER2_AGG_MISSING::{dataset}", flush=True)

    rows_by_config = matrix.rows_by_dataset
    timing = matrix.timing_by_dataset
    report = aggregate_results(
        rows_by_config,
        supported_labels=GLINER2_SUPPORTED_LABELS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
        result_sha256_by_config=matrix.result_sha256_by_dataset,
        timing_by_config=timing,
    )
    assert_frozen_fixture(report)
    report_path = Path(BASELINE_VOLUME_MOUNT) / "reports" / "gliner2-aggregate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline_volume.commit()
    print("GLINER2_AGG_BEGIN", flush=True)
    print(format_aggregate_report(report, model="gliner2"), flush=True)
    print("GLINER2_AGG_END", flush=True)


@app.local_entrypoint()
def run_aggregate_gliner2() -> None:
    aggregate_gliner2.remote()
