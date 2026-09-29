"""Modal first-light harness for external PII baselines.

Gate 1 wires the shared eval loop plus the OpenMed adapter only:

    MODAL_PROFILE=retraction modal run scripts/ops/run_model_baseline.py --max-rows 32

Results are written to a Modal Volume. The function prints a compact
``OPENMED_SMOKE_RESULT::`` line and returns no payload, avoiding Modal's large
result/log blob path.
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

from meddies_pii.eval_baseline.baseline.datasets import (
    EVAL_DATASETS,
    EVAL_EXPECTED_ROWS,
    EXTERNAL_CONFIGS,
)
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)

if TYPE_CHECKING:
    from meddies_pii.evaluation.identity import EvaluationContract


def _evaluation_contract() -> EvaluationContract:
    from meddies_pii import __file__ as package_file
    from meddies_pii.eval_baseline.adapters import openmed
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
        model=manifest_artifact(f"hf://{openmed.MODEL_ID}", openmed.MODEL_REVISION),
        vendor_inference_source=payload_artifact("python-package:transformers", "5.11.0"),
        local_adapter_source=runtime_source,
        applied_label_prediction_contract=payload_artifact("openmed-label-prediction", "simple+clean_spans"),
        decoder_contract=payload_artifact("openmed-decoder", "pipeline-simple-512-stride-64"),
        resolved_runtime_environment=payload_artifact("resolved-runtime", resolved_environment_fingerprint()),
        scorer_contract=runtime_source,
        supported_labels=tuple(sorted(openmed.OPENMED_SUPPORTED_LABELS)),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )


FIRST_LIGHT_MAX_ROWS = 32

BASELINE_VOLUME_NAME = "meddies-pii-baseline-eval"
BASELINE_VOLUME_MOUNT = "/baseline"
H100_GPU = "H100"
TIMEOUT_SECONDS = 3 * 60 * 60
"""3h: full-scale cells (nemotron ~100k rows) can run long on a single H100."""

baseline_volume = modal.Volume.from_name(BASELINE_VOLUME_NAME, create_if_missing=True)

openmed_image = add_source_pythonpath(
    modal.Image
    .from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.11.0",
        "datasets==4.5.0",
        "huggingface_hub==1.19.0",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }),
).add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
"""add_local_dir MUST be the last image step (current Modal forbids any build step — incl.

.env — after add_local_*). So set both env vars first, then mount.

Long-text eval cells (v2-eval, v2-eval-challenge, creddata_en) OOM'd at batch_size=128 under GPU contention ("71.71 GiB in
use"). The adapter default is now 32; expandable_segments lets the CUDA caching allocator grow/shrink instead of
fragmenting fixed blocks, which is what tips a long-sequence batch over the edge.

"""

app = modal.App("meddies-pii-model-baseline-eval")


@app.function(
    image=openmed_image,
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BASELINE_VOLUME_MOUNT: baseline_volume},
)
def openmed_first_light(max_rows: int = 32) -> None:
    from meddies_pii.eval_baseline.adapters.openmed import OpenMedAdapter
    from meddies_pii.eval_baseline.baseline.datasets import load_first_light_subset
    from meddies_pii.eval_baseline.baseline.run import (
        ShardSpec,
        run_shard,
        shard_result_json,
    )
    from meddies_pii.eval_baseline.baseline.subset import write_subset_manifest

    if max_rows > FIRST_LIGHT_MAX_ROWS:
        msg = "Gate-1 first-light is capped at 32 rows"
        raise ValueError(msg)

    subset = load_first_light_subset(limit=max_rows)
    rows = list(subset.rows)
    write_subset_manifest(
        Path(BASELINE_VOLUME_MOUNT) / "subsets" / f"{subset.name}-{subset.sha256[:12]}.json",
        subset,
    )
    adapter = OpenMedAdapter()
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
    print(f"OPENMED_SMOKE_RESULT::{shard_result_json(result)}", flush=True)


@app.local_entrypoint()
def main(max_rows: int = 32) -> None:
    if max_rows > FIRST_LIGHT_MAX_ROWS:
        msg = "Gate-1 first-light is capped at 32 rows"
        raise ValueError(msg)
    openmed_first_light.remote(max_rows=max_rows)
    print(json.dumps({"dispatched": "openmed_first_light", "max_rows": max_rows}))


EXTERNAL_CELL_LIMIT = 2000


@app.function(
    image=openmed_image,
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BASELINE_VOLUME_MOUNT: baseline_volume},
)
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def openmed_eval_cell(dataset: str, max_rows: int = 0, full: bool = False, force: bool = False) -> str:  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    """One (model, dataset) eval cell on its own H100: predict + write to Volume.

    dataset is a v2 split or one external config. ``full`` drops the per-config
    2k external cap to score the WHOLE eval split (v2 splits are always full).
    ``force`` re-runs past an existing .done (e.g. to overwrite a subset run with
    the full-scale one). Resume-safe when the persisted fixture identity matches.
    """
    from meddies_pii.eval_baseline.adapters.openmed import OpenMedAdapter
    from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell
    from meddies_pii.eval_baseline.baseline.run import ShardSpec, run_shard, shard_result_json

    rows = load_eval_cell(
        dataset,
        v2_limit=max_rows or None,
        external_limit=None if full else (max_rows or EXTERNAL_CELL_LIMIT),
    )

    adapter = OpenMedAdapter()
    adapter.load()
    result = run_shard(
        adapter=adapter,
        rows=rows,
        output_root=Path(BASELINE_VOLUME_MOUNT),
        spec=ShardSpec(model="openmed", dataset=dataset, shard="full"),
        evaluation_contract=_evaluation_contract(),
        volume=baseline_volume,
        force=force,
    )
    line = f"OPENMED_EVAL_RESULT::{shard_result_json(result)}"
    print(line, flush=True)
    return line


@app.function(
    image=openmed_image,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def inspect_external() -> None:
    """CPU check: load every external config's eval split, report rows + failures.

    Surfaces the ai4privacy_ko preview failure (and any other broken config).
    """
    from datasets import load_dataset

    for config in EXTERNAL_CONFIGS:
        try:
            table = load_dataset("Meddies/meddies-pii-external", config, split="eval")
            sample = table[0] if table.num_rows else {}
            spans = len(sample.get("label") or [])
            print(
                f"OK   {config:<18} eval_rows={table.num_rows:>6} text_len={len(sample.get('text', ''))} spans={spans}",
                flush=True,
            )
        # reason: a config that will not load raises whatever its loader raises — network, schema or auth — and every one
        # reason: means the same thing here: record this config as an error row and keep sweeping the others.
        except Exception as exc:  # ruff: ignore[blind-except,try-except-in-loop]
            print(f"FAIL {config:<18} {type(exc).__name__}: {str(exc)[:90]}", flush=True)


@app.local_entrypoint()
def run_inspect_external() -> None:
    inspect_external.remote()


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def run_openmed_matrix(max_rows: int = 0, datasets: str = "", full: bool = False, force: bool = False) -> None:  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    """Fan out openmed across v2 + 15 external configs — one concurrent H100 per cell.

    Pass --datasets "a,b,c" to re-run only specific cells. Cells with a matching persisted fixture \
are skipped, but each skip still cold-starts an H100 and loads the model
    before the fixture check, so on a resume target the failed cells directly.
    Pass --full to score the WHOLE eval split (drops the 2k external cap) and
    --force to overwrite an existing subset run with the full-scale one.
    """
    from meddies_pii.eval_baseline.baseline.datasets import select_eval_datasets

    selected = select_eval_datasets(datasets)
    handles = [(dataset, openmed_eval_cell.spawn(dataset, max_rows, full, force)) for dataset in selected]
    print(json.dumps({"dispatched": "openmed_matrix", "cells": len(handles), "full": full}))
    for dataset, handle in handles:
        try:
            print(handle.get())
        # reason: a Modal cell fails inside a remote container, so its exception reaches this process as a type this
        # reason: process never imported; the loop records the cell that failed and the rest of the fan-in still prints,
        # reason: which is exactly what the per-iteration `try` buys.
        except Exception as exc:  # ruff: ignore[blind-except,try-except-in-loop]
            print(f"CELL_FAILED::{dataset}::{type(exc).__name__}: {str(exc)[:90]}")


@app.function(
    image=openmed_image,
    timeout=TIMEOUT_SECONDS,
    volumes={BASELINE_VOLUME_MOUNT: baseline_volume},
)
def aggregate_openmed() -> None:
    """Score every results/openmed/<config>/full.jsonl on the Volume, on CPU and with no GPU.

    Scoring runs through the shared aggregator, prints the per-config, per-language and
    per-label F1 grid, and persists the full nested report to reports/openmed-aggregate.json.
    """
    from meddies_pii.eval_baseline.adapters.openmed import OPENMED_SUPPORTED_LABELS
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
        "openmed",
        EVAL_DATASETS,
        require_done=True,
        expected_rows=EVAL_EXPECTED_ROWS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=expected_matrix_shard_identities(contract, EVAL_DATASETS),
    )
    for dataset in matrix.missing_datasets:
        print(f"OPENMED_AGG_MISSING::{dataset}", flush=True)

    rows_by_config = matrix.rows_by_dataset
    timing = matrix.timing_by_dataset
    report = aggregate_results(
        rows_by_config,
        supported_labels=OPENMED_SUPPORTED_LABELS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
        result_sha256_by_config=matrix.result_sha256_by_dataset,
        timing_by_config=timing,
    )
    assert_frozen_fixture(report)
    report_path = Path(BASELINE_VOLUME_MOUNT) / "reports" / "openmed-aggregate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline_volume.commit()
    print("OPENMED_AGG_BEGIN", flush=True)
    print(format_aggregate_report(report, model="openmed"), flush=True)
    print("OPENMED_AGG_END", flush=True)


@app.local_entrypoint()
def run_aggregate_openmed() -> None:
    aggregate_openmed.remote()
