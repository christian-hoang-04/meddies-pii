"""Modal eval harness for the LFM2.5-350M BIOES spike on the Meddies eval matrix.

The first checkpoint of the own-architecture PII tagger (150 steps,
``bioes/20260604_h100_8192_r128a256_pack_bs128_150step_ckpt10``) scored as a 5th
baseline model, side-by-side with OPF, OpenMed, and GLiNER2 through the SAME eval loop
(``run_shard`` -> ``aggregate_results``), so the numbers are directly
comparable. One H100 cell scores one dataset/config and writes to a dedicated Volume;
the checkpoint (adapter + classifier) is read from the training artifact Volume.

    # CPU preflight: unit test + label/checkpoint-file check, no model load
    MODAL_PROFILE=<ws> modal run scripts/ops/run_lfm_bioes_baseline.py::run_preflight
    # GPU smoke: reassemble the checkpoint, print GOLD vs PRED on a few rows
    MODAL_PROFILE=<ws> modal run scripts/ops/run_lfm_bioes_baseline.py --max-rows 8
    # Full 17-cell matrix, one concurrent H100 per cell, then aggregate
    MODAL_PROFILE=<ws> modal run scripts/ops/run_lfm_bioes_baseline.py::run_lfm_matrix --full
    MODAL_PROFILE=<ws> modal run scripts/ops/run_lfm_bioes_baseline.py::run_aggregate_lfm
"""

from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: this module is launched as `uv run modal run <this path>` and is never imported, so it is a script
# reason: rather than a package member. An `__init__.py` would declare this directory a package it is not, and
# reason: the sibling scripts here that run under `python` say so with a shebang instead.
# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import json

# reason: the adapter check runs as pytest in its own process inside the Modal container, so its exit
# reason: status and captured output are the evidence this script reports.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from meddies_pii.eval_baseline.baseline.datasets import EVAL_DATASETS, EVAL_EXPECTED_ROWS
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
    use_pinned_debian_snapshot,
)

if TYPE_CHECKING:
    from meddies_pii.evaluation.identity import EvaluationContract


def _evaluation_contract() -> EvaluationContract:
    from meddies_pii import __file__ as package_file
    from meddies_pii.eval_baseline.adapters import lfm_bioes
    from meddies_pii.eval_baseline.baseline.run import resolved_environment_fingerprint
    from meddies_pii.evaluation.identity import (
        checkpoint_tree_artifact,
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
    base_model = manifest_artifact(f"hf://{lfm_bioes.MODEL_ID}", lfm_bioes.MODEL_REVISION)
    checkpoint = checkpoint_tree_artifact(f"modal-volume://{ARTIFACT_VOLUME_NAME}/{CHECKPOINT_DIR}", CHECKPOINT_DIR)
    return evaluation_contract(
        model=payload_artifact(
            "lfm-bioes-base-and-checkpoint",
            {
                "base_model": base_model.to_payload(),
                "checkpoint": checkpoint.to_payload(),
            },
        ),
        vendor_inference_source=payload_artifact("python-packages", PINNED_PACKAGES),
        local_adapter_source=runtime_source,
        applied_label_prediction_contract=payload_artifact("lfm-bioes-label-prediction", "bioes-full-nine-label-taxonomy"),
        decoder_contract=payload_artifact("lfm-bioes-decoder", "viterbi-bioes-max-seq-length-8192"),
        resolved_runtime_environment=payload_artifact("resolved-runtime", resolved_environment_fingerprint()),
        scorer_contract=runtime_source,
        supported_labels=tuple(sorted(lfm_bioes.LFM_BIOES_SUPPORTED_LABELS)),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )


FIRST_LIGHT_MAX_ROWS = 32

BASELINE_VOLUME_NAME = "meddies-pii-baseline-eval-lfm-bioes"
BASELINE_VOLUME_MOUNT = "/baseline"
ARTIFACT_VOLUME_NAME = "meddies-pii-bioes-artifacts"
ARTIFACT_VOLUME_MOUNT = "/artifacts"
CACHE_VOLUME_NAME = "hf-cache"
CACHE_MOUNT = "/cache"
CHECKPOINT_DIR = f"{ARTIFACT_VOLUME_MOUNT}/bioes/20260604_h100_8192_r128a256_pack_bs128_150step_ckpt10/unsloth"
"""The spike checkpoint on the training artifact Volume (profile that owns the volume)."""
TESTS_REMOTE_ROOT = "/root/tests"
H100_GPU = "H100"
A10_GPU = "A10G"
TIMEOUT_SECONDS = 3 * 60 * 60

PINNED_PACKAGES = (
    "torch==2.13.0",
    "transformers==5.11.0",
    "datasets==4.5.0",
    "peft==0.19.1",
    "safetensors==0.8.0",
    "python-dotenv==1.2.2",
    "pytest==9.1.1",
)
"""Native Transformers + PEFT reconstruction replaces the historical Unsloth runtime.

Every direct package is exact so this eval remains reproducible and auditable.

"""
COMPILER_APT_PACKAGES = ("gcc=4:12.2.0-3",)

baseline_volume = modal.Volume.from_name(BASELINE_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


image = (
    add_source_pythonpath(
        use_pinned_debian_snapshot(
            modal.Image.from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941"),
        )
        .apt_install(*COMPILER_APT_PACKAGES)
        .pip_install(*PINNED_PACKAGES)
        .env({"HF_HOME": f"{CACHE_MOUNT}/huggingface"}),
    )
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
    .add_local_dir("tests", remote_path=TESTS_REMOTE_ROOT)
)
"""add_local_dir MUST be the last image steps (Modal forbids build steps after local mounts).

Set env first, then mount src (the package) and tests (for the preflight).

"""

app = modal.App("meddies-pii-lfm-bioes-baseline-eval")

EXTERNAL_CELL_LIMIT = 2000
MODEL_NAME = "lfm-bioes-spike"


@app.function(
    image=image,
    timeout=TIMEOUT_SECONDS,
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def preflight() -> None:
    """CPU gate: run the decode unit test + verify labels and checkpoint files.

    No model load — the risky adapter+classifier reassembly happens at first-light on
    a GPU. This confirms the code is importable, the labels are the full nine, the
    checkpoint files are present on the mounted Volume, and the decode logic is green.
    """
    from meddies_pii.eval_baseline.adapters.lfm_bioes import (
        LFM_BIOES_SUPPORTED_LABELS,
        LfmBioesAdapter,
    )

    adapter = LfmBioesAdapter(checkpoint_dir=CHECKPOINT_DIR)
    adapter_dir = Path(CHECKPOINT_DIR) / "backbone_adapter"
    classifier = Path(CHECKPOINT_DIR) / "classifier.pt"
    print(
        f"LFM_INSPECT:: name={adapter.name} "
        f"supported={sorted(LFM_BIOES_SUPPORTED_LABELS)} "
        f"adapter_dir_exists={adapter_dir.is_dir()} "
        f"classifier_exists={classifier.is_file()}",
        flush=True,
    )

    # reason: argv is a list, so no shell parses it, and every element is a constant of this module —
    # reason: `TESTS_REMOTE_ROOT` is a container path baked into the image, not caller input.
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        # reason: `python` resolves against the PATH pinned on the `env=` line below, so the lookup is
        # reason: confined to the three root-owned directories of the container image.
        [  # ruff: ignore[start-process-with-partial-path]
            "python",
            "-m",
            "pytest",
            f"{TESTS_REMOTE_ROOT}/test_lfm_bioes_adapter.py",
            "-q",
        ],
        env={"PYTHONPATH": MODAL_SOURCE_ROOT, "PATH": "/usr/local/bin:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    print(completed.stdout, flush=True)
    if completed.returncode != 0:
        print(completed.stderr, flush=True)
        msg = f"preflight pytest failed (exit {completed.returncode})"
        raise RuntimeError(msg)
    print("LFM_PREFLIGHT_OK", flush=True)


@app.local_entrypoint()
def run_preflight() -> None:
    preflight.remote()


@app.function(
    image=image,
    gpu=A10_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        BASELINE_VOLUME_MOUNT: baseline_volume,
        ARTIFACT_VOLUME_MOUNT: artifact_volume,
        CACHE_MOUNT: cache_volume,
    },
)
# reason: one fail-closed parity gate validates persisted row identity, span shape, and exact counts before approval.
def native_span_parity(max_rows: int = 8) -> None:  # ruff: ignore[complex-structure,too-many-locals]
    """Require exact spans against a persisted pre-migration Unsloth shard.

    No insecure reference runtime is rebuilt. The old shard is evidence only; an
    absent or mismatched reference fails closed instead of silently changing a
    historical baseline.
    """
    from collections import Counter

    from meddies_pii.eval_baseline.adapters.lfm_bioes import LfmBioesAdapter
    from meddies_pii.eval_baseline.baseline.datasets import load_first_light_subset
    from meddies_pii.eval_baseline.baseline.run import resolved_environment_fingerprint
    from meddies_pii.json_types import is_str_mapping
    from meddies_pii.jsonl import read_jsonl

    if not 1 <= max_rows <= FIRST_LIGHT_MAX_ROWS:
        msg = "native parity max_rows must be in [1, 32]"
        raise ValueError(msg)
    subset = load_first_light_subset(limit=max_rows)
    reference_paths = sorted((Path(BASELINE_VOLUME_MOUNT) / "results" / MODEL_NAME).rglob("*.jsonl"))
    if not reference_paths:
        msg = "missing persisted Unsloth predictions; migration cannot be proven"
        raise RuntimeError(msg)
    reference: dict[str, list[tuple[str, int, int]]] = {}
    for path in reference_paths:
        for record in read_jsonl(path):
            doc_id = record.get("doc_id")
            raw_spans = record.get("pred_spans")
            if not isinstance(doc_id, str) or not isinstance(raw_spans, list):
                msg = "persisted Unsloth prediction has malformed identity or spans"
                # reason: this validates persisted run evidence, not a live caller argument;
                # reason: malformed evidence aborts the run.
                raise RuntimeError(msg)  # ruff: ignore[type-check-without-type-error]
            spans: list[tuple[str, int, int]] = []
            for raw_span in raw_spans:
                if not is_str_mapping(raw_span):
                    msg = "persisted Unsloth prediction contains a malformed span"
                    raise RuntimeError(msg)
                label = raw_span.get("label")
                start = raw_span.get("start")
                end = raw_span.get("end")
                if (
                    not isinstance(label, str)
                    or isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                ):
                    msg = "persisted Unsloth prediction contains a malformed span"
                    # reason: this validates persisted run evidence, not a live caller argument;
                    # reason: malformed evidence aborts the run.
                    raise RuntimeError(msg)  # ruff: ignore[type-check-without-type-error]
                spans.append((label, start, end))
            reference[doc_id] = spans
    rows = list(subset.rows)
    missing = [row.stable_id for row in rows if row.stable_id not in reference]
    if missing:
        msg = f"persisted reference lacks fixed parity rows: {missing[:3]}"
        raise RuntimeError(msg)
    adapter = LfmBioesAdapter(checkpoint_dir=CHECKPOINT_DIR)
    adapter.load()
    actual = adapter.predict([row.text for row in rows])
    expected = [reference[row.stable_id] for row in rows]
    observed = [[(span.label, span.start, span.end) for span in spans] for spans in actual]
    row_mismatches = sum(
        actual_spans != expected_spans for actual_spans, expected_spans in zip(observed, expected, strict=True)
    )
    missing_spans = sum(
        (Counter(expected_spans) - Counter(actual_spans)).total()
        for actual_spans, expected_spans in zip(observed, expected, strict=True)
    )
    unexpected_spans = sum(
        (Counter(actual_spans) - Counter(expected_spans)).total()
        for actual_spans, expected_spans in zip(observed, expected, strict=True)
    )
    if row_mismatches:
        msg = (
            "native span parity failed against persisted Unsloth reference: "
            f"row_mismatches={row_mismatches} missing_spans={missing_spans} "
            f"unexpected_spans={unexpected_spans}"
        )
        raise RuntimeError(
            msg,
        )
    print(
        "LFM_NATIVE_PARITY_OK::"
        + json.dumps({
            "rows": len(rows),
            "subset_sha256": subset.sha256,
            "gpu": A10_GPU,
            "reference_paths": [str(path) for path in reference_paths],
            "row_mismatches": row_mismatches,
            "missing_spans": missing_spans,
            "unexpected_spans": unexpected_spans,
            "environment_fingerprint": resolved_environment_fingerprint(),
        }),
        flush=True,
    )


@app.local_entrypoint()
def run_native_parity(max_rows: int = 8) -> None:
    native_span_parity.remote(max_rows=max_rows)


@app.function(
    image=image,
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        BASELINE_VOLUME_MOUNT: baseline_volume,
        ARTIFACT_VOLUME_MOUNT: artifact_volume,
        CACHE_MOUNT: cache_volume,
    },
)
def lfm_first_light(max_rows: int = 8) -> None:
    """Eye-verify checkpoint load + char-offset alignment on real.

    Multilingual) text before trusting the matrix: GOLD vs PRED should line up on multibyte scripts.

    """
    from meddies_pii.eval_baseline.adapters.lfm_bioes import LfmBioesAdapter
    from meddies_pii.eval_baseline.baseline.datasets import load_first_light_subset
    from meddies_pii.eval_baseline.baseline.run import ShardSpec, run_shard, shard_result_json
    from meddies_pii.eval_baseline.baseline.subset import write_subset_manifest

    if max_rows > FIRST_LIGHT_MAX_ROWS:
        msg = "first-light is capped at 32 rows"
        raise ValueError(msg)

    subset = load_first_light_subset(limit=max_rows)
    rows = list(subset.rows)
    write_subset_manifest(
        Path(BASELINE_VOLUME_MOUNT) / "subsets" / f"{subset.name}-{subset.sha256[:12]}.json",
        subset,
    )

    adapter = LfmBioesAdapter(checkpoint_dir=CHECKPOINT_DIR)
    adapter.load()

    preview = adapter.predict([row.text for row in rows[:3]])
    for row, pred in zip(rows[:3], preview, strict=True):
        gold = [(span.label, span.text) for span in row.gold_spans]
        pred_pairs = [(span.label, span.text) for span in pred]
        print(f"LFM_ROW lang={row.language}", flush=True)
        print(f"  GOLD={gold}", flush=True)
        print(f"  PRED={pred_pairs}", flush=True)

    result = run_shard(
        adapter=adapter,
        rows=rows,
        output_root=Path(BASELINE_VOLUME_MOUNT),
        spec=ShardSpec(
            model=MODEL_NAME,
            dataset="smoke",
            shard=f"rows-{len(rows)}-{subset.sha256[:12]}",
        ),
        evaluation_contract=_evaluation_contract(),
        volume=baseline_volume,
        force=True,
    )
    print(f"LFM_SMOKE_RESULT::{shard_result_json(result)}", flush=True)


@app.local_entrypoint()
def main(max_rows: int = 8) -> None:
    if max_rows > FIRST_LIGHT_MAX_ROWS:
        msg = "first-light is capped at 32 rows"
        raise ValueError(msg)
    lfm_first_light.remote(max_rows=max_rows)
    print(json.dumps({"dispatched": "lfm_first_light", "max_rows": max_rows}))


@app.function(
    image=image,
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        BASELINE_VOLUME_MOUNT: baseline_volume,
        ARTIFACT_VOLUME_MOUNT: artifact_volume,
        CACHE_MOUNT: cache_volume,
    },
)
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def lfm_eval_cell(dataset: str, max_rows: int = 0, full: bool = False, force: bool = False) -> str:  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    """One (lfm-bioes-spike, dataset) eval cell on its own H100: predict + write.

    ``full`` drops the per-config 2k external cap to score the WHOLE split; ``force``
    re-runs past an existing ``.done``. Resume-safe when the persisted fixture identity matches.
    """
    from meddies_pii.eval_baseline.adapters.lfm_bioes import LfmBioesAdapter
    from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell
    from meddies_pii.eval_baseline.baseline.run import ShardSpec, run_shard, shard_result_json

    rows = load_eval_cell(
        dataset,
        v2_limit=max_rows or None,
        external_limit=None if full else (max_rows or EXTERNAL_CELL_LIMIT),
    )

    adapter = LfmBioesAdapter(checkpoint_dir=CHECKPOINT_DIR)
    adapter.load()
    result = run_shard(
        adapter=adapter,
        rows=rows,
        output_root=Path(BASELINE_VOLUME_MOUNT),
        spec=ShardSpec(model=MODEL_NAME, dataset=dataset, shard="full"),
        evaluation_contract=_evaluation_contract(),
        volume=baseline_volume,
        force=force,
    )
    line = f"LFM_EVAL_RESULT::{shard_result_json(result)}"
    print(line, flush=True)
    return line


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def run_lfm_matrix(max_rows: int = 0, datasets: str = "", full: bool = False, force: bool = False) -> None:  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    """Fan out the spike across v2 + 15 external configs — one concurrent H100 per cell.

    Pass --datasets "a,b,c" to target cells, --full to score whole splits, and --force
    to overwrite an existing ``.done``.
    """
    from meddies_pii.eval_baseline.baseline.datasets import select_eval_datasets

    selected = select_eval_datasets(datasets)
    handles = [(dataset, lfm_eval_cell.spawn(dataset, max_rows, full, force)) for dataset in selected]
    print(json.dumps({"dispatched": "lfm_matrix", "cells": len(handles), "full": full}))
    for dataset, handle in handles:
        try:
            print(handle.get())
        # reason: a Modal cell fails inside a remote container, so its exception reaches this process as a type this
        # reason: process never imported; the loop records the cell that failed and the rest of the fan-in still prints,
        # reason: which is exactly what the per-iteration `try` buys.
        except Exception as exc:  # ruff: ignore[blind-except,try-except-in-loop]
            print(f"CELL_FAILED::{dataset}::{type(exc).__name__}: {str(exc)[:90]}")


@app.function(
    image=image,
    timeout=TIMEOUT_SECONDS,
    volumes={
        BASELINE_VOLUME_MOUNT: baseline_volume,
        ARTIFACT_VOLUME_MOUNT: artifact_volume,
    },
)
def aggregate_lfm() -> None:
    """CPU: score every results/lfm-bioes-spike/<config>/full.jsonl -> aggregate JSON."""
    from meddies_pii.eval_baseline.adapters.lfm_bioes import LFM_BIOES_SUPPORTED_LABELS
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
        MODEL_NAME,
        EVAL_DATASETS,
        require_done=True,
        expected_rows=EVAL_EXPECTED_ROWS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=expected_matrix_shard_identities(contract, EVAL_DATASETS),
    )
    for dataset in matrix.missing_datasets:
        print(f"LFM_AGG_MISSING::{dataset}", flush=True)

    rows_by_config = matrix.rows_by_dataset
    timing = matrix.timing_by_dataset
    report = aggregate_results(
        rows_by_config,
        supported_labels=LFM_BIOES_SUPPORTED_LABELS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
        result_sha256_by_config=matrix.result_sha256_by_dataset,
        timing_by_config=timing,
    )
    assert_frozen_fixture(report)
    report_path = Path(BASELINE_VOLUME_MOUNT) / "reports" / f"{MODEL_NAME}-aggregate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    baseline_volume.commit()
    print("LFM_AGG_BEGIN", flush=True)
    print(format_aggregate_report(report, model=MODEL_NAME), flush=True)
    print("LFM_AGG_END", flush=True)


@app.local_entrypoint()
def run_aggregate_lfm() -> None:
    aggregate_lfm.remote()
