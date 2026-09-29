"""Profile evaluation-report aggregation without models or external datasets.

The fixture is deliberately synthetic: it matches the frozen matrix's 17 cells
and 263,785 rows, but is **shape evidence only**, not benchmark evidence.  It
lets us measure aggregation's resident-memory behaviour without downloading a
corpus or running inference.

Run on Modal CPU:

    MODAL_PROFILE=meddies-run uv run modal run \
      scripts/quality/profile_eval_aggregation.py::profile
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
import hashlib
import time
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, overload, override

import modal

from meddies_pii.eval_baseline.baseline.datasets import EVAL_EXPECTED_ROWS
from meddies_pii.json_types import JsonObject, JsonValue
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)
from meddies_pii.runtime_memory import peak_rss_bytes

if TYPE_CHECKING:
    from meddies_pii.taxonomy import PiiLabel

WRONG_LABEL_OUTCOME = 2

APP_NAME = "meddies-pii-aggregation-profile"
FULL_MATRIX_ROWS = sum(EVAL_EXPECTED_ROWS.values())
SUPPORTED_LABELS: frozenset[PiiLabel] = frozenset({"email_address", "human_name"})

image = add_source_pythonpath(
    modal.Image.from_registry(
        "python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d",
    ).pip_install("typing-extensions==4.15.0"),
).add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
app = modal.App(APP_NAME, image=image)


def _peak_rss_bytes() -> int:
    """Return portable process peak resident bytes."""
    return peak_rss_bytes()


def _span(start: int, end: int, text: str, label: str) -> JsonObject:
    return {"start": start, "end": end, "text": text, "label": label}


def _row(config: str, index: int) -> JsonObject:
    """Return one deterministic valid result row with varied scoring outcomes."""
    text_id = hashlib.sha256(f"{config}:{index}".encode()).hexdigest()
    gold = _span(0, 4, "John", "human_name")
    predicted: list[JsonValue] = []
    outcome = index % 4
    if outcome == 0:
        predicted.append(gold)
    elif outcome == 1:
        predicted.append(_span(1, 4, "ohn", "human_name"))
    elif outcome == WRONG_LABEL_OUTCOME:
        predicted.append(_span(0, 4, "John", "email_address"))
    return {
        "id": text_id,
        "doc_id": f"{config}:{index}",
        "pred_spans": predicted,
        "gold_spans": [gold],
        "language": config.rsplit("_", maxsplit=1)[-1],
        "slice": ["shape-only"],
    }


class _GeneratedRows(Sequence[JsonObject]):
    """A deterministic re-iterable fixture with no retained result rows."""

    def __init__(self, config: str, count: int) -> None:
        self._config = config
        self._count = count

    @override
    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> JsonObject: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[JsonObject]: ...

    @override
    def __getitem__(self, index: int | slice) -> JsonObject | Sequence[JsonObject]:
        if isinstance(index, slice):
            return tuple(_row(self._config, item) for item in range(*index.indices(self._count)))
        resolved = index if index >= 0 else self._count + index
        if resolved < 0 or resolved >= self._count:
            raise IndexError(index)
        return _row(self._config, resolved)

    @override
    def __iter__(self) -> Iterator[JsonObject]:
        return (_row(self._config, index) for index in range(self._count))


def _profile_fixture(rows_per_cell: int) -> tuple[dict[str, _GeneratedRows], int]:
    rows_by_config = {
        config: _GeneratedRows(config, min(rows_per_cell, count)) for config, count in EVAL_EXPECTED_ROWS.items()
    }
    return rows_by_config, sum(len(rows) for rows in rows_by_config.values())


@app.function(cpu=8.0, memory=16_384, timeout=30 * 60)
def profile(rows_per_cell: int = 0) -> JsonObject:
    """Return aggregation RSS/time for the fixed matrix shape or a small probe."""
    from meddies_pii.eval_baseline.baseline.aggregate import aggregate_results
    from meddies_pii.evaluation.identity import (
        ArtifactIdentity,
        EvaluationContract,
        canonical_sha256,
        dataset_shard_identity_from_result_rows,
    )

    per_cell = rows_per_cell or max(EVAL_EXPECTED_ROWS.values())
    rows_by_config, total_rows = _profile_fixture(per_cell)
    if rows_per_cell == 0 and total_rows != FULL_MATRIX_ROWS:
        msg = f"shape fixture row count changed: expected {FULL_MATRIX_ROWS}, got {total_rows}"
        raise RuntimeError(msg)

    def artifact(reference: str, character: str) -> ArtifactIdentity:
        return ArtifactIdentity(reference, "f" * 40, character * 64)

    contract = EvaluationContract(
        model=artifact("shape-model", "a"),
        vendor_inference_source=artifact("shape-vendor", "b"),
        local_adapter_source=artifact("shape-adapter", "c"),
        applied_label_prediction_contract=artifact("shape-labels", "d"),
        decoder_contract=artifact("shape-decoder", "e"),
        resolved_runtime_environment=artifact("shape-runtime", "f"),
        scorer_contract=artifact("shape-scorer", "0"),
        supported_labels=tuple(sorted(SUPPORTED_LABELS)),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )
    identities = {
        config: dataset_shard_identity_from_result_rows(contract, dataset=config, shard="full", rows=rows)
        for config, rows in rows_by_config.items()
    }
    result_sha256_by_config = {
        config: canonical_sha256({
            "shape_only_result": config,
            "rows": identity.row_count,
            "fixture_sha256": identity.fixture_identity.digest,
        })
        for config, identity in identities.items()
    }
    before_rss = _peak_rss_bytes()
    started = time.perf_counter()
    report = aggregate_results(
        rows_by_config,
        supported_labels=SUPPORTED_LABELS,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=identities,
        result_sha256_by_config=result_sha256_by_config,
    )
    elapsed = time.perf_counter() - started
    payload: JsonObject = {
        "evidence": "shape-only synthetic fixture; no model, corpus, or inference",
        "rows": total_rows,
        "cells": len(rows_by_config),
        "peak_rss_mib": round(_peak_rss_bytes() / (1024 * 1024), 2),
        "rss_before_aggregate_mib": round(before_rss / (1024 * 1024), 2),
        "aggregate_wall_seconds": round(elapsed, 3),
        "report_sha256": canonical_sha256(report),
        "fixture_sha256": report["fixture"]["sha256"],
    }
    print(f"AGGREGATION_PROFILE::{payload}", flush=True)
    return payload


@app.local_entrypoint()
def run(rows_per_cell: int = 0) -> None:
    """Launch the profile and print a JSON-compatible result."""
    print(profile.remote(rows_per_cell))
