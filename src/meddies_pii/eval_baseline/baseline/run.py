from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the sibling is imported inside the call so a test patching it on its own module is seen; binding it at
# reason: import time would bypass that seam.
# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from meddies_pii.eval_baseline.baseline.adapter import PiiAdapter
from meddies_pii.eval_baseline.baseline.datasets import (
    EVAL_EXPECTED_ROWS,
    EVAL_FIXTURE_SHA256,
    EvalRow,
)
from meddies_pii.eval_baseline.baseline.views import (
    EVALUATION_VIEWS,
    MODEL_CORE_VIEW,
    DualViewPrediction,
    EvaluationView,
)
from meddies_pii.evaluation.identity import (
    DatasetShardIdentity,
    EvaluationContract,
    canonical_json_bytes,
    canonical_sha256,
    dataset_shard_identity,
    dataset_shard_identity_from_result_rows,
    file_sha256,
    is_sha256,
    parse_dataset_shard_identity,
    parse_evaluation_contract,
)
from meddies_pii.evaluation.span_metrics import (
    containment_span_prf_by_label,
    containment_span_report_by_doc,
    exact_span_prf_by_label,
    exact_span_report_by_doc,
)
from meddies_pii.json_types import JsonObject, as_json_object, is_str_list, is_str_mapping
from meddies_pii.jsonl import read_jsonl, write_jsonl
from meddies_pii.spans import CharSpan, char_span_to_dict

if TYPE_CHECKING:
    from meddies_pii.taxonomy import PiiLabel


class VolumeLike(Protocol):
    def commit(self) -> None: ...


class DualViewPiiAdapter(PiiAdapter, Protocol):
    def predict_views(self, texts: list[str]) -> list[DualViewPrediction]: ...


@dataclass(frozen=True, slots=True)
class ShardSpec:
    model: str
    dataset: str
    shard: str
    view: EvaluationView | None = None


@dataclass(frozen=True, slots=True)
class ShardSample:
    doc_id: str
    label: str
    start: int
    end: int
    surface: str
    text_slice: str


@dataclass(frozen=True, slots=True)
class ShardResult:
    spec: ShardSpec
    skipped: bool
    rows_in: int
    spans_out: int
    labels_seen: frozenset[str]
    output_path: Path
    done_path: Path
    samples: tuple[ShardSample, ...] = ()
    elapsed_seconds: float = 0.0
    """Wall-clock of the predict() call ONLY (excludes model load/compile, which is cold-start, not inference).

    0.0 for a skipped shard with a matching fixture.

    """
    environment_fingerprint_sha256: str = ""


@dataclass(frozen=True, slots=True)
class DualViewShardResult:
    """Two separately scoreable shards tied to one document inference pass."""

    by_view: dict[EvaluationView, ShardResult]
    shared_inference_identity: str


@dataclass(frozen=True, slots=True)
class JsonlResultSource:
    """Re-open one persisted JSONL shard for the aggregate's single pass."""

    path: Path
    expected_result_sha256: str

    def __call__(self) -> Iterator[JsonObject]:
        _require_result_file_digest(self.path, self.expected_result_sha256)
        try:
            for row in read_jsonl(self.path):
                json_row = as_json_object(row)
                if json_row is not None:
                    yield json_row
        finally:
            _require_result_file_digest(self.path, self.expected_result_sha256)


@dataclass(frozen=True, slots=True)
class MatrixResults:
    rows_by_dataset: dict[str, JsonlResultSource]
    result_sha256_by_dataset: dict[str, str]
    missing_datasets: tuple[str, ...]
    timing_by_dataset: dict[str, dict[str, float]]
    shard_identities_by_dataset: dict[str, DatasetShardIdentity]


def result_dir(output_root: str | Path, spec: ShardSpec) -> Path:
    root = Path(output_root) / "results" / spec.model
    if spec.view is not None:
        root /= spec.view
    return root / spec.dataset


def shard_output_path(output_root: str | Path, spec: ShardSpec) -> Path:
    return result_dir(output_root, spec) / f"{spec.shard}.jsonl"


def shard_done_path(output_root: str | Path, spec: ShardSpec) -> Path:
    return result_dir(output_root, spec) / f"{spec.shard}.done"


def shard_meta_path(output_root: str | Path, spec: ShardSpec) -> Path:
    return result_dir(output_root, spec) / f"{spec.shard}.meta.json"


# reason: write shard meta exposes path/provenance as its public contract; bundling would break callers.
def write_shard_meta(  # ruff: ignore[too-many-arguments]
    path: str | Path,
    *,
    elapsed_seconds: float,
    rows_in: int,
    spans_out: int,
    fixture_sha256: str,
    result_sha256: str,
    environment_fingerprint: Mapping[str, object],
    evaluation_contract: EvaluationContract,
    dataset_shard_identity: DatasetShardIdentity,
    provenance: Mapping[str, object] | None = None,
) -> None:
    """Persist per-shard inference timing so the (separate) aggregate can read it.

    Timing lives on the Volume in a sidecar (not the row JSONL — it is a shard-level
    metric, not per-row), so a resumed/skipped shard keeps its original measurement.
    """
    payload: dict[str, object] = {
        "elapsed_seconds": round(float(elapsed_seconds), 4),
        "rows_in": int(rows_in),
        "spans_out": int(spans_out),
    }
    payload["fixture_sha256"] = fixture_sha256
    payload["result_sha256"] = result_sha256
    payload["environment_fingerprint"] = dict(environment_fingerprint)
    payload["evaluation_contract"] = evaluation_contract.to_payload()
    payload["evaluation_contract_sha256"] = evaluation_contract.digest
    payload["dataset_shard_identity"] = dataset_shard_identity.to_payload()
    payload["dataset_shard_identity_sha256"] = dataset_shard_identity.digest
    if provenance is not None:
        payload.update(provenance)
    atomic_write_bytes(Path(path), canonical_json_bytes(payload))


def resolved_environment_fingerprint() -> dict[str, object]:
    """Describe the installed evaluation environment, including transitive packages.

    Direct image pins are an input contract. This fingerprint captures what the
    resolver actually installed for a persisted evaluation result.

    Returns:
        The Python version, platform, and every installed distribution pinned as
        ``name==version``, plus the ``sha256`` of that record so two runs can be
        compared without diffing the whole list.

    """
    distributions = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    identity = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "distribution_count": len(distributions),
        "distributions": distributions,
    }
    digest = canonical_sha256(identity)
    return {**identity, "sha256": digest}


def read_shard_timing(
    output_root: str | Path,
    model: str,
    datasets: Sequence[str],
    *,
    view: EvaluationView | None = None,
) -> dict[str, dict[str, float]]:
    """Read each config's ``<shard>.meta.json`` into ``{config: {seconds, rows}}``.

    Configs with no meta (an old run, or a never-run cell) are skipped, so a model
    evaluated before timing existed still aggregates -- it just has no rows/s.

    Returns:
        One entry per config that has a sidecar, holding its ``elapsed_seconds``
        and the ``rows`` it scored. A config with no sidecar is absent rather
        than zero, so a caller can tell 'not timed' from 'took no time'.

    """
    timing: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        spec = ShardSpec(model=model, dataset=dataset, shard="full", view=view)
        meta_path = shard_meta_path(output_root, spec)
        if not meta_path.exists():
            continue
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        timing[dataset] = {
            "elapsed_seconds": float(payload["elapsed_seconds"]),
            "rows": float(payload["rows_in"]),
        }
    return timing


def expected_matrix_shard_identities(
    evaluation_contract: EvaluationContract,
    datasets: Sequence[str],
) -> dict[str, DatasetShardIdentity]:
    """Materialize each pinned source fixture before aggregation accepts a cell.

    Returns:
        The identity each dataset's cell must carry to be read, computed from the
        fixture rows themselves rather than from what a result claims.

    """
    from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell

    return {
        dataset: dataset_shard_identity(
            evaluation_contract,
            dataset=dataset,
            shard="full",
            rows=load_eval_cell(dataset, v2_limit=None, external_limit=None),
        )
        for dataset in datasets
    }


# reason: read matrix exposes output root/view as its public contract; bundling would break callers.
def read_matrix_results(  # ruff: ignore[too-many-arguments]
    output_root: str | Path,
    model: str,
    datasets: Sequence[str],
    *,
    require_done: bool = False,
    expected_rows: Mapping[str, int] = EVAL_EXPECTED_ROWS,
    expected_evaluation_contract: EvaluationContract,
    expected_dataset_shard_identities: Mapping[str, DatasetShardIdentity],
    view: EvaluationView | None = None,
) -> MatrixResults:
    """Load complete shard outputs, report missing cells, and retain their timings.

    A result only participates in an aggregate when its sidecar proves it was
    produced for the expected row count and fixture represented by its output
    rows. Older or malformed sidecars are deliberately reported as missing
    rather than mixed into a benchmark comparison.

    Actual JSONL rows are intentionally not parsed here. The aggregate consumes this source once, compares every row to
    this identity, and only returns a report when that complete comparison succeeds.

    Returns:
        The readable cells as row sources keyed by dataset, their result digests,
        the datasets reported missing, their timings, and the identity each cell
        was accepted against.

    Raises:
        ValueError: when a requested dataset has no frozen row count, or when the
            expected identities do not all carry one evaluation contract - mixing
            contracts in a single read would produce an incomparable aggregate.

    """
    try:
        requested_rows = {dataset: expected_rows[dataset] for dataset in datasets}
    except KeyError as exc:
        msg = f"missing frozen row count for requested dataset {exc.args[0]!r}"
        raise ValueError(msg) from None
    if any(
        identity.evaluation_contract != expected_evaluation_contract
        for identity in expected_dataset_shard_identities.values()
    ):
        msg = "mixed evaluation contracts cannot be read"
        raise ValueError(msg)

    rows_by_dataset: dict[str, JsonlResultSource] = {}
    result_sha256_by_dataset: dict[str, str] = {}
    shard_identities_by_dataset: dict[str, DatasetShardIdentity] = {}
    missing_datasets: list[str] = []
    for dataset in datasets:
        spec = ShardSpec(model=model, dataset=dataset, shard="full", view=view)
        output_path = shard_output_path(output_root, spec)
        done_path = shard_done_path(output_root, spec)
        meta_path = shard_meta_path(output_root, spec)
        if not output_path.exists() or (require_done and not done_path.exists()):
            missing_datasets.append(dataset)
            continue
        metadata = _read_shard_metadata(meta_path)
        expected_identity = expected_dataset_shard_identities.get(dataset)
        if metadata is None or expected_identity is None:
            missing_datasets.append(dataset)
            continue
        if metadata.get("rows_in") != requested_rows[dataset]:
            msg = (
                "frozen matrix row counts changed: "
                f"{dataset} expected {requested_rows[dataset]}, "
                f"sidecar={metadata.get('rows_in')}"
            )
            raise ValueError(msg)
        if not _metadata_matches_identity(
            metadata,
            expected_evaluation_contract,
            expected_identity,
        ):
            missing_datasets.append(dataset)
            continue
        if not _result_file_matches(output_path, metadata.get("result_sha256")):
            missing_datasets.append(dataset)
            continue
        result_sha256 = metadata["result_sha256"]
        if not isinstance(result_sha256, str):
            missing_datasets.append(dataset)
            continue
        rows_by_dataset[dataset] = JsonlResultSource(
            output_path,
            result_sha256,
        )
        result_sha256_by_dataset[dataset] = result_sha256
        shard_identities_by_dataset[dataset] = expected_identity

    return MatrixResults(
        rows_by_dataset=rows_by_dataset,
        result_sha256_by_dataset=result_sha256_by_dataset,
        missing_datasets=tuple(missing_datasets),
        timing_by_dataset=read_shard_timing(output_root, model, tuple(rows_by_dataset), view=view),
        shard_identities_by_dataset=shard_identities_by_dataset,
    )


def assert_frozen_fixture(report: Mapping[str, Any]) -> None:
    fixture = report.get("fixture")
    observed = fixture.get("sha256") if isinstance(fixture, Mapping) else None
    if observed != EVAL_FIXTURE_SHA256:
        msg = f"frozen fixture hash changed: expected {EVAL_FIXTURE_SHA256}, got {observed}"
        raise ValueError(msg)


def filter_gold_to_supported(
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
    supported_labels: frozenset[PiiLabel],
) -> dict[str, list[CharSpan]]:
    return {doc_id: [span for span in spans if span.label in supported_labels] for doc_id, spans in gold_by_doc.items()}


def score_reports(
    predicted_by_doc: Mapping[str, Sequence[CharSpan]],
    gold_by_doc: Mapping[str, Sequence[CharSpan]],
    *,
    supported_labels: frozenset[PiiLabel],
) -> dict[str, Any]:
    supported_gold = filter_gold_to_supported(gold_by_doc, supported_labels)
    return {
        "full_exact": exact_span_report_by_doc(predicted_by_doc, gold_by_doc),
        "supported_exact": exact_span_report_by_doc(predicted_by_doc, supported_gold),
        "full_containment": containment_span_report_by_doc(predicted_by_doc, gold_by_doc),
        "supported_containment": containment_span_report_by_doc(predicted_by_doc, supported_gold),
        "full_exact_by_label": exact_span_prf_by_label(predicted_by_doc, gold_by_doc),
        "supported_exact_by_label": exact_span_prf_by_label(predicted_by_doc, supported_gold),
        "full_containment_by_label": containment_span_prf_by_label(predicted_by_doc, gold_by_doc),
        "supported_containment_by_label": containment_span_prf_by_label(predicted_by_doc, supported_gold),
    }


# reason: run shard exposes adapter/force as its public contract; bundling would break callers.
def run_shard(  # ruff: ignore[too-many-arguments]
    *,
    adapter: PiiAdapter,
    rows: Sequence[EvalRow],
    output_root: str | Path,
    spec: ShardSpec,
    evaluation_contract: EvaluationContract,
    volume: VolumeLike,
    force: bool = False,
) -> ShardResult:
    """Score one shard and commit it behind a digest-bearing sidecar.

    Publish the JSONL first, then publish its digest-bearing sidecar. A crash can therefore expose either the old valid
    pair or a result/sidecar digest mismatch, never a sidecar that authenticates incomplete or older result bytes.

    Returns:
        The shard's paths and digests, with ``skipped`` set when a completed
        artifact already matched this identity and nothing was re-scored.

    Raises:
        RuntimeError: when the adapter returns a different number of predictions
            than rows it was given, which would silently misalign every span.

    """
    shard_identity = dataset_shard_identity(
        evaluation_contract,
        dataset=spec.dataset,
        shard=spec.shard,
        rows=rows,
    )
    output_path = shard_output_path(output_root, spec)
    done_path = shard_done_path(output_root, spec)
    meta_path = shard_meta_path(output_root, spec)
    environment_fingerprint = resolved_environment_fingerprint()
    environment_sha256 = str(environment_fingerprint["sha256"])
    if not force and _completed_fixture_matches(
        output_path,
        done_path,
        meta_path,
        shard_identity,
    ):
        return ShardResult(
            spec=spec,
            skipped=True,
            rows_in=0,
            spans_out=0,
            labels_seen=frozenset(),
            output_path=output_path,
            done_path=done_path,
            environment_fingerprint_sha256=environment_sha256,
        )

    predict_start = time.perf_counter()
    predictions = adapter.predict([row.text for row in rows])
    elapsed_seconds = time.perf_counter() - predict_start
    if len(predictions) != len(rows):
        msg = f"{adapter.name} returned {len(predictions)} predictions for {len(rows)} rows"
        raise RuntimeError(msg)

    records: list[dict[str, object]] = []
    labels_seen: set[str] = set()
    samples: list[ShardSample] = []
    spans_out = 0
    for row, spans in zip(rows, predictions, strict=True):
        validate_spans(row, spans, adapter.supported_labels)
        labels_seen.update(span.label for span in spans)
        spans_out += len(spans)
        if not samples:
            samples.extend(sample_spans(row, spans, limit=3))
        records.append({
            "id": row.text_sha256,
            "doc_id": row.stable_id,
            "pred_spans": [char_span_to_dict(span) for span in spans],
            "gold_spans": [char_span_to_dict(span) for span in row.gold_spans],
            "language": row.language,
            "slice": sorted(row.slices),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomically(output_path, records)
    result_sha256 = file_sha256(str(output_path))
    write_shard_meta(
        meta_path,
        elapsed_seconds=elapsed_seconds,
        rows_in=len(rows),
        spans_out=spans_out,
        fixture_sha256=shard_identity.digest,
        result_sha256=result_sha256,
        environment_fingerprint=environment_fingerprint,
        evaluation_contract=evaluation_contract,
        dataset_shard_identity=shard_identity,
    )
    atomic_write_bytes(done_path, b"done\n")
    volume.commit()
    return ShardResult(
        spec=spec,
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


# reason: run dual view combines write jsonl and check spans; splitting would split cleanup from writes.
def run_dual_view_shards(  # ruff: ignore[complex-structure,too-many-arguments,too-many-locals]
    *,
    adapter: DualViewPiiAdapter,
    rows: Sequence[EvalRow],
    output_root: str | Path,
    specs: Mapping[EvaluationView, ShardSpec],
    evaluation_contracts: Mapping[EvaluationView, EvaluationContract],
    volume: VolumeLike,
    force: bool = False,
) -> DualViewShardResult:
    """Persist named core and hybrid outputs from one adapter ``predict_views`` call.

    Returns:
        Both views' shard results plus the shared inference identity that binds
        them to the single ``predict_views`` call they came from.

    Raises:
        ValueError: when the specs or contracts do not name both views, do not
            score the same model dataset shard, or do not bind each view to its
            own spec - any of which would let two unrelated runs be paired.
        RuntimeError: when the adapter returns a different number of dual-view
            predictions than rows it was given.

    """
    if set(specs) != set(EVALUATION_VIEWS):
        msg = "dual-view evaluation requires both named view specs"
        raise ValueError(msg)
    if set(evaluation_contracts) != set(EVALUATION_VIEWS):
        msg = "dual-view evaluation requires both named view contracts"
        raise ValueError(msg)
    first_spec = specs[MODEL_CORE_VIEW]
    if any(
        (spec.model, spec.dataset, spec.shard) != (first_spec.model, first_spec.dataset, first_spec.shard)
        for spec in specs.values()
    ):
        msg = "dual-view specs must score the same model dataset shard"
        raise ValueError(msg)
    if any(specs[view].view != view for view in EVALUATION_VIEWS):
        msg = "dual-view specs must bind each named view explicitly"
        raise ValueError(msg)

    identities: dict[EvaluationView, DatasetShardIdentity] = {
        view: dataset_shard_identity(
            evaluation_contracts[view],
            dataset=specs[view].dataset,
            shard=specs[view].shard,
            rows=rows,
        )
        for view in EVALUATION_VIEWS
    }
    environment_fingerprint = resolved_environment_fingerprint()
    environment_sha256 = str(environment_fingerprint["sha256"])
    paths: dict[EvaluationView, tuple[Path, Path, Path]] = {
        view: (
            shard_output_path(output_root, specs[view]),
            shard_done_path(output_root, specs[view]),
            shard_meta_path(output_root, specs[view]),
        )
        for view in EVALUATION_VIEWS
    }
    existing_identity = _completed_dual_view_identity(paths, identities) if not force else None
    if existing_identity is not None:
        return DualViewShardResult(
            by_view={
                view: ShardResult(
                    spec=specs[view],
                    skipped=True,
                    rows_in=0,
                    spans_out=0,
                    labels_seen=frozenset(),
                    output_path=paths[view][0],
                    done_path=paths[view][1],
                    environment_fingerprint_sha256=environment_sha256,
                )
                for view in EVALUATION_VIEWS
            },
            shared_inference_identity=existing_identity,
        )

    for view in EVALUATION_VIEWS:
        paths[view][1].unlink(missing_ok=True)

    predict_start = time.perf_counter()
    predicted_views = adapter.predict_views([row.text for row in rows])
    elapsed_seconds = time.perf_counter() - predict_start
    if len(predicted_views) != len(rows):
        msg = f"{adapter.name} returned {len(predicted_views)} dual-view predictions for {len(rows)} rows"
        raise RuntimeError(msg)
    shared_inference_identity = _shared_inference_identity(
        rows=rows,
        predictions=predicted_views,
    )

    results: dict[EvaluationView, ShardResult] = {}
    for view in EVALUATION_VIEWS:
        records: list[dict[str, object]] = []
        labels_seen: set[str] = set()
        samples: list[ShardSample] = []
        spans_out = 0
        for row, prediction in zip(rows, predicted_views, strict=True):
            spans = prediction.for_view(view)
            validate_spans(row, spans, adapter.supported_labels)
            labels_seen.update(span.label for span in spans)
            spans_out += len(spans)
            if not samples:
                samples.extend(sample_spans(row, spans, limit=3))
            records.append({
                "id": row.text_sha256,
                "doc_id": row.stable_id,
                "pred_spans": [char_span_to_dict(span) for span in spans],
                "gold_spans": [char_span_to_dict(span) for span in row.gold_spans],
                "language": row.language,
                "slice": sorted(row.slices),
                "evaluation_view": view,
                "shared_inference_identity": shared_inference_identity,
            })
        output_path, done_path, meta_path = paths[view]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomically(output_path, records)
        result_sha256 = file_sha256(str(output_path))
        write_shard_meta(
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
                "failure_state": None,
            },
        )
        results[view] = ShardResult(
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
    for view in EVALUATION_VIEWS:
        atomic_write_bytes(paths[view][1], b"done\n")
    volume.commit()
    return DualViewShardResult(
        by_view=results,
        shared_inference_identity=shared_inference_identity,
    )


def _completed_dual_view_identity(
    paths: Mapping[EvaluationView, tuple[Path, Path, Path]],
    identities: Mapping[EvaluationView, DatasetShardIdentity],
) -> str | None:
    return completed_paired_view_identity(paths, identities, EVALUATION_VIEWS)


@dataclass(frozen=True, slots=True)
class PairedViewExpectation:
    regex_manifest_sha256: str | None = None
    comparison_id: str | None = None
    corpus_manifest_sha256: str | None = None
    corpus_manifest_payload: Mapping[str, object] | None = None


# reason: each early return rejects one incomplete persisted-evidence condition before reuse is authorized.
def completed_paired_view_identity(  # ruff: ignore[too-many-return-statements]
    paths: Mapping[EvaluationView, tuple[Path, Path, Path]],
    identities: Mapping[EvaluationView, DatasetShardIdentity],
    views: Sequence[EvaluationView],
    *,
    expectation: PairedViewExpectation | None = None,
) -> str | None:
    expectation = expectation or PairedViewExpectation()
    if not all(
        _completed_fixture_matches(paths[view][0], paths[view][1], paths[view][2], identities[view]) for view in views
    ):
        return None
    shared_identities: set[str] = set()
    for view in views:
        metadata = _read_shard_metadata(paths[view][2])
        if metadata is None or metadata.get("evaluation_view") != view:
            return None
        if (
            expectation.regex_manifest_sha256 is not None
            and metadata.get("regex_manifest_sha256") != expectation.regex_manifest_sha256
        ):
            return None
        if expectation.comparison_id is not None and metadata.get("comparison_id") != expectation.comparison_id:
            return None
        if (
            expectation.corpus_manifest_sha256 is not None
            and metadata.get("corpus_manifest_sha256") != expectation.corpus_manifest_sha256
        ):
            return None
        if (
            expectation.corpus_manifest_payload is not None
            and metadata.get("corpus_manifest") != expectation.corpus_manifest_payload
        ):
            return None
        shared_identity = metadata.get("shared_inference_identity")
        if not is_sha256(shared_identity):
            return None
        shared_identities.add(shared_identity)
    if len(shared_identities) != 1:
        return None
    observed_identity = shared_identities.pop()
    persisted_identity = _persisted_paired_inference_identity(
        paths,
        views=views,
        expected_identity=observed_identity,
        expectation=expectation,
    )
    return observed_identity if persisted_identity == observed_identity else None


def _shared_inference_identity(
    *,
    rows: Sequence[EvalRow],
    predictions: Sequence[DualViewPrediction],
) -> str:
    """Bind one paired inference's collision-safe inputs and both mapped outputs.

    Returns:
        One digest over every row's stable id, text digest, language, slices, and
        both views' spans, so a later read can prove the two views came from the
        same inference rather than from two runs that happen to agree.

    """
    return shared_inference_identity_from_rows(
        {
            "stable_id": row.stable_id,
            "text_sha256": row.text_sha256,
            "language": row.language,
            "slices": sorted(row.slices),
            "outputs": {
                view: [{"start": span.start, "end": span.end, "label": span.label} for span in prediction.for_view(view)]
                for view in EVALUATION_VIEWS
            },
        }
        for row, prediction in zip(rows, predictions, strict=True)
    )


# reason: paired artifact rows must pass every schema and provenance check before they contribute to one identity digest.
def _persisted_paired_inference_identity(  # ruff: ignore[complex-structure]
    paths: Mapping[EvaluationView, tuple[Path, Path, Path]],
    *,
    views: Sequence[EvaluationView],
    expected_identity: str,
    expectation: PairedViewExpectation,
) -> str | None:
    if not views or views[0] != MODEL_CORE_VIEW:
        msg = "paired evaluation must begin with model_core"
        raise ValueError(msg)
    rows_by_view = {view: _strict_result_rows(paths[view][0]) for view in views}

    def paired_rows() -> Iterator[dict[str, object]]:
        for paired in zip_longest(*(rows_by_view[view] for view in views)):
            if any(row is None for row in paired):
                msg = "paired result artifacts have different row counts"
                raise ValueError(msg)
            present = tuple(row for row in paired if row is not None)
            core = present[0]
            identity = (
                core.get("id"),
                core.get("doc_id"),
                core.get("language"),
                core.get("slice"),
            )
            if any(
                (
                    row.get("id"),
                    row.get("doc_id"),
                    row.get("language"),
                    row.get("slice"),
                )
                != identity
                for row in present[1:]
            ):
                msg = "paired result artifacts have mismatched row identity"
                raise ValueError(msg)
            if any(
                row.get("evaluation_view") != view or row.get("shared_inference_identity") != expected_identity
                for view, row in zip(views, present, strict=True)
            ):
                msg_0 = "paired result artifacts have mismatched provenance"
                raise ValueError(msg_0)
            if expectation.regex_manifest_sha256 is not None and any(
                row.get("regex_manifest_sha256") != expectation.regex_manifest_sha256 for row in present
            ):
                msg_0 = "paired result artifacts have mismatched regex manifests"
                raise ValueError(msg_0)
            if expectation.comparison_id is not None and any(
                row.get("comparison_id") != expectation.comparison_id for row in present
            ):
                msg_0 = "paired result artifacts have mismatched comparison IDs"
                raise ValueError(msg_0)
            if expectation.corpus_manifest_sha256 is not None and any(
                row.get("corpus_manifest_sha256") != expectation.corpus_manifest_sha256 for row in present
            ):
                msg_0 = "paired result artifacts have mismatched corpus manifests"
                raise ValueError(msg_0)
            yield {
                "stable_id": _result_string(core, "doc_id"),
                "text_sha256": _result_string(core, "id"),
                "language": _result_string(core, "language"),
                "slices": _result_string_list(core, "slice"),
                "outputs": {
                    view: _result_span_identity(row, "pred_spans") for view, row in zip(views, present, strict=True)
                },
            }

    try:
        return shared_inference_identity_from_rows(paired_rows())
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def shared_inference_identity_from_rows(rows: Iterator[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"schema_version": 1}))
    for row in rows:
        digest.update(canonical_json_bytes(row))
    return digest.hexdigest()


def _strict_result_rows(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not is_str_mapping(value):
                msg = "result artifact rows must be JSON objects"
                raise ValueError(msg)
            yield dict(value)


def _result_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        msg = f"result row is missing {key}"
        raise ValueError(msg)
    return value


def _result_string_list(row: Mapping[str, object], key: str) -> list[str]:
    value = row.get(key)
    if not is_str_list(value):
        msg = f"result row is missing {key}"
        raise ValueError(msg)
    return value


def _result_span_identity(row: Mapping[str, object], key: str) -> list[dict[str, object]]:
    value = row.get(key)
    if not isinstance(value, list):
        msg = f"result row is missing {key}"
        raise ValueError(msg)
    spans: list[dict[str, object]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            msg = "result span is malformed"
            raise ValueError(msg)
        start, end, label = raw.get("start"), raw.get("end"), raw.get("label")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not isinstance(label, str)
        ):
            msg = "result span is malformed"
            raise ValueError(msg)
        spans.append({"start": start, "end": end, "label": label})
    return spans


def shard_result_json(result: ShardResult) -> str:
    payload = {
        **asdict(result),
        "labels_seen": sorted(result.labels_seen),
        "output_path": str(result.output_path),
        "done_path": str(result.done_path),
    }
    return json.dumps(payload, ensure_ascii=False)


def _completed_fixture_matches(
    output_path: Path,
    done_path: Path,
    meta_path: Path,
    expected_identity: DatasetShardIdentity,
) -> bool:
    if not output_path.exists() or not done_path.exists() or not meta_path.exists():
        return False
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping) or not _metadata_matches_identity(
        payload,
        expected_identity.evaluation_contract,
        expected_identity,
    ):
        return False
    if not _result_file_matches(output_path, payload.get("result_sha256")):
        return False
    try:
        persisted_identity = dataset_shard_identity_from_result_rows(
            expected_identity.evaluation_contract,
            dataset=expected_identity.dataset,
            shard=expected_identity.shard,
            rows=list(read_jsonl(output_path)),
        )
    except ValueError:
        return False
    return persisted_identity == expected_identity


def _metadata_matches_identity(
    metadata: Mapping[str, object],
    expected_contract: EvaluationContract,
    expected_identity: DatasetShardIdentity,
) -> bool:
    try:
        observed_contract = parse_evaluation_contract(metadata.get("evaluation_contract"))
        observed = parse_dataset_shard_identity(metadata.get("dataset_shard_identity"))
    except ValueError:
        return False
    return (
        observed_contract == expected_contract
        and metadata.get("evaluation_contract_sha256") == expected_contract.digest
        and metadata.get("dataset_shard_identity_sha256") == expected_identity.digest
        and metadata.get("fixture_sha256") == expected_identity.digest
        and observed == expected_identity
        and observed.evaluation_contract == expected_contract
    )


def _read_shard_metadata(meta_path: Path) -> Mapping[str, object] | None:
    try:
        payload: object = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if is_str_mapping(payload) else None


def _result_file_matches(path: Path, expected_sha256: object) -> bool:
    if not is_sha256(expected_sha256):
        return False
    try:
        return file_sha256(str(path)) == expected_sha256
    except (OSError, RuntimeError):
        return False


def _require_result_file_digest(path: Path, expected_sha256: str) -> None:
    if not _result_file_matches(path, expected_sha256):
        msg = f"result artifact digest mismatch: {path}"
        raise ValueError(msg)


def write_jsonl_atomically(path: Path, records: Sequence[dict[str, object]]) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        write_jsonl(temporary_path, records)
        _fsync_file(temporary_path)
        Path(temporary_path).replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        with temporary_path.open("wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        Path(temporary_path).replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _temporary_sibling(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary_name)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_spans(
    row: EvalRow,
    spans: Sequence[CharSpan],
    supported_labels: frozenset[PiiLabel],
) -> None:
    for span in spans:
        if span.label not in supported_labels:
            msg = f"adapter emitted unsupported label {span.label!r} for {row.stable_id}"
            raise RuntimeError(msg)
        if span.start < 0 or span.end <= span.start or span.end > len(row.text):
            msg = f"adapter emitted invalid bounds {span!r}"
            raise RuntimeError(msg)
        if row.text[span.start : span.end] != span.text:
            msg = f"adapter emitted misaligned span for {row.stable_id}: {span!r}"
            raise RuntimeError(msg)


def sample_spans(row: EvalRow, spans: Sequence[CharSpan], *, limit: int) -> list[ShardSample]:
    return [
        ShardSample(
            doc_id=row.stable_id,
            label=span.label,
            start=span.start,
            end=span.end,
            surface=span.text,
            text_slice=row.text[span.start : span.end],
        )
        for span in spans[:limit]
    ]
