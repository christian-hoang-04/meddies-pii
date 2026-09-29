"""Offline child execution and evidence serialization for the regex release gate."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, cast

from meddies_pii.bioes_inference.artifacts import PUBLIC_Q8_ARTIFACT_SPEC
from meddies_pii.bioes_inference.contracts import ExpectedFileIdentity
from meddies_pii.eval_baseline.baseline.run import ShardSpec
from meddies_pii.eval_baseline.baseline.views import MODEL_CORE_VIEW, MODEL_PLUS_REGEX_VIEW
from meddies_pii.eval_baseline.regex_release.paired_evaluation import (
    paired_regex_document_counts,
    run_regex_view_shards,
)
from meddies_pii.eval_baseline.regex_release.regex_bootstrap import (
    LengthBucket,
    PairedDocumentCounts,
    length_bucket,
    source_row_sha256,
)
from meddies_pii.eval_baseline.regex_release.regex_ignore_list import count_ignored_false_positives
from meddies_pii.eval_baseline.regex_release.regex_release_contract import (
    ReleaseGateContract,
    require_paid_run_receipts,
)
from meddies_pii.eval_baseline.regex_release.regex_release_gate import (
    ChildRegexEvidence,
    LabelMetricCounts,
    ProjectedInferenceAdapter,
)
from meddies_pii.eval_baseline.regex_release.regex_report import RegexCorpusManifest
from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    EvaluationContract,
    canonical_sha256,
    dataset_shard_identity,
    payload_artifact,
)
from meddies_pii.evaluation.span_metrics import exact_span_counts
from meddies_pii.spans import char_span_from_value
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from meddies_pii.eval_baseline.baseline.run import VolumeLike
    from meddies_pii.eval_baseline.baseline.views import EvaluationView
    from meddies_pii.eval_baseline.regex_release.regex_ignore_list import RegexIgnoreList

CACHE_MOUNT = "/cache"
BASE_CPU_RATE_USD_PER_SECOND = 0.00014032
"""Measured all-in Modal rate for the original 8-core / 16 GiB envelope; it reproduces the 2026-08-09 live projection
exactly (16,399.921s * this = $2.301237). Modal bills cores and memory linearly, and our envelope scales memory with
cores (8/16 GiB -> 32/64 GiB), so one core ratio carries both.
"""
BASE_CPU_CORES = 8
BASE_MEMORY_MIB = 16 * 1024
PROJECTION_RESERVE_SECONDS = 300.0


def cost_rate_usd_per_second(cpu_cores: int, memory_mib: int) -> float:
    if memory_mib * BASE_CPU_CORES != BASE_MEMORY_MIB * cpu_cores:
        msg = "cost rate assumes memory scales with cores"
        raise ValueError(msg)
    return BASE_CPU_RATE_USD_PER_SECOND * (cpu_cores / BASE_CPU_CORES)


# reason: child execution retains dataset, receipt, corpus, paired-view, and aggregate identities in one transaction.
def execute_child(  # ruff: ignore[too-many-locals]
    contract: ReleaseGateContract,
    config: str,
    *,
    volume: VolumeLike,
    ignore_list: RegexIgnoreList | None = None,
) -> ChildRegexEvidence:
    # reason: tests patch the dataset module before this call to prove digest mismatch refuses
    # reason: before row loading; binding the function at import time would bypass that seam.
    from meddies_pii.eval_baseline.baseline.datasets import load_v2_eval_rows  # ruff: ignore[import-outside-top-level]

    if contract.ignore_list_sha256 is not None and (
        ignore_list is None or ignore_list.sha256 != contract.ignore_list_sha256
    ):
        msg = "ignore-list digest does not match contract; refusing before inference"
        raise ValueError(msg)

    rows = load_v2_eval_rows(config, revision=contract.dataset_revision)
    if len(rows) != contract.dataset_rows[config]:
        msg = "offline dataset row count does not match contract"
        raise RuntimeError(msg)
    contracts = evaluation_contracts(contract)
    source_hashes = tuple(
        source_row_sha256(
            stable_id=row.stable_id,
            text_sha256=row.text_sha256,
            gold_spans=row.gold_spans,
            language=row.language,
            slices=tuple(sorted(row.slices)),
        )
        for row in rows
    )
    fixture_sha256 = dataset_shard_identity(
        contracts[MODEL_CORE_VIEW],
        dataset=contract.dataset_repo,
        shard=config,
        rows=rows,
    ).fixture_identity.digest
    hydration = require_paid_run_receipts(contract.output_root, contract)
    hydrated_fixtures = hydration.get("dataset_fixture_sha256s")
    if not isinstance(hydrated_fixtures, dict) or hydrated_fixtures.get(config) != fixture_sha256:
        msg = "offline dataset fixture does not match hydration receipt"
        raise RuntimeError(msg)
    manifest = RegexCorpusManifest.create(
        corpus_id=f"v2-{config}",
        dataset=contract.dataset_repo,
        dataset_revision=contract.dataset_revision,
        shard=config,
        fixture_sha256=fixture_sha256,
        source_row_sha256s=source_hashes,
        historically_exposed=True,
        blinded=False,
        preregistered_strata=tuple(sorted({(row.language, length_bucket(len(row.text))) for row in rows})),
    )
    adapter = _build_adapter(contract)
    adapter.load()
    specs: dict[EvaluationView, ShardSpec] = {
        view: ShardSpec("meddies-pii-v2-q8", contract.dataset_repo, config, view)
        for view in (MODEL_CORE_VIEW, MODEL_PLUS_REGEX_VIEW)
    }
    result = run_regex_view_shards(
        adapter=adapter,
        rows=rows,
        output_root=contract.output_root,
        specs=specs,
        evaluation_contracts=contracts,
        volume=volume,
        comparison_id=contract.comparison_id,
        corpus_manifest=manifest,
    )
    model_rows = _read_jsonl(result.by_view[MODEL_CORE_VIEW].output_path)
    regex_rows = _read_jsonl(result.by_view[MODEL_PLUS_REGEX_VIEW].output_path)
    documents = paired_regex_document_counts(model_rows, regex_rows, ignore_list=ignore_list, split=config)
    labels = _label_counts(model_rows, regex_rows, ignore_list=ignore_list, split=config)
    evaluation_contract_sha256 = canonical_sha256({view: item.digest for view, item in contracts.items()})
    runtime_sha256 = contracts[MODEL_CORE_VIEW].resolved_runtime_environment.sha256
    ignore_list_sha256 = ignore_list.sha256 if ignore_list is not None else None
    evidence_sha256 = _child_evidence_sha256(
        config=config,
        manifest_sha256=manifest.sha256,
        documents=documents,
        labels=labels,
        contract_sha256=contract.sha256,
        evaluation_contract_sha256=evaluation_contract_sha256,
        runtime_sha256=runtime_sha256,
        ignore_list_sha256=ignore_list_sha256,
    )
    return ChildRegexEvidence(
        config=config,
        corpus_manifest=manifest,
        documents=documents,
        label_counts=labels,
        contract_sha256=contract.sha256,
        evaluation_contract_sha256=evaluation_contract_sha256,
        ignore_list_sha256=ignore_list_sha256,
        runtime_sha256=runtime_sha256,
        evidence_sha256=evidence_sha256,
    )


def _build_adapter(contract: ReleaseGateContract) -> ProjectedInferenceAdapter:
    # reason: tests replace the inference backend classes before constructing the adapter, and
    # reason: production loads this ONNX stack only inside the remote execution path.
    from meddies_pii.bioes_inference.detector import (  # ruff: ignore[import-outside-top-level]
        BioesSpanDetector,
        OnnxRuntimeBackend,
    )

    root = (
        Path(CACHE_MOUNT)
        / "regex-release-artifacts"
        / PUBLIC_Q8_ARTIFACT_SPEC.artifact_id
        / PUBLIC_Q8_ARTIFACT_SPEC.revision
    )
    model = next(file for file in PUBLIC_Q8_ARTIFACT_SPEC.files if file.role == "model")
    detector = BioesSpanDetector(
        model_path=root / "files" / model.relative_path,
        tokenizer_path=root / "tokenizer",
        expected_model_size_bytes=model.size_bytes,
        expected_model_sha256=model.sha256,
        expected_tokenizer_files={
            file.relative_path: ExpectedFileIdentity(file.size_bytes, file.sha256)
            for file in PUBLIC_Q8_ARTIFACT_SPEC.files
            if file.role == "tokenizer"
        },
        backend=OnnxRuntimeBackend(threads=contract.cpu_cores),
    )
    return ProjectedInferenceAdapter(
        detector=detector,
        supported_labels=PII_LABEL_SET,
        sample_rows=contract.projection_sample_rows,
        projected_total_rows=sum(contract.dataset_rows.values()),
        reserve_seconds=PROJECTION_RESERVE_SECONDS,
        timeout_seconds=contract.timeout_seconds,
        cost_rate_usd_per_second=cost_rate_usd_per_second(contract.cpu_cores, contract.memory_mib),
        cost_ceiling_usd=contract.projected_cost_ceiling_usd,
    )


def evaluation_contracts(
    contract: ReleaseGateContract,
) -> dict[EvaluationView, EvaluationContract]:
    model_pin = contract.model_artifacts[0]
    model = ArtifactIdentity(
        f"{contract.model_repo}/{model_pin.path}",
        contract.model_revision,
        model_pin.sha256,
    )
    runtime = payload_artifact(
        "modal-ort-runtime",
        {
            "image": contract.image_digest,
            "packages": list(contract.runtime_packages),
            "cpu": contract.cpu_cores,
            "memory_mib": contract.memory_mib,
        },
    )
    vendor = payload_artifact(
        "bioes-ort",
        {"backend": "onnxruntime-cpu", "threads": contract.cpu_cores},
    )
    adapter = payload_artifact("regex-release-adapter", {"source_commit": contract.source_commit})
    scorer = payload_artifact("exact-character-span", {"version": 1})
    result_schema = (
        "id",
        "doc_id",
        "pred_spans",
        "gold_spans",
        "language",
        "slice",
        "length_bucket",
        "document_chars",
        "evaluation_view",
        "shared_inference_identity",
        "regex_manifest_sha256",
        "comparison_id",
        "source_row_sha256",
        "corpus_manifest_sha256",
    )
    return {
        view: EvaluationContract(
            model=model,
            vendor_inference_source=vendor,
            local_adapter_source=adapter,
            applied_label_prediction_contract=payload_artifact(
                f"labels-{view}",
                {"labels": sorted(PII_LABEL_SET), "view": view},
            ),
            decoder_contract=payload_artifact(
                f"decoder-{view}",
                {
                    "bioes": 1,
                    "regex_manifest_sha256": contract.regex_manifest_sha256 if view == MODEL_PLUS_REGEX_VIEW else None,
                },
            ),
            resolved_runtime_environment=runtime,
            scorer_contract=scorer,
            supported_labels=tuple(sorted(PII_LABEL_SET)),
            result_schema=result_schema,
        )
        for view in (MODEL_CORE_VIEW, MODEL_PLUS_REGEX_VIEW)
    }


def _label_counts(
    model_rows: list[dict[str, object]],
    regex_rows: list[dict[str, object]],
    *,
    ignore_list: RegexIgnoreList | None = None,
    split: str | None = None,
) -> tuple[LabelMetricCounts, ...]:
    if ignore_list is not None and split is None:
        msg = "label counting needs split when ignore_list is set"
        raise ValueError(msg)
    totals: dict[str, list[int]] = defaultdict(lambda: [0] * 6)
    for model_row, regex_row in zip(model_rows, regex_rows, strict=True):
        uid = cast("str", model_row["doc_id"])
        gold = [span for value in cast("list[object]", model_row["gold_spans"]) if (span := char_span_from_value(value))]
        model = [span for value in cast("list[object]", model_row["pred_spans"]) if (span := char_span_from_value(value))]
        regex = [span for value in cast("list[object]", regex_row["pred_spans"]) if (span := char_span_from_value(value))]
        for label in {span.label for span in (*gold, *model, *regex)}:
            label_model = [span for span in model if span.label == label]
            label_regex = [span for span in regex if span.label == label]
            label_gold = [span for span in gold if span.label == label]
            model_counts = exact_span_counts(label_model, label_gold)
            regex_counts = exact_span_counts(label_regex, label_gold)
            model_fp_ignored = count_ignored_false_positives(
                label_model,
                label_gold,
                ignore_list,
                split=split or "",
                uid=uid,
            )
            regex_fp_ignored = count_ignored_false_positives(
                label_regex,
                label_gold,
                ignore_list,
                split=split or "",
                uid=uid,
            )
            values = (
                model_counts["tp"],
                model_counts["pred_total"] - model_counts["tp"] - model_fp_ignored,
                model_counts["gold_total"] - model_counts["tp"],
                regex_counts["tp"],
                regex_counts["pred_total"] - regex_counts["tp"] - regex_fp_ignored,
                regex_counts["gold_total"] - regex_counts["tp"],
            )
            totals[label] = [left + right for left, right in zip(totals[label], values, strict=True)]
    return tuple(LabelMetricCounts(label, *totals[label]) for label in sorted(totals))


def child_to_payload(child: ChildRegexEvidence) -> dict[str, object]:
    return {
        "schema_version": 1,
        "config": child.config,
        "corpus_manifest": child.corpus_manifest.to_payload(),
        "corpus_manifest_sha256": child.corpus_manifest.sha256,
        "documents": [_document_to_payload(item) for item in child.documents],
        "label_counts": [asdict(item) for item in child.label_counts],
        "contract_sha256": child.contract_sha256,
        "evaluation_contract_sha256": child.evaluation_contract_sha256,
        "runtime_sha256": child.runtime_sha256,
        "evidence_sha256": child.evidence_sha256,
        "ignore_list_sha256": child.ignore_list_sha256,
    }


def child_from_payload(payload: dict[str, object]) -> ChildRegexEvidence:
    document_payloads = [_object_payload(item) for item in cast("list[object]", payload["documents"])]
    label_payloads = [_object_payload(item) for item in cast("list[object]", payload["label_counts"])]
    manifest = RegexCorpusManifest.from_payload(
        cast("dict[str, object]", payload["corpus_manifest"]),
        expected_sha256=cast("str", payload["corpus_manifest_sha256"]),
    )
    child = ChildRegexEvidence(
        config=cast("str", payload["config"]),
        corpus_manifest=manifest,
        documents=tuple(_document_from_payload(document) for document in document_payloads),
        label_counts=tuple(_label_from_payload(label) for label in label_payloads),
        contract_sha256=cast("str", payload["contract_sha256"]),
        evaluation_contract_sha256=cast("str", payload["evaluation_contract_sha256"]),
        runtime_sha256=cast("str", payload["runtime_sha256"]),
        evidence_sha256=cast("str", payload["evidence_sha256"]),
        ignore_list_sha256=cast("str | None", payload.get("ignore_list_sha256")),
    )
    expected_evidence = _child_payload_evidence_sha256(
        config=child.config,
        manifest_sha256=child.corpus_manifest.sha256,
        documents=document_payloads,
        labels=label_payloads,
        contract_sha256=child.contract_sha256,
        evaluation_contract_sha256=child.evaluation_contract_sha256,
        runtime_sha256=child.runtime_sha256,
        ignore_list_sha256=child.ignore_list_sha256,
    )
    if child.evidence_sha256 != expected_evidence:
        msg = "child evidence payload does not match its SHA-256"
        raise ValueError(msg)
    return child


# reason: the keyword-only signature mirrors every hashed evidence field so omissions remain visible at the call site.
def _child_evidence_sha256(  # ruff: ignore[too-many-arguments]
    *,
    config: str,
    manifest_sha256: str,
    documents: tuple[PairedDocumentCounts, ...],
    labels: tuple[LabelMetricCounts, ...],
    contract_sha256: str,
    evaluation_contract_sha256: str,
    runtime_sha256: str,
    ignore_list_sha256: str | None = None,
) -> str:
    return _child_payload_evidence_sha256(
        config=config,
        manifest_sha256=manifest_sha256,
        documents=[_document_to_payload(document) for document in documents],
        labels=[asdict(label) for label in labels],
        contract_sha256=contract_sha256,
        evaluation_contract_sha256=evaluation_contract_sha256,
        runtime_sha256=runtime_sha256,
        ignore_list_sha256=ignore_list_sha256,
    )


# reason: the keyword-only signature mirrors every persisted hash field; a wrapper could silently omit provenance.
def _child_payload_evidence_sha256(  # ruff: ignore[too-many-arguments]
    *,
    config: str,
    manifest_sha256: str,
    documents: list[dict[str, object]],
    labels: list[dict[str, object]],
    contract_sha256: str,
    evaluation_contract_sha256: str,
    runtime_sha256: str,
    ignore_list_sha256: str | None = None,
) -> str:
    return canonical_sha256({
        "config": config,
        "manifest": manifest_sha256,
        "documents": documents,
        "labels": labels,
        "contract_sha256": contract_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "runtime_sha256": runtime_sha256,
        "ignore_list_sha256": ignore_list_sha256,
    })


def _document_from_payload(payload: dict[str, object]) -> PairedDocumentCounts:
    """Reconstruct paired document counts from a persisted payload.

    Over-redacted counts are only meaningful, and only persisted, alongside a known document_chars; a row without it
    predates the schema addition and reads as 0, matching PairedDocumentCounts' own default and validation.

    Returns:
        The reconstructed paired-document count record.

    Raises:
        ValueError: If the persisted document payload is missing fields or contains invalid values.

    """
    bucket = _required_length_bucket(payload)
    document_chars = _optional_int(payload, "document_chars")
    if document_chars is None and ("model_overredacted_chars" in payload or "regex_overredacted_chars" in payload):
        msg = "over-redacted characters require document_chars"
        raise ValueError(msg)
    model_overredacted_chars = _required_int(payload, "model_overredacted_chars") if document_chars is not None else 0
    regex_overredacted_chars = _required_int(payload, "regex_overredacted_chars") if document_chars is not None else 0
    return PairedDocumentCounts(
        document_id=_required_str(payload, "document_id"),
        language=_required_str(payload, "language"),
        length_bucket=bucket,
        source_row_sha256=_required_str(payload, "source_row_sha256"),
        corpus_manifest_sha256=_required_str(payload, "corpus_manifest_sha256"),
        model_tp=_required_int(payload, "model_tp"),
        model_fp=_required_int(payload, "model_fp"),
        model_fn=_required_int(payload, "model_fn"),
        regex_tp=_required_int(payload, "regex_tp"),
        regex_fp=_required_int(payload, "regex_fp"),
        regex_fn=_required_int(payload, "regex_fn"),
        model_fp_ignored=_required_int(payload, "model_fp_ignored"),
        regex_fp_ignored=_required_int(payload, "regex_fp_ignored"),
        gold_pii_chars=_required_int(payload, "gold_pii_chars"),
        model_leaked_chars=_required_int(payload, "model_leaked_chars"),
        regex_leaked_chars=_required_int(payload, "regex_leaked_chars"),
        document_chars=document_chars,
        model_overredacted_chars=model_overredacted_chars,
        regex_overredacted_chars=regex_overredacted_chars,
    )


def _required_length_bucket(payload: dict[str, object]) -> LengthBucket:
    bucket = _required_str(payload, "length_bucket")
    if _is_length_bucket(bucket):
        return bucket
    msg = "child document length bucket is malformed"
    raise ValueError(msg)


def _is_length_bucket(value: str) -> TypeGuard[LengthBucket]:
    return value in {"short", "medium", "long"}


def _document_to_payload(document: PairedDocumentCounts) -> dict[str, object]:
    payload: dict[str, object] = {
        "document_id": document.document_id,
        "language": document.language,
        "length_bucket": document.length_bucket,
        "source_row_sha256": document.source_row_sha256,
        "corpus_manifest_sha256": document.corpus_manifest_sha256,
        "model_tp": document.model_tp,
        "model_fp": document.model_fp,
        "model_fn": document.model_fn,
        "regex_tp": document.regex_tp,
        "regex_fp": document.regex_fp,
        "regex_fn": document.regex_fn,
        "model_fp_ignored": document.model_fp_ignored,
        "regex_fp_ignored": document.regex_fp_ignored,
        "gold_pii_chars": document.gold_pii_chars,
        "model_leaked_chars": document.model_leaked_chars,
        "regex_leaked_chars": document.regex_leaked_chars,
        "document_chars": document.document_chars,
        "model_overredacted_chars": document.model_overredacted_chars,
        "regex_overredacted_chars": document.regex_overredacted_chars,
    }
    if document.document_chars is None:
        del payload["document_chars"]
        del payload["model_overredacted_chars"]
        del payload["regex_overredacted_chars"]
    return payload


def _label_from_payload(payload: dict[str, object]) -> LabelMetricCounts:
    return LabelMetricCounts(
        label=_required_str(payload, "label"),
        model_tp=_required_int(payload, "model_tp"),
        model_fp=_required_int(payload, "model_fp"),
        model_fn=_required_int(payload, "model_fn"),
        regex_tp=_required_int(payload, "regex_tp"),
        regex_fp=_required_int(payload, "regex_fp"),
        regex_fn=_required_int(payload, "regex_fn"),
    )


def _object_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = "child evidence row is malformed"
        raise ValueError(msg)
    return cast("dict[str, object]", value)


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"child evidence {key} is malformed"
        # reason: every malformed persisted child-evidence field shares the reader's ValueError schema contract.
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    return value


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"child evidence {key} is malformed"
        # reason: every malformed persisted child-evidence field shares the reader's ValueError schema contract.
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    return value


def _optional_int(payload: dict[str, object], key: str) -> int | None:
    """Read a possibly-absent int field, strict whenever it is present.

    Absence means the row predates this field's schema addition. A
    present-but-wrong type is always malformed, never treated as absence.

    Returns:
        The integer value, or `None` when the field is absent.

    Raises:
        ValueError: If a present field is not an integer.

    """
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"child evidence {key} is malformed"
        # reason: every malformed persisted child-evidence field shares the reader's ValueError schema contract.
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [cast("dict[str, object]", json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
