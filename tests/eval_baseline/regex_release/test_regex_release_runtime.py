from __future__ import annotations

# ruff: file-ignore[no-self-use]
# reason: the stateless volume double retains Modal's bound commit method shape.
# ruff: file-ignore[float-equality-comparison]
# reason: exact equality pins the literal base CPU rate and its identical base-envelope route;
# reason: derived scaled costs below use pytest.approx instead.
# ruff: file-ignore[docstring-missing-returns]
# reason: private test-fixture helpers are readable at their call sites; boilerplate return sections add no contract value.
import hashlib
from dataclasses import asdict, replace

import pytest

import meddies_pii.bioes_inference.detector as inference
from meddies_pii.eval_baseline.baseline.datasets import V2_DATASET_REVISION, V2_REPO_ID
from meddies_pii.eval_baseline.baseline.views import MODEL_CORE_VIEW
from meddies_pii.eval_baseline.regex_release.regex_bootstrap import PairedDocumentCounts
from meddies_pii.eval_baseline.regex_release.regex_ignore_list import ignore_list_from_rows
from meddies_pii.eval_baseline.regex_release.regex_release_contract import ReleaseGateContract
from meddies_pii.eval_baseline.regex_release.regex_release_gate import (
    ChildRegexEvidence,
    LabelMetricCounts,
)
from meddies_pii.eval_baseline.regex_release.regex_release_runtime import (
    BASE_CPU_RATE_USD_PER_SECOND,
    _build_adapter,
    _child_evidence_sha256,
    _document_from_payload,
    child_from_payload,
    child_to_payload,
    cost_rate_usd_per_second,
    evaluation_contracts,
    execute_child,
)
from meddies_pii.eval_baseline.regex_release.regex_report import RegexCorpusManifest
from meddies_pii.evaluation.identity import canonical_sha256, payload_artifact


def _child_with_ignored_false_positives() -> ChildRegexEvidence:
    """Build child evidence containing ignored false positives.

    evidence_sha256 binds every field above, including the ignored counts; child_from_payload recomputes and refuses a
    mismatch, so the fixture must carry the digest its own payload would actually produce.
    """
    source_hash = hashlib.sha256(b"eval:0").hexdigest()
    manifest = RegexCorpusManifest.create(
        corpus_id="v2-eval",
        dataset=V2_REPO_ID,
        dataset_revision=V2_DATASET_REVISION,
        shard="eval",
        fixture_sha256=hashlib.sha256(b"fixture:eval").hexdigest(),
        source_row_sha256s=(source_hash,),
        historically_exposed=True,
        blinded=False,
        preregistered_strata=(("vi", "short"),),
    )
    documents = (
        PairedDocumentCounts(
            document_id="eval-0",
            language="vi",
            length_bucket="short",
            source_row_sha256=source_hash,
            corpus_manifest_sha256=manifest.sha256,
            model_tp=8,
            model_fp=1,
            model_fn=2,
            regex_tp=9,
            regex_fp=0,
            regex_fn=1,
            model_fp_ignored=3,
            regex_fp_ignored=1,
            gold_pii_chars=25,
            model_leaked_chars=7,
            regex_leaked_chars=0,
            document_chars=40,
            model_overredacted_chars=5,
            regex_overredacted_chars=3,
        ),
    )
    label_counts = (LabelMetricCounts("company_name", 8, 1, 2, 9, 0, 1),)
    child = ChildRegexEvidence(
        config="eval",
        corpus_manifest=manifest,
        documents=documents,
        label_counts=label_counts,
        contract_sha256="a" * 64,
        evaluation_contract_sha256="b" * 64,
        runtime_sha256="c" * 64,
        evidence_sha256="0" * 64,
        ignore_list_sha256="e" * 64,
    )
    evidence_sha256 = _child_evidence_sha256(
        config=child.config,
        manifest_sha256=child.corpus_manifest.sha256,
        documents=child.documents,
        labels=child.label_counts,
        contract_sha256=child.contract_sha256,
        evaluation_contract_sha256=child.evaluation_contract_sha256,
        runtime_sha256=child.runtime_sha256,
        ignore_list_sha256=child.ignore_list_sha256,
    )
    return replace(child, evidence_sha256=evidence_sha256)


def test_child_payload_round_trip_preserves_ignored_false_positive_counts() -> None:
    child = _child_with_ignored_false_positives()

    payload = child_to_payload(child)
    rehydrated = child_from_payload(payload)

    assert rehydrated.documents[0].model_fp_ignored == 3
    assert rehydrated.documents[0].regex_fp_ignored == 1
    assert rehydrated.ignore_list_sha256 == child.ignore_list_sha256
    assert rehydrated == child


def test_child_payload_round_trip_preserves_leaked_character_counts() -> None:
    child = _child_with_ignored_false_positives()

    payload = child_to_payload(child)
    rehydrated = child_from_payload(payload)

    assert rehydrated.documents[0].gold_pii_chars == 25
    assert rehydrated.documents[0].model_leaked_chars == 7
    assert rehydrated.documents[0].regex_leaked_chars == 0
    assert rehydrated.evidence_sha256 == child.evidence_sha256
    assert rehydrated == child


def test_child_payload_round_trip_preserves_overredacted_character_counts() -> None:
    child = _child_with_ignored_false_positives()

    payload = child_to_payload(child)
    rehydrated = child_from_payload(payload)

    assert rehydrated.documents[0].document_chars == 40
    assert rehydrated.documents[0].model_overredacted_chars == 5
    assert rehydrated.documents[0].regex_overredacted_chars == 3
    assert rehydrated.evidence_sha256 == child.evidence_sha256
    assert rehydrated == child


def test_document_payload_missing_document_chars_reads_as_pre_schema_evidence() -> None:
    """Read a payload without document_chars as pre-schema evidence.

    A document payload persisted before the document_chars addition must
    not crash the reader. Simulates a real old evidence artifact: the document
    dict simply never had document_chars/model_overredacted_chars/
    regex_overredacted_chars keys, the shape of a JSON file written before
    2026-08-11.
    """
    payload = asdict(_child_with_ignored_false_positives().documents[0])
    del payload["document_chars"]
    del payload["model_overredacted_chars"]
    del payload["regex_overredacted_chars"]

    document = _document_from_payload(payload)

    assert document.document_chars is None
    assert document.model_overredacted_chars == 0
    assert document.regex_overredacted_chars == 0


def test_document_payload_present_null_document_chars_is_malformed() -> None:
    payload = asdict(_child_with_ignored_false_positives().documents[0])
    payload["document_chars"] = None

    with pytest.raises(ValueError, match="document_chars is malformed"):
        _document_from_payload(payload)


def test_document_payload_overredacted_counts_without_document_chars_is_malformed() -> None:
    payload = asdict(_child_with_ignored_false_positives().documents[0])
    del payload["document_chars"]

    with pytest.raises(ValueError, match="require document_chars"):
        _document_from_payload(payload)


def test_child_payload_missing_document_chars_preserves_pre_schema_digest() -> None:
    child = _child_with_ignored_false_positives()
    payload = child_to_payload(child)
    documents = payload["documents"]
    assert isinstance(documents, list)
    document = documents[0]
    assert isinstance(document, dict)
    del document["document_chars"]
    del document["model_overredacted_chars"]
    del document["regex_overredacted_chars"]
    payload["evidence_sha256"] = canonical_sha256({
        "config": payload["config"],
        "manifest": payload["corpus_manifest_sha256"],
        "documents": payload["documents"],
        "labels": payload["label_counts"],
        "contract_sha256": payload["contract_sha256"],
        "evaluation_contract_sha256": payload["evaluation_contract_sha256"],
        "runtime_sha256": payload["runtime_sha256"],
        "ignore_list_sha256": payload["ignore_list_sha256"],
    })

    rehydrated = child_from_payload(payload)

    assert rehydrated.documents[0].document_chars is None
    assert rehydrated.evidence_sha256 == payload["evidence_sha256"]
    reserialized = child_to_payload(rehydrated)
    reserialized_documents = reserialized["documents"]
    assert isinstance(reserialized_documents, list)
    reserialized_document = reserialized_documents[0]
    assert isinstance(reserialized_document, dict)
    assert "document_chars" not in reserialized_document
    assert child_from_payload(reserialized) == rehydrated


def test_evaluation_contract_schema_declares_document_chars() -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    schemas = {evaluation_contract.result_schema for evaluation_contract in evaluation_contracts(contract).values()}

    assert len(schemas) == 1
    assert "document_chars" in schemas.pop()


class _StubVolume:
    def commit(self) -> None:
        return None


def _refuse_if_called(*_args: object, **_kwargs: object) -> object:
    msg = "execute_child must refuse before loading dataset rows"
    raise AssertionError(msg)


def test_execute_child_refuses_before_inference_on_ignore_list_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ReleaseGateContract.default(
        source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf",
        ignore_list_sha256="a" * 64,
    )
    wrong_list = ignore_list_from_rows([
        {
            "split": "eval",
            "uid": "doc",
            "start": 0,
            "end": 3,
            "text": "abc",
            "category": "company_name",
        },
    ])
    assert wrong_list.sha256 != contract.ignore_list_sha256
    monkeypatch.setattr("meddies_pii.eval_baseline.baseline.datasets.load_v2_eval_rows", _refuse_if_called)

    with pytest.raises(ValueError, match="ignore-list digest does not match contract"):
        execute_child(contract, "eval", volume=_StubVolume(), ignore_list=wrong_list)


def test_execute_child_refuses_before_inference_when_ignore_list_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ReleaseGateContract.default(
        source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf",
        ignore_list_sha256="a" * 64,
    )
    monkeypatch.setattr("meddies_pii.eval_baseline.baseline.datasets.load_v2_eval_rows", _refuse_if_called)

    with pytest.raises(ValueError, match="ignore-list digest does not match contract"):
        execute_child(contract, "eval", volume=_StubVolume(), ignore_list=None)


def test_cost_rate_scales_with_the_contract_compute_envelope() -> None:
    """The base reproduces the live 8-core projection exactly: 16,399.921s * 0.00014032 = $2.301237.

    32 cores bills 4x the base, so the same 16,400s costs ~$9.2, not ~$2.3.
    """
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    assert BASE_CPU_RATE_USD_PER_SECOND == 0.00014032
    assert cost_rate_usd_per_second(8, 16 * 1024) == BASE_CPU_RATE_USD_PER_SECOND

    rate = cost_rate_usd_per_second(contract.cpu_cores, contract.memory_mib)
    assert rate == pytest.approx(0.00056128)
    assert 16_400 * rate == pytest.approx(9.205, abs=0.001)


def test_inference_threads_match_the_contract_cores_in_run_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record the thread count used to produce hydrated evidence.

    Only the digest survives into the contract, so recompute the identity the evidence must carry: threads recorded as
    what actually ran.
    """
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    captured: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, *, threads: int) -> None:
            captured["threads"] = threads

    monkeypatch.setattr(inference, "OnnxRuntimeBackend", FakeBackend)
    monkeypatch.setattr(inference, "BioesSpanDetector", lambda **kwargs: kwargs["backend"])

    _build_adapter(contract)
    assert captured["threads"] == contract.cpu_cores

    vendor = evaluation_contracts(contract)[MODEL_CORE_VIEW].vendor_inference_source
    assert vendor == payload_artifact(
        "bioes-ort",
        {"backend": "onnxruntime-cpu", "threads": contract.cpu_cores},
    )
    assert vendor != payload_artifact("bioes-ort", {"backend": "onnxruntime-cpu", "threads": 8})
