"""Decision report for a paired model-core versus regex quality evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from meddies_pii.eval_baseline.regex_release.regex_bootstrap import (
    LengthBucket,
    PairedBootstrapResult,
    PairedDocumentCounts,
    paired_document_bootstrap,
)
from meddies_pii.eval_baseline.regex_release.regex_corpus_registry import (
    TRUSTED_CORPUS_REGISTRY_SHA256,
    is_known_exposed,
    matches_approved_blinded_shipping_corpus,
)
from meddies_pii.eval_baseline.regex_release.regex_fixtures import (
    LANGUAGE_SCOPED_LOCKED_NEGATIVES,
    LOCKED_CLINICAL_NEGATIVES,
    fixture_manifest,
)
from meddies_pii.evaluation.identity import canonical_sha256, is_sha256
from meddies_pii.regex_runtime import regex_manifest
from meddies_pii.regex_runtime.postprocess import apply_regex_postprocess

if TYPE_CHECKING:
    from collections.abc import Mapping

RELEASE_GATE_CHALLENGE_CORPUS = "v2-eval-challenge"
ELIGIBILITY_BOOTSTRAP_REPLICATES = 10_000
MINIMUM_ELIGIBILITY_DOCUMENTS = 40
"""Forty documents prevents a supported slice from becoming an eligibility claim from a tiny convenience sample; every
declared stratum still needs replication.
"""
MINIMUM_DOCUMENTS_PER_STRATUM = 2


@dataclass(frozen=True, slots=True)
class RegexCorpusManifest:
    """Content-addressed corpus identity and contamination status."""

    corpus_id: str
    dataset: str
    dataset_revision: str
    shard: str
    fixture_sha256: str
    source_rows_sha256: str
    document_count: int
    historically_exposed: bool
    blinded: bool
    preregistered_strata: tuple[tuple[str, LengthBucket], ...]
    approval_audit_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.corpus_id or not self.dataset or not self.dataset_revision or not self.shard:
            msg = "corpus manifest identity fields are required"
            raise ValueError(msg)
        if not is_sha256(self.fixture_sha256) or not is_sha256(self.source_rows_sha256):
            msg = "corpus manifest requires SHA-256 evidence identities"
            raise ValueError(msg)
        if self.document_count <= 0:
            msg = "corpus manifest document_count must be positive"
            raise ValueError(msg)
        if self.historically_exposed and self.blinded:
            msg = "a historically exposed corpus cannot be marked blinded"
            raise ValueError(msg)
        if self.approval_audit_sha256 is not None and not is_sha256(self.approval_audit_sha256):
            msg = "corpus manifest approval audit must be a SHA-256"
            raise ValueError(msg)
        if not self.preregistered_strata or len(set(self.preregistered_strata)) != len(self.preregistered_strata):
            msg = "corpus manifest requires unique preregistered strata"
            raise ValueError(msg)

    @classmethod
    # reason: this keyword-only factory constructs the manifest identity itself; no existing object can own its fields.
    def create(  # ruff: ignore[too-many-arguments]
        cls,
        *,
        corpus_id: str,
        dataset: str,
        dataset_revision: str,
        shard: str,
        fixture_sha256: str,
        source_row_sha256s: tuple[str, ...],
        historically_exposed: bool,
        blinded: bool,
        preregistered_strata: tuple[tuple[str, LengthBucket], ...],
        approval_audit_sha256: str | None = None,
    ) -> RegexCorpusManifest:
        if not source_row_sha256s or any(not is_sha256(digest) for digest in source_row_sha256s):
            msg = "corpus manifest source-row identities are required"
            raise ValueError(msg)
        if len(set(source_row_sha256s)) != len(source_row_sha256s):
            msg = "corpus manifest source-row identities must be unique"
            raise ValueError(msg)
        return cls(
            corpus_id=corpus_id,
            dataset=dataset,
            dataset_revision=dataset_revision,
            shard=shard,
            fixture_sha256=fixture_sha256,
            source_rows_sha256=_source_rows_sha256(source_row_sha256s),
            document_count=len(source_row_sha256s),
            historically_exposed=historically_exposed,
            blinded=blinded,
            preregistered_strata=preregistered_strata,
            approval_audit_sha256=approval_audit_sha256,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], *, expected_sha256: str) -> RegexCorpusManifest:
        if payload.get("schema_version") != 1:
            msg = "unsupported corpus manifest schema version"
            raise ValueError(msg)
        strata = payload.get("preregistered_strata")
        if not isinstance(strata, list):
            msg = "corpus manifest preregistered strata are malformed"
            # reason: every invalid manifest shape and value shares this constructor's ValueError schema contract.
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
        parsed_strata: list[tuple[str, LengthBucket]] = []
        for raw in strata:
            if (
                not isinstance(raw, list)
                or len(raw) != 2  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
                or not isinstance(raw[0], str)
                or raw[1] not in {"short", "medium", "long"}
            ):
                msg = "corpus manifest preregistered strata are malformed"
                raise ValueError(msg)
            parsed_strata.append((raw[0], cast("LengthBucket", raw[1])))
        values = {
            key: payload.get(key)
            for key in (
                "corpus_id",
                "dataset",
                "dataset_revision",
                "shard",
                "fixture_sha256",
                "source_rows_sha256",
            )
        }
        if not all(isinstance(value, str) for value in values.values()):
            msg = "corpus manifest identity fields are malformed"
            raise ValueError(msg)
        document_count = payload.get("document_count")
        historically_exposed = payload.get("historically_exposed")
        blinded = payload.get("blinded")
        approval_audit_sha256 = payload.get("approval_audit_sha256")
        # reason: count and exposure fields form one persisted status-schema predicate with one failure contract.
        if (
            not isinstance(document_count, int)  # ruff: ignore[too-many-boolean-expressions]
            or isinstance(document_count, bool)
            or not isinstance(historically_exposed, bool)
            or not isinstance(blinded, bool)
            or (approval_audit_sha256 is not None and not isinstance(approval_audit_sha256, str))
        ):
            msg = "corpus manifest status fields are malformed"
            raise ValueError(msg)
        manifest = cls(
            corpus_id=cast("str", values["corpus_id"]),
            dataset=cast("str", values["dataset"]),
            dataset_revision=cast("str", values["dataset_revision"]),
            shard=cast("str", values["shard"]),
            fixture_sha256=cast("str", values["fixture_sha256"]),
            source_rows_sha256=cast("str", values["source_rows_sha256"]),
            document_count=document_count,
            historically_exposed=historically_exposed,
            blinded=blinded,
            preregistered_strata=tuple(parsed_strata),
            approval_audit_sha256=approval_audit_sha256,
        )
        if manifest.sha256 != expected_sha256:
            msg = "corpus manifest payload does not match its SHA-256"
            raise ValueError(msg)
        return manifest

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "corpus_id": self.corpus_id,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "shard": self.shard,
            "fixture_sha256": self.fixture_sha256,
            "source_rows_sha256": self.source_rows_sha256,
            "document_count": self.document_count,
            "historically_exposed": self.historically_exposed,
            "blinded": self.blinded,
            "approval_audit_sha256": self.approval_audit_sha256,
            "preregistered_strata": [list(stratum) for stratum in self.preregistered_strata],
        }

    def validate_persisted_fixture(
        self,
        *,
        dataset: str,
        shard: str,
        fixture_sha256: str,
        source_row_sha256s: tuple[str, ...],
    ) -> None:
        if (
            self.dataset != dataset
            or self.shard != shard
            or self.fixture_sha256 != fixture_sha256
            or self.document_count != len(source_row_sha256s)
            or self.source_rows_sha256 != _source_rows_sha256(source_row_sha256s)
        ):
            msg = "corpus manifest does not bind the persisted fixture"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RegexQualityReport:
    corpus_manifest: RegexCorpusManifest
    documents: int
    model_f1: float
    regex_f1: float
    model_precision: float
    regex_precision: float
    locked_negative_new_false_positives: int
    bootstrap: PairedBootstrapResult
    regex_manifest_sha256: str
    fixture_manifest_sha256: str
    corpus_manifest_sha256: str
    evidence_sha256: str
    source_row_sha256s: tuple[str, ...]
    document_ids: tuple[str, ...]
    sample_adequate_for_inference: bool
    quality_gate_passed: bool
    development_release_gate_passed: bool
    blinded_shipping_confirmation_passed: bool
    trusted_corpus_registry_sha256: str


@dataclass(frozen=True, slots=True)
class RegexShippingVerdict:
    shipping_eligible: bool
    release_corpus_manifest_sha256: str
    blinded_corpus_manifest_sha256: str


# reason: raw counts, bootstrap evidence, manifests, and final gates must remain visibly bound in one report assembly.
def build_regex_quality_report(  # ruff: ignore[too-many-locals]
    *,
    corpus_manifest: RegexCorpusManifest,
    documents: tuple[PairedDocumentCounts, ...],
    seed: int,
) -> RegexQualityReport:
    known_exposed = is_known_exposed(
        dataset_id=corpus_manifest.dataset,
        revision=corpus_manifest.dataset_revision,
        shard=corpus_manifest.shard,
    )
    if known_exposed and (not corpus_manifest.historically_exposed or corpus_manifest.blinded):
        msg = "trusted registry marks corpus as exposed"
        raise ValueError(msg)
    if corpus_manifest.document_count != len(documents):
        msg = "corpus manifest document_count does not match observations"
        raise ValueError(msg)
    if any(document.corpus_manifest_sha256 != corpus_manifest.sha256 for document in documents):
        msg = "paired documents contain mixed or mismatched corpus manifests"
        raise ValueError(msg)
    source_row_sha256s = tuple(sorted(document.source_row_sha256 for document in documents))
    if len(set(source_row_sha256s)) != len(source_row_sha256s):
        msg = "paired documents contain duplicate source-row identities"
        raise ValueError(msg)
    if _source_rows_sha256(source_row_sha256s) != corpus_manifest.source_rows_sha256:
        msg = "paired documents do not match the corpus source-row identity"
        raise ValueError(msg)
    document_ids = tuple(sorted(document.document_id for document in documents))

    model_tp = sum(document.model_tp for document in documents)
    model_fp = sum(document.model_fp for document in documents)
    model_fn = sum(document.model_fn for document in documents)
    regex_tp = sum(document.regex_tp for document in documents)
    regex_fp = sum(document.regex_fp for document in documents)
    regex_fn = sum(document.regex_fn for document in documents)
    bootstrap = paired_document_bootstrap(documents, replicates=ELIGIBILITY_BOOTSTRAP_REPLICATES, seed=seed)
    model_precision = _precision(model_tp, model_fp)
    regex_precision = _precision(regex_tp, regex_fp)
    rules = regex_manifest()
    fixtures = fixture_manifest()
    global_negative_false_positives = sum(
        len(apply_regex_postprocess(fixture.text, (), language=None).spans) for fixture in LOCKED_CLINICAL_NEGATIVES
    )
    scoped_negative_false_positives = sum(
        len(apply_regex_postprocess(fixture.text, (), language=fixture.language).spans)
        for fixture in LANGUAGE_SCOPED_LOCKED_NEGATIVES
    )
    locked_negative_new_false_positives = global_negative_false_positives + scoped_negative_false_positives
    sample_adequate = _sample_adequate_for_inference(documents, corpus_manifest.preregistered_strata)
    quality_gate_passed = (
        sample_adequate
        and bootstrap.replicates == ELIGIBILITY_BOOTSTRAP_REPLICATES
        and bootstrap.lower_95 > 0
        and regex_precision >= model_precision
        and locked_negative_new_false_positives == 0
    )
    evidence_sha256 = canonical_sha256({
        "schema_version": 1,
        "corpus_manifest_sha256": corpus_manifest.sha256,
        "trusted_corpus_registry_sha256": TRUSTED_CORPUS_REGISTRY_SHA256,
        "regex_manifest_sha256": rules["sha256"],
        "fixture_manifest_sha256": fixtures["sha256"],
        "locked_negative_new_false_positives": (locked_negative_new_false_positives),
        "documents": [
            {
                "source_row_sha256": document.source_row_sha256,
                "document_id": document.document_id,
                "language": document.language,
                "length_bucket": document.length_bucket,
                "model": [document.model_tp, document.model_fp, document.model_fn],
                "regex": [document.regex_tp, document.regex_fp, document.regex_fn],
            }
            for document in sorted(documents, key=lambda item: item.source_row_sha256)
        ],
    })
    return RegexQualityReport(
        corpus_manifest=corpus_manifest,
        documents=len(documents),
        model_f1=_f1(model_tp, model_fp, model_fn),
        regex_f1=_f1(regex_tp, regex_fp, regex_fn),
        model_precision=model_precision,
        regex_precision=regex_precision,
        locked_negative_new_false_positives=locked_negative_new_false_positives,
        bootstrap=bootstrap,
        regex_manifest_sha256=str(rules["sha256"]),
        fixture_manifest_sha256=str(fixtures["sha256"]),
        corpus_manifest_sha256=corpus_manifest.sha256,
        evidence_sha256=evidence_sha256,
        source_row_sha256s=source_row_sha256s,
        document_ids=document_ids,
        sample_adequate_for_inference=sample_adequate,
        quality_gate_passed=quality_gate_passed,
        development_release_gate_passed=(
            corpus_manifest.corpus_id == RELEASE_GATE_CHALLENGE_CORPUS and known_exposed and quality_gate_passed
        ),
        blinded_shipping_confirmation_passed=(
            not corpus_manifest.historically_exposed
            and corpus_manifest.blinded
            and matches_approved_blinded_shipping_corpus(
                dataset_id=corpus_manifest.dataset,
                revision=corpus_manifest.dataset_revision,
                shard=corpus_manifest.shard,
                fixture_sha256=corpus_manifest.fixture_sha256,
                source_rows_sha256=corpus_manifest.source_rows_sha256,
                approval_audit_sha256=corpus_manifest.approval_audit_sha256,
            )
            and quality_gate_passed
        ),
        trusted_corpus_registry_sha256=TRUSTED_CORPUS_REGISTRY_SHA256,
    )


def combine_regex_shipping_verdict(
    release_gate: RegexQualityReport,
    blinded_confirmation: RegexQualityReport,
) -> RegexShippingVerdict:
    release_manifest = release_gate.corpus_manifest
    blind_manifest = blinded_confirmation.corpus_manifest
    registry_release_corpus = (
        release_manifest.corpus_id == RELEASE_GATE_CHALLENGE_CORPUS
        and is_known_exposed(
            dataset_id=release_manifest.dataset,
            revision=release_manifest.dataset_revision,
            shard=release_manifest.shard,
        )
        and release_manifest.historically_exposed
        and not release_manifest.blinded
    )
    registry_blinded_corpus = (
        not blind_manifest.historically_exposed
        and blind_manifest.blinded
        and matches_approved_blinded_shipping_corpus(
            dataset_id=blind_manifest.dataset,
            revision=blind_manifest.dataset_revision,
            shard=blind_manifest.shard,
            fixture_sha256=blind_manifest.fixture_sha256,
            source_rows_sha256=blind_manifest.source_rows_sha256,
            approval_audit_sha256=blind_manifest.approval_audit_sha256,
        )
    )
    separate_corpora = (
        release_gate.corpus_manifest_sha256 != blinded_confirmation.corpus_manifest_sha256
        and release_gate.corpus_manifest.dataset_revision != blinded_confirmation.corpus_manifest.dataset_revision
        and set(release_gate.source_row_sha256s).isdisjoint(blinded_confirmation.source_row_sha256s)
        and set(release_gate.document_ids).isdisjoint(blinded_confirmation.document_ids)
    )
    same_locked_intervention = (
        release_gate.regex_manifest_sha256 == blinded_confirmation.regex_manifest_sha256
        and release_gate.fixture_manifest_sha256 == blinded_confirmation.fixture_manifest_sha256
    )
    same_trusted_registry = (
        release_gate.trusted_corpus_registry_sha256
        == TRUSTED_CORPUS_REGISTRY_SHA256
        == blinded_confirmation.trusted_corpus_registry_sha256
    )
    return RegexShippingVerdict(
        shipping_eligible=(
            release_gate.development_release_gate_passed
            and blinded_confirmation.blinded_shipping_confirmation_passed
            and registry_release_corpus
            and registry_blinded_corpus
            and separate_corpora
            and same_locked_intervention
            and same_trusted_registry
        ),
        release_corpus_manifest_sha256=release_gate.corpus_manifest_sha256,
        blinded_corpus_manifest_sha256=blinded_confirmation.corpus_manifest_sha256,
    )


def _sample_adequate_for_inference(
    documents: tuple[PairedDocumentCounts, ...],
    preregistered_strata: tuple[tuple[str, LengthBucket], ...],
) -> bool:
    strata = Counter((document.language, document.length_bucket) for document in documents)
    if any(stratum not in preregistered_strata for stratum in strata):
        msg = "observed document stratum was not preregistered"
        raise ValueError(msg)
    return len(documents) >= MINIMUM_ELIGIBILITY_DOCUMENTS and all(
        strata[stratum] >= MINIMUM_DOCUMENTS_PER_STRATUM for stratum in preregistered_strata
    )


def _source_rows_sha256(source_row_sha256s: tuple[str, ...]) -> str:
    return canonical_sha256({"schema_version": 1, "source_row_sha256s": sorted(source_row_sha256s)})


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0
