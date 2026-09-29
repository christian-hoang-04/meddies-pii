from __future__ import annotations

# ruff: file-ignore[too-many-arguments]
# reason: the fixture factory names every evidence coordinate; bundling would hide which dimension a test varies.
import hashlib
import inspect
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from meddies_pii.eval_baseline.regex_release import regex_report as report_module
from meddies_pii.eval_baseline.regex_release.regex_bootstrap import (
    LengthBucket,
    PairedDocumentCounts,
    paired_document_bootstrap,
)
from meddies_pii.eval_baseline.regex_release.regex_corpus_registry import (
    APPROVED_BLINDED_SHIPPING_CORPORA,
    MEDDIES_PII_V2_EXPOSED_REVISION,
    TRUSTED_CORPUS_REGISTRY_SHA256,
    is_known_exposed,
)
from meddies_pii.eval_baseline.regex_release.regex_fixtures import RegexFixture
from meddies_pii.eval_baseline.regex_release.regex_report import (
    RegexCorpusManifest,
    build_regex_quality_report,
    combine_regex_shipping_verdict,
)


def _source_hashes(namespace: str, count: int) -> tuple[str, ...]:
    return tuple(hashlib.sha256(f"{namespace}:{index}".encode()).hexdigest() for index in range(count))


def _evidence(
    corpus_id: str,
    *,
    namespace: str,
    count: int = 40,
    historically_exposed: bool,
    blinded: bool,
    dataset_revision: str,
    dataset: str | None = None,
    shard: str = "full",
    approval_audit_sha256: str | None = None,
    preregistered_strata: tuple[tuple[str, LengthBucket], ...] = (("vi", "short"),),
    document_strata: tuple[tuple[str, LengthBucket], ...] | None = None,
) -> tuple[RegexCorpusManifest, tuple[PairedDocumentCounts, ...]]:
    source_hashes = _source_hashes(namespace, count)
    manifest = RegexCorpusManifest.create(
        corpus_id=corpus_id,
        dataset=dataset or corpus_id,
        dataset_revision=dataset_revision,
        shard=shard,
        fixture_sha256=hashlib.sha256(f"fixture:{namespace}".encode()).hexdigest(),
        source_row_sha256s=source_hashes,
        historically_exposed=historically_exposed,
        blinded=blinded,
        approval_audit_sha256=approval_audit_sha256,
        preregistered_strata=preregistered_strata,
    )
    strata = document_strata or tuple(("vi", "short") for _ in range(count))
    documents = tuple(
        PairedDocumentCounts(
            document_id=f"{namespace}-{index}",
            language=strata[index][0],
            length_bucket=strata[index][1],
            source_row_sha256=source_hash,
            corpus_manifest_sha256=manifest.sha256,
            model_tp=8,
            model_fp=1,
            model_fn=2,
            regex_tp=10,
            regex_fp=1,
            regex_fn=0,
        )
        for index, source_hash in enumerate(source_hashes)
    )
    return manifest, documents


def test_paired_bootstrap_is_deterministic_and_resamples_within_strata() -> None:
    manifest, documents = _evidence(
        "bootstrap",
        namespace="bootstrap",
        count=3,
        historically_exposed=True,
        blinded=False,
        dataset_revision="rev-bootstrap",
        preregistered_strata=(("vi", "short"), ("en", "long")),
        document_strata=(("vi", "short"), ("vi", "short"), ("en", "long")),
    )
    assert all(document.corpus_manifest_sha256 == manifest.sha256 for document in documents)

    first = paired_document_bootstrap(documents, replicates=200, seed=19)
    second = paired_document_bootstrap(documents, replicates=200, seed=19)

    assert first == second
    assert first.observed_delta > 0
    assert first.replicates == 200
    assert first.strata == 2


def test_shipping_requires_release_gate_and_blinded_confirmation() -> None:
    release_manifest, release_documents = _evidence(
        "v2-eval-challenge",
        namespace="release",
        historically_exposed=True,
        blinded=False,
        dataset="Meddies/meddies-pii-v2",
        dataset_revision=MEDDIES_PII_V2_EXPOSED_REVISION,
        shard="eval-challenge",
    )
    blind_manifest, blind_documents = _evidence(
        "blinded-confirmation",
        namespace="blind",
        historically_exposed=False,
        blinded=True,
        dataset_revision="dataset-blind",
        approval_audit_sha256="c" * 64,
    )
    release = build_regex_quality_report(corpus_manifest=release_manifest, documents=release_documents, seed=23)
    blinded = build_regex_quality_report(corpus_manifest=blind_manifest, documents=blind_documents, seed=29)

    assert release.development_release_gate_passed is True
    assert release.blinded_shipping_confirmation_passed is False
    assert blinded.development_release_gate_passed is False
    assert blinded.quality_gate_passed is True
    assert blinded.blinded_shipping_confirmation_passed is False
    assert combine_regex_shipping_verdict(release, blinded).shipping_eligible is False


def test_trusted_registry_denies_known_exposed_family_and_has_no_blind_corpus() -> None:
    assert TRUSTED_CORPUS_REGISTRY_SHA256 == ("36a62ae9d535838807fac5234cc92dcc977381b1db5eb626b1da2e8540f33060")
    assert APPROVED_BLINDED_SHIPPING_CORPORA == ()
    assert is_known_exposed(
        dataset_id="Meddies/meddies-pii-v2",
        revision=MEDDIES_PII_V2_EXPOSED_REVISION,
        shard="eval",
    )
    assert is_known_exposed(
        dataset_id="Meddies/meddies-pii-v2",
        revision=MEDDIES_PII_V2_EXPOSED_REVISION,
        shard="eval-challenge",
    )


def test_exposed_family_cannot_be_relabelled_or_revision_laundered_as_blind() -> None:
    relabelled_manifest, relabelled_documents = _evidence(
        "claimed-blind-subset",
        namespace="disjoint-exposed-subset",
        dataset="Meddies/meddies-pii-v2",
        dataset_revision=MEDDIES_PII_V2_EXPOSED_REVISION,
        shard="eval-challenge",
        historically_exposed=False,
        blinded=True,
        approval_audit_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="trusted registry marks corpus as exposed"):
        build_regex_quality_report(
            corpus_manifest=relabelled_manifest,
            documents=relabelled_documents,
            seed=31,
        )

    new_revision_manifest, new_revision_documents = _evidence(
        "claimed-blind-new-revision",
        namespace="disjoint-new-revision",
        dataset="Meddies/meddies-pii-v2",
        dataset_revision="1" * 40,
        shard="eval-challenge",
        historically_exposed=False,
        blinded=True,
        approval_audit_sha256="e" * 64,
    )
    report = build_regex_quality_report(
        corpus_manifest=new_revision_manifest,
        documents=new_revision_documents,
        seed=37,
    )

    assert report.quality_gate_passed is True
    assert report.blinded_shipping_confirmation_passed is False

    release_manifest, release_documents = _evidence(
        "v2-eval-challenge",
        namespace="hostile-release",
        dataset="Meddies/meddies-pii-v2",
        dataset_revision=MEDDIES_PII_V2_EXPOSED_REVISION,
        shard="eval-challenge",
        historically_exposed=True,
        blinded=False,
    )
    release = build_regex_quality_report(
        corpus_manifest=release_manifest,
        documents=release_documents,
        seed=41,
    )
    forged_flag = replace(report, blinded_shipping_confirmation_passed=True)
    assert combine_regex_shipping_verdict(release, forged_flag).shipping_eligible is False


def test_identical_source_rows_cannot_be_relabelled_as_blinded_evidence() -> None:
    source_hashes = _source_hashes("reused", 40)
    release_manifest = RegexCorpusManifest.create(
        corpus_id="v2-eval-challenge",
        dataset="v2-eval-challenge",
        dataset_revision="release-revision",
        shard="full",
        fixture_sha256="a" * 64,
        source_row_sha256s=source_hashes,
        historically_exposed=True,
        blinded=False,
        preregistered_strata=(("vi", "short"),),
    )
    claimed_blind_manifest = RegexCorpusManifest.create(
        corpus_id="claimed-blind",
        dataset="claimed-blind",
        dataset_revision="claimed-blind-revision",
        shard="full",
        fixture_sha256="b" * 64,
        source_row_sha256s=source_hashes,
        historically_exposed=False,
        blinded=True,
        preregistered_strata=(("vi", "short"),),
    )

    def documents(manifest: RegexCorpusManifest) -> tuple[PairedDocumentCounts, ...]:
        return tuple(
            PairedDocumentCounts(
                f"same-{index}",
                "vi",
                "short",
                source_hash,
                manifest.sha256,
                8,
                1,
                2,
                10,
                1,
                0,
            )
            for index, source_hash in enumerate(source_hashes)
        )

    release = build_regex_quality_report(
        corpus_manifest=release_manifest,
        documents=documents(release_manifest),
        seed=23,
    )
    claimed_blind = build_regex_quality_report(
        corpus_manifest=claimed_blind_manifest,
        documents=documents(claimed_blind_manifest),
        seed=29,
    )

    assert release.source_row_sha256s == claimed_blind.source_row_sha256s
    assert combine_regex_shipping_verdict(release, claimed_blind).shipping_eligible is False


def test_report_requires_positive_ci_precision_and_negative_safety() -> None:
    manifest, documents = _evidence(
        "v2-eval-challenge",
        namespace="quality",
        historically_exposed=True,
        blinded=False,
        dataset="Meddies/meddies-pii-v2",
        dataset_revision=MEDDIES_PII_V2_EXPOSED_REVISION,
        shard="eval-challenge",
    )
    report = build_regex_quality_report(corpus_manifest=manifest, documents=documents, seed=23)

    assert report.development_release_gate_passed is True
    assert report.blinded_shipping_confirmation_passed is False
    assert report.bootstrap.lower_95 > 0
    assert report.bootstrap.replicates == 10_000
    assert report.regex_precision >= report.model_precision
    assert len(report.regex_manifest_sha256) == 64
    assert len(report.evidence_sha256) == 64
    changed_counts = (
        replace(documents[0], regex_fp=documents[0].regex_fp + 1),
        *documents[1:],
    )
    changed_report = build_regex_quality_report(corpus_manifest=manifest, documents=changed_counts, seed=23)
    assert changed_report.evidence_sha256 != report.evidence_sha256


def test_report_rejects_mixed_corpus_manifest_evidence() -> None:
    manifest, documents = _evidence(
        "v2-eval-challenge",
        namespace="mixed",
        historically_exposed=True,
        blinded=False,
        dataset_revision="mixed-revision",
    )
    altered = (
        replace(documents[0], corpus_manifest_sha256="f" * 64),
        *documents[1:],
    )

    with pytest.raises(ValueError, match="mixed or mismatched corpus manifests"):
        build_regex_quality_report(corpus_manifest=manifest, documents=altered, seed=23)


def test_report_computes_locked_negative_failures_internally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(build_regex_quality_report)
    assert "locked_negative_new_false_positives" not in signature.parameters
    assert "replicates" not in signature.parameters
    monkeypatch.setattr(
        report_module,
        "LOCKED_CLINICAL_NEGATIVES",
        (RegexFixture("injected-email", "alpha@example.invalid"),),
    )
    manifest, documents = _evidence(
        "v2-eval-challenge",
        namespace="negative",
        historically_exposed=True,
        blinded=False,
        dataset_revision="negative-revision",
    )

    report = build_regex_quality_report(corpus_manifest=manifest, documents=documents, seed=23)

    assert report.locked_negative_new_false_positives == 1
    assert report.quality_gate_passed is False
    assert report.development_release_gate_passed is False


def test_report_counts_language_scoped_locked_negative_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoped_fixture = RegexFixture("injected-language-scoped-violation", "scoped violation", "en")
    monkeypatch.setattr(report_module, "LOCKED_CLINICAL_NEGATIVES", ())
    monkeypatch.setattr(report_module, "LANGUAGE_SCOPED_LOCKED_NEGATIVES", (scoped_fixture,))

    def scoped_only_postprocess(
        text: str,
        model_spans: tuple[object, ...],
        *,
        language: str | None = None,
    ) -> SimpleNamespace:
        del model_spans
        spans = (object(),) if text == scoped_fixture.text and language == "en" else ()
        return SimpleNamespace(spans=spans)

    monkeypatch.setattr(report_module, "apply_regex_postprocess", scoped_only_postprocess)
    manifest, documents = _evidence(
        "v2-eval-challenge",
        namespace="scoped-negative",
        historically_exposed=True,
        blinded=False,
        dataset_revision="scoped-negative-revision",
    )

    report = build_regex_quality_report(corpus_manifest=manifest, documents=documents, seed=23)

    assert report.locked_negative_new_false_positives == 1
    assert report.quality_gate_passed is False


def test_aggregate_report_is_descriptive_not_a_release_gate() -> None:
    manifest, documents = _evidence(
        "aggregate",
        namespace="aggregate",
        historically_exposed=True,
        blinded=False,
        dataset_revision="aggregate-revision",
    )
    report = build_regex_quality_report(corpus_manifest=manifest, documents=documents, seed=23)

    assert report.bootstrap.lower_95 > 0
    assert report.quality_gate_passed is True
    assert report.development_release_gate_passed is False
    assert report.blinded_shipping_confirmation_passed is False


def test_small_or_singleton_strata_results_are_descriptive_only() -> None:
    small_manifest, small_documents = _evidence(
        "v2-eval-challenge",
        namespace="small",
        count=39,
        historically_exposed=True,
        blinded=False,
        dataset_revision="small-revision",
    )
    short: LengthBucket = "short"
    long: LengthBucket = "long"
    singleton_strata = cast(
        "tuple[tuple[str, LengthBucket], ...]",
        (*(("vi", short) for _ in range(39)), ("en", long)),
    )
    singleton_manifest, singleton_documents = _evidence(
        "v2-eval-challenge",
        namespace="singleton",
        historically_exposed=True,
        blinded=False,
        dataset_revision="singleton-revision",
        preregistered_strata=(("vi", "short"), ("en", "long")),
        document_strata=singleton_strata,
    )
    too_small = build_regex_quality_report(corpus_manifest=small_manifest, documents=small_documents, seed=23)
    singleton = build_regex_quality_report(corpus_manifest=singleton_manifest, documents=singleton_documents, seed=23)

    assert too_small.bootstrap.replicates == 10_000
    assert too_small.sample_adequate_for_inference is False
    assert too_small.development_release_gate_passed is False
    assert singleton.sample_adequate_for_inference is False
    assert singleton.development_release_gate_passed is False


def _minimal_document(
    *,
    document_chars: int | None = None,
    gold_pii_chars: int = 0,
    model_overredacted_chars: int = 0,
    regex_overredacted_chars: int = 0,
) -> PairedDocumentCounts:
    return PairedDocumentCounts(
        document_id="doc-0",
        language="vi",
        length_bucket="short",
        source_row_sha256=hashlib.sha256(b"row").hexdigest(),
        corpus_manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
        model_tp=0,
        model_fp=0,
        model_fn=0,
        regex_tp=0,
        regex_fp=0,
        regex_fn=0,
        gold_pii_chars=gold_pii_chars,
        document_chars=document_chars,
        model_overredacted_chars=model_overredacted_chars,
        regex_overredacted_chars=regex_overredacted_chars,
    )


def test_document_chars_none_leaves_overredacted_chars_at_zero() -> None:
    document = _minimal_document()

    assert document.document_chars is None
    assert document.model_overredacted_chars == 0
    assert document.regex_overredacted_chars == 0


def test_overredacted_chars_without_document_chars_is_refused() -> None:
    with pytest.raises(ValueError, match="require a known document_chars"):
        _minimal_document(model_overredacted_chars=1)


def test_overredacted_chars_exceeding_non_gold_total_is_refused() -> None:
    """Refuse over-redacted counts larger than the non-gold text.

    document_chars=10, gold_pii_chars=4 -> 6 non-gold chars available; 7 is one character more than the row could
    possibly have over-masked.
    """
    with pytest.raises(ValueError, match="cannot exceed the non-gold total"):
        _minimal_document(document_chars=10, gold_pii_chars=4, model_overredacted_chars=7)


def test_document_chars_smaller_than_gold_pii_chars_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be smaller than gold_pii_chars"):
        _minimal_document(document_chars=2, gold_pii_chars=4)
