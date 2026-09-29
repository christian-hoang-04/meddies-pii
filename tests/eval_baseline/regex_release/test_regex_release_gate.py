from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: exact equality is limited to frozen cost, deterministic fake-clock, and serialized contract values;
# reason: calculated metrics in this file use pytest.approx instead.
# ruff: file-ignore[docstring-missing-returns]
# reason: private test-fixture helpers are readable at their call sites; boilerplate return sections add no contract value.
# ruff: file-ignore[too-many-arguments]
# reason: the child-evidence fixture names each persisted schema field so tests can vary one dimension explicitly.
import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from meddies_pii.eval_baseline.baseline.datasets import V2_DATASET_REVISION, V2_REPO_ID
from meddies_pii.eval_baseline.regex_release.regex_bootstrap import (
    LengthBucket,
    PairedBootstrapResult,
    PairedDocumentCounts,
)
from meddies_pii.eval_baseline.regex_release.regex_release_contract import (
    REGEX_RELEASE_SEED,
    ReleaseGateContract,
    render_launch_commands,
    require_paid_run_receipts,
    stage_approval_receipt,
    write_hydration_receipt,
    write_terminal_receipt,
)
from meddies_pii.eval_baseline.regex_release.regex_release_gate import (
    BudgetProjectionError,
    ChildRegexEvidence,
    LabelMetricCounts,
    LeakMetrics,
    OverRedactionMetrics,
    ProjectedInferenceAdapter,
    RegexAggregateReport,
    _overredaction_metrics,
    aggregate_exposed_children,
)
from meddies_pii.eval_baseline.regex_release.regex_report import RegexCorpusManifest
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from meddies_pii.spans import CharSpan


def test_default_contract_content_addresses_every_paid_run_input() -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    assert contract.seed == REGEX_RELEASE_SEED == 20260807
    assert contract.dataset_rows == {"eval": 1_700, "eval-challenge": 3_400}
    assert contract.exposed_evidence_only is True
    assert contract.development_release_allowed is False
    assert contract.shipping_allowed is False
    assert len(contract.sha256) == 64
    assert replace(contract, comparison_id="another-run").sha256 != contract.sha256

    with pytest.raises(ValueError, match="cannot authorize release or shipping"):
        replace(contract, shipping_allowed=True)


def test_default_contract_pins_the_measured_compute_envelope() -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    assert contract.cpu_cores == 32
    assert contract.memory_mib == 64 * 1024
    assert contract.gpu is False
    assert contract.timeout_seconds == 3 * 60 * 60
    assert contract.projected_cost_ceiling_usd == 6.50

    with pytest.raises(ValueError, match="resource envelope is frozen"):
        replace(contract, cpu_cores=8)


def test_paid_run_requires_matching_approval_and_hydration_receipts(
    tmp_path: Path,
) -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    with pytest.raises(RuntimeError, match="approval receipt is missing"):
        require_paid_run_receipts(tmp_path, contract)

    stage_approval_receipt(tmp_path, contract, approved_contract_sha256=contract.sha256)
    with pytest.raises(RuntimeError, match="hydration receipt is missing"):
        require_paid_run_receipts(tmp_path, contract)

    write_hydration_receipt(
        tmp_path,
        contract,
        model_artifact_sha256s=tuple(artifact.sha256 for artifact in contract.model_artifacts),
        dataset_row_counts=contract.dataset_rows,
        dataset_fixture_sha256s={"eval": "a" * 64, "eval-challenge": "b" * 64},
    )
    require_paid_run_receipts(tmp_path, contract)

    approval_path = tmp_path / contract.comparison_id / "approval.json"
    payload = json.loads(approval_path.read_text(encoding="utf-8"))
    payload["contract_sha256"] = "0" * 64
    approval_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="approval receipt does not match"):
        require_paid_run_receipts(tmp_path, contract)


def test_hydration_refuses_an_ignore_list_digest_that_does_not_match_the_contract(
    tmp_path: Path,
) -> None:
    contract = ReleaseGateContract.default(
        source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf",
        ignore_list_sha256="a" * 64,
    )
    stage_approval_receipt(tmp_path, contract, approved_contract_sha256=contract.sha256)

    with pytest.raises(ValueError, match="hydrated ignore-list digest does not match"):
        write_hydration_receipt(
            tmp_path,
            contract,
            model_artifact_sha256s=tuple(artifact.sha256 for artifact in contract.model_artifacts),
            dataset_row_counts=contract.dataset_rows,
            dataset_fixture_sha256s={"eval": "b" * 64, "eval-challenge": "c" * 64},
            ignore_list_sha256="d" * 64,
        )

    write_hydration_receipt(
        tmp_path,
        contract,
        model_artifact_sha256s=tuple(artifact.sha256 for artifact in contract.model_artifacts),
        dataset_row_counts=contract.dataset_rows,
        dataset_fixture_sha256s={"eval": "b" * 64, "eval-challenge": "c" * 64},
        ignore_list_sha256="a" * 64,
    )
    require_paid_run_receipts(tmp_path, contract)


def _fixture_sha256(config: str) -> str:
    return hashlib.sha256(f"fixture:{config}".encode()).hexdigest()


def _hydrated_fixture_sha256s() -> dict[str, str]:
    return {config: _fixture_sha256(config) for config in ("eval", "eval-challenge")}


def _child_evidence(
    config: str,
    count: int,
    contract_sha256: str,
    *,
    model_counts: tuple[int, int, int] = (8, 1, 2),
    regex_counts: tuple[int, int, int] = (9, 1, 1),
    length_buckets: tuple[LengthBucket, ...] | None = None,
    ignore_list_sha256: str | None = None,
    gold_pii_chars: int = 0,
    model_leaking_rows: int = 0,
    regex_leaking_rows: int = 0,
    document_chars: int | None = None,
    model_overredacting_rows: int = 0,
    regex_overredacting_rows: int = 0,
) -> ChildRegexEvidence:
    """Make the first N rows leak every gold character and the rest leak none.

    The first N rows over-mask every non-gold character; the rest over-mask none.
    """
    source_hashes = tuple(hashlib.sha256(f"{config}:{index}".encode()).hexdigest() for index in range(count))
    buckets = length_buckets or ("short",) * count
    manifest = RegexCorpusManifest.create(
        corpus_id=f"v2-{config}",
        dataset=V2_REPO_ID,
        dataset_revision=V2_DATASET_REVISION,
        shard=config,
        fixture_sha256=_fixture_sha256(config),
        source_row_sha256s=source_hashes,
        historically_exposed=True,
        blinded=False,
        preregistered_strata=tuple(sorted({("vi", bucket) for bucket in buckets})),
    )
    model_tp, model_fp, model_fn = model_counts
    regex_tp, regex_fp, regex_fn = regex_counts
    documents = tuple(
        PairedDocumentCounts(
            document_id=f"{config}-{index}",
            language="vi",
            length_bucket=buckets[index],
            source_row_sha256=source_hash,
            corpus_manifest_sha256=manifest.sha256,
            model_tp=model_tp,
            model_fp=model_fp,
            model_fn=model_fn,
            regex_tp=regex_tp,
            regex_fp=regex_fp,
            regex_fn=regex_fn,
            gold_pii_chars=gold_pii_chars,
            model_leaked_chars=gold_pii_chars if index < model_leaking_rows else 0,
            regex_leaked_chars=gold_pii_chars if index < regex_leaking_rows else 0,
            document_chars=document_chars,
            model_overredacted_chars=(
                document_chars - gold_pii_chars if document_chars is not None and index < model_overredacting_rows else 0
            ),
            regex_overredacted_chars=(
                document_chars - gold_pii_chars if document_chars is not None and index < regex_overredacting_rows else 0
            ),
        )
        for index, source_hash in enumerate(source_hashes)
    )
    return ChildRegexEvidence(
        config=config,
        corpus_manifest=manifest,
        documents=documents,
        label_counts=(
            LabelMetricCounts(
                "human_name",
                model_tp * count,
                model_fp * count,
                model_fn * count,
                regex_tp * count,
                regex_fp * count,
                regex_fn * count,
            ),
        ),
        contract_sha256=contract_sha256,
        evaluation_contract_sha256="a" * 64,
        runtime_sha256="b" * 64,
        evidence_sha256=hashlib.sha256(f"evidence:{config}".encode()).hexdigest(),
        ignore_list_sha256=ignore_list_sha256,
    )


def _freeze_bootstrap(monkeypatch: pytest.MonkeyPatch, *, lower_95: float = 0.08) -> dict[str, int]:
    """Replace both bootstrap call sites and record the row count each one saw."""
    seen: dict[str, int] = {}

    def fake(module: str) -> Callable[..., PairedBootstrapResult]:
        def bootstrap(documents: tuple[PairedDocumentCounts, ...], *, replicates: int, seed: int) -> PairedBootstrapResult:
            seen[module] = len(documents)
            return PairedBootstrapResult(0.1, lower_95, 0.12, replicates, seed, len(documents), 1)

        return bootstrap

    monkeypatch.setattr(
        "meddies_pii.eval_baseline.regex_release.regex_release_gate.paired_document_bootstrap",
        fake("pooled"),
    )
    monkeypatch.setattr(
        "meddies_pii.eval_baseline.regex_release.regex_report.paired_document_bootstrap",
        fake("challenge"),
    )
    return seen


def test_aggregate_is_child_bound_disjoint_and_never_a_release_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tp=8, fn=2 per model document; tp=9, fn=1 per regex document."""
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    eval_child = _child_evidence("eval", 1_700, contract.sha256)
    challenge_child = _child_evidence("eval-challenge", 3_400, contract.sha256)
    seen = _freeze_bootstrap(monkeypatch)

    report = aggregate_exposed_children(
        contract=contract,
        children=(eval_child, challenge_child),
        hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
    )

    assert isinstance(report, RegexAggregateReport)
    assert report.documents == 5_100
    assert seen == {"pooled": 5_100, "challenge": 3_400}
    assert report.descriptive_pooled_bootstrap.replicates == 10_000
    assert report.challenge.documents == 3_400
    assert report.challenge.sample_adequate_for_inference is True
    assert report.quality_gate_passed is True
    assert report.development_release_gate_passed is False
    assert report.shipping_confirmation_passed is False
    languages = report.descriptive_pooled_by_language
    assert languages[0].name == "vi"
    assert {metric.name for metric in report.descriptive_pooled_by_label} == set(PII_LABEL_SET)
    assert languages[0].model_recall == pytest.approx(0.8)
    assert languages[0].regex_recall == pytest.approx(0.9)
    human_name = next(metric for metric in report.descriptive_pooled_by_label if metric.name == "human_name")
    assert human_name.model_recall == pytest.approx(0.8)
    assert human_name.regex_recall == pytest.approx(0.9)

    overlapping = replace(
        challenge_child,
        documents=(
            replace(challenge_child.documents[0], document_id="eval-0"),
            *challenge_child.documents[1:],
        ),
    )
    with pytest.raises(ValueError, match="disjoint"):
        aggregate_exposed_children(
            contract=contract,
            children=(eval_child, overlapping),
            hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
        )


def test_quality_gate_reads_challenge_only_evidence_not_the_pooled_5100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a pooled pass when the challenge shard regresses.

    The eval shard removes 20 model false positives per document, so pooled precision improves; the challenge shard
    alone shows a precision decline.
    """
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    eval_child = _child_evidence(
        "eval",
        1_700,
        contract.sha256,
        model_counts=(8, 20, 2),
        regex_counts=(9, 0, 1),
    )
    challenge_child = _child_evidence(
        "eval-challenge",
        3_400,
        contract.sha256,
        model_counts=(8, 0, 2),
        regex_counts=(9, 1, 1),
    )
    seen = _freeze_bootstrap(monkeypatch)

    report = aggregate_exposed_children(
        contract=contract,
        children=(eval_child, challenge_child),
        hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
    )

    assert report.regex_precision >= report.model_precision
    assert report.challenge.regex_precision < report.challenge.model_precision
    assert report.challenge.quality_gate_passed is False
    assert report.quality_gate_passed is False
    assert seen == {"pooled": 5_100, "challenge": 3_400}


def test_quality_gate_refuses_a_challenge_stratum_below_the_replication_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject inference when a preregistered stratum has one document.

    One preregistered stratum carries a single document, so no bootstrap replicate can resample it and the shard cannot
    support inference.
    """
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    eval_child = _child_evidence("eval", 1_700, contract.sha256)
    challenge_child = _child_evidence(
        "eval-challenge",
        3_400,
        contract.sha256,
        length_buckets=("long", *("short",) * 3_399),
    )
    _freeze_bootstrap(monkeypatch)

    report = aggregate_exposed_children(
        contract=contract,
        children=(eval_child, challenge_child),
        hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
    )

    assert report.challenge.sample_adequate_for_inference is False
    assert report.challenge.quality_gate_passed is False
    assert report.quality_gate_passed is False


def test_aggregate_verifies_fixtures_against_the_hydration_receipt_not_the_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    eval_child = _child_evidence("eval", 1_700, contract.sha256)
    challenge_child = _child_evidence("eval-challenge", 3_400, contract.sha256)
    _freeze_bootstrap(monkeypatch)

    disagreeing = {**_hydrated_fixture_sha256s(), "eval-challenge": "c" * 64}
    with pytest.raises(ValueError, match="does not bind the persisted fixture"):
        aggregate_exposed_children(
            contract=contract,
            children=(eval_child, challenge_child),
            hydrated_fixture_sha256s=disagreeing,
        )

    missing = {"eval": _fixture_sha256("eval")}
    with pytest.raises(ValueError, match="no fixture digest for eval-challenge"):
        aggregate_exposed_children(
            contract=contract,
            children=(eval_child, challenge_child),
            hydrated_fixture_sha256s=missing,
        )


def test_no_ignore_list_contract_scores_identically_to_before_the_ignore_list_existed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve legacy behavior when the contract has no ignore-list.

    A contract that declares no ignore-list produces identical counts and gate
    outcomes to before the ignore-list existed. Its content-addressed digests
    changed one time when the ignore-list fields were added to the payload shape
    (see test_contract_digest_changed_once_for_the_ignore_list_field_addition) —
    that is an accepted, intentional one-time bump, not a behavior change.
    """
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    assert contract.ignore_list_sha256 is None
    eval_child = _child_evidence("eval", 1_700, contract.sha256)
    challenge_child = _child_evidence("eval-challenge", 3_400, contract.sha256)
    _freeze_bootstrap(monkeypatch)

    report = aggregate_exposed_children(
        contract=contract,
        children=(eval_child, challenge_child),
        hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
    )

    assert report.quality_gate_passed is True
    assert report.model_fp_ignored == 0
    assert report.regex_fp_ignored == 0
    assert report.challenge.model_fp_ignored == 0
    assert report.challenge.regex_fp_ignored == 0


def test_contract_digest_changed_once_for_the_ignore_list_field_addition() -> None:
    """Record the intentional digest change caused by the ignore-list field.

    The gate has never executed a paid run, so no receipt is invalidated by
    this digest change — accepted as a one-time, intentional bump (not a
    regression) when the ignore-list field was added to the frozen contract.
    """
    pre_ignore_list_digest = "3d89170022e9384cfbe1a06ea8495454fc044cd798f8dcf8f1a29c701a5dad9b"
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    assert contract.sha256 != pre_ignore_list_digest


def test_ignore_list_digest_mismatch_between_contract_and_child_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a child scored with an ignore-list other than the contract pin."""
    contract = ReleaseGateContract.default(
        source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf",
        ignore_list_sha256="a" * 64,
    )
    eval_child = _child_evidence("eval", 1_700, contract.sha256, ignore_list_sha256="a" * 64)
    challenge_child = _child_evidence("eval-challenge", 3_400, contract.sha256, ignore_list_sha256="b" * 64)
    _freeze_bootstrap(monkeypatch)

    with pytest.raises(ValueError, match="ignore-list digest does not match"):
        aggregate_exposed_children(
            contract=contract,
            children=(eval_child, challenge_child),
            hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
        )


def test_missing_child_ignore_list_is_refused_when_the_contract_declares_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a child scored before the ignore-list contract existed."""
    contract = ReleaseGateContract.default(
        source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf",
        ignore_list_sha256="a" * 64,
    )
    eval_child = _child_evidence("eval", 1_700, contract.sha256, ignore_list_sha256="a" * 64)
    challenge_child = _child_evidence("eval-challenge", 3_400, contract.sha256)
    _freeze_bootstrap(monkeypatch)

    with pytest.raises(ValueError, match="ignore-list digest does not match"):
        aggregate_exposed_children(
            contract=contract,
            children=(eval_child, challenge_child),
            hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
        )


def test_projection_infers_each_row_once_and_aborts_before_unapproved_cost() -> None:
    class Detector:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def detect(self, texts: Sequence[str]) -> list[tuple[CharSpan, ...]]:
            self.calls.append(tuple(texts))
            return [() for _ in texts]

    times = iter((0.0, 1.0))
    detector = Detector()
    adapter = ProjectedInferenceAdapter(
        detector=detector,
        supported_labels=PII_LABEL_SET,
        sample_rows=2,
        projected_total_rows=4,
        reserve_seconds=0,
        timeout_seconds=10,
        cost_rate_usd_per_second=0.01,
        cost_ceiling_usd=1,
        clock=lambda: next(times),
    )

    assert adapter.predict(["a", "b", "c", "d"]) == [[], [], [], []]
    assert detector.calls == [("a", "b"), ("c", "d")]
    assert adapter.projection is not None
    assert adapter.projection.projected_total_seconds == 2.0

    slow_times = iter((0.0, 10.0))
    slow_detector = Detector()
    slow_adapter = ProjectedInferenceAdapter(
        detector=slow_detector,
        supported_labels=PII_LABEL_SET,
        sample_rows=2,
        projected_total_rows=5_100,
        reserve_seconds=300,
        timeout_seconds=7_200,
        cost_rate_usd_per_second=0.00014032,
        cost_ceiling_usd=1.010304,
        clock=lambda: next(slow_times),
    )
    with pytest.raises(BudgetProjectionError) as caught:
        slow_adapter.predict(["secret-a", "secret-b", "secret-c"])
    assert slow_detector.calls == [("secret-a", "secret-b")]
    assert "secret" not in str(caught.value)


def test_rendered_launch_commands_require_the_exact_contract_digest() -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    commands = render_launch_commands(contract)

    assert tuple(commands) == (
        "stage_approval",
        "hydrate",
        "run_eval",
        "run_eval_challenge",
        "aggregate_5100",
    )
    assert all(contract.sha256 in command for command in commands.values())
    assert all("MODAL_PROFILE=huyhoang041100" in command for command in commands.values())
    assert "--approved-contract-sha256" in commands["stage_approval"]
    assert "--contract-sha256" in commands["run_eval"]


def test_rendered_launch_commands_carry_the_ignore_list_pin_when_present() -> None:
    """Carry the ignore-list pin through every rendered launch command.

    A pinned contract's runbook must carry --ignore-list-sha256 on every
    stage, or a copy-paster walks straight back into the local/remote
    handshake mismatch the pin exists to prevent.
    """
    pinned = ReleaseGateContract.default(
        source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf",
        ignore_list_sha256="e" * 64,
    )
    unpinned = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    pinned_commands = render_launch_commands(pinned)
    unpinned_commands = render_launch_commands(unpinned)

    flag = f"--ignore-list-sha256 {'e' * 64}"
    assert all(flag in command for command in pinned_commands.values())
    assert not any("--ignore-list-sha256" in command for command in unpinned_commands.values())


def test_failure_receipt_records_only_safe_failure_metadata(tmp_path: Path) -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    path = write_terminal_receipt(
        tmp_path,
        contract,
        stage="eval",
        status="failed",
        exception_type="RuntimeError",
    )
    payload = path.read_text(encoding="utf-8")

    assert "RuntimeError" in payload
    assert "text" not in payload
    assert "pred_spans" not in payload
    assert "gold_spans" not in payload


def test_budget_stop_receipt_records_the_projection_that_stopped_the_run(
    tmp_path: Path,
) -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    path = write_terminal_receipt(
        tmp_path,
        contract,
        stage="eval",
        status="budget_stop",
        aggregate={
            "sample_rows": 64,
            "sample_seconds": 12.5,
            "projected_total_seconds": 332.0,
            "projected_cost_usd": 1.25,
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "budget_stop"
    assert payload["aggregate"]["projected_cost_usd"] == 1.25


def test_terminal_receipts_refuse_metric_shapes_their_status_cannot_carry(
    tmp_path: Path,
) -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    with pytest.raises(ValueError, match="only completed and budget-stop receipts carry aggregates"):
        write_terminal_receipt(tmp_path, contract, stage="eval", status="completed", aggregate=None)
    with pytest.raises(ValueError, match="only completed and budget-stop receipts carry aggregates"):
        write_terminal_receipt(tmp_path, contract, stage="eval", status="budget_stop", aggregate=None)
    with pytest.raises(ValueError, match="only completed and budget-stop receipts carry aggregates"):
        write_terminal_receipt(
            tmp_path,
            contract,
            stage="eval",
            status="failed",
            exception_type="RuntimeError",
            aggregate={"documents": 1},
        )


def _leak(report: RegexAggregateReport, scope: str, view: str) -> LeakMetrics:
    return next(item for item in report.descriptive_leak_metrics if item.scope == scope and item.view == view)


def test_leak_metrics_report_the_distribution_per_scope_and_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """340 of 3,400 challenge rows leak all 10 gold characters under model_core; model_plus_regex redacts every row.

    Sorted ascending the 3,400 fractions are 3,060 zeros then 340 ones, so the P95 position (3399 * 0.95 = 3229.05)
    lands inside the leaking tail.
    """
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    eval_child = _child_evidence("eval", 1_700, contract.sha256, gold_pii_chars=10)
    challenge_child = _child_evidence(
        "eval-challenge",
        3_400,
        contract.sha256,
        gold_pii_chars=10,
        model_leaking_rows=340,
    )
    _freeze_bootstrap(monkeypatch)

    report = aggregate_exposed_children(
        contract=contract,
        children=(eval_child, challenge_child),
        hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
    )

    challenge_core = _leak(report, "eval-challenge", "model_core")
    assert challenge_core.rows_with_gold == 3_400
    assert challenge_core.any_leak_rows == 340
    assert challenge_core.any_leak_row_rate == pytest.approx(0.1)
    assert challenge_core.max_row_leak_fraction == pytest.approx(1.0)
    assert challenge_core.p95_row_leak_fraction == pytest.approx(1.0)
    assert challenge_core.char_weighted_leak_fraction == pytest.approx(0.1)

    challenge_regex = _leak(report, "eval-challenge", "model_plus_regex")
    assert challenge_regex.any_leak_rows == 0
    assert challenge_regex.max_row_leak_fraction == pytest.approx(0.0)
    assert challenge_regex.char_weighted_leak_fraction == pytest.approx(0.0)

    pooled_core = _leak(report, "pooled", "model_core")
    assert pooled_core.rows_with_gold == 5_100
    assert pooled_core.any_leak_rows == 340
    assert pooled_core.char_weighted_leak_fraction == pytest.approx(340 / 5_100)


def test_gate_verdict_is_insensitive_to_leak_because_leak_is_report_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    def aggregate(*, model_leaking_rows: int) -> RegexAggregateReport:
        _freeze_bootstrap(monkeypatch)
        return aggregate_exposed_children(
            contract=contract,
            children=(
                _child_evidence("eval", 1_700, contract.sha256, gold_pii_chars=10),
                _child_evidence(
                    "eval-challenge",
                    3_400,
                    contract.sha256,
                    gold_pii_chars=10,
                    model_leaking_rows=model_leaking_rows,
                ),
            ),
            hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
        )

    clean = aggregate(model_leaking_rows=0)
    catastrophic = aggregate(model_leaking_rows=3_400)

    assert _leak(clean, "eval-challenge", "model_core").any_leak_rows == 0
    assert _leak(catastrophic, "eval-challenge", "model_core").any_leak_rows == 3_400
    assert catastrophic.quality_gate_passed == clean.quality_gate_passed is True
    assert catastrophic.challenge == clean.challenge


def _overredaction(report: RegexAggregateReport, scope: str, view: str) -> OverRedactionMetrics:
    return next(item for item in report.descriptive_overredaction_metrics if item.scope == scope and item.view == view)


def _overredaction_document(
    document_id: str,
    *,
    document_chars: int | None,
    gold_pii_chars: int,
    model_overredacted_chars: int = 0,
    regex_overredacted_chars: int = 0,
) -> PairedDocumentCounts:
    return PairedDocumentCounts(
        document_id=document_id,
        language="vi",
        length_bucket="short",
        source_row_sha256=hashlib.sha256(f"row:{document_id}".encode()).hexdigest(),
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


def test_overredaction_metrics_match_hand_calculated_tiny_documents() -> None:
    """10 total - 2 gold = 8 non-gold; model masks 4/8, regex 0/8.

    5 total - 1 gold = 4 non-gold; model masks 0/4, regex 4/4.

    3 total - 3 gold = 0 non-gold, so this row is excluded from rates.
    """
    documents = (
        _overredaction_document(
            "doc-a",
            document_chars=10,
            gold_pii_chars=2,
            model_overredacted_chars=4,
        ),
        _overredaction_document(
            "doc-b",
            document_chars=5,
            gold_pii_chars=1,
            regex_overredacted_chars=4,
        ),
        _overredaction_document("doc-c", document_chars=3, gold_pii_chars=3),
    )

    model_core, model_plus_regex = _overredaction_metrics("eval", documents)

    assert model_core.rows_with_text == 2
    assert model_core.any_overredaction_rows == 1
    assert model_core.any_overredaction_row_rate == pytest.approx(1 / 2)
    assert model_core.p95_row_overredaction_fraction == pytest.approx(0.475)
    assert model_core.max_row_overredaction_fraction == pytest.approx(1 / 2)
    assert model_core.char_weighted_overredaction_fraction == pytest.approx(4 / 12)
    assert model_plus_regex.rows_with_text == 2
    assert model_plus_regex.any_overredaction_rows == 1
    assert model_plus_regex.p95_row_overredaction_fraction == pytest.approx(0.95)
    assert model_plus_regex.max_row_overredaction_fraction == pytest.approx(1.0)
    assert model_plus_regex.char_weighted_overredaction_fraction == pytest.approx(4 / 12)


def test_overredaction_metrics_omit_mixed_old_and_new_rows_in_one_scope() -> None:
    documents = (
        _overredaction_document("new", document_chars=10, gold_pii_chars=2),
        _overredaction_document("old", document_chars=None, gold_pii_chars=2),
    )

    assert _overredaction_metrics("eval", documents) == ()


def test_overredaction_metrics_emit_both_views_when_all_rows_are_new() -> None:
    documents = (
        _overredaction_document("new-a", document_chars=10, gold_pii_chars=2),
        _overredaction_document("new-b", document_chars=5, gold_pii_chars=1),
    )

    metrics = _overredaction_metrics("eval", documents)

    assert {(metric.scope, metric.view) for metric in metrics} == {
        ("eval", "model_core"),
        ("eval", "model_plus_regex"),
    }


def test_overredaction_metrics_report_the_distribution_per_scope_and_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """110-char documents, 10 gold chars each -> 100 non-gold chars per row.

    340 of 3,400 challenge rows over-mask every non-gold character under model_core; model_plus_regex over-masks
    nothing.

    Sorted ascending the 3,400 fractions are 3,060 zeros then 340 ones, so the P95 position (3399 * 0.95 = 3229.05)
    lands inside the over-masking tail.
    """
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    eval_child = _child_evidence(
        "eval",
        1_700,
        contract.sha256,
        gold_pii_chars=10,
        document_chars=110,
    )
    challenge_child = _child_evidence(
        "eval-challenge",
        3_400,
        contract.sha256,
        gold_pii_chars=10,
        document_chars=110,
        model_overredacting_rows=340,
    )
    _freeze_bootstrap(monkeypatch)

    report = aggregate_exposed_children(
        contract=contract,
        children=(eval_child, challenge_child),
        hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
    )

    challenge_core = _overredaction(report, "eval-challenge", "model_core")
    assert challenge_core.rows_with_text == 3_400
    assert challenge_core.any_overredaction_rows == 340
    assert challenge_core.any_overredaction_row_rate == pytest.approx(0.1)
    assert challenge_core.max_row_overredaction_fraction == pytest.approx(1.0)
    assert challenge_core.p95_row_overredaction_fraction == pytest.approx(1.0)
    assert challenge_core.char_weighted_overredaction_fraction == pytest.approx(0.1)

    challenge_regex = _overredaction(report, "eval-challenge", "model_plus_regex")
    assert challenge_regex.any_overredaction_rows == 0
    assert challenge_regex.max_row_overredaction_fraction == pytest.approx(0.0)
    assert challenge_regex.char_weighted_overredaction_fraction == pytest.approx(0.0)

    pooled_core = _overredaction(report, "pooled", "model_core")
    assert pooled_core.rows_with_text == 5_100
    assert pooled_core.any_overredaction_rows == 340
    assert pooled_core.char_weighted_overredaction_fraction == pytest.approx(340 / 5_100)


def test_overredaction_block_is_omitted_when_a_config_predates_document_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old evidence (no document_chars) must not produce partial numbers.

    eval-challenge here never set document_chars, simulating a child scored
    before the schema addition. Its own scope, and pooled (which spans both
    children), must be entirely absent from the report rather than computed
    over only the rows that happen to carry the field.
    """
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    eval_child = _child_evidence("eval", 1_700, contract.sha256, gold_pii_chars=10, document_chars=110)
    challenge_child = _child_evidence("eval-challenge", 3_400, contract.sha256, gold_pii_chars=10)
    _freeze_bootstrap(monkeypatch)

    report = aggregate_exposed_children(
        contract=contract,
        children=(eval_child, challenge_child),
        hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
    )

    scopes = {item.scope for item in report.descriptive_overredaction_metrics}
    assert scopes == {"eval"}


def test_overredaction_block_is_populated_when_every_row_carries_document_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    eval_child = _child_evidence("eval", 1_700, contract.sha256, gold_pii_chars=10, document_chars=110)
    challenge_child = _child_evidence(
        "eval-challenge",
        3_400,
        contract.sha256,
        gold_pii_chars=10,
        document_chars=110,
    )
    _freeze_bootstrap(monkeypatch)

    report = aggregate_exposed_children(
        contract=contract,
        children=(eval_child, challenge_child),
        hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
    )

    scopes = {item.scope for item in report.descriptive_overredaction_metrics}
    assert scopes == {"eval", "eval-challenge", "pooled"}


def test_gate_verdict_is_insensitive_to_overredaction_because_it_is_report_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    def aggregate(*, model_overredacting_rows: int) -> RegexAggregateReport:
        _freeze_bootstrap(monkeypatch)
        return aggregate_exposed_children(
            contract=contract,
            children=(
                _child_evidence(
                    "eval",
                    1_700,
                    contract.sha256,
                    gold_pii_chars=10,
                    document_chars=110,
                ),
                _child_evidence(
                    "eval-challenge",
                    3_400,
                    contract.sha256,
                    gold_pii_chars=10,
                    document_chars=110,
                    model_overredacting_rows=model_overredacting_rows,
                ),
            ),
            hydrated_fixture_sha256s=_hydrated_fixture_sha256s(),
        )

    clean = aggregate(model_overredacting_rows=0)
    catastrophic = aggregate(model_overredacting_rows=3_400)

    assert _overredaction(clean, "eval-challenge", "model_core").any_overredaction_rows == 0
    assert _overredaction(catastrophic, "eval-challenge", "model_core").any_overredaction_rows == 3_400
    assert catastrophic.quality_gate_passed == clean.quality_gate_passed is True
    assert catastrophic.challenge == clean.challenge
