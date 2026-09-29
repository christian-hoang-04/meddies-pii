from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from meddies_pii.pdf_redaction.benchmark.results import (
    TIMING_STAGES,
    ArtifactIdentity,
    FailedCandidateResult,
    LicenseClass,
    MeasuredCandidateResult,
    NoEligibleCandidateError,
    StageTimingRecord,
    product_ranking,
    result_to_safe_dict,
    results_to_safe_jsonl,
    technical_ranking,
    validate_results,
)

if TYPE_CHECKING:
    from meddies_pii.pdf_redaction.benchmark.registry import CandidateId

_DIGEST = "a" * 64


# reason: The factory exposes scoring, geometry, structure, and licensing axes so ranking tests vary one dimension.
def _measured(  # ruff: ignore[too-many-arguments]
    candidate_id: CandidateId,
    *,
    license_class: LicenseClass = "agpl",
    latency: float = 1.0,
    rss: int = 100,
    output_bytes: int = 1_000,
    recoverable: int = 0,
    negative_control_touches: int = 0,
    oracle_region_recall: float = 1.0,
    geometry_undercoverage_rate: float = 0.0,
    geometry_overredaction_rate: float = 0.0,
    outside_mask_changed_pixel_rate: float = 0.0,
    outside_mask_changed_region_count: int = 0,
    output_structure_valid: bool = True,
    page_invariants_exact: bool = True,
) -> MeasuredCandidateResult:
    timings = tuple(
        StageTimingRecord(
            stage=stage,
            seconds=((latency,) if stage in {"cold_start", "model_load"} else (latency, latency + 0.1, latency + 0.2)),
        )
        for stage in TIMING_STAGES
    )
    return MeasuredCandidateResult(
        candidate_id=candidate_id,
        stage_timings=timings,
        peak_rss_bytes=rss,
        input_bytes=800,
        output_bytes=output_bytes,
        ocr_pages=5,
        native_pages=5,
        negative_control_touches=negative_control_touches,
        recoverable_gold_canaries=recoverable,
        oracle_region_recall=oracle_region_recall,
        geometry_undercoverage_rate=geometry_undercoverage_rate,
        geometry_overredaction_rate=geometry_overredaction_rate,
        outside_mask_changed_pixel_rate=outside_mask_changed_pixel_rate,
        outside_mask_changed_region_count=outside_mask_changed_region_count,
        output_structure_valid=output_structure_valid,
        page_invariants_exact=page_invariants_exact,
        license_class=license_class,
        artifacts=(
            ArtifactIdentity(
                artifact_id="pii_model",
                version="1.0.0",
                sha256=_DIGEST,
            ),
        ),
    )


def _failure(candidate_id: CandidateId) -> FailedCandidateResult:
    return FailedCandidateResult(
        candidate_id=candidate_id,
        status="setup_failure",
        failure_code="runtime.import_failed",
        error_digest="b" * 64,
        license_class="permissive",
        artifacts=(),
    )


def test_result_serialization_is_schema_versioned_and_phi_free() -> None:
    result = _measured("H1")

    safe = result_to_safe_dict(result)
    jsonl = results_to_safe_jsonl((result,))

    assert safe["schema_version"] == 2
    assert safe["candidate_id"] == "H1"
    stage_timings = safe["stage_timings"]
    assert isinstance(stage_timings, list)
    assert len(stage_timings) == len(TIMING_STAGES)
    assert f'"sha256":"{_DIGEST}"' in jsonl
    assert "page_text" not in jsonl
    assert "raw_canary" not in jsonl


def test_measured_result_requires_every_stage_exactly_once() -> None:
    result = _measured("H1")

    with pytest.raises(ValueError, match="stage_timings must contain each stage exactly once"):
        MeasuredCandidateResult(
            candidate_id="H1",
            stage_timings=result.stage_timings[:-1],
            peak_rss_bytes=result.peak_rss_bytes,
            input_bytes=result.input_bytes,
            output_bytes=result.output_bytes,
            ocr_pages=result.ocr_pages,
            native_pages=result.native_pages,
            negative_control_touches=result.negative_control_touches,
            recoverable_gold_canaries=result.recoverable_gold_canaries,
            oracle_region_recall=result.oracle_region_recall,
            geometry_undercoverage_rate=result.geometry_undercoverage_rate,
            geometry_overredaction_rate=result.geometry_overredaction_rate,
            outside_mask_changed_pixel_rate=(result.outside_mask_changed_pixel_rate),
            outside_mask_changed_region_count=(result.outside_mask_changed_region_count),
            output_structure_valid=result.output_structure_valid,
            page_invariants_exact=result.page_invariants_exact,
            license_class=result.license_class,
            artifacts=result.artifacts,
        )


def test_cold_and_model_load_are_single_samples_while_warm_stages_use_three() -> None:
    StageTimingRecord(stage="cold_start", seconds=(0.7,))
    StageTimingRecord(stage="model_load", seconds=(1.2,))
    StageTimingRecord(stage="verify", seconds=(0.1, 0.2, 0.15))

    with pytest.raises(ValueError, match="cold_start requires exactly 1"):
        StageTimingRecord(stage="cold_start", seconds=(0.7, 0.6, 0.5))
    with pytest.raises(ValueError, match="verify requires exactly 3"):
        StageTimingRecord(stage="verify", seconds=(0.1,))


def test_registry_completeness_is_enforced_for_results() -> None:
    with pytest.raises(ValueError, match="missing required candidates: H4"):
        validate_results((_measured("H1"), _measured("H2"), _failure("H3")))


def test_technical_ranking_blocks_insecure_rows_and_exposes_fast_and_small() -> None:
    results = (
        _measured("H1", recoverable=1, latency=0.1, rss=50),
        _measured("H2", latency=1.0, rss=100, output_bytes=1_000),
        _measured("H3", latency=0.5, rss=200, output_bytes=700),
        _failure("H4"),
    )

    ranking = technical_ranking(results)

    assert tuple(row.candidate_id for row in ranking.ordered) == ("H3", "H2")
    assert ranking.fast.candidate_id == "H3"
    assert ranking.small.candidate_id == "H2"


def test_every_blocking_security_gate_excludes_a_measured_row() -> None:
    blocked_rows = (
        _measured("H1", recoverable=1),
        _measured("H1", negative_control_touches=1),
        _measured("H1", geometry_undercoverage_rate=0.01),
        _measured("H1", outside_mask_changed_pixel_rate=0.000001),
        _measured("H1", outside_mask_changed_region_count=1),
        _measured("H1", oracle_region_recall=0.99),
        _measured("H1", output_structure_valid=False),
        _measured("H1", page_invariants_exact=False),
    )
    failures = (_failure("H2"), _failure("H3"), _failure("H4"))

    for row in blocked_rows:
        with pytest.raises(NoEligibleCandidateError, match="technical ranking"):
            technical_ranking((row, *failures))


def test_product_ranking_excludes_nonpermissive_until_terms_are_accepted() -> None:
    results = (
        _measured("H1", latency=0.3),
        _measured("H2", latency=0.4),
        _measured("H3", latency=0.5),
        _measured("H4", license_class="permissive", latency=1.5),
    )

    assert product_ranking(results).winner.candidate_id == "H4"
    assert product_ranking(results, license_terms_accepted=True).winner.candidate_id == "H1"


def test_product_ranking_blocks_conditional_model_license_by_default() -> None:
    results = (
        _failure("H1"),
        _failure("H2"),
        _failure("H3"),
        _measured("H4", license_class="conditional"),
    )

    with pytest.raises(NoEligibleCandidateError, match="product ranking"):
        product_ranking(results)


def test_product_ranking_fails_explicitly_without_an_eligible_row() -> None:
    results = (
        _measured("H1"),
        _measured("H2"),
        _measured("H3"),
        _failure("H4"),
    )

    with pytest.raises(NoEligibleCandidateError, match="product ranking"):
        product_ranking(results)
