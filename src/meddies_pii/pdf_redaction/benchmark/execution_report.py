from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.bioes_inference.file_identity import file_sha256
from meddies_pii.pdf_redaction.benchmark.execution_environment import (
    _automated_redaction_status,
    _current_rss_bytes,
    _environment_identity,
    _page_invariants,
    _peak_rss_bytes,
)
from meddies_pii.pdf_redaction.benchmark.results import (
    ArtifactIdentity,
    FailedCandidateResult,
    MeasuredCandidateResult,
    NoEligibleCandidateError,
    StageTimingRecord,
    product_ranking,
    result_to_safe_dict,
    technical_ranking,
)
from meddies_pii.pdf_redaction.benchmark.runtime import (
    ModelGate,
    OcrSuccess,
    manifest_sha256,
)

if TYPE_CHECKING:
    from meddies_pii.bioes_inference.artifacts import HydratedArtifactIdentity
    from meddies_pii.pdf_redaction.benchmark.execution_types import FullMatrixInputs
    from meddies_pii.pdf_redaction.benchmark.fidelity import RenderFidelity
    from meddies_pii.pdf_redaction.benchmark.registry import CandidateId
    from meddies_pii.pdf_redaction.benchmark.spatial import SpatialCoverage
    from meddies_pii.pdf_redaction.contracts import GeometryPage, PageRegion
    from meddies_pii.pdf_redaction.document import DocumentInspection
    from meddies_pii.pdf_redaction.verification import VerificationReport
    from meddies_pii.pdf_redaction.writers import RedactionWriteResult


@dataclass(frozen=True, slots=True)
class FullMatrixEvidence:
    """Measured evidence assembled into the public full-matrix report."""

    inputs: FullMatrixInputs
    artifact: HydratedArtifactIdentity
    ocr_rows: list[dict[str, object]]
    best_ocr: OcrSuccess
    model_gate: ModelGate
    inspection: DocumentInspection
    ocr_indices: tuple[int, ...]
    prepared_pages: tuple[GeometryPage, ...]
    model_regions: tuple[PageRegion, ...]
    model_coverage: SpatialCoverage
    model_coverage_rows: dict[str, object]
    oracle_coverage: SpatialCoverage
    stage_timings: tuple[StageTimingRecord, ...]
    promoted_write: RedactionWriteResult
    final_report: VerificationReport
    render_fidelity: RenderFidelity
    rasterization_fidelity: RenderFidelity
    model_reports: list[VerificationReport]
    process_setup_seconds: float
    hydration_seconds: float
    allocated_cpu_cores: float
    started_at: float


def build_full_matrix_result(evidence: FullMatrixEvidence) -> dict[str, object]:
    """Build the serializable report after execution has collected all evidence.

    Returns:
        The serializable full-matrix report — schema version 2 — assembled only after execution has
        collected every piece of evidence it references.

    """
    recoverable = len({digest for finding in evidence.final_report.findings for digest in finding.matched_value_sha256})
    structure_valid = evidence.final_report.finding("structure").passed
    page_invariants_exact, normalized_rotations = _page_invariants(
        evidence.inputs.source,
        evidence.promoted_write.output_path,
    )
    artifact_identities = _artifact_identities(evidence)
    candidate_results = _candidate_results(
        evidence,
        artifact_identities=artifact_identities,
        recoverable=recoverable,
        structure_valid=structure_valid,
        page_invariants_exact=page_invariants_exact,
    )
    technical = technical_ranking(candidate_results)
    try:
        product = product_ranking(candidate_results)
    except NoEligibleCandidateError:
        product = None
    model_outputs_verified = bool(evidence.model_reports) and all(report.passed for report in evidence.model_reports)
    automated_redaction_eligible, automated_redaction_status = _automated_redaction_status(
        evidence.model_coverage,
        outputs_verified=model_outputs_verified,
    )
    return {
        "schema_version": 2,
        "stage": "full_matrix",
        "environment": _environment_identity(allocated_cpu_cores=evidence.allocated_cpu_cores),
        "process_setup_seconds": evidence.process_setup_seconds,
        "cold_start_definition": "selected_ocr_initialization_and_first_routed_pass",
        "artifact_hydration_seconds": evidence.hydration_seconds,
        "ocr_gate": evidence.ocr_rows,
        "ocr_winner": evidence.best_ocr.candidate_id,
        "ocr_selection_policy": ("verifier_independent_rapidocr_then_clean_misses_recall_speed_bytes"),
        "pii_gate": list(evidence.model_gate.component_rows),
        "pii_winner": evidence.model_gate.selected_runtime_id,
        "pii_selection_policy": ("fixed_q8_artifact_exact_span_parity_then_warm_median_load_runtime_id"),
        "model_spatial_metrics": _model_spatial_metrics(evidence),
        "candidate_results": [result_to_safe_dict(candidate) for candidate in candidate_results],
        "oracle_writer_technical_winner": technical.ordered[0].candidate_id,
        "oracle_writer_fast_winner": technical.fast.candidate_id,
        "oracle_writer_small_winner": technical.small.candidate_id,
        "oracle_writer_product_winner": (product.winner.candidate_id if product is not None else None),
        "oracle_writer_product_status": (
            "eligible_unconditionally_permissive" if product is not None else "license_review_required"
        ),
        "license_scope": {
            "H1_H3": "agpl_or_commercial_PyMuPDF",
            "H4_libraries": "permissive",
            "H4_model": "conditional_LFM_Open_License_v1.0",
            "benchmark_verifier": "includes_external_GPL_Poppler_tooling",
        },
        "winner_scope": "oracle_regions_writer_and_verifier_only",
        "automated_redaction_winner": (technical.ordered[0].candidate_id if automated_redaction_eligible else None),
        "automated_redaction_status": automated_redaction_status,
        "model_outputs_independently_verified": model_outputs_verified,
        "oracle_render_fidelity": _oracle_render_fidelity(evidence.render_fidelity),
        "rasterization_fidelity": _rasterization_fidelity(evidence.rasterization_fidelity),
        "pii_artifact_manifest": {
            "artifact_id": evidence.artifact.artifact_id,
            "repo_id": evidence.artifact.repo_id,
            "revision": evidence.artifact.revision,
            "files": [
                {
                    "role": file.role,
                    "relative_path": file.relative_path,
                    "size_bytes": file.size_bytes,
                    "sha256": file.sha256,
                }
                for file in evidence.artifact.files
            ],
        },
        "page_invariant_policy": (
            "page_count_media_box_crop_box_and_rotation_exact_except_full_box_rotation_normalization"
        ),
        "normalized_rotation_pages": list(normalized_rotations),
        "dual_mode_contract": {
            "model_mode": "measures_detection_and_geometry_without_promotion",
            "oracle_mode": "proves_writer_and_verifier_security",
        },
        "memory_metrics": {
            "benchmark_process_peak_rss_bytes": _peak_rss_bytes(),
            "post_matrix_current_rss_bytes": _current_rss_bytes(),
            "scope": ("single_process_high_water_and_post_matrix_snapshot; candidate runtimes may remain resident"),
        },
        "wall_seconds": time.perf_counter() - evidence.started_at,
    }


def _artifact_identities(
    evidence: FullMatrixEvidence,
) -> tuple[ArtifactIdentity, ...]:
    return (
        *(
            ArtifactIdentity(
                artifact_id=f"pii_{file.role}_{Path(file.relative_path).name}",
                version=evidence.artifact.revision,
                sha256=file.sha256,
            )
            for file in evidence.artifact.files
        ),
        ArtifactIdentity(
            artifact_id="ocr_manifest",
            version=evidence.best_ocr.candidate_id,
            sha256=manifest_sha256(evidence.best_ocr.model_manifest),
        ),
        ArtifactIdentity(
            artifact_id="challenge_corpus",
            version="schema1",
            sha256=file_sha256(evidence.inputs.source),
        ),
        ArtifactIdentity(
            artifact_id="oracle_writer_output",
            version=evidence.promoted_write.writer_id,
            sha256=evidence.final_report.output_sha256,
        ),
    )


def _candidate_results(
    evidence: FullMatrixEvidence,
    *,
    artifact_identities: tuple[ArtifactIdentity, ...],
    recoverable: int,
    structure_valid: bool,
    page_invariants_exact: bool,
) -> tuple[FailedCandidateResult | MeasuredCandidateResult, ...]:
    failure_code = "writer.pruned_after_preflight_dynamic_object_failure"
    failure_digest = hashlib.sha256(failure_code.encode("ascii")).hexdigest()
    pruned_candidates: tuple[CandidateId, ...] = ("H1", "H2", "H3")
    return (
        *(
            FailedCandidateResult(
                candidate_id=candidate_id,
                status="capability_failure",
                failure_code=failure_code,
                error_digest=failure_digest,
                license_class="agpl",
                artifacts=(),
            )
            for candidate_id in pruned_candidates
        ),
        MeasuredCandidateResult(
            candidate_id="H4",
            stage_timings=evidence.stage_timings,
            peak_rss_bytes=_peak_rss_bytes(),
            input_bytes=evidence.inputs.source.stat().st_size,
            output_bytes=evidence.promoted_write.output_path.stat().st_size,
            ocr_pages=len(evidence.ocr_indices),
            native_pages=len(evidence.inspection.pages) - len(evidence.ocr_indices),
            negative_control_touches=evidence.oracle_coverage.negative_control_touches,
            recoverable_gold_canaries=recoverable,
            oracle_region_recall=evidence.oracle_coverage.gold_region_recall,
            geometry_undercoverage_rate=evidence.oracle_coverage.undercoverage_rate,
            geometry_overredaction_rate=evidence.oracle_coverage.overredaction_rate,
            outside_mask_changed_pixel_rate=(evidence.render_fidelity.changed_outside_mask_rate),
            outside_mask_changed_region_count=len(evidence.render_fidelity.unexplained_changed_regions),
            output_structure_valid=structure_valid,
            page_invariants_exact=page_invariants_exact,
            license_class="conditional",
            artifacts=artifact_identities,
        ),
    )


def _model_spatial_metrics(evidence: FullMatrixEvidence) -> dict[str, object]:
    return {
        "gold_regions_fully_covered": evidence.model_coverage.gold_regions_fully_covered,
        "gold_region_recall": evidence.model_coverage.gold_region_recall,
        "gold_regions_partially_covered": (evidence.model_coverage.gold_regions_partially_covered),
        "gold_region_partial_recall": (evidence.model_coverage.gold_region_partial_recall),
        "undercoverage_rate": evidence.model_coverage.undercoverage_rate,
        "overredaction_rate": evidence.model_coverage.overredaction_rate,
        "negative_control_touches": evidence.model_coverage.negative_control_touches,
        "breakdown": evidence.model_coverage_rows,
        "detected_spans_by_page": [
            {
                "page_index": page_index,
                "count": len(detection.spans),
            }
            for page_index, detection in enumerate(evidence.model_gate.detections)
        ],
        "mapped_regions_by_page": [
            {
                "page_index": page_index,
                "count": sum(region.page_index == page_index for region in evidence.model_regions),
            }
            for page_index in range(len(evidence.prepared_pages))
        ],
    }


def _oracle_render_fidelity(fidelity: RenderFidelity) -> dict[str, object]:
    return {
        "render_dpi": fidelity.render_dpi,
        "pixel_difference_threshold": fidelity.pixel_difference_threshold,
        "mask_dilation_pixels": fidelity.mask_dilation_pixels,
        "total_outside_mask_pixels": fidelity.total_outside_mask_pixels,
        "changed_outside_mask_pixels": fidelity.changed_outside_mask_pixels,
        "changed_outside_mask_rate": fidelity.changed_outside_mask_rate,
        "changed_outside_mask_region_count": len(fidelity.unexplained_changed_regions),
        "required_changed_outside_mask_rate": 0.0,
        "required_changed_outside_mask_region_count": 0,
        "comparison": "unmasked_raster_reference_vs_redacted_raster_output",
    }


def _rasterization_fidelity(fidelity: RenderFidelity) -> dict[str, object]:
    return {
        "render_dpi": fidelity.render_dpi,
        "pixel_difference_threshold": fidelity.pixel_difference_threshold,
        "total_pixels": fidelity.total_outside_mask_pixels,
        "changed_pixels": fidelity.changed_outside_mask_pixels,
        "changed_pixel_rate": fidelity.changed_outside_mask_rate,
        "changed_region_count": len(fidelity.unexplained_changed_regions),
        "comparison": "source_vector_pdf_vs_unmasked_raster_reference",
        "ranking_role": "reported_utility_loss_not_unexplained_redaction_drift",
    }
