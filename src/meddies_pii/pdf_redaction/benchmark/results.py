from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

from meddies_pii.pdf_redaction.benchmark.registry import (
    CandidateId,
    CandidateOutcome,
    OutcomeStatus,
    validate_candidate_outcomes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.json_types import JsonValue

LicenseClass = Literal["permissive", "agpl", "commercial", "conditional"]
FailureStatus = Literal["setup_failure", "capability_failure"]
TimingStage = Literal[
    "cold_start",
    "model_load",
    "page_classification",
    "extraction_ocr",
    "pii_inference",
    "geometry_mapping",
    "write_sanitize",
    "verify",
    "end_to_end",
]

SCHEMA_VERSION = 2
TIMING_STAGES: tuple[TimingStage, ...] = (
    "cold_start",
    "model_load",
    "page_classification",
    "extraction_ocr",
    "pii_inference",
    "geometry_mapping",
    "write_sanitize",
    "verify",
    "end_to_end",
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class NoEligibleCandidateError(ValueError):
    """Raised when blocking gates leave no candidate to rank."""


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    artifact_id: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        _require_safe_id("artifact_id", self.artifact_id)
        _require_safe_id("version", self.version)
        _require_sha256("sha256", self.sha256)


@dataclass(frozen=True, slots=True)
class StageTimingRecord:
    stage: TimingStage
    seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        expected_samples = 1 if self.stage in {"cold_start", "model_load"} else 3
        if len(self.seconds) != expected_samples:
            msg = f"{self.stage} requires exactly {expected_samples} timing sample(s)"
            raise ValueError(msg)
        for value in self.seconds:
            _require_nonnegative_finite("stage timing", value)

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.seconds)


@dataclass(frozen=True, slots=True)
class MeasuredCandidateResult:
    candidate_id: CandidateId
    stage_timings: tuple[StageTimingRecord, ...]
    peak_rss_bytes: int
    input_bytes: int
    output_bytes: int
    ocr_pages: int
    native_pages: int
    negative_control_touches: int
    recoverable_gold_canaries: int
    oracle_region_recall: float
    geometry_undercoverage_rate: float
    geometry_overredaction_rate: float
    outside_mask_changed_pixel_rate: float
    outside_mask_changed_region_count: int
    output_structure_valid: bool
    page_invariants_exact: bool
    license_class: LicenseClass
    artifacts: tuple[ArtifactIdentity, ...]
    status: Literal["measured"] = field(default="measured", init=False)

    def __post_init__(self) -> None:
        if tuple(timing.stage for timing in self.stage_timings) != TIMING_STAGES:
            msg = "stage_timings must contain each stage exactly once in schema order"
            raise ValueError(msg)
        _require_positive_int("peak_rss_bytes", self.peak_rss_bytes)
        _require_positive_int("input_bytes", self.input_bytes)
        _require_nonnegative_int("output_bytes", self.output_bytes)
        _require_nonnegative_int("ocr_pages", self.ocr_pages)
        _require_nonnegative_int("native_pages", self.native_pages)
        if self.ocr_pages + self.native_pages == 0:
            msg = "ocr_pages plus native_pages must be positive"
            raise ValueError(msg)
        _require_nonnegative_int("negative_control_touches", self.negative_control_touches)
        _require_nonnegative_int("recoverable_gold_canaries", self.recoverable_gold_canaries)
        _require_rate("oracle_region_recall", self.oracle_region_recall)
        _require_rate("geometry_undercoverage_rate", self.geometry_undercoverage_rate)
        _require_nonnegative_finite("geometry_overredaction_rate", self.geometry_overredaction_rate)
        _require_rate(
            "outside_mask_changed_pixel_rate",
            self.outside_mask_changed_pixel_rate,
        )
        _require_nonnegative_int(
            "outside_mask_changed_region_count",
            self.outside_mask_changed_region_count,
        )
        if not self.artifacts:
            msg = "measured results require artifact identities"
            raise ValueError(msg)
        _require_license_class(self.license_class)
        _require_unique_artifact_ids(self.artifacts)

    @property
    def warm_end_to_end_median_seconds(self) -> float:
        return self.stage_timings[-1].median_seconds


@dataclass(frozen=True, slots=True)
class FailedCandidateResult:
    candidate_id: CandidateId
    status: FailureStatus
    failure_code: str
    error_digest: str
    license_class: LicenseClass
    artifacts: tuple[ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        if self.status not in {"setup_failure", "capability_failure"}:
            msg = "failed result status must identify its failure class"
            raise ValueError(msg)
        _require_safe_id("failure_code", self.failure_code)
        _require_sha256("error_digest", self.error_digest)
        _require_license_class(self.license_class)
        _require_unique_artifact_ids(self.artifacts)


CandidateResult: TypeAlias = MeasuredCandidateResult | FailedCandidateResult


@dataclass(frozen=True, slots=True)
class TechnicalRanking:
    ordered: tuple[MeasuredCandidateResult, ...]
    pareto_frontier: tuple[MeasuredCandidateResult, ...]
    fast: MeasuredCandidateResult
    small: MeasuredCandidateResult


@dataclass(frozen=True, slots=True)
class ProductRanking:
    ordered: tuple[MeasuredCandidateResult, ...]
    winner: MeasuredCandidateResult


def validate_results(results: Sequence[CandidateResult]) -> tuple[CandidateResult, ...]:
    """Order records as H1-H4 after the shared completeness gate.

    Returns:
        The candidate results ordered H1-H4. Ordering is delegated to
        `validate_candidate_outcomes`, so that function's refusals — a missing required candidate,
        an unknown one, or a duplicate — propagate unchanged and gate this call too.

    """
    outcomes = tuple(_as_registry_outcome(result) for result in results)
    ordered_outcomes = validate_candidate_outcomes(outcomes)
    by_id = {result.candidate_id: result for result in results}
    return tuple(by_id[outcome.candidate_id] for outcome in ordered_outcomes)


def technical_ranking(results: Sequence[CandidateResult]) -> TechnicalRanking:
    eligible = _security_eligible(validate_results(results))
    if not eligible:
        msg = "technical ranking has no measured candidate passing all blocking gates"
        raise NoEligibleCandidateError(msg)
    ordered = tuple(sorted(eligible, key=_technical_key))
    frontier = _pareto_frontier(eligible)
    fast = min(
        frontier,
        key=lambda row: (
            row.warm_end_to_end_median_seconds,
            row.peak_rss_bytes,
            row.output_bytes,
        ),
    )
    small = min(
        frontier,
        key=lambda row: (
            row.peak_rss_bytes,
            row.output_bytes,
            row.warm_end_to_end_median_seconds,
        ),
    )
    return TechnicalRanking(
        ordered=ordered,
        pareto_frontier=frontier,
        fast=fast,
        small=small,
    )


def product_ranking(
    results: Sequence[CandidateResult],
    *,
    license_terms_accepted: bool = False,
) -> ProductRanking:
    eligible = tuple(
        row
        for row in _security_eligible(validate_results(results))
        if row.license_class == "permissive" or license_terms_accepted
    )
    if not eligible:
        msg = "product ranking has no security- and license-eligible candidate"
        raise NoEligibleCandidateError(msg)
    ordered = tuple(sorted(eligible, key=_technical_key))
    return ProductRanking(ordered=ordered, winner=ordered[0])


def result_to_safe_dict(result: CandidateResult) -> dict[str, JsonValue]:
    common: dict[str, JsonValue] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": result.candidate_id,
        "status": result.status,
        "license_class": result.license_class,
        "artifacts": [_artifact_to_safe_dict(item) for item in result.artifacts],
    }
    if isinstance(result, FailedCandidateResult):
        return {
            **common,
            "failure_code": result.failure_code,
            "error_digest": result.error_digest,
        }
    return {
        **common,
        "stage_timings": [{"stage": timing.stage, "seconds": list(timing.seconds)} for timing in result.stage_timings],
        "peak_rss_bytes": result.peak_rss_bytes,
        "input_bytes": result.input_bytes,
        "output_bytes": result.output_bytes,
        "ocr_pages": result.ocr_pages,
        "native_pages": result.native_pages,
        "negative_control_touches": result.negative_control_touches,
        "recoverable_gold_canaries": result.recoverable_gold_canaries,
        "oracle_region_recall": result.oracle_region_recall,
        "geometry_undercoverage_rate": result.geometry_undercoverage_rate,
        "geometry_overredaction_rate": result.geometry_overredaction_rate,
        "outside_mask_changed_pixel_rate": (result.outside_mask_changed_pixel_rate),
        "outside_mask_changed_region_count": (result.outside_mask_changed_region_count),
        "output_structure_valid": result.output_structure_valid,
        "page_invariants_exact": result.page_invariants_exact,
    }


def results_to_safe_jsonl(results: Sequence[CandidateResult]) -> str:
    return "\n".join(json.dumps(result_to_safe_dict(result), sort_keys=True, separators=(",", ":")) for result in results)


def _as_registry_outcome(result: CandidateResult) -> CandidateOutcome:
    failure_reason = result.failure_code if isinstance(result, FailedCandidateResult) else None
    status: OutcomeStatus = result.status
    return CandidateOutcome(
        candidate_id=result.candidate_id,
        status=status,
        failure_reason=failure_reason,
    )


def _security_eligible(
    results: Sequence[CandidateResult],
) -> tuple[MeasuredCandidateResult, ...]:
    return tuple(
        result for result in results if isinstance(result, MeasuredCandidateResult) and _passes_blocking_gates(result)
    )


def _passes_blocking_gates(result: MeasuredCandidateResult) -> bool:
    return (
        result.recoverable_gold_canaries == 0
        and result.negative_control_touches == 0
        and result.geometry_undercoverage_rate == 0
        and result.geometry_overredaction_rate == 0
        and result.outside_mask_changed_pixel_rate == 0
        and result.outside_mask_changed_region_count == 0
        and result.oracle_region_recall == 1
        and result.output_structure_valid
        and result.page_invariants_exact
    )


def _technical_key(result: MeasuredCandidateResult) -> tuple[float | int, ...]:
    return (
        result.recoverable_gold_canaries,
        result.negative_control_touches,
        result.geometry_undercoverage_rate,
        result.geometry_overredaction_rate,
        result.outside_mask_changed_pixel_rate,
        result.warm_end_to_end_median_seconds,
        result.peak_rss_bytes,
        result.output_bytes,
    )


def _pareto_frontier(
    results: Sequence[MeasuredCandidateResult],
) -> tuple[MeasuredCandidateResult, ...]:
    return tuple(
        result for result in results if not any(other is not result and _dominates(other, result) for other in results)
    )


def _dominates(left: MeasuredCandidateResult, right: MeasuredCandidateResult) -> bool:
    left_values = (
        left.geometry_overredaction_rate,
        left.outside_mask_changed_pixel_rate,
        left.warm_end_to_end_median_seconds,
        left.peak_rss_bytes,
        left.output_bytes,
    )
    right_values = (
        right.geometry_overredaction_rate,
        right.outside_mask_changed_pixel_rate,
        right.warm_end_to_end_median_seconds,
        right.peak_rss_bytes,
        right.output_bytes,
    )
    return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a < b for a, b in zip(left_values, right_values, strict=True)
    )


def _artifact_to_safe_dict(artifact: ArtifactIdentity) -> dict[str, JsonValue]:
    return {
        "artifact_id": artifact.artifact_id,
        "version": artifact.version,
        "sha256": artifact.sha256,
    }


def _require_unique_artifact_ids(artifacts: Sequence[ArtifactIdentity]) -> None:
    artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
    if len(set(artifact_ids)) != len(artifact_ids):
        msg = "artifact identities must have unique artifact_id values"
        raise ValueError(msg)


def _require_safe_id(name: str, value: str) -> None:
    if _SAFE_ID_RE.fullmatch(value) is None:
        msg = f"{name} must be a PHI-free identifier"
        raise ValueError(msg)


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        msg = f"{name} must be a lowercase SHA-256 digest"
        raise ValueError(msg)


def _require_license_class(value: str) -> None:
    if value not in {"permissive", "agpl", "commercial", "conditional"}:
        msg = "license_class must be permissive, agpl, commercial, or conditional"
        raise ValueError(msg)


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)


def _require_nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        msg = f"{name} must be finite and non-negative"
        raise ValueError(msg)


def _require_rate(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        msg = f"{name} must be between zero and one"
        raise ValueError(msg)
