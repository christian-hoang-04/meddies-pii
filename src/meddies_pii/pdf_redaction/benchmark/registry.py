from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

CandidateId = Literal["H1", "H2", "H3", "H4"]
OutcomeStatus = Literal["measured", "capability_failure", "setup_failure"]


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: CandidateId
    native_locator: str
    scan_locator: str
    writer: str
    pii_runtime: str
    benchmark_role: str


REQUIRED_CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        candidate_id="H1",
        native_locator="PyMuPDF characters",
        scan_locator="RapidOCR PP-OCRv6 small OpenVINO word polygons",
        writer="PyMuPDF true redaction plus object scrub",
        pii_runtime="OpenVINO",
        benchmark_role="primary hypothesis",
    ),
    CandidateSpec(
        candidate_id="H2",
        native_locator="PyMuPDF characters",
        scan_locator="RapidOCR PP-OCRv6 tiny word polygons",
        writer="PyMuPDF true redaction plus object scrub",
        pii_runtime="OpenVINO",
        benchmark_role="speed challenger",
    ),
    CandidateSpec(
        candidate_id="H3",
        native_locator="PyMuPDF characters",
        scan_locator="Tesseract TSV",
        writer="PyMuPDF true redaction plus object scrub",
        pii_runtime="OpenVINO",
        benchmark_role="simplicity baseline",
    ),
    CandidateSpec(
        candidate_id="H4",
        native_locator="pypdfium2 character boxes",
        scan_locator="best mobile OCR from Gate 1",
        writer="fresh full-page raster rebuild",
        pii_runtime="OpenVINO",
        benchmark_role="security fallback",
    ),
)


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    candidate_id: CandidateId
    status: OutcomeStatus
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status == "measured":
            if self.failure_reason is not None:
                msg = "measured outcomes must not have failure_reason"
                raise ValueError(msg)
            return
        if self.failure_reason is None or not self.failure_reason.strip():
            msg = "failed outcomes require a non-empty failure_reason"
            raise ValueError(msg)


def validate_candidate_outcomes(
    outcomes: Sequence[CandidateOutcome],
) -> tuple[CandidateOutcome, ...]:
    """Require evidence for every mandatory end-to-end candidate.

    Returns:
        The mandatory candidate outcomes in required order.

    Raises:
        ValueError: if any required candidate has no outcome; if an outcome names a candidate that
            is not in the registry; or if the same candidate is reported twice. The gate exists so
            a partial or duplicated matrix cannot be read as a complete one.

    """
    required_ids: tuple[CandidateId, ...] = tuple(spec.candidate_id for spec in REQUIRED_CANDIDATES)
    by_id: dict[CandidateId, CandidateOutcome] = {}
    for outcome in outcomes:
        if outcome.candidate_id not in required_ids:
            msg = f"unknown candidate: {outcome.candidate_id}"
            raise ValueError(msg)
        if outcome.candidate_id in by_id:
            msg = f"duplicate candidate outcome: {outcome.candidate_id}"
            raise ValueError(msg)
        by_id[outcome.candidate_id] = outcome

    missing = tuple(candidate_id for candidate_id in required_ids if candidate_id not in by_id)
    if missing:
        msg = f"missing required candidates: {', '.join(missing)}"
        raise ValueError(msg)
    return tuple(by_id[candidate_id] for candidate_id in required_ids)
