from __future__ import annotations

import pytest

from meddies_pii.pdf_redaction.benchmark.registry import (
    REQUIRED_CANDIDATES,
    CandidateOutcome,
    OutcomeStatus,
    validate_candidate_outcomes,
)


def test_registry_rejects_a_silently_missing_required_candidate() -> None:
    outcomes = tuple(
        CandidateOutcome(candidate_id=candidate.candidate_id, status="measured") for candidate in REQUIRED_CANDIDATES[:-1]
    )

    with pytest.raises(ValueError, match=r"missing required candidates: H4"):
        validate_candidate_outcomes(outcomes)


def test_registry_accepts_measurements_and_exact_failures_for_every_candidate() -> None:
    outcomes = (
        CandidateOutcome(candidate_id="H1", status="measured"),
        CandidateOutcome(
            candidate_id="H2",
            status="capability_failure",
            failure_reason="word polygons unavailable: RuntimeError('synthetic')",
        ),
        CandidateOutcome(candidate_id="H3", status="measured"),
        CandidateOutcome(
            candidate_id="H4",
            status="setup_failure",
            failure_reason="renderer import failed: ModuleNotFoundError('synthetic')",
        ),
    )

    validated = validate_candidate_outcomes(outcomes)

    assert tuple(item.candidate_id for item in validated) == ("H1", "H2", "H3", "H4")


@pytest.mark.parametrize(
    ("status", "reason"),
    [("measured", "unexpected note"), ("setup_failure", None)],
)
def test_registry_rejects_ambiguous_failure_evidence(status: OutcomeStatus, reason: str | None) -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        CandidateOutcome(candidate_id="H1", status=status, failure_reason=reason)
