"""Named, independently scoreable evaluation views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from meddies_pii.spans import CharSpan

EvaluationView: TypeAlias = Literal["model_core", "vendor_hybrid", "model_plus_regex"]

MODEL_CORE_VIEW: EvaluationView = "model_core"
VENDOR_HYBRID_VIEW: EvaluationView = "vendor_hybrid"
MODEL_PLUS_REGEX_VIEW: EvaluationView = "model_plus_regex"
EVALUATION_VIEWS: tuple[EvaluationView, ...] = (
    MODEL_CORE_VIEW,
    VENDOR_HYBRID_VIEW,
)
REGEX_EVALUATION_VIEWS: tuple[EvaluationView, ...] = (
    MODEL_CORE_VIEW,
    MODEL_PLUS_REGEX_VIEW,
)


@dataclass(frozen=True, slots=True)
class DualViewPrediction:
    """Both outputs derived from a single document inference."""

    model_core: tuple[CharSpan, ...]
    vendor_hybrid: tuple[CharSpan, ...]

    def for_view(self, view: EvaluationView) -> tuple[CharSpan, ...]:
        if view == MODEL_CORE_VIEW:
            return self.model_core
        return self.vendor_hybrid
