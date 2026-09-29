"""Pinned BIOES detector artifacts, hydration, and CPU inference backends."""

from meddies_pii.bioes_inference.artifacts import (
    PUBLIC_Q8_ARTIFACT_SPEC,
    ArtifactHydrationError,
    HydratedArtifactIdentity,
    HydratedFileIdentity,
    PinnedArtifactSpec,
    PinnedFileSpec,
    create_default_detector,
    hydrate_artifact,
)
from meddies_pii.bioes_inference.contracts import ExpectedFileIdentity, PiiSpanDetector, SpanDetection
from meddies_pii.bioes_inference.detector import (
    BioesInferenceBackend,
    BioesSpanDetector,
    OnnxRuntimeBackend,
    OpenVinoBackend,
)

__all__ = (
    "PUBLIC_Q8_ARTIFACT_SPEC",
    "ArtifactHydrationError",
    "BioesInferenceBackend",
    "BioesSpanDetector",
    "ExpectedFileIdentity",
    "HydratedArtifactIdentity",
    "HydratedFileIdentity",
    "OnnxRuntimeBackend",
    "OpenVinoBackend",
    "PiiSpanDetector",
    "PinnedArtifactSpec",
    "PinnedFileSpec",
    "SpanDetection",
    "create_default_detector",
    "hydrate_artifact",
)
