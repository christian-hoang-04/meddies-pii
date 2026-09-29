"""Production-owned regex recovery for model PII spans."""

from __future__ import annotations

# ruff: file-ignore[non-empty-init-module]
# reason: this package root is the stable public runtime facade; moving these definitions would change import identity.
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from meddies_pii.regex_runtime.language import detect_language
from meddies_pii.regex_runtime.manifest import regex_manifest
from meddies_pii.regex_runtime.postprocess import apply_regex_postprocess

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.spans import CharSpan

RegexLanguageSource = Literal["caller", "detected", "undetected"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegexRuntimeResult:
    spans: tuple[tuple[CharSpan, ...], ...]
    regex_language: str | None
    regex_language_source: RegexLanguageSource | None
    manifest_sha256: str | None


def apply(
    texts: Sequence[str],
    model_spans: Sequence[Sequence[CharSpan]],
    *,
    language: str | None = None,
    enabled: bool = True,
) -> RegexRuntimeResult:
    """Apply one document-level regex policy to aligned text and model spans.

    Returns:
        The aligned model-plus-regex spans and document-level language provenance.

    Raises:
        ValueError: If text and model-span collections are not aligned.

    """
    if len(texts) != len(model_spans):
        msg = "text and model span collections must have equal lengths"
        raise ValueError(msg)
    aligned_model_spans = tuple(tuple(spans) for spans in model_spans)
    if not enabled:
        return RegexRuntimeResult(aligned_model_spans, None, None, None)

    resolved_language = language
    language_source: RegexLanguageSource
    if resolved_language is not None:
        language_source = "caller"
    else:
        resolved_language = detect_language("\n".join(texts))
        language_source = "detected" if resolved_language is not None else "undetected"
        if resolved_language is None:
            logger.warning(
                "Language detection was undecided; language-scoped organization "
                "regex packs remain disabled for this document.",
            )

    return RegexRuntimeResult(
        spans=tuple(
            apply_regex_postprocess(
                text,
                spans,
                language=resolved_language,
            ).spans
            for text, spans in zip(texts, aligned_model_spans, strict=True)
        ),
        regex_language=resolved_language,
        regex_language_source=language_source,
        manifest_sha256=str(regex_manifest()["sha256"]),
    )


__all__ = (
    "RegexLanguageSource",
    "RegexRuntimeResult",
    "apply",
    "regex_manifest",
)
