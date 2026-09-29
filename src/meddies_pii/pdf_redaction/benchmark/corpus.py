"""Deterministic PDF challenge-corpus facade."""

from meddies_pii.pdf_redaction.benchmark.corpus_builder import _CorpusBuilder
from meddies_pii.pdf_redaction.benchmark.corpus_cases import (
    _add_hostile_document_objects,
    _build_hostile_page,
    _build_hybrid_hidden_ocr,
    _build_image_clean,
    _build_image_complex,
    _build_image_degraded,
    _build_multilingual_hybrid,
    _build_native_discharge,
    _build_native_form,
    _build_native_rotated,
    _build_native_two_column,
)
from meddies_pii.pdf_redaction.benchmark.corpus_records import (
    PAGE_CLASSES,
    CanaryGold,
    CorpusFixture,
    CorpusGold,
    Degradation,
    GoldChannel,
    NegativeControlGold,
    PageClass,
    PageGold,
)


def generate_challenge_corpus(seed: int = 20260713) -> CorpusFixture:
    """Build the fixed ten-page benchmark without deriving gold from PDF output.

    Returns:
        The fixed ten-page corpus fixture with its gold spans. The gold is built from the generator
        input, never derived from the PDF the generator produced, so the benchmark cannot mark its
        own homework.

    """
    builder = _CorpusBuilder(seed)
    _build_native_discharge(builder)
    _build_native_form(builder)
    _build_native_two_column(builder)
    _build_native_rotated(builder)
    _build_hybrid_hidden_ocr(builder)
    _build_image_clean(builder)
    _build_image_degraded(builder)
    _build_image_complex(builder)
    _build_multilingual_hybrid(builder)
    _build_hostile_page(builder)
    builder.set_fixed_metadata()
    _add_hostile_document_objects(builder)
    return builder.finish()


__all__ = [
    "PAGE_CLASSES",
    "CanaryGold",
    "CorpusFixture",
    "CorpusGold",
    "Degradation",
    "GoldChannel",
    "NegativeControlGold",
    "PageClass",
    "PageGold",
    "generate_challenge_corpus",
]
