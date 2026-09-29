"""Public seams for the focused annotation domain primitives."""

from __future__ import annotations

from meddies_pii import taxonomy
from meddies_pii.annotations.bioes import (
    TokenizedExample,
    build_bioes_label_space,
    decode_bioes_from_offsets,
)
from meddies_pii.annotations.source_mapping import map_native_label_to_pii_label
from meddies_pii.constants import (
    LABEL_MAP,
    PII_LABEL_SET,
    VALID_LABELS,
    VALID_LABELS_BRACKETED,
)
from meddies_pii.generation import prompts, text_formats
from meddies_pii.spans import CharSpan


def test_constants_facade_reexports_domain_owners() -> None:
    assert PII_LABEL_SET == taxonomy.PII_LABEL_SET
    assert VALID_LABELS == taxonomy.PII_LABEL_SET
    assert VALID_LABELS_BRACKETED == taxonomy.PII_LABELS_BRACKETED
    assert isinstance(VALID_LABELS, frozenset)
    assert isinstance(VALID_LABELS_BRACKETED, frozenset)
    assert LABEL_MAP["first_name"] == "<human_name>"


def test_bioes_facade_decodes_single_token_entity() -> None:
    vocabulary = build_bioes_label_space(("human_name",))
    label_ids = {label: index for index, label in enumerate(vocabulary)}
    offsets = [(0, 0), (0, 4), (0, 0)]
    decoded = decode_bioes_from_offsets(
        "John",
        offsets,
        [-100, label_ids["S-human_name"], -100],
        dict(enumerate(vocabulary)),
    )

    assert decoded == (CharSpan(0, 4, "John", "human_name"),)
    assert (
        TokenizedExample(
            raw="John",
            spans=decoded,
            input_ids=[101, 102],
            attention_mask=[1, 1],
            labels=[-100, -100],
            offset_mapping=[(0, 0), (0, 0)],
        ).truncated
        is False
    )


def test_generation_primitives_preserve_prompt_and_format_values() -> None:
    assert prompts.EXTRACTION_PROMPT.startswith("Extract <address>")
    assert text_formats.canonical_text_format("FHIR-like JSON") == "FHIR_JSON"
    assert map_native_label_to_pii_label("B-FIRST_NAME") == "human_name"
