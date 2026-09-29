"""Phase-B prompt-quality contracts.

The single email exemplar anchors the model: a spoken "at ... dot" exemplar drove
~46% of generated emails into obfuscated form (2026-06-20 audit). The exemplar
must be a clean canonical email; obfuscation belongs only to the adversarial
scenarios. The doc-type/format name must not leak into the document body.

mistral-medium defaults to lazy descending/sequential placeholders (0987654321); the gate rejects them but a prompt nudge
stops the model wasting attempts on them.

"""

from __future__ import annotations

from meddies_pii.generation.label_corpus.catalog import SCENARIOS
from meddies_pii.generation.label_corpus.prompts import (
    targeted_system_prompt,
    targeted_user_prompt,
)
from meddies_pii.languages import normalize_language


def test_email_exemplar_is_clean_not_spoken_at_dot() -> None:
    sysp = targeted_system_prompt(normalize_language("Vietnamese"), domain_hint="x")
    assert "[maya.green@example.com]<email_address>" in sysp
    assert "at example dot com" not in sysp


def test_system_prompt_steers_to_reserved_example_domains() -> None:
    sysp = targeted_system_prompt(normalize_language("Vietnamese"), domain_hint="x")
    assert "example.com" in sysp


def test_system_prompt_forbids_templated_sequential_digits() -> None:
    sysp = targeted_system_prompt(normalize_language("Vietnamese"), domain_hint="x")
    low = sysp.lower()
    assert "0987654321" in sysp or "sequential" in low
    assert "realistic" in low


def test_user_prompt_forbids_narrating_the_document_type() -> None:
    usr = targeted_user_prompt(
        profile=normalize_language("Vietnamese"),
        scenario=SCENARIOS[0],
        document_type="ambient voice scribe transcript",
        text_format="voice-scribe transcript",
        min_spans=10,
        min_unique_labels=6,
        required_labels=("human_name",),
    )
    assert "Do not name or describe the document type or format" in usr
