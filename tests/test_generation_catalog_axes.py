"""Phase-B catalog-axis contracts.

Edge-cases are a rated orthogonal axis, and scribe/OCR is a single text-format channel rather
than also a scenario and a doc-type.

The 2026-06-20 audit found ~29% of docs carried a scribe/voice signal because it
was injected by three independent axes (scenario + document_type + text_format)
that compounded. Collapsing it to one channel makes the proportion controllable.

The catalog is the single source of obfuscation techniques; it carries the hard real-data cases (partial masking,
run-together fusion, truncation) on top of the original spoken/digit-word/line-break/full-width/OCR/fragment set.

"""

from __future__ import annotations

import random

from meddies_pii.generation.label_corpus.catalog import (
    DOCUMENT_TYPES,
    MEDICAL_SCENARIO_NAMES,
    SCENARIOS,
    TEXT_FORMATS,
    _scenario_by_name,
)
from meddies_pii.generation.label_corpus.edge_cases import (
    ADVERSARIAL_EDGE_CASE_RANGE,
    EDGE_CASE_RATE,
    EDGE_CASES,
    sample_edge_cases,
)
from meddies_pii.generation.label_corpus.prompts import targeted_user_prompt
from meddies_pii.languages import normalize_language


def test_edge_cases_are_a_rated_orthogonal_axis() -> None:
    assert EDGE_CASES
    assert 0 < EDGE_CASE_RATE < 1


def test_scribe_is_one_text_format_channel_not_three() -> None:
    assert "voice_ocr_transcription" not in MEDICAL_SCENARIO_NAMES
    assert "ambient voice scribe transcript" not in DOCUMENT_TYPES
    assert "voice-scribe transcript" in TEXT_FORMATS


def test_dialogue_transcript_text_format_added() -> None:
    assert "clinical dialogue transcript" in TEXT_FORMATS


def test_user_prompt_injects_all_edge_cases_when_given() -> None:
    """Every perturbation in the heavy regime is injected, not just the first."""
    prof = normalize_language("English")
    sc = SCENARIOS[0]
    one = ["SENTINEL-EDGE-1"]
    many = ["SENTINEL-EDGE-1", "SENTINEL-EDGE-2", "SENTINEL-EDGE-3"]

    def prompt(edge_cases: list[str]) -> str:
        return targeted_user_prompt(
            profile=prof,
            scenario=sc,
            document_type="d",
            text_format="f",
            min_spans=5,
            min_unique_labels=5,
            required_labels=(),
            edge_cases=edge_cases,
        )

    with_one = prompt(one)
    with_many = prompt(many)
    without = prompt([])

    assert "SENTINEL-EDGE-1" in with_one
    for marker in many:
        assert marker in with_many
    assert "SENTINEL-EDGE-1" not in without
    assert "perturbation" not in without.lower()


def test_edge_case_catalog_includes_real_world_hard_cases() -> None:
    blob = " ".join(EDGE_CASES).lower()
    assert "mask" in blob
    assert "no separator" in blob
    assert "truncate" in blob
    assert "mixing numerals" in blob
    assert len(EDGE_CASES) >= 9


def test_adversarial_scenarios_are_marked_and_delegate_to_catalog() -> None:
    """The 10-technique blob is gone from the prose.

    Techniques are now drawn from EDGE_CASES at generation time, not enumerated inside the scenario.

    """
    adv = _scenario_by_name("adversarial_obfuscation")
    fmt = _scenario_by_name("adversarial_formatting")
    assert adv.adversarial is True
    assert fmt.adversarial is True
    assert "(1)" not in adv.description
    assert "obfuscation techniques" not in adv.description.lower()
    assert _scenario_by_name("all_labels_dense").adversarial is False


def test_sample_edge_cases_draws_several_distinct_for_adversarial() -> None:
    # reason: the fixed seed pins the 30 draws below, so a range violation is reproducible.
    rng = random.Random(7)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    sc = _scenario_by_name("adversarial_obfuscation")
    low, high = ADVERSARIAL_EDGE_CASE_RANGE
    for _ in range(30):
        drawn = sample_edge_cases(rng, sc)
        assert low <= len(drawn) <= high
        assert len(set(drawn)) == len(drawn)
        assert all(ec in EDGE_CASES for ec in drawn)


def test_sample_edge_cases_is_light_and_rated_for_normal_scenarios() -> None:
    """Never more than one for a normal doc rated: both clean and lightly-perturbed docs occur."""
    # reason: the fixed seed pins the 300 draws below, so the `sizes <= {0, 1}` claim is a fact about
    # reason: this sequence rather than a probabilistic one that could flake.
    rng = random.Random(7)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    sc = _scenario_by_name("all_labels_dense")
    sizes = {len(sample_edge_cases(rng, sc)) for _ in range(300)}
    assert sizes <= {0, 1}
    assert sizes == {0, 1}
