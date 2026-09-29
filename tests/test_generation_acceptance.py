"""Phase-B junk-gate contract: templated / ordered-digit phone spans are rejected.

The 2026-06-20 audit found phone spans that are lazy digit dumps — ordered runs
like "0123456789" or all-same-digit — produced under span-count pressure. These
are not plausible numbers and must be rejected. A low-floor test scenario isolates
the new gate from the (separate) span-count acceptance checks.

The EDGE_CASES axis instructs unicode-digit obfuscation (subscript / superscript / circled). The junk gate extracts digits
with int(), but str.isdigit() admits glyphs int() rejects ('₈', '①') -> ValueError. The gate must tolerate these as a
normal (non-templated) span, not crash the document.

Correctness-only acceptance: a well-formed 2-span doc is kept (was rejected under the old fixed span floor). Density
diversity is wanted, not penalized.

The applied edge cases are auditable on the accepted corpus (a list, since the adversarial regime applies several), not
just on the raw attempts.

"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import random
from typing import TYPE_CHECKING

from meddies_pii.generation.label_corpus.catalog import (
    SCENARIOS,
    SPAN_TARGET_RANGE,
    sample_span_target,
)
from meddies_pii.generation.label_corpus.validate import validate_tagged_document

if TYPE_CHECKING:
    from meddies_pii.generation.label_corpus.validate import ValidationResult

_PAD = (
    "Discharge note for the outpatient clinic follow-up visit. The attending "
    "physician reviewed the chart and confirmed the ongoing care plan in detail "
    "with the multidisciplinary care team before the patient was finally "
    "discharged from the ward late that afternoon, with instructions to follow up."
)
"""English skips the native-script marker; pad past the 240-char minimum."""


def _doc(*spans: str) -> str:
    return f"{_PAD} " + " ".join(spans) + " End of clinical note for the record."


def _validate(doc: str) -> ValidationResult:
    return validate_tagged_document(doc, language="English", required_labels=())


def test_ordered_digit_phone_is_rejected_as_templated() -> None:
    v = _validate(_doc("Patient [John Smith]<human_name>", "call [0123456789]<phone_number>"))
    assert not v.ok
    assert "templated_phone_span" in v.errors


def test_all_same_digit_phone_is_rejected_as_templated() -> None:
    v = _validate(_doc("Patient [John Smith]<human_name>", "call [5555555555]<phone_number>"))
    assert "templated_phone_span" in v.errors


def test_realistic_phone_is_not_flagged_templated() -> None:
    v = _validate(_doc("Patient [John Smith]<human_name>", "call [+1 415 555 0132]<phone_number>"))
    assert "templated_phone_span" not in v.errors
    assert v.ok, v.errors


def test_unicode_digit_phone_does_not_crash_junk_gate() -> None:
    v = _validate(_doc("Patient [John Smith]<human_name>", "call [+1 415 555 01₈2]<phone_number>"))
    assert "templated_phone_span" not in v.errors


def test_correct_doc_with_two_spans_is_accepted() -> None:
    v = _validate(_doc("Patient [John Smith]<human_name>", "phone [+1 415 555 0132]<phone_number>"))
    assert v.ok, v.errors


def test_two_spans_same_label_not_rejected_for_low_diversity() -> None:
    v = _validate(_doc("Dr [Jane Roe]<human_name>", "nurse [Mary Lin]<human_name>"))
    assert v.ok, v.errors
    assert "too_few_unique_labels" not in v.errors


def test_sample_span_target_in_range_and_feasible() -> None:
    # reason: a fixed seed is what makes this test reproducible — an unseeded or cryptographic
    # reason: source would turn a failure into one nobody can re-run.
    rng = random.Random(0)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    sc = SCENARIOS[0]
    for _ in range(50):
        target = sample_span_target(rng, sc, required_labels=("private_url", "secret"))
        assert target <= SPAN_TARGET_RANGE[1]
        assert target >= sc.min_unique_labels


def test_accepted_record_carries_edge_cases_for_observability() -> None:
    from meddies_pii.generation.label_corpus.catalog import SCENARIOS
    from meddies_pii.generation.label_corpus.validate import accepted_record

    v = _validate(_doc("Patient [John Smith]<human_name>", "phone [+1 415 555 0132]<phone_number>"))
    rec = accepted_record(
        validation=v,
        language="English",
        scenario=SCENARIOS[0],
        document_type="d",
        text_format="f",
        model="m",
        provider="p",
        attempt_index=1,
        edge_cases=["EDGE-X", "EDGE-Y"],
    )
    assert rec["info"]["edge_cases"] == ["EDGE-X", "EDGE-Y"]
