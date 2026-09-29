from __future__ import annotations

import hashlib
import inspect
import json
import re
from typing import TYPE_CHECKING, TypeGuard

import pytest

from meddies_pii import regex_runtime
from meddies_pii.eval_baseline.baseline.views import (
    MODEL_CORE_VIEW,
    MODEL_PLUS_REGEX_VIEW,
    REGEX_EVALUATION_VIEWS,
)
from meddies_pii.eval_baseline.regex_release.regex_fixtures import (
    LANGUAGE_SCOPED_LOCKED_NEGATIVES,
    LOCKED_CLINICAL_NEGATIVES,
    POSITIVE_SELECTION_FIXTURES,
)
from meddies_pii.regex_runtime import language as regex_language
from meddies_pii.regex_runtime import manifest as regex_manifest_module
from meddies_pii.regex_runtime import postprocess as regex_postprocess
from meddies_pii.regex_runtime import regex_manifest
from meddies_pii.regex_runtime import rules as regex_rules
from meddies_pii.regex_runtime.postprocess import apply_regex_postprocess
from meddies_pii.regex_runtime.rules import regex_candidates
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from types import ModuleType

NON_BREAKING_HYPHEN = "\N{NON-BREAKING HYPHEN}"


def _fixture_text(fixture_id: str) -> str:
    return next(fixture.text for fixture in POSITIVE_SELECTION_FIXTURES if fixture.fixture_id == fixture_id)


def _test_canonical_sha256(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _is_string_dict(value: object) -> TypeGuard[dict[str, str]]:
    return isinstance(value, dict) and all(isinstance(key, str) and isinstance(item, str) for key, item in value.items())


def test_high_precision_rules_add_only_explicit_candidate_families() -> None:
    text = (
        "Email: alpha@example.invalid; dien thoai: 0900 000 001; "
        "CCCD: 001099000001; lien ket rieng: "
        "https://portal.example.invalid/case?access_token=Abc12345; "
        "API key: sk_test_Abc12345."
    )

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [
        ("alpha@example.invalid", "email_address"),
        ("0900 000 001", "phone_number"),
        ("001099000001", "id_number"),
        (
            "https://portal.example.invalid/case?access_token=Abc12345",
            "private_url",
        ),
        ("sk_test_Abc12345", "secret"),
    ]


def test_locked_clinical_negatives_add_zero_false_positives() -> None:
    observed = {
        (fixture.fixture_id, variant): apply_regex_postprocess(variant, (), language=fixture.language).spans
        for fixture in (*LOCKED_CLINICAL_NEGATIVES, *LANGUAGE_SCOPED_LOCKED_NEGATIVES)
        for variant in _negative_variants(fixture.text)
    }

    assert observed == dict.fromkeys(observed, ())


def _negative_variants(text: str) -> tuple[str, ...]:
    variants = {text, text.upper(), text.title()}
    for match in re.finditer(r"(?<=\w)[ \t\u00a0]+(?=\w)", text):
        variants.update(text[: match.start()] + hyphen + text[match.end() :] for hyphen in ("-", NON_BREAKING_HYPHEN))
    return tuple(sorted(variants))


def test_dense_candidate_overlap_resolution_is_linear_in_emitted_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Hospital Santa. " * 512
    overlap_calls = 0
    original_overlaps = regex_rules._overlaps

    def counted_overlaps(left: CharSpan, right: CharSpan) -> bool:
        nonlocal overlap_calls
        overlap_calls += 1
        return original_overlaps(left, right)

    monkeypatch.setattr(regex_rules, "_overlaps", counted_overlaps)

    candidates = regex_candidates(text, language="es")

    assert len(candidates) == 512
    assert overlap_calls <= len(candidates) * 3


def test_normalized_match_maps_back_to_exact_original_codepoints() -> None:
    text = "Email: ﬁ@example.invalid"

    (span,) = apply_regex_postprocess(text, ()).spans

    assert (span.start, span.end, span.text, span.label) == (
        7,
        len(text),
        "ﬁ@example.invalid",
        "email_address",
    )


def test_different_label_overlap_preserves_model_and_records_conflict() -> None:
    text = "Contact alpha@example.invalid or phone: 0900 000 001."
    model_span = CharSpan(8, 13, "alpha", "human_name")

    result = apply_regex_postprocess(text, (model_span,))

    assert result.spans == (
        model_span,
        CharSpan(13, 29, "@example.invalid", "email_address"),
        CharSpan(40, 52, "0900 000 001", "phone_number"),
    )
    assert len(result.conflicts) == 1
    assert result.conflicts[0].reason == "different_label_trimmed_remainder"


def test_exact_duplicate_collapses_without_replacing_model_span() -> None:
    text = "Email: alpha@example.invalid"
    model_span = CharSpan(7, 28, "alpha@example.invalid", "email_address")

    result = apply_regex_postprocess(text, (model_span,))

    assert result.spans == (model_span,)
    assert result.conflicts == ()


def test_same_label_overlap_unions_only_to_a_valid_family_value() -> None:
    text = "Email: alpha@example.invalid"
    partial_model_span = CharSpan(7, 20, "alpha@example", "email_address")

    result = apply_regex_postprocess(text, (partial_model_span,))

    assert result.spans == (CharSpan(7, 28, "alpha@example.invalid", "email_address"),)


def test_private_url_spans_a_parameter_that_precedes_the_sensitive_key() -> None:
    text = _fixture_text("private-url-parameter-before-the-key")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [(text, "private_url")]


def test_private_url_spans_a_parameter_that_follows_the_sensitive_key() -> None:
    text = _fixture_text("private-url-parameter-after-the-key")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [(text, "private_url")]


def test_private_url_rejects_a_seven_character_sensitive_value() -> None:
    text = _fixture_text("private-url-token-below-the-minimum")

    assert regex_candidates(text) == ()


def test_private_url_accepts_an_eight_character_sensitive_value() -> None:
    text = _fixture_text("private-url-token-at-the-minimum")

    assert [(candidate.span.text, candidate.span.label) for candidate in regex_candidates(text)] == [(text, "private_url")]


def test_private_url_stops_at_the_whitespace_that_ends_the_link() -> None:
    text = _fixture_text("private-url-then-prose")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("https://p.test/r?token=abc123xyz456", "private_url")]


def test_email_does_not_reach_across_an_hl7_field_separator() -> None:
    text = _fixture_text("email-in-an-hl7-segment")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user@host.test", "email_address")]


def test_email_rule_leaves_a_correct_model_span_on_an_hl7_line_unchanged() -> None:
    text = _fixture_text("email-in-an-hl7-segment")
    model_span = CharSpan(9, len(text), "user@host.test", "email_address")

    result = apply_regex_postprocess(text, (model_span,))

    assert result.spans == (model_span,)
    assert result.conflicts == ()


def test_email_stops_at_a_url_query_parameter_boundary() -> None:
    text = _fixture_text("email-in-a-url-query-parameter")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user@host.test", "email_address")]


def test_email_does_not_reach_across_an_hl7_subcomponent_separator() -> None:
    text = _fixture_text("email-after-an-hl7-subcomponent-separator")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user@host.test", "email_address")]


def test_email_does_not_reach_across_an_hl7_repetition_separator() -> None:
    text = _fixture_text("email-after-an-hl7-repetition-separator")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user@host.test", "email_address")]


def test_email_rule_leaves_a_model_span_inside_a_url_query_unchanged() -> None:
    text = _fixture_text("email-in-a-url-query-parameter")
    start = text.index("user@host.test")
    model_span = CharSpan(start, start + len("user@host.test"), "user@host.test", "email_address")

    result = apply_regex_postprocess(text, (model_span,))

    assert result.spans == (model_span,)
    assert result.conflicts == ()


def test_private_url_stops_before_ampersand_joined_prose() -> None:
    text = _fixture_text("private-url-then-ampersand-prose")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("https://p.test/r?token=abc123xyz456", "private_url")]


def test_private_url_stops_before_an_ampersand_joined_initialism() -> None:
    text = _fixture_text("private-url-then-ampersand-initialism")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("https://p.test/r?token=abc123xyz456", "private_url")]


def test_private_url_spans_every_trailing_name_value_parameter() -> None:
    text = _fixture_text("private-url-with-two-trailing-parameters")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [(text, "private_url")]


def test_email_starts_after_a_leading_quote() -> None:
    text = _fixture_text("email-in-single-quotes")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user@host.test", "email_address")]


def test_email_starts_after_a_leading_asterisk() -> None:
    text = _fixture_text("email-between-asterisks")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user@host.test", "email_address")]


def test_email_starts_after_a_leading_hash() -> None:
    text = _fixture_text("email-after-a-hash")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user@host.test", "email_address")]


def test_email_starts_after_leading_dots() -> None:
    text = _fixture_text("email-after-leading-dots")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user@host.test", "email_address")]


def test_email_rule_leaves_a_quoted_model_span_unchanged() -> None:
    text = _fixture_text("email-in-single-quotes")
    start = text.index("user@host.test")
    model_span = CharSpan(start, start + len("user@host.test"), "user@host.test", "email_address")

    result = apply_regex_postprocess(text, (model_span,))

    assert result.spans == (model_span,)
    assert result.conflicts == ()


def test_email_rejects_a_local_part_ending_in_a_dot() -> None:
    text = _fixture_text("email-local-part-ending-in-a-dot")

    result = apply_regex_postprocess(text, ())

    assert result.spans == ()


def test_email_accepts_a_local_part_ending_in_a_hyphen() -> None:
    text = _fixture_text("email-local-part-ending-in-a-hyphen")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user-@host.test", "email_address")]


def test_private_url_chains_past_a_parameter_with_no_value() -> None:
    text = _fixture_text("private-url-with-an-empty-parameter")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [(text, "private_url")]


def test_private_url_includes_a_trailing_parameter_with_no_value() -> None:
    text = _fixture_text("private-url-ending-in-an-empty-parameter")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [(text, "private_url")]


def test_email_accepts_a_local_part_starting_in_an_underscore() -> None:
    text = _fixture_text("email-local-part-starting-in-an-underscore")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("_user@host.test", "email_address")]


def test_email_keeps_a_word_interior_underscore_inside_one_span() -> None:
    text = _fixture_text("email-local-part-with-an-interior-underscore")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("source_user@host.test", "email_address")]


def test_email_starts_after_a_prose_ellipsis() -> None:
    text = _fixture_text("email-after-a-prose-ellipsis")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("user@host.test", "email_address")]


def test_email_rule_leaves_a_model_span_after_an_ellipsis_unchanged() -> None:
    text = _fixture_text("email-after-a-prose-ellipsis")
    start = text.index("user@host.test")
    model_span = CharSpan(start, start + len("user@host.test"), "user@host.test", "email_address")

    result = apply_regex_postprocess(text, (model_span,))

    assert result.spans == (model_span,)
    assert result.conflicts == ()


def test_email_with_consecutive_dots_matches_only_the_valid_suffix() -> None:
    text = _fixture_text("email-local-part-with-consecutive-dots")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [("b@host.test", "email_address")]


def test_email_over_the_length_limit_matches_only_the_valid_suffix() -> None:
    text = _fixture_text("email-over-the-length-limit-with-a-valid-suffix")

    assert [(candidate.span.text, candidate.span.label) for candidate in regex_candidates(text)] == [
        ("bbbbbbbbbb@host.test", "email_address"),
    ]


def test_email_keeps_a_dot_separated_local_part_whole() -> None:
    text = _fixture_text("email-local-part-with-two-atoms")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [(text, "email_address")]


def test_email_keeps_a_plus_tagged_local_part_whole() -> None:
    text = _fixture_text("email-local-part-with-a-plus-tag")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [(text, "email_address")]


def test_private_url_chains_past_a_punctuation_only_value() -> None:
    text = _fixture_text("private-url-with-a-punctuation-only-value")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [(text, "private_url")]


def test_private_url_drops_a_punctuation_only_value_before_prose() -> None:
    text = _fixture_text("private-url-punctuation-only-value-then-prose")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [
        ("https://p.test/r?token=abc123xyz456&view=", "private_url"),
    ]


def test_email_length_bound_matches_the_validator() -> None:
    """The pattern's 64-character bound and _valid_email's must agree.

    The bound exists to stop a long punctuation run from restarting the scan at every
    position, not to filter addresses. If either side moves alone, one of these two
    assertions fails rather than the pair drifting silently.
    """
    at_limit = _fixture_text("email-local-part-at-the-length-limit")
    over_limit = _fixture_text("email-local-part-over-the-length-limit")

    assert [(span.text, span.label) for span in apply_regex_postprocess(at_limit, ()).spans] == [
        (at_limit, "email_address"),
    ]
    assert apply_regex_postprocess(over_limit, ()).spans == ()


def test_private_url_does_not_end_on_a_full_stop() -> None:
    text = _fixture_text("private-url-ending-a-sentence")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [
        ("https://p.test/r?token=abc123xyz456&view=pdf", "private_url"),
    ]


def test_private_url_does_not_end_on_a_comma_mid_sentence() -> None:
    text = _fixture_text("private-url-inside-a-sentence")

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [
        ("https://p.test/r?token=abc123xyz456&view=pdf", "private_url"),
    ]


def test_private_url_stops_before_a_parenthesized_duplicate() -> None:
    url = "https://portal.haeon-test.kr/patient/PK-8831?session=st-4f9a&auth=Qm93Lk9r"
    text = f"{url}({url}"

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [
        (url, "private_url"),
        (url, "private_url"),
    ]


def test_private_url_keeps_parentheses_that_are_part_of_one_value() -> None:
    text = "https://p.test/r?token=abc(123)xyz"

    result = apply_regex_postprocess(text, ())

    assert [(span.text, span.label) for span in result.spans] == [(text, "private_url")]


def test_regex_ablation_has_two_explicit_paired_views() -> None:
    assert REGEX_EVALUATION_VIEWS == (MODEL_CORE_VIEW, MODEL_PLUS_REGEX_VIEW)


def test_regex_manifest_hashes_every_frozen_policy_component() -> None:
    manifest = regex_manifest()
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    component_sha256 = manifest["component_sha256"]
    manifest_sha256 = manifest["sha256"]
    assert _is_string_dict(component_sha256)
    assert isinstance(manifest_sha256, str)

    assert set(component_sha256) == {
        "rules",
        "validators",
        "normalization",
        "offset_map",
        "boundary_policy",
        "applicability_policy",
        "overlap_policy",
        "packs",
        "rules_executable",
        "postprocess_executable",
        "language_executable",
        "runtime_executable",
    }
    assert all(len(digest) == 64 for digest in component_sha256.values())
    assert len(manifest_sha256) == 64
    assert _test_canonical_sha256(payload) == manifest_sha256
    assert (
        component_sha256["rules_executable"] == hashlib.sha256(inspect.getsource(regex_rules).encode("utf-8")).hexdigest()
    )
    assert (
        component_sha256["postprocess_executable"]
        == hashlib.sha256(inspect.getsource(regex_postprocess).encode("utf-8")).hexdigest()
    )
    assert (
        component_sha256["language_executable"]
        == hashlib.sha256(inspect.getsource(regex_language).encode("utf-8")).hexdigest()
    )
    assert (
        component_sha256["runtime_executable"]
        == hashlib.sha256(inspect.getsource(regex_runtime).encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize("component", ["language_executable", "runtime_executable"])
def test_regex_manifest_digest_rotates_when_runtime_component_digest_changes(
    component: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = regex_manifest()
    target_module = {
        "language_executable": regex_language,
        "runtime_executable": regex_runtime,
    }[component]
    source_sha256 = regex_manifest_module._source_sha256

    def changed_source_sha256(module: ModuleType) -> str:
        if module is target_module:
            return "0" * 64
        return source_sha256(module)

    monkeypatch.setattr(regex_manifest_module, "_source_sha256", changed_source_sha256)
    rotated = regex_manifest()
    baseline_components = baseline["component_sha256"]
    rotated_components = rotated["component_sha256"]
    assert _is_string_dict(baseline_components)
    assert _is_string_dict(rotated_components)

    assert rotated_components[component] != baseline_components[component]
    assert rotated["sha256"] != baseline["sha256"]


def test_regex_manifest_records_cross_label_remainder_policy() -> None:
    assert regex_manifest()["model_overlap"] == (
        "collapse exact duplicate; add non-overlap; same-label union only when the "
        "family validator accepts; different-label model span preserved and conflict "
        "recorded; structured labels emit non-overlapping remainders of at least 3 "
        "chars; other labels, including company_name, discard the candidate"
    )
