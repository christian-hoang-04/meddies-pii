from __future__ import annotations

from meddies_pii.regex_runtime.rules import REGEX_RULES, regex_candidates


def test_every_shape_validated_rule_carries_the_authoritative_tier() -> None:
    shape_rules = {
        "email",
        "vietnam_phone",
        "cccd",
        "private_access_url",
        "explicit_secret",
    }

    assert {rule.name: rule.tier for rule in REGEX_RULES if rule.name in shape_rules} == dict.fromkeys(shape_rules, "AUTH")


def test_candidates_report_the_tier_and_rule_that_produced_them() -> None:
    text = "Email: alpha@example.invalid"

    (candidate,) = regex_candidates(text)

    assert (candidate.tier, candidate.rule_name) == ("AUTH", "email")
    assert (candidate.span.start, candidate.span.end, candidate.span.label) == (
        7,
        28,
        "email_address",
    )


def test_regex_validators_do_not_cache_document_text() -> None:
    assert all(not hasattr(rule.validator, "cache_info") for rule in REGEX_RULES)
