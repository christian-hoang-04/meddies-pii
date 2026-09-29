from __future__ import annotations

from meddies_pii.training.bioes.eval.audit import find_label_candidates


def test_public_reference_url_is_not_a_private_url_candidate() -> None:
    candidates = find_label_candidates("Schema reference: https://example.com/clinical/schema.")

    assert not any(candidate.label == "private_url" for candidate in candidates)


def test_phone_pattern_requires_enough_digits_not_only_enough_separators() -> None:
    candidates = find_label_candidates("Call 1.......2 for help.")

    assert not any(candidate.label == "phone_number" for candidate in candidates)
