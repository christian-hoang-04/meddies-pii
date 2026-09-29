"""Characterization tests for OpenAI-compatible quota accounting."""

from __future__ import annotations

from datetime import date

import pytest

from meddies_pii.exceptions import DailyBudgetExceeded
from meddies_pii.generation.openai_compatible.quota import (
    DailyTokenBudget,
    billable_tokens,
)


def test_quota_accounting_preserves_cached_input_discount_and_daily_guard() -> None:
    budget = DailyTokenBudget({"openai": 1_000})
    day = date(2026, 6, 11)
    usage = {
        "total_tokens": 1_000,
        "prompt_tokens_details": {"cached_tokens": 400},
    }

    assert billable_tokens(usage) == 700
    budget.check_and_add("openai", 700, today=day)
    with pytest.raises(DailyBudgetExceeded, match="openai"):
        budget.check_and_add("openai", 301, today=day)
