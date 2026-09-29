from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from meddies_pii.generation.openai_compatible import quota
from meddies_pii.generation.openai_compatible.quota import (
    AsyncRateLimiter,
    DailyTokenBudget,
    billable_tokens,
)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({}, None),
        ({"total_tokens": "100"}, None),
        ({"total_tokens": 100}, 100),
        (
            {
                "total_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": -1},
            },
            100,
        ),
        (
            {
                "total_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 200},
            },
            25,
        ),
    ],
)
def test_billable_tokens_handles_missing_and_untrusted_usage_details(
    usage: dict[str, object],
    expected: int | None,
) -> None:
    assert billable_tokens(usage) == expected


def test_uncapped_providers_do_not_accrue_budget_usage() -> None:
    day = date(2026, 7, 29)
    budget = DailyTokenBudget({"paid": 100})

    budget.check_and_add("local", 500, today=day)
    budget.reconcile("local", reserved=500, actual=900, today=day)

    assert budget.spent("local", day) == 0


def test_reconcile_replaces_a_capped_provider_reservation() -> None:
    day = date(2026, 7, 29)
    budget = DailyTokenBudget({"paid": 100})
    budget.check_and_add("paid", 80, today=day)

    budget.reconcile("paid", reserved=80, actual=30, today=day)
    assert budget.spent("paid", day) == 30

    budget.reconcile("paid", reserved=80, actual=0, today=day)
    assert budget.spent("paid", day) == 0


def test_rate_limiter_waits_until_a_token_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    immediate = AsyncRateLimiter(requests_per_minute=60)
    delayed = AsyncRateLimiter(requests_per_minute=60)
    delayed.tokens = 0
    delayed.last_update = 0
    clock = iter([0.25, 1.0])
    sleeps: list[float] = []

    monkeypatch.setattr(quota, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    # reason: this replaces `asyncio.sleep` inside the quota module, which `quota.py:194` awaits when it throttles.
    async def record_sleep(seconds: float) -> None:  # ruff: ignore[unused-async]
        sleeps.append(seconds)

    monkeypatch.setattr(quota.asyncio, "sleep", record_sleep)

    async def exercise() -> None:
        await immediate.wait()
        await delayed.wait()

    asyncio.run(exercise())

    assert immediate.tokens == 59
    assert delayed.tokens == 0
    assert sleeps == [0.75]
