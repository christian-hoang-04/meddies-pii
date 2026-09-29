"""Token estimates, daily budgets, and request-rate limiting.

The client uses these calculations to reserve provider spend before a request,
reconcile actual usage afterward, and pace calls across concurrent tasks.

UTF-8 bytes//3 is a safe upper bound across scripts (CJK ~3 bytes/char ~= 1 token/char; Latin ~1 byte/char). Plus a small
per-message chat overhead.

"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

from meddies_pii.exceptions import DailyBudgetExceeded

_CHARS_PER_TOKEN = 4


_TYPICAL_COMPLETION_TOKENS = 1024
"""Realistic completion length for rate-limit (TPM/TPD) gating.

The clinical documents are short (~300-800 tokens); the conservative `max_tokens` budget (4096) over-states a request ~2x,
which would false-skip tight-TPM providers (groq TPM 6000) before they send a single request. The ledger records ACTUAL raw
tokens after each call, so the running TPM/TPD stays exact — this only needs to predict the typical cost well enough to
gate.

"""


def _estimate_prompt_tokens(system: str, user: str) -> int:
    return len((system + user).encode("utf-8")) // 3 + 8


def _estimate_request_tokens(system: str, user: str, max_tokens: int) -> int:
    """Conservative pre-call token estimate for the zero-spend budget guard.

    Prompt tokens are over-counted and the full `max_tokens` completion budget is
    assumed spent. Over-counting is the safe direction for a hard COST cap: it
    can only stop a call *earlier* than the true cost would, never later.

    Returns:
        The over-counted prompt estimate plus the whole ``max_tokens`` completion allowance, as
        though every call ran to its limit. This is checked and reserved before the request, so
        a refusal means zero spend rather than a spend that is later found to have crossed the
        cap.

    """
    return _estimate_prompt_tokens(system, user) + max_tokens


def _estimate_rate_limit_tokens(system: str, user: str, max_tokens: int) -> int:
    """Estimate pre-call tokens realistically for the ledger's TPM/TPD gate.

    The estimate is prompt plus a
    *typical* completion, not the full max_tokens budget (see the constant).

    Returns:
        The prompt estimate plus whichever is smaller, ``max_tokens`` or the typical completion
        size. Deliberately lower than the cost estimate: the rate-limit gate only decides
        whether to wait, so over-counting here would false-skip a provider with a tight TPM
        that could in fact have taken the request.

    """
    return _estimate_prompt_tokens(system, user) + min(max_tokens, _TYPICAL_COMPLETION_TOKENS)


_CACHED_INPUT_BILLING_RATE = 0.25
"""OpenAI bills cached input tokens at 25% of the regular input price (GPT-5.x family.

Verified 2026-06 against OpenAI pricing). `usage.total_tokens` counts cached prefixes at full weight, so a workload that
reuses one large system prompt (synthetic generation) reports throughput ~1.5x its charged tokens.

"""


def billable_tokens(usage: Mapping[str, object]) -> int | None:
    """Charged-token count for one response, discounting cached input.

    The daily budget guards an *allowance* (free-tier daily grant / paid spend),
    so it must meter what the provider charges, not raw model throughput. Cached
    prompt tokens are billed at ``_CACHED_INPUT_BILLING_RATE`` of full price but
    appear in ``total_tokens`` at full weight; this removes that discount. Returns
    ``None`` when the response carries no integer ``total_tokens`` (nothing to
    reconcile).

    Returns:
        The charged-token count with the cached-prefix discount applied, or ``None`` when the
        response carries no integer ``total_tokens``. ``None`` means unreconcilable, not zero:
        the caller keeps its pre-call estimate rather than crediting the budget back, so a
        provider that omits usage cannot make a call look free.

    """
    total = usage.get("total_tokens")
    if not isinstance(total, int):
        return None
    details = usage.get("prompt_tokens_details")
    cached_raw = cast("dict[str, object]", details).get("cached_tokens") if isinstance(details, dict) else None
    cached = cached_raw if isinstance(cached_raw, int) and cached_raw >= 0 else 0
    cached = min(cached, total)
    discount = round(cached * (1.0 - _CACHED_INPUT_BILLING_RATE))
    return max(0, total - discount)


class DailyTokenBudget:
    """Per-provider daily token cap — zero-spend protection for paid providers.

    Tracks tokens spent per (provider, UTC date). `check_and_add` raises
    `DailyBudgetExceeded` when the add would push that day's total past the
    provider's cap, so the caller stops before spending. Providers absent
    from the cap table are uncapped (free tiers, local endpoints).

    Caps are injected, not hardcoded: real provider numbers come from a
    separate research pass and land as a constant later.

    Wired into `OpenAICompatibleClient.chat()` as a pre-call reservation with post-call \
reconciliation when provider usage metadata is returned.
    """

    def __init__(self, caps: dict[str, int]) -> None:
        self._caps: dict[str, int] = dict(caps)
        self._usage: dict[tuple[str, date], int] = {}

    def spent(self, provider: str, day: date) -> int:
        return self._usage.get((provider, day), 0)

    def check_and_add(self, provider: str, tokens: int, *, today: date | None = None) -> None:
        cap = self._caps.get(provider)
        if cap is None:
            return
        day = today if today is not None else datetime.now(UTC).date()
        projected = self.spent(provider, day) + tokens
        if projected > cap:
            msg = f"Daily token budget exceeded for provider {provider!r}."
            raise DailyBudgetExceeded(
                msg,
                context={
                    "provider": provider,
                    "utc_date": day.isoformat(),
                    "cap": cap,
                    "spent": self.spent(provider, day),
                    "requested": tokens,
                },
            )
        self._usage[provider, day] = projected

    def reconcile(
        self,
        provider: str,
        *,
        reserved: int,
        actual: int,
        today: date | None = None,
    ) -> None:
        """Replace a pre-call reservation with the real token count.

        Called once the
        response is in. Adjusts the day's tally by ``actual - reserved`` so the
        hard cap tracks true spend instead of the conservative estimate (which
        assumes the full ``max_tokens`` completion every call). No-op for
        uncapped providers — they reserve nothing, so there is nothing to true
        up. The pre-call zero-spend guarantee is unaffected: this only corrects
        the running total *after* a call has already succeeded.
        """
        if provider not in self._caps:
            return
        day = today if today is not None else datetime.now(UTC).date()
        adjusted = self.spent(provider, day) + (actual - reserved)
        self._usage[provider, day] = max(0, adjusted)


class AsyncRateLimiter:
    """Token-bucket rate limiter for async API calls."""

    def __init__(self, requests_per_minute: int = 100) -> None:
        self.rate: float = requests_per_minute / 60.0
        self.capacity: int = requests_per_minute
        self.tokens: float = float(self.capacity)
        self.last_update: float = time.monotonic()
        self.lock: asyncio.Lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            while self.tokens < 1:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(float(self.capacity), self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens < 1:
                    wait_time = (1 - self.tokens) / self.rate
                    await asyncio.sleep(wait_time)

            self.tokens -= 1
