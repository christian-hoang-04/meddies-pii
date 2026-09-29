"""Per-account rate-limit ledger shared across one daily run.

Each free-tier account (one API key, for one model) carries up to four
independent limits — requests/minute (RPM), requests/day (RPD), tokens/minute
(TPM), tokens/day (TPD). Providers enforce every axis and return 429 (then ban
repeat offenders) when any is exceeded, so a run that wants to *maximize* daily
quota must respect all four. The ledger is created once per run and gates every
request: given an account and a token estimate it answers "send now", "wait N
seconds", or "this account is done for the day" (a daily ceiling is reached).

Token axes count RAW provider tokens (what the provider meters for TPM/TPD), not
the cost-discounted figure used for paid-spend budgeting. A run lives within a
day, so the daily windows (sliding 24h) cap in-run usage at each account's daily
ceiling. The clock is injectable so the windowing logic is unit-testable without
sleeping.
"""

from __future__ import annotations

import operator
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_MINUTE = 60.0
_DAY = 86_400.0


@dataclass(frozen=True)
class AccountLimits:
    """Per-account limits for one provider/model. ``None`` = that axis unbounded.

    ``rpm``/``rpd`` are requests per minute/day; ``tpm``/``tpd`` are tokens per
    minute/day (raw provider tokens, the figure the provider meters).
    """

    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None


class AccountLedger:
    def __init__(
        self,
        limits: Mapping[str, AccountLimits],
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Request timestamps per (provider, account) -> RPM (60s) / RPD (24h).

        (timestamp, tokens) per (provider, account) -> TPM (60s) / TPD (24h).

        (provider, account) -> monotonic time the account is blocked until, set by trip() after a rate-limit rejection. inf
        = blocked for the run.

        """
        self._limits = dict(limits)
        self._now = now
        self._requests: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._token_events: dict[tuple[str, str], deque[tuple[float, int]]] = defaultdict(deque)
        self._req_count: dict[tuple[str, str], int] = defaultdict(int)
        self._tokens: dict[tuple[str, str], int] = defaultdict(int)
        self._seen: dict[str, list[str]] = defaultdict(list)
        self._blocked_until: dict[tuple[str, str], float] = {}

    def _usage(self, key: tuple[str, str], now: float) -> tuple[int, int, int, int]:
        """(rpm, rpd, tpm, tpd) currently inside the windows for one account.

        Prunes events older than 24h as a side effect (keeps the deques bounded).

        Returns:
            Requests in the last 60s, requests in the last 24h, tokens in the last 60s and
            tokens in the last 24h, in that order. The daily figures are exact rather than
            sampled because pruning runs first, so the deques hold only the last 24h. Token
            counts are RAW provider tokens, never the cost-discounted figure.

        """
        reqs = self._requests[key]
        while reqs and reqs[0] <= now - _DAY:
            reqs.popleft()
        toks = self._token_events[key]
        while toks and toks[0][0] <= now - _DAY:
            toks.popleft()
        rpm = sum(1 for ts in reqs if ts > now - _MINUTE)
        tpm = sum(t for ts, t in toks if ts > now - _MINUTE)
        tpd = sum(t for _, t in toks)
        return rpm, len(reqs), tpm, tpd

    def _blocked_wait(self, key: tuple[str, str], now: float) -> float | None:
        """Remaining block for a trip()'d account.

        Returns:
            ``None`` when the account is free to send, ``inf`` when ``trip()`` blocked it for
            the rest of the run, else the seconds left on a transient cooldown. The three
            cases are distinct on purpose: ``choose`` treats ``inf`` like a daily ceiling and
            drops the account, while a finite wait keeps it in rotation as merely throttled.

        """
        until = self._blocked_until.get(key)
        if until is None or now >= until:
            return None
        return float("inf") if until == float("inf") else until - now

    # reason: now and blocked wait share AccountLedger's state; extraction would desync retries and counters.
    def choose(self, provider: str, key_ids: Sequence[str], est_tokens: int = 0) -> tuple[str | None, float]:  # ruff: ignore[complex-structure,too-many-branches]
        """Pick an account for a request of ~``est_tokens`` and the wait before it.

        Returns ``(account_id, wait_seconds)``. ``account_id`` is ``None`` only
        when every account has hit a DAILY ceiling (RPD or TPD) — the caller
        should stop using the provider for the day. When accounts are only at a
        per-minute ceiling (RPM or TPM) the soonest-free one is returned with a
        positive wait. Among accounts that can send now, the least daily-loaded
        one is chosen so usage spreads evenly across accounts.

        A trip()'d account: inf -> out for the run (like a daily ceiling); a finite cooldown -> throttled, routed around.

        Returns:
            The account to send on and the seconds to wait first. A ``None`` account means
            every account has hit a DAILY ceiling and the caller should stop using the
            provider for the day -- it never means "wait and retry", which is the positive-wait
            case. A returned wait of 0.0 with an account means send now. When the provider has
            no configured limits the least daily-loaded unblocked account comes back at 0.0,
            so an unlimited provider still spreads load rather than pinning one account.

        """
        now = self._now()
        limits = self._limits.get(provider)
        if limits is None:
            usable = [k for k in key_ids if self._blocked_wait((provider, k), now) is None]
            if not usable:
                return None, 0.0
            best = min(usable, key=lambda k: self._usage((provider, k), now)[1])
            return best, 0.0

        ready: list[tuple[str, int, int]] = []
        throttled: list[tuple[str, float]] = []
        for account in key_ids:
            blocked = self._blocked_wait((provider, account), now)
            if blocked is not None:
                if blocked != float("inf"):
                    throttled.append((account, blocked))
                continue
            rpm, rpd, tpm, tpd = self._usage((provider, account), now)
            if limits.rpd is not None and rpd >= limits.rpd:
                continue
            if limits.tpd is not None and tpd + est_tokens > limits.tpd:
                continue
            if limits.tpm is not None and est_tokens > limits.tpm:
                continue
            wait = 0.0
            if limits.rpm is not None and rpm >= limits.rpm:
                wait = max(wait, self._request_minute_wait((provider, account), now))
            if limits.tpm is not None and tpm + est_tokens > limits.tpm:
                wait = max(
                    wait,
                    self._token_minute_wait((provider, account), now, est_tokens, limits.tpm),
                )
            if wait <= 0:
                ready.append((account, rpd, tpd))
            else:
                throttled.append((account, wait))

        if ready:
            chosen = min(ready, key=operator.itemgetter(1, 2))
            return chosen[0], 0.0
        if throttled:
            account, wait = min(throttled, key=operator.itemgetter(1))
            return account, max(0.0, wait)
        return None, 0.0

    def _request_minute_wait(self, key: tuple[str, str], now: float) -> float:
        in_minute = [ts for ts in self._requests[key] if ts > now - _MINUTE]
        return (in_minute[0] + _MINUTE) - now if in_minute else 0.0

    def _token_minute_wait(self, key: tuple[str, str], now: float, est_tokens: int, tpm: int) -> float:
        """Seconds until enough token events age out of the 60s window.

        Returns:
            The wait until enough token events leave the trailing 60s window for
            ``est_tokens`` to fit under ``tpm``, never negative. 0.0 when nothing is in the
            window. The scan credits events in insertion order and stops at the first one
            that frees enough, so the answer is the EARLIEST sufficient wait rather than the
            wait for the whole window to clear.

        """
        events = [(ts, t) for ts, t in self._token_events[key] if ts > now - _MINUTE]
        if not events:
            return 0.0
        need_to_free = sum(t for _, t in events) + est_tokens - tpm
        freed = 0
        for ts, tokens in events:
            freed += tokens
            if freed >= need_to_free:
                return max(0.0, (ts + _MINUTE) - now)
        return max(0.0, (events[-1][0] + _MINUTE) - now)

    def reserve(self, provider: str, account: str) -> None:
        self._mark_seen(provider, account)
        self._requests[provider, account].append(self._now())
        self._req_count[provider, account] += 1

    def trip(self, provider: str, account: str, cooldown: float | None = None) -> None:
        """Take an account out of rotation after a rate-limit rejection.

        ``cooldown=None`` blocks it for the rest of the run — a HARD daily-cap
        signal (Cloudflare 4006, OpenRouter free-models-per-day, an RPD 429) that
        won't clear before the provider's next daily reset, after this run ends. A
        positive ``cooldown`` (e.g. a ``Retry-After`` value) blocks it transiently
        so ``choose`` routes around it, then frees it. This stops the ledger from
        hammering a capped account — the sustained-429 pattern that gets free-tier
        accounts flagged.
        """
        self._mark_seen(provider, account)
        until = float("inf") if cooldown is None else self._now() + cooldown
        self._blocked_until[provider, account] = until

    def add_tokens(self, provider: str, account: str, tokens: int) -> None:
        """Record RAW provider tokens for one completed request (feeds TPM/TPD)."""
        self._mark_seen(provider, account)
        amount = int(tokens)
        self._token_events[provider, account].append((self._now(), amount))
        self._tokens[provider, account] += amount

    def _mark_seen(self, provider: str, account: str) -> None:
        if account not in self._seen[provider]:
            self._seen[provider].append(account)

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Per-(provider, account) usage for the dashboard, in first-seen order.

        Returns:
            Provider -> account -> ``{"requests", "tokens"}``, counting every account the
            ledger has SEEN rather than only those still under a limit, so an account tripped
            out for the run still appears with the usage it accrued. The counters are run
            totals and are not pruned by the 24h windows, unlike the figures ``_usage``
            reports. Token counts are RAW provider tokens.

        """
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for provider, accounts in self._seen.items():
            out[provider] = {
                account: {
                    "requests": self._req_count[provider, account],
                    "tokens": self._tokens[provider, account],
                }
                for account in accounts
            }
        return out
