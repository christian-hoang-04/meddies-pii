"""Requests above the real documented daily cap = the 429-storm/ban signature.

The exhaustion signature: made calls but produced nothing (all 429'd / hit a daily cap). Must NOT read as "lagging" or a
low % — it's spent, not idle. This is the cloudflare acct2-12 case (1 request, 0 tokens, neuron-exhausted).

Even an old record that stored a cloudflare rpd proxy (122) must show exhausted, not "1/122 = 0.8% idle".

cerebras case: symmetric workers should converge, so acct9 at <60% of the cohort's best (360) means a
transport/event-loop/provider bottleneck. Surfaces the throughput gap the dashboard's daily-headroom view hides.

Funded acct1 (705, cap 1000) and unfunded acct2 (30, cap 50) are DIFFERENT cap cohorts — the unfunded account must not read
as "lagging" vs the funded one. Each is healthy against its own cap (70% and 60%).

A ban-risk account (over documented cap) is the one verdict that must fail the verify check — exit non-zero so a cron/CI
gate catches it.

A stray non-date file must not shadow the latest real record ("notes" sorts after "2026-..." lexicographically).

"""

from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
import json
from typing import TYPE_CHECKING

import pytest

from meddies_pii.generation.pool_monitor import (
    format_pool_status,
    load_usage_record,
    pool_health,
)
from meddies_pii.generation.pool_policy import account_rpd_cap
from meddies_pii.generation.usage.records import (
    AccountUsage,
    ProviderLimits,
    ProviderUsage,
    UsageRecord,
    UsageTotals,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record(providers: dict[str, ProviderUsage]) -> UsageRecord:
    return UsageRecord(
        date="2026-06-20",
        providers=providers,
        totals=UsageTotals(accepted=0, rejected=0, tokens=0, requests=0),
    )


def _provider(accounts: dict[str, tuple[int, int]], rpd_per_account: int | None) -> ProviderUsage:
    """Build one provider's record from ``{account: (requests, tokens)}``.

    Only ``accounts`` and ``limits`` drive pool health; the remaining
    ``ProviderUsage`` counters are zeroed so the fixture carries the same shape
    the persisted record does.

    Returns:
        A provider record whose accounts hold the given request and token counts.

    """
    return ProviderUsage(
        model="",
        accepted=0,
        rejected=0,
        tokens=0,
        requests=0,
        by_language={},
        accounts={a: AccountUsage(requests=r, tokens=t) for a, (r, t) in accounts.items()},
        limits=ProviderLimits(
            accounts=len(accounts),
            token_cap_per_account=None,
            rpd_per_account=rpd_per_account,
        ),
    )


@pytest.mark.parametrize(
    ("provider", "requests", "configured_rpd", "expected_cap"),
    [
        ("groq", 1000, 1000, 1000),
        ("openrouter", 50, 1000, 50),
        ("openrouter", 51, 1000, 1000),
        ("cloudflare", 1, 122, None),
    ],
)
def test_account_rpd_cap_applies_the_shared_quota_policy(
    provider: str,
    requests: int,
    configured_rpd: int | None,
    expected_cap: int | None,
) -> None:
    assert account_rpd_cap(provider, requests, configured_rpd) == expected_cap


def test_account_over_documented_cap_is_ban_risk() -> None:
    rec = _record({"groq": _provider({"acct1": (1100, 5)}, 1000)})
    health = {h.account: h for h in pool_health(rec)}
    assert health["acct1"].status == "ban-risk"


def test_zero_request_account_is_idle() -> None:
    rec = _record({"groq": _provider({"acct1": (0, 0)}, 1000)})
    assert pool_health(rec)[0].status == "idle"


def test_account_within_cap_is_healthy() -> None:
    rec = _record({"groq": _provider({"acct1": (700, 5)}, 1000)})
    h = pool_health(rec)[0]
    assert h.status == "healthy"
    assert h.util_pct == 70.0


def test_account_with_requests_but_zero_tokens_is_exhausted() -> None:
    rec = _record({"cloudflare": _provider({"acct1": (40, 75_750), "acct2": (1, 0)}, None)})
    health = {h.account: h for h in pool_health(rec)}
    assert health["acct2"].status == "exhausted"
    assert health["acct1"].status == "healthy"


def test_exhausted_still_detected_when_legacy_record_has_rpd() -> None:
    rec = _record({"cloudflare": _provider({"acct2": (1, 0)}, 122)})
    assert pool_health(rec)[0].status == "exhausted"


def test_account_far_below_same_cap_peers_is_lagging() -> None:
    accts = {f"acct{i}": (360, 100) for i in range(1, 9)}
    accts["acct9"] = (170, 50)
    rec = _record({"cerebras": _provider(accts, 2400)})
    health = {h.account: h for h in pool_health(rec)}
    assert health["acct1"].status == "healthy"
    assert health["acct9"].status == "lagging"


def test_openrouter_funding_cohorts_do_not_false_flag() -> None:
    rec = _record({"openrouter": _provider({"acct1": (705, 9), "acct2": (30, 3)}, 1000)})
    health = {h.account: h for h in pool_health(rec)}
    assert health["acct1"].cap == 1000
    assert health["acct1"].status == "healthy"
    assert health["acct2"].cap == 50
    assert health["acct2"].status == "healthy"


def test_format_flags_ban_risk_with_nonzero_exit() -> None:
    rec = _record({"groq": _provider({"acct1": (1100, 5)}, 1000)})
    text, exit_code = format_pool_status(rec)
    assert "ban-risk" in text
    assert "acct1" in text
    assert exit_code != 0


def test_format_clean_pool_has_zero_exit() -> None:
    rec = _record({"groq": _provider({"acct1": (700, 5), "acct2": (650, 5)}, 1000)})
    text, exit_code = format_pool_status(rec)
    assert exit_code == 0
    assert "2026-06-20" in text


def test_load_usage_record_picks_latest_date(tmp_path: Path) -> None:
    for d in ("2026-06-18", "2026-06-20", "2026-06-19"):
        (tmp_path / f"{d}.json").write_text(json.dumps({"date": d, "providers": {}, "totals": {}}), encoding="utf-8")
    rec = load_usage_record(tmp_path)
    assert rec is not None
    assert rec["date"] == "2026-06-20"
    by_date = load_usage_record(tmp_path, date="2026-06-19")
    assert by_date is not None
    assert by_date["date"] == "2026-06-19"


def test_load_usage_record_missing_returns_none(tmp_path: Path) -> None:
    assert load_usage_record(tmp_path) is None


def test_load_usage_record_ignores_non_date_json(tmp_path: Path) -> None:
    (tmp_path / "2026-06-20.json").write_text(
        json.dumps({"date": "2026-06-20", "providers": {}, "totals": {}}),
        encoding="utf-8",
    )
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    rec = load_usage_record(tmp_path)
    assert rec is not None
    assert rec["date"] == "2026-06-20"
