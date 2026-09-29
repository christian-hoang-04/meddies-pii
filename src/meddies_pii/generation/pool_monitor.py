"""Read-only health check for the weak-label provider pool.

Turns a usage record (the shape ``build_usage_record`` produces) into a per-account
verdict — idle / exhausted / lagging / ban-risk / healthy — so an operator can
VERIFY a run utilized accounts correctly and never crossed a documented rate cap
(the ban signature). Backs the ``meddies-pii pool-status`` command. Pure functions here;
I/O lives in the CLI. Reuses ``account_rpd_cap`` so the verdict and the dashboard
share one definition of each account's real binding cap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from meddies_pii.generation.pool_policy import account_rpd_cap
from meddies_pii.generation.usage.records import UsageRecord, decode_usage_record


def load_usage_record(usage_dir: str | Path, date: str | None = None) -> UsageRecord | None:
    """Load one day's usage record from ``<usage_dir>/<date>.json``.

    The latest date is used
    when ``date`` is None. Returns None when there's nothing to read — the CLI
    turns that into a clear 'no run yet' message rather than a stack trace.

    Match only YYYY-MM-DD.json so a stray non-date file (e.g. notes.json) can't sort last and shadow the real latest
    record.

    Returns:
        The decoded record, or ``None`` when there is nothing to read -- no directory, no
        matching file, or a file that fails to parse or validate. ``None`` is the "no run yet"
        signal the CLI turns into a plain message; it is deliberately not an exception, because
        an empty usage directory is a normal state rather than a fault.

    """
    directory = Path(usage_dir)
    if date is not None:
        path = directory / f"{date}.json"
        candidates = [path] if path.exists() else []
    else:
        candidates = sorted(directory.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
    if not candidates:
        return None
    try:
        return decode_usage_record(json.loads(candidates[-1].read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, ValueError):
        return None


MIN_LAGGING_COHORT_SIZE = 2

LAGGING_FRACTION = 0.6
"""An account doing less than this fraction of its same-cap cohort's best performer is "lagging".

The account_worker gives every account an identical independent loop, so same-cap accounts should converge — a large gap
means a transport / event-loop / provider bottleneck, not unequal assigned work. Diagnostic, not a ban signal.

"""


@dataclass(frozen=True)
class AccountHealth:
    provider: str
    account: str
    requests: int
    tokens: int
    cap: int | None
    """real documented daily request cap (None = uncapped/neuron-metered)."""
    util_pct: float | None
    status: str
    """"idle" | "ban-risk" | "exhausted" | "lagging" | "healthy"."""


def pool_health(record: UsageRecord) -> list[AccountHealth]:
    """Classify every account in a usage record against its real binding cap.

    Per-account real cap, then the best performer + size of each same-cap cohort (lagging is judged only against accounts
    that share a cap).

    Made calls but produced nothing — every request failed (a daily cap / 429). "spent", never "idle": shown as exhausted,
    not a low %.

    Returns:
        One health entry per account, sorted by provider then account, each carrying its
        requests, tokens, real cap, utilization and verdict. ``lagging`` is judged only within
        a same-cap cohort, so an account on a small cap is never called slow for trailing one
        on a large cap. A request-uncapped provider gets ``None`` for cap and utilization
        rather than a fabricated denominator.

    """
    out: list[AccountHealth] = []
    for provider, stats in sorted(record.get("providers", {}).items()):
        rpd = stats.get("limits", {}).get("rpd_per_account")
        base_cap = int(rpd) if rpd else None
        accounts = stats.get("accounts", {})
        caps = {a: account_rpd_cap(provider, int(e.get("requests", 0)), base_cap) for a, e in accounts.items()}
        cohort_best: dict[int | None, int] = {}
        cohort_size: dict[int | None, int] = {}
        for a, e in accounts.items():
            cohort_best[caps[a]] = max(cohort_best.get(caps[a], 0), int(e.get("requests", 0)))
            cohort_size[caps[a]] = cohort_size.get(caps[a], 0) + 1
        for account, entry in sorted(accounts.items()):
            requests = int(entry.get("requests", 0))
            tokens = int(entry.get("tokens", 0))
            cap = caps[account]
            util = (100 * requests / cap) if cap else None
            best = cohort_best[cap]
            if requests == 0:
                status = "idle"
            elif cap is not None and requests > cap:
                status = "ban-risk"
            elif tokens == 0:
                status = "exhausted"
            elif cohort_size[cap] >= MIN_LAGGING_COHORT_SIZE and best > 0 and requests < LAGGING_FRACTION * best:
                status = "lagging"
            else:
                status = "healthy"
            out.append(
                AccountHealth(
                    provider=provider,
                    account=account,
                    requests=requests,
                    tokens=tokens,
                    cap=cap,
                    util_pct=util,
                    status=status,
                ),
            )
    return out


_STATUS_ORDER = ("ban-risk", "exhausted", "idle", "lagging", "healthy")
"""Status order for the summary line + legend.

Full words, no cryptic codes — the verdict must be readable with zero decoding.

"""
_LEGEND = (
    "legend  healthy=producing under its cap · lagging=well below same-cap peers",
    "        exhausted=made requests but 0 output (hit a daily cap) · idle=no requests",
    "        ban-risk=ABOVE a documented cap   |   cap ∞ = token/neuron-metered (not request-capped)",
)


def format_pool_status(record: UsageRecord) -> tuple[str, int]:
    """Render the per-account health table and return ``(text, exit_code)``.

    ``exit_code`` is 1 when any account is over a documented cap (the ban
    signature), else 0 — so ``meddies-pii pool-status`` doubles as a cron/CI gate
    that fails loudly the moment the pool risks a rate-limit ban.

    Returns:
        The rendered table with its legend, and the exit code. The code is 1 only when some
        account is above a documented cap -- the ban signature -- so an idle or exhausted pool
        still exits 0. Being over a cap is the one state worth waking someone for; running out
        of quota is the expected end of a day's run.

    """
    health = pool_health(record)
    date = record.get("date", "")
    lines = [
        f"Pool status — {date}",
        "",
        f"{'provider':<13} {'account':<8} {'requests':>9} {'cap':>10} {'util':>7}  status",
    ]
    counts: dict[str, int] = dict.fromkeys(_STATUS_ORDER, 0)
    for h in health:
        counts[h.status] = counts.get(h.status, 0) + 1
        cap = f"{h.cap:,}" if h.cap is not None else "∞"
        util = f"{h.util_pct:.1f}%" if h.util_pct is not None else "—"
        lines.append(f"{h.provider:<13} {h.account:<8} {h.requests:>9,} {cap:>10} {util:>7}  {h.status}")
    lines += ["", " · ".join(f"{counts[s]} {s}" for s in _STATUS_ORDER), "", *_LEGEND]
    return "\n".join(lines), (1 if counts["ban-risk"] else 0)
