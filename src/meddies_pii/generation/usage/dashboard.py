"""Self-contained HTML usage dashboard rendering."""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the dependency is optional at runtime and deferred to the call that needs it.
# ruff: file-ignore[try-except-in-loop]
# reason: a record-and-continue harness: each item's outcome is the product, so an exception has to
# reason: become that item's recorded result. Hoisting the try would hide every item after the first.
import json
from html import escape
from pathlib import Path

from meddies_pii.generation.pool_policy import NEURON_METERED, account_rpd_cap
from meddies_pii.generation.usage.records import UsageRecord, decode_usage_record


def render_usage_dashboard(usage_dir: str | Path) -> Path:
    """Render every valid daily record; malformed records are skipped deliberately.

    Returns:
        The path of the written ``index.html``. A record that fails to read, parse or validate
        is skipped rather than aborting the render, because the dashboard is an operational
        view: one corrupt day should cost that day's row, not the whole page.

    """
    directory = Path(usage_dir)
    records: list[UsageRecord] = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(decode_usage_record(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    out = directory / "index.html"
    out.write_text(_render_html(records), encoding="utf-8")
    return out


def _empty_usage_record() -> UsageRecord:
    return {
        "date": "",
        "providers": {},
        "totals": {"accepted": 0, "rejected": 0, "tokens": 0, "requests": 0},
    }


def _accept_rate(accepted: int, rejected: int) -> str:
    total = accepted + rejected
    return f"{(100 * accepted / total):.0f}%" if total else "—"


def _pct(used: int, cap: int | None) -> str:
    return f"{100 * used / cap:.1f}%" if cap else "—"


def _of_cap(used: int, cap: int | None) -> str:
    return f"{used:,} / {cap:,}" if cap else f"{used:,} / ∞"


def _utilization_provider_rows(record: UsageRecord) -> str:
    """Render per-provider utilization against un-headroomed daily caps.

    Returns:
        The provider table rows as HTML. Caps are multiplied out to pool totals by account
        count, and a neuron-metered provider is shown request-uncapped rather than against its
        throttle-proxy RPD, which would render an exhausted account as a near-idle percentage.

    """
    rows = []
    for provider, stats in sorted(record["providers"].items()):
        limits = stats["limits"]
        accounts = limits.get("accounts", 0) or 1
        token_cap = limits.get("token_cap_per_account")
        rpd = None if provider in NEURON_METERED else limits.get("rpd_per_account")
        token_cap_total = token_cap * accounts if token_cap else None
        rpd_total = rpd * accounts if rpd else None
        tokens = stats["tokens"]
        requests = stats["requests"]
        fraction = tokens / token_cap_total if token_cap_total else (requests / rpd_total if rpd_total else 0.0)
        rows.append(
            f'<tr><td class="p">{escape(provider)}</td>'
            f'<td class="n">{_of_cap(tokens, token_cap_total)}</td>'
            f'<td class="n">{_pct(tokens, token_cap_total)}</td>'
            f'<td class="n">{_of_cap(requests, rpd_total)}</td>'
            f'<td class="n">{_pct(requests, rpd_total)}</td>'
            f'<td class="bar"><span style="width:{min(100, round(100 * fraction))}%">'
            f"</span></td></tr>",
        )
    return "\n".join(rows)


def _utilization_account_rows(record: UsageRecord) -> str:
    """Render per-account utilization and the shared pool-health status verdict.

    Returns:
        The per-account table rows as HTML, each carrying the status the pool monitor assigned.
        The verdict is read from ``pool_health`` rather than recomputed here, so the dashboard
        and the ``pool-status`` command can never disagree about whether an account is at risk.

    """
    from meddies_pii.generation.pool_monitor import pool_health

    status_by = {(health.provider, health.account): health.status for health in pool_health(record)}
    rows = []
    for provider, stats in sorted(record["providers"].items()):
        rpd = stats["limits"].get("rpd_per_account")
        base_cap = rpd or None
        for account, entry in sorted(stats["accounts"].items()):
            requests = entry["requests"]
            cap = account_rpd_cap(provider, requests, base_cap)
            status = status_by.get((provider, account), "")
            rows.append(
                f'<tr><td class="p">{escape(provider)}</td>'
                f"<td>{escape(account)}</td>"
                f'<td class="n">{_of_cap(requests, cap)}</td>'
                f'<td class="n">{_pct(requests, cap)}</td>'
                f'<td class="n">{entry["tokens"]:,}</td>'
                f'<td class="st st-{escape(status)}">{escape(status)}</td></tr>',
            )
    return "\n".join(rows)


def _render_html(records: list[UsageRecord]) -> str:
    cumulative: dict[str, dict[str, int]] = {}
    accounts: dict[tuple[str, str], dict[str, int]] = {}
    models: dict[str, str] = {}
    grand = {"accepted": 0, "rejected": 0, "tokens": 0}
    for record in records:
        for provider, stats in record["providers"].items():
            accumulated = cumulative.setdefault(provider, {"accepted": 0, "rejected": 0, "tokens": 0})
            accumulated["accepted"] += stats["accepted"]
            accumulated["rejected"] += stats["rejected"]
            accumulated["tokens"] += stats["tokens"]
            if stats["model"]:
                models[provider] = stats["model"]
            for account_id, entry in stats["accounts"].items():
                bucket = accounts.setdefault((provider, account_id), {"requests": 0, "tokens": 0})
                bucket["requests"] += entry["requests"]
                bucket["tokens"] += entry["tokens"]
        totals = record["totals"]
        grand["accepted"] += totals["accepted"]
        grand["rejected"] += totals["rejected"]
        grand["tokens"] += totals["tokens"]

    latest = records[-1]["date"] if records else "—"
    peak = max((count["accepted"] for count in cumulative.values()), default=1) or 1
    latest_record = records[-1] if records else _empty_usage_record()
    utilization_provider_rows = _utilization_provider_rows(latest_record)
    utilization_account_rows = _utilization_account_rows(latest_record)

    provider_rows = "\n".join(
        f'<tr><td class="p">{escape(provider)}</td>'
        f'<td class="m">{escape(models.get(provider, "—"))}</td>'
        f'<td class="n">{counts["accepted"]:,}</td>'
        f'<td class="n">{counts["tokens"]:,}</td>'
        f'<td class="n">{_accept_rate(counts["accepted"], counts["rejected"])}</td>'
        f'<td class="bar"><span style="width:{100 * counts["accepted"] // peak}%"></span></td></tr>'
        for provider, counts in sorted(cumulative.items(), key=lambda entry: -entry[1]["accepted"])
    )
    account_rows = "\n".join(
        f'<tr><td class="p">{escape(provider)}</td>'
        f"<td>{escape(account_id)}</td>"
        f'<td class="n">{bucket["requests"]:,}</td>'
        f'<td class="n">{bucket["tokens"]:,}</td></tr>'
        for (provider, account_id), bucket in sorted(accounts.items())
    )
    day_rows = "\n".join(
        f'<tr><td class="p">{escape(record["date"])}</td>'
        f"<td>{escape(', '.join(sorted(record['providers'])))}</td>"
        f'<td class="n">{record["totals"]["accepted"]:,}</td>'
        f'<td class="n">{record["totals"]["rejected"]:,}</td>'
        f'<td class="n">{record["totals"]["tokens"]:,}</td></tr>'
        for record in reversed(records)
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="3600">
<title>Weak-Label Usage</title>
<style>
  :root {{ --ink:#0f2a33; --muted:#5b7682; --line:#e3ebee; --accent:#0e7c86; --bg:#f7fafb; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:920px; margin:0 auto; padding:40px 24px 64px; }}
  h1 {{ font-size:22px; margin:0 0 2px; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); margin:0 0 28px; font-size:13px; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:32px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
  .card .v {{ font-size:24px; font-weight:650; letter-spacing:-.02em; }}
  .card .k {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
    margin:32px 0 10px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff;
    border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
  th,td {{ padding:10px 14px; text-align:left; border-bottom:1px solid var(--line); font-size:14px; }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
  tr:last-child td {{ border-bottom:none; }}
  td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.p {{ font-weight:600; }}
  td.m {{ color:var(--muted); font-size:12.5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  td.bar {{ width:30%; }}
  td.bar span {{ display:block; height:8px; border-radius:4px; background:var(--accent); min-width:2px; }}
  td.st {{ font-weight:600; font-size:12.5px; }}
  td.st-healthy {{ color:#15803d; }}
  td.st-lagging, td.st-exhausted {{ color:#b45309; }}
  td.st-ban-risk {{ color:#b91c1c; }}
  td.st-idle {{ color:var(--muted); font-weight:400; }}
</style></head>
<body><div class="wrap">
  <h1>Weak-Label Generation — Daily Usage</h1>
  <p class="sub">Free-pool provider pool · {len(records)} day(s) tracked · through {escape(latest)} \
· auto-refreshes hourly</p>
  <div class="cards">
    <div class="card"><div class="v">{grand["accepted"]:,}</div><div class="k">Samples accepted</div></div>
    <div class="card"><div class="v">{grand["tokens"]:,}</div><div class="k">Tokens spent</div></div>
    <div class="card"><div class="v">{_accept_rate(grand["accepted"], grand["rejected"])}</div><div \
class="k">Accept rate</div></div>
    <div class="card"><div class="v">{len(cumulative)}</div><div class="k">Providers</div></div>
  </div>
  <h2>Utilization — {escape(latest)} (vs real 100% caps)</h2>
  <table><thead><tr><th>Provider</th><th class="n">Tokens used / cap</th><th class="n">Token \
%</th><th class="n">Requests / cap</th><th class="n">Req %</th><th>Usage</th></tr></thead>
  <tbody>
  {utilization_provider_rows or '<tr><td colspan="6">No data yet.</td></tr>'}
  </tbody></table>
  <h2>Utilization by account — {escape(latest)}</h2>
  <table><thead><tr><th>Provider</th><th>Account</th><th class="n">Requests / cap</th><th \
class="n">Req %</th><th class="n">Tokens</th><th>Status</th></tr></thead>
  <tbody>
  {utilization_account_rows or '<tr><td colspan="5">No data yet.</td></tr>'}
  </tbody></table>
  <h2>By provider (cumulative)</h2>
  <table><thead><tr><th>Provider</th><th>Model</th><th class="n">Samples</th><th \
class="n">Tokens</th><th class="n">Accept</th><th>Share</th></tr></thead>
  <tbody>
  {provider_rows or '<tr><td colspan="6">No data yet.</td></tr>'}
  </tbody></table>
  <h2>By account</h2>
  <table><thead><tr><th>Provider</th><th>Account</th><th class="n">Requests</th><th class="n">Tokens</th></tr></thead>
  <tbody>
  {account_rows or '<tr><td colspan="4">No data yet.</td></tr>'}
  </tbody></table>
  <h2>By day</h2>
  <table><thead><tr><th>Date</th><th>Providers</th><th class="n">Accepted</th><th \
class="n">Rejected</th><th class="n">Tokens</th></tr></thead>
  <tbody>
  {day_rows or '<tr><td colspan="5">No data yet.</td></tr>'}
  </tbody></table>
</div></body></html>
"""
