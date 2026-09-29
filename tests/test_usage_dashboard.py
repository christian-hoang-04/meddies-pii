"""Token cap is per-account; the provider total is cap * accounts.

13,423 of 8 * 2,000,000 = 16,000,000 is ~0.1%. RPD likewise: 6 of 8 * 1000 = 8000.

OpenRouter unfunded accounts cap at 50 :free requests/day, funded at 1000. The dashboard must show each account against its
REAL cap, not 1000 for all: an unfunded account at 30/50 (60%) otherwise reads as idle (3% of 1000). Funding is inferred
from the data: >50 requests proves funded (unfunded trips at 50), otherwise treat as unfunded.

A legacy record stored cloudflare rpd=122; the dashboard must still render it neuron-metered (∞) in BOTH the provider and
account tables — never "/ 122", which reads as "0.8% idle" when the account is actually neuron-exhausted.

The account table carries a STATUS verdict so a 0-output account reads as "exhausted", not a misleading low %. Same
classifier as `pool-status`, so the HTML and the CLI never contradict each other.

llm7/nim have no token or RPD cap — utilization must render "/ ∞", not error.

A later same-day run without a model field must not blank an earlier model.

nim/cloudflare have no token cap but the ledger still records their spend.

Persisted JSON is untrusted. A string counter must not be silently coerced into a dashboard value or prevent the valid
daily record from rendering.

"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from meddies_pii.generation.usage import (
    AccountUsage,
    ProviderLimits,
    ProviderRunStats,
    UsageRecord,
    build_usage_record,
    decode_usage_record,
    render_usage_dashboard,
    write_usage_record,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_usage_public_imports_build_and_decode_a_record() -> None:
    stats: ProviderRunStats = {
        "accepted": 1,
        "rejected": 0,
        "by_language": {"vi": 1},
    }
    account: AccountUsage = {"requests": 1, "tokens": 2}
    limits: ProviderLimits = {"rpd_per_account": 10}
    record: UsageRecord = build_usage_record(
        "2026-06-19",
        {"groq": stats},
        {"groq": {"acct1": account}},
        {"groq": limits},
    )

    assert decode_usage_record(record) == record


def test_build_usage_record_merges_samples_and_account_snapshot() -> None:
    per_provider: dict[str, ProviderRunStats] = {
        "groq": {
            "accepted": 10,
            "rejected": 2,
            "by_language": {"vietnamese": 6, "english": 4},
        },
    }
    accounts: dict[str, dict[str, AccountUsage]] = {
        "groq": {
            "acct1": {"requests": 5, "tokens": 8_000},
            "acct2": {"requests": 3, "tokens": 4_345},
        },
    }
    record = build_usage_record("2026-06-19", per_provider, accounts)

    groq = record["providers"]["groq"]
    assert groq["accepted"] == 10
    assert groq["rejected"] == 2
    assert groq["tokens"] == 12_345
    assert groq["requests"] == 8
    assert groq["accounts"]["acct1"]["tokens"] == 8_000
    assert record["totals"]["tokens"] == 12_345
    assert record["totals"]["accepted"] == 10


def test_provider_with_no_account_snapshot_has_zero_tokens() -> None:
    record = build_usage_record(
        "2026-06-19",
        {"llm7": {"accepted": 4, "rejected": 0, "by_language": {"vietnamese": 4}}},
        {},
    )
    assert record["providers"]["llm7"]["tokens"] == 0
    assert record["providers"]["llm7"]["accounts"] == {}


def test_model_is_recorded_and_rendered(tmp_path: Path) -> None:
    """auto-refresh so an open dashboard reflects each new run without a reload."""
    record = build_usage_record(
        "2026-06-19",
        {
            "groq": {
                "model": "qwen/qwen3-32b",
                "accepted": 5,
                "rejected": 0,
                "by_language": {"vietnamese": 5},
            },
        },
        {"groq": {"acct1": {"requests": 3, "tokens": 900}}},
    )
    assert record["providers"]["groq"]["model"] == "qwen/qwen3-32b"

    write_usage_record(record, tmp_path)
    text = render_usage_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "<th>Model</th>" in text
    assert "qwen/qwen3-32b" in text
    assert 'http-equiv="refresh"' in text


def test_utilization_computed_against_real_caps(tmp_path: Path) -> None:
    record = build_usage_record(
        "2026-06-19",
        {
            "groq": {
                "model": "qwen/qwen3-32b",
                "accepted": 13,
                "rejected": 0,
                "by_language": {"vi": 13},
            },
        },
        {
            "groq": {
                "acct1": {"requests": 4, "tokens": 9930},
                "acct2": {"requests": 2, "tokens": 3493},
            },
        },
        {
            "groq": {
                "accounts": 8,
                "token_cap_per_account": 2_000_000,
                "rpd_per_account": 1000,
            },
        },
    )
    assert record["providers"]["groq"]["limits"]["rpd_per_account"] == 1000

    write_usage_record(record, tmp_path)
    text = render_usage_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "Utilization — 2026-06-19" in text
    assert "13,423 / 16,000,000" in text
    assert "0.1%" in text
    assert "6 / 8,000" in text


def test_openrouter_per_account_cap_reflects_funding(tmp_path: Path) -> None:
    record = build_usage_record(
        "2026-06-19",
        {"openrouter": {"accepted": 100, "rejected": 0, "by_language": {"vi": 100}}},
        {
            "openrouter": {
                "acct1": {"requests": 705, "tokens": 900_000},
                "acct2": {"requests": 30, "tokens": 30_000},
            },
        },
        {
            "openrouter": {
                "accounts": 2,
                "token_cap_per_account": None,
                "rpd_per_account": 1000,
            },
        },
    )
    write_usage_record(record, tmp_path)
    text = render_usage_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "705 / 1,000" in text
    assert "30 / 50" in text
    assert "60.0%" in text


def test_cloudflare_renders_neuron_metered_not_request_capped(tmp_path: Path) -> None:
    record = build_usage_record(
        "2026-06-19",
        {"cloudflare": {"accepted": 0, "rejected": 1, "by_language": {"vi": 0}}},
        {"cloudflare": {"acct2": {"requests": 1, "tokens": 0}}},
        {
            "cloudflare": {
                "accounts": 1,
                "token_cap_per_account": None,
                "rpd_per_account": 122,
            },
        },
    )
    write_usage_record(record, tmp_path)
    text = render_usage_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "/ 122" not in text
    assert "1 / ∞" in text


def test_dashboard_account_table_shows_status(tmp_path: Path) -> None:
    record = build_usage_record(
        "2026-06-19",
        {"cloudflare": {"accepted": 0, "rejected": 1, "by_language": {"vi": 0}}},
        {
            "cloudflare": {
                "acct1": {"requests": 40, "tokens": 75_750},
                "acct2": {"requests": 1, "tokens": 0},
            },
        },
        {
            "cloudflare": {
                "accounts": 2,
                "token_cap_per_account": None,
                "rpd_per_account": 122,
            },
        },
    )
    write_usage_record(record, tmp_path)
    text = render_usage_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "<th>Status</th>" in text
    assert "exhausted" in text
    assert "healthy" in text


def test_uncapped_provider_shows_infinity_not_crash(tmp_path: Path) -> None:
    record = build_usage_record(
        "2026-06-19",
        {"nim": {"accepted": 8, "rejected": 0, "by_language": {"vi": 8}}},
        {"nim": {"acct1": {"requests": 4, "tokens": 8254}}},
        {
            "nim": {
                "accounts": 1,
                "token_cap_per_account": None,
                "rpd_per_account": None,
            },
        },
    )
    write_usage_record(record, tmp_path)
    html = render_usage_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "/ ∞" in html


def test_model_survives_same_day_merge(tmp_path: Path) -> None:
    r1 = build_usage_record(
        "2026-06-19",
        {
            "groq": {
                "model": "qwen/qwen3-32b",
                "accepted": 3,
                "rejected": 0,
                "by_language": {"vietnamese": 3},
            },
        },
        {"groq": {"acct1": {"requests": 3, "tokens": 100}}},
    )
    r2 = build_usage_record(
        "2026-06-19",
        {"groq": {"accepted": 2, "rejected": 0, "by_language": {"english": 2}}},
        {"groq": {"acct2": {"requests": 2, "tokens": 80}}},
    )
    write_usage_record(r1, tmp_path)
    write_usage_record(r2, tmp_path)

    merged = json.loads((tmp_path / "2026-06-19.json").read_text(encoding="utf-8"))
    assert merged["providers"]["groq"]["model"] == "qwen/qwen3-32b"


def test_tokens_tracked_for_uncapped_providers() -> None:
    record = build_usage_record(
        "2026-06-19",
        {"nim": {"accepted": 2, "rejected": 0, "by_language": {"vietnamese": 2}}},
        {"nim": {"acct1": {"requests": 2, "tokens": 900}}},
    )
    assert record["providers"]["nim"]["tokens"] == 900


def test_same_date_write_merges_providers(tmp_path: Path) -> None:

    r1 = build_usage_record(
        "2026-06-19",
        {"groq": {"accepted": 3, "rejected": 0, "by_language": {"vietnamese": 3}}},
        {"groq": {"acct1": {"requests": 3, "tokens": 100}}},
    )
    r2 = build_usage_record(
        "2026-06-19",
        {"llm7": {"accepted": 2, "rejected": 1, "by_language": {"french": 2}}},
        {"llm7": {"acct1": {"requests": 2, "tokens": 500}}},
    )
    write_usage_record(r1, tmp_path)
    write_usage_record(r2, tmp_path)

    merged = json.loads((tmp_path / "2026-06-19.json").read_text(encoding="utf-8"))
    assert set(merged["providers"]) == {"groq", "llm7"}
    assert merged["totals"]["tokens"] == 600


def test_same_date_same_provider_sums_tokens_max_accepted(tmp_path: Path) -> None:
    """Ledger deltas summed file-cumulative count → max, not sum."""
    r1 = build_usage_record(
        "2026-06-19",
        {"groq": {"accepted": 3, "rejected": 0, "by_language": {"vietnamese": 3}}},
        {"groq": {"acct1": {"requests": 3, "tokens": 100}}},
    )
    r2 = build_usage_record(
        "2026-06-19",
        {"groq": {"accepted": 5, "rejected": 1, "by_language": {"vietnamese": 5}}},
        {"groq": {"acct1": {"requests": 2, "tokens": 50}}},
    )
    write_usage_record(r1, tmp_path)
    write_usage_record(r2, tmp_path)

    groq = json.loads((tmp_path / "2026-06-19.json").read_text(encoding="utf-8"))["providers"]["groq"]
    assert groq["tokens"] == 150
    assert groq["accepted"] == 5
    assert groq["accounts"]["acct1"]["tokens"] == 150


def test_write_and_render_dashboard_shows_per_account(tmp_path: Path) -> None:
    """Match the rendered numeric cells, not a bare substring — "10"/"600" also occur in CSS (e.g.

    padding:10px) and would pass for the wrong reason.

    """
    rec1 = build_usage_record(
        "2026-06-18",
        {"groq": {"accepted": 3, "rejected": 0, "by_language": {"vietnamese": 3}}},
        {"groq": {"acct1": {"requests": 3, "tokens": 100}}},
    )
    rec2 = build_usage_record(
        "2026-06-19",
        {"groq": {"accepted": 7, "rejected": 1, "by_language": {"english": 7}}},
        {
            "groq": {
                "acct1": {"requests": 4, "tokens": 300},
                "acct2": {"requests": 3, "tokens": 200},
            },
        },
    )
    write_usage_record(rec1, tmp_path)
    write_usage_record(rec2, tmp_path)

    dashboard = render_usage_dashboard(tmp_path)
    text = dashboard.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert "groq" in text
    assert "acct1" in text
    assert "acct2" in text
    assert "2026-06-19" in text
    assert '<td class="n">10</td>' in text
    assert '<td class="n">600</td>' in text


def test_dashboard_skips_malformed_persisted_record(tmp_path: Path) -> None:
    (tmp_path / "2026-06-18.json").write_text(
        '{"date":"2026-06-18","providers":{"groq":{"accepted":"7"}}}',
        encoding="utf-8",
    )
    valid = build_usage_record(
        "2026-06-19",
        {"groq": {"accepted": 2, "rejected": 0, "by_language": {"vi": 2}}},
        {"groq": {"acct1": {"requests": 2, "tokens": 9}}},
    )
    write_usage_record(valid, tmp_path)

    html = render_usage_dashboard(tmp_path).read_text(encoding="utf-8")

    assert "through 2026-06-19" in html
    assert "2026-06-18" not in html


def test_same_day_malformed_record_is_replaced_by_valid_contract(tmp_path: Path) -> None:
    path = tmp_path / "2026-06-19.json"
    path.write_text('{"date":"2026-06-19","providers":[]}', encoding="utf-8")
    valid = build_usage_record(
        "2026-06-19",
        {"groq": {"accepted": 2, "rejected": 1, "by_language": {"vi": 2}}},
        {"groq": {"acct1": {"requests": 3, "tokens": 11}}},
    )

    write_usage_record(valid, tmp_path)

    assert json.loads(path.read_text(encoding="utf-8")) == valid


def test_dashboard_escapes_persisted_provider_model_account_and_date(tmp_path: Path) -> None:
    record = build_usage_record(
        "2026-06-19<script>",
        {
            "<groq>": {
                "model": '<img src=x onerror="alert(1)">',
                "accepted": 1,
                "rejected": 0,
                "by_language": {"vi": 1},
            },
        },
        {"<groq>": {"acct&1": {"requests": 1, "tokens": 2}}},
    )
    write_usage_record(record, tmp_path)

    html = render_usage_dashboard(tmp_path).read_text(encoding="utf-8")

    assert "&lt;groq&gt;" in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    assert "acct&amp;1" in html
    assert "2026-06-19&lt;script&gt;" in html
    assert '<img src=x onerror="alert(1)">' not in html
