"""glm_cn + sambanova verified 2026-06-30.

Both produce all 9 labels incl private_url/ secret, in the same prose+JSON mix as the existing pool
(JSON/FHIR/portal-audit-log output is INTENDED corpus diversity — see the structured_payload scenarios + the JSON
text_formats in catalog.py, ~25% across every provider). sambanova defaults to gpt-oss-120b (our proven model;
Meta-Llama-3.3-70B was too weak); its free tier is tiny (per-model 20 RPD) but adds DeepSeek/gpt-oss diversity. glm_cn
(glm-4.5-flash) is free + uncapped (the flagships are paid) -> high serial volume.

Cost budget = billable tokens, OpenAI grant only. Free providers are rate-limited per account by the AccountLedger (raw
tokens), not here.

Default 1 for: cerebras (hour-capped, already at it), llm7 (1 rpm), cloudflare (neuron-capped), opencode_zen (unknown cap
-> stay serial), and groq — groq is the pool's tightest-TPM provider (4800), where serial requests keep the TPM gate exact:
in-flight requests' tokens aren't recorded until completion, so concurrency would let N requests slip past the per-minute
token check.

"""

from __future__ import annotations

from typing import Any

from meddies_pii.generation.account_ledger import AccountLedger, AccountLimits
from meddies_pii.generation.label_corpus.provider_access import provider_keys
from meddies_pii.generation.openai_compatible.quota import DailyTokenBudget
from meddies_pii.languages import LANGUAGE_PROFILES

OUT_ROOT = "data/run2a"
SYNTHETIC_ROOT = "data/bioes-v2/synthetic"
HEADROOM = 0.9
PER_ACCOUNT_DAILY_TOKEN_CAPS = {
    "openai": 2_400_000,
}
"""Cost budget (billable tokens, provider-wide).

Only the OpenAI data-sharing grant is metered this way — 2.4M/day, calibrated just under the hard 2.5M ceiling (commit
61efdb4), used as-is with no headroom. The FREE providers are NOT here: their real per-account daily ceilings (TPD/RPD) and
per-minute throttles (TPM/RPM) are enforced by the AccountLedger below, against RAW provider tokens — the figure providers
actually meter for rate limits.

"""
DEFAULT_LANGUAGE_KEYS = tuple(LANGUAGE_PROFILES)
WEAK_LABEL_FREE_PROVIDERS = (
    "groq",
    "cerebras",
    "mistral",
    "llm7",
    "nim",
    "cloudflare",
    "opencode_zen",
    "openrouter",
    "gemini",
    "glm_cn",
    "sambanova",
    "minimax",
)
"""Every provider runs as its own concurrent worker.

So a slow one (llm7 at 1 rpm) never blocks the others. The run's hard wall-clock deadline (max_minutes) then cancels any
worker still going, so llm7 contributes what it can in the window without dragging the run. Keyless OVH/Pollinations stay
off (per-IP throttled / too small to follow the inline-tag format). mistral re-enabled 2026-07-01: its bind is a MONTHLY
token budget (see the AccountLimits note below) that exhausted + paused it 2026-06-25; the month boundary reset it.
Verified via provider_pingpong (real chat completion per key + a live generation): keys authenticate and produce all 9
labels. Two revoked keys (401, NOT budget-related) were pruned from MISTRAL_API_KEYS: a 401 is non-retryable and surfaces
as a fatal_api_error that ABORTS the provider's generation (synthetic.py:237), so one dead key in the pool kills the whole
mistral worker on a fresh run — keep the pool free of revoked keys.

"""


def _account_count(provider: str) -> int:
    return max(1, len(provider_keys(provider)))


def build_budget(openai_cap: int | None = None) -> DailyTokenBudget:
    caps = dict(PER_ACCOUNT_DAILY_TOKEN_CAPS)
    if openai_cap is not None:
        caps["openai"] = openai_cap
    return DailyTokenBudget(caps)


ACCOUNT_RATE_LIMITS = {
    "groq": AccountLimits(rpm=54, rpd=900, tpm=4_800, tpd=450_000),
    "cerebras": AccountLimits(rpm=2, rpd=2_160, tpm=27_000, tpd=900_000),
    "mistral": AccountLimits(rpm=20, rpd=3_600, tpm=300_000, tpd=30_000_000),
    "llm7": AccountLimits(rpm=1, rpd=None, tpm=None, tpd=None),
    "nim": AccountLimits(rpm=24, rpd=1_000, tpm=None, tpd=None),
    "cloudflare": AccountLimits(rpm=None, rpd=110, tpm=None, tpd=None),
    "opencode_zen": AccountLimits(rpm=5, rpd=None, tpm=None, tpd=450_000),
    "openrouter": AccountLimits(rpm=10, rpd=900, tpm=None, tpd=None),
    "gemini": AccountLimits(rpm=3, rpd=None, tpm=None, tpd=None),
    "glm_cn": AccountLimits(rpm=30, rpd=None, tpm=None, tpd=None),
    "sambanova": AccountLimits(rpm=11, rpd=18, tpm=None, tpd=120_000),
    "minimax": AccountLimits(rpm=15, rpd=None, tpm=None, tpd=None),
}
"""Per-account limits VERIFIED 2026-06-19 from each provider's live API response headers.

Primary) or official docs, then deliberately set BELOW the documented ceiling — RPM targets at ~50-60% of the published
limit — so we read as a polite client, not a quota extractor. The AccountLedger enforces all four axes per account against
RAW provider tokens (RPM/TPM throttle per minute; RPD/TPD stop an account once spent); on top of it the client jitters
every request, honors Retry-After, and trips an account out of rotation on a hard daily-cap 429 (no 429 storms). None =
that axis is unbounded.

qwen3-32b. Verified: 60 RPM / 1000 RPD / 6000 TPM / 500k TPD (live header + Groq docs). TPM is the bind (4800 = 0.8
headroom, ~8/min); at ~500 tok/req that caps groq ~10/min, so RPM never binds — kept at 0.9x60 for consistency.

gpt-oss-120b. Verified live header: 5/min, 150/HOUR, 2400/day, 30k TPM. The ledger has no hourly axis, so rpm=2 (=120/hr)
respects the 150/hr cap that actually binds a multi-hour run — already where production sat.

medium-2505. Verified live header: 25 req/min, 375k TPM, no daily req cap — the limit is a MONTHLY token budget
(unverified), which is why it exhausted and paused 2026-06-25. Under the 24/7 long window (max-minutes 1200) rpm alone
would run it ~24k/day and re-exhaust the month in days, so cap rpd to ~the old 3h volume (20/min x 180min = 3600). NOTE:
3600/day still drained the month before its reset, so to last all the way to the monthly reset set this LOWER once verified
(rpd ~= monthly_token_budget / 30 / ~500 tok per sample).

Verified docs (docs.llm7.io): free tier ~30 RPM, 100 req/HOUR, 5M tok/day. We run anonymous; kept at 1/min (=60/hr, under
the 100/hr cap) — gentle.

llama-3.3-70b. Verified (NVIDIA forum + secondaries): 40 RPM, FINITE credit pool with NO daily reset (it never refills).
rpd is a SCHEDULE knob, not a yield knob: nim's total output = the pool size regardless, so rpd only sets how fast it's
spent. 1000/day spreads the pool for steady model diversity + resilience (vs a ~2-day burst then dead at 4320 under the
24/7 window). To get MORE total, add NIM accounts (each = its own pool) — rpd can't. Tune toward pool_size/~30d once known
(watch the dashboard for the exhaustion trip); don't set it so low the credits expire unused. circuit-break still covers
exhaustion.

Verified (4006 body): 10,000 neurons/day, neuron-metered. Modeled as a conservative daily request cap; the client trips on
the 4006 daily signal.

Limits UNPUBLISHED (web-verify 2026-06-19 found none). Ultra-conservative 5/min; the circuit-breaker on any daily-cap 429
is the real safety net.

Verified (429 body): 50/day unfunded, 1000/day funded. rpd at 0.9 of the funded 1000; the client trips unfunded accounts at
their 50 on the free-models-per-day 429 (no storm). RPM held well under 18.

gemini-3.5-flash free tier, per PROJECT (each AI_STUDIO key = a separate project, so keys stack quota). Verified live
2026-06-30: 429 after ~6 rapid calls, per-minute window (recovered <60s); the 429 body carried both ...PerMinute-FreeTier
and ...PerDay-FreeTier metrics, so a small daily cap also exists (value unverified). rpm=3 sits ~0.6x the observed ~5/min
so we never trip the limit — polite, doesn't read as a quota extractor. rpd left None: the daily 429 trips the account via
the circuit-breaker (like opencode_zen), and a 403 "denied access" on a restricted project trips it for the run.
Concurrency stays at the default 1 (serial = exact pacing).

Z.AI (GLM) glm-4.5-flash — the permanently-free model (flagships are paid). No published numeric token cap; z.ai free is
concurrency-limited (default serial=1 worker gives exact pacing). rpm=30 is a polite throttle that the ~15-25s serial
latency never reaches; tighten if 429/concurrency appears.

SambaNova free tier (no card linked): PER-MODEL 20 RPM / 20 RPD / 200K TPD. rpd=18 sits under the hard 20/day wall
(dashboard recovers 20 via /0.9); tpd=120k = 0.6x. Low-volume by design (~18 docs/day per model) -> rotate SAMBANOVA_MODEL
across the 6 free models to multiply. Serial (default conc 1).

MiniMax-M3 free tier on PUBLICLY-SHARED dumped keys (ephemeral). No documented per-minute limit; the bind is the token plan
(429 "Token Plan usage limit reached"), absorbed by the rate-limit path. Conservative rpm; serial (default concurrency 1).
Swap in own-account keys for reliable volume.

"""

PROVIDER_CONCURRENCY = {
    "mistral": 4,
    "nim": 2,
    "openrouter": 2,
}
"""Per-account in-flight request cap.

Concurrency only lets an account REACH its ledger rate cap when request latency would otherwise hold it below; the ledger
still throttles the actual rate, so a higher value never exceeds the limits above. Default 1 (one in-flight request ->
precise per-minute accounting), raised only where a provider has real rate headroom a single serial worker can't fill at
its ~15-25s latency.

conc 4 reaches the rpm=20 cap vs real latency (spike: clean to 32/min).

"""


def build_account_ledger() -> AccountLedger:
    return AccountLedger(ACCOUNT_RATE_LIMITS)


def _provider_limits(provider: str) -> dict[str, Any]:
    """Return the real, un-headroomed per-account daily caps.

    These feed the usage dashboard's
    utilization view. ``ACCOUNT_RATE_LIMITS`` stores TPD/RPD with the 0.9 headroom
    applied, so divide it back out to recover the true ceiling. The per-account
    daily TOKEN cap is the ledger's TPD for free providers and the cost-budget cap
    for the OpenAI grant. ``None`` means that axis is uncapped (request-metered).

    Cloudflare is neuron-metered (10k neurons/day, error 4006), not request-metered — its rpd=110 is only a ledger throttle
    proxy, meaningless as a utilization denominator (an exhausted account reads as "1/122 idle"). Show it request-uncapped
    so the bogus percentage is gone; the token-spend column carries cloudflare's real signal, and the monitor flags
    exhaustion.

    Returns:
        The account count, the per-account daily token cap and the per-account RPD, all with
        the 0.9 headroom divided back out so they read as true ceilings rather than the
        throttle values the ledger enforces. ``None`` on either cap axis means uncapped, which
        the dashboard renders as no denominator instead of a misleading percentage.

    """
    rate_limit = ACCOUNT_RATE_LIMITS.get(provider)
    token_cap: int | None
    if rate_limit and rate_limit.tpd:
        token_cap = round(rate_limit.tpd / HEADROOM)
    else:
        token_cap = PER_ACCOUNT_DAILY_TOKEN_CAPS.get(provider)
    rpd_per_account = (
        None if provider == "cloudflare" else (round(rate_limit.rpd / HEADROOM) if rate_limit and rate_limit.rpd else None)
    )
    return {
        "accounts": _account_count(provider),
        "token_cap_per_account": token_cap,
        "rpd_per_account": rpd_per_account,
    }
