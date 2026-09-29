"""Behavior tests for OpenAICompatibleClient.

HTTP boundary mocked via respx — tests assert on the wire contract, not
on internal implementation. Tests should remain valid through refactors
that don't change what the client sends or returns.

the pre-call reservation over-counts max_tokens; reconcile trues the day's tally down to the response's real usage so the
cap reflects actual spend.

usage.total_tokens counts cached prompt tokens at full weight, but OpenAI bills cached input at 25% of the input price.
billable_tokens removes the 75% cached discount so the daily guard meters charged tokens, not raw throughput. total=1845,
cached=1280 -> 1845 - round(1280*0.75) = 885.

no cached prefix -> billable equals the full total_tokens (no discount).

a response without an int total_tokens cannot be reconciled.

uncapped providers track nothing pre-call, so reconcile must not create a negative tally out of a reservation that never
happened.

gpt-5.4-mini live-verified 2026-06-11: rejects `max_tokens` (needs `max_completion_tokens`) and
`reasoning_effort:"minimal"` (needs "none").

A plain per-minute 429 is NOT a daily cap — it must stay a transient cooldown, not block the account for the whole run.

"""

from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import json
import os
from datetime import date
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from meddies_pii.exceptions import ConfigurationError, DailyBudgetExceeded
from meddies_pii.generation.account_ledger import AccountLedger, AccountLimits
from meddies_pii.generation.openai_compatible.client import (
    OpenAICompatibleClient,
    _is_hard_daily_cap,
    _parse_retry_after,
)
from meddies_pii.generation.openai_compatible.providers import (
    get_provider_spec,
    resolve_provider_base_urls,
    resolve_provider_model,
)
from meddies_pii.generation.openai_compatible.quota import (
    AsyncRateLimiter,
    DailyTokenBudget,
    billable_tokens,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable rate limiting in tests, where RPM gating would cost time for no gain.

    The token bucket is well understood; these tests are about wire behavior.
    """

    # reason: this replaces `AsyncRateLimiter.wait`, which `client.py:420` awaits before every request.
    async def _instant(_self: AsyncRateLimiter) -> None:  # ruff: ignore[unused-async]
        return None

    monkeypatch.setattr(AsyncRateLimiter, "wait", _instant)


@pytest.fixture(autouse=True)
def _no_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the per-request random sleep so ledger-path tests stay fast and deterministic.

    The jitter is timing-only and changes no wire behavior.
    """
    monkeypatch.setattr(
        "meddies_pii.generation.openai_compatible.client._REQUEST_JITTER_RANGE",
        (0.0, 0.0),
    )


def _success_response(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


MIMO_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MIMO_AMS_URL = "https://token-plan-ams.xiaomimimo.com/v1/chat/completions"
MIMO_DEDICATED_URL = "https://token-plan-sgp.xiaomimimo.com/v1/chat/completions"
OPENCODE_ZEN_URL = "https://opencode.ai/zen/v1/chat/completions"


@pytest.mark.anyio
async def test_chat_returns_assistant_text() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.post(MIMO_URL).mock(return_value=httpx.Response(200, json=_success_response("hello")))
        async with OpenAICompatibleClient(["k1"], model="m") as client:
            text = await client.chat("sys", "user", temperature=0.7, max_tokens=10)
        assert text == "hello"


@pytest.mark.anyio
async def test_keys_rotate_round_robin() -> None:
    auths_seen: list[str] = []

    def capture(request: httpx.Request) -> httpx.Response:
        auths_seen.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json=_success_response("ok"))

    with respx.mock(assert_all_called=False) as router:
        router.post(MIMO_URL).mock(side_effect=capture)
        async with OpenAICompatibleClient(["k1", "k2", "k3"], model="m") as client:
            for _ in range(5):
                await client.chat("s", "u", temperature=0.5, max_tokens=10)

    assert auths_seen == [
        "Bearer k1",
        "Bearer k2",
        "Bearer k3",
        "Bearer k1",
        "Bearer k2",
    ]


@pytest.mark.anyio
async def test_keys_rotate_with_aligned_base_urls() -> None:
    seen: list[tuple[str, str]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization", "")))
        return httpx.Response(200, json=_success_response("ok"))

    with respx.mock(assert_all_called=False) as router:
        router.post(MIMO_AMS_URL).mock(side_effect=capture)
        router.post(MIMO_DEDICATED_URL).mock(side_effect=capture)
        async with OpenAICompatibleClient(
            ["ams-key", "sgp-key"],
            model="m",
            base_urls=[
                "https://token-plan-ams.xiaomimimo.com/v1",
                "https://token-plan-sgp.xiaomimimo.com/v1",
            ],
        ) as client:
            await client.chat("s", "u", temperature=0.5, max_tokens=10)
            await client.chat("s", "u", temperature=0.5, max_tokens=10)

    assert seen == [
        (MIMO_AMS_URL, "Bearer ams-key"),
        (MIMO_DEDICATED_URL, "Bearer sgp-key"),
    ]


@pytest.mark.anyio
async def test_5xx_triggers_retry() -> None:
    responses = [
        httpx.Response(503, text="service unavailable"),
        httpx.Response(200, json=_success_response("recovered")),
    ]
    with respx.mock(assert_all_called=True) as router:
        route = router.post(MIMO_URL).mock(side_effect=responses)
        async with OpenAICompatibleClient(["k1"], model="m") as client:
            text = await client.chat("s", "u", temperature=0.5, max_tokens=10)
        assert text == "recovered"
        assert route.call_count == 2


@pytest.mark.anyio
async def test_429_triggers_retry() -> None:
    responses = [
        httpx.Response(429),
        httpx.Response(200, json=_success_response("ok")),
    ]
    with respx.mock(assert_all_called=True) as router:
        route = router.post(MIMO_URL).mock(side_effect=responses)
        async with OpenAICompatibleClient(["k1"], model="m") as client:
            text = await client.chat("s", "u", temperature=0.5, max_tokens=10)
        assert text == "ok"
        assert route.call_count == 2


@pytest.mark.anyio
async def test_401_fails_fast_without_retry() -> None:
    """A bad/expired API key shouldn't burn the entire retry budget.

    Regression test: pre-fix, retry-on-HTTPStatusError caught 401 too,
    wasting 5 attempts on the same broken key. Now 4xx-not-429 fails fast.
    """
    with respx.mock(assert_all_called=True) as router:
        route = router.post(MIMO_URL).mock(return_value=httpx.Response(401))
        async with OpenAICompatibleClient(["k1"], model="m") as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.chat("s", "u", temperature=0.5, max_tokens=10)
        assert route.call_count == 1


@pytest.mark.anyio
async def test_404_fails_fast_without_retry() -> None:
    """Bad endpoint / route should also fail fast (4xx-not-429)."""
    with respx.mock(assert_all_called=True) as router:
        route = router.post(MIMO_URL).mock(return_value=httpx.Response(404))
        async with OpenAICompatibleClient(["k1"], model="m") as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.chat("s", "u", temperature=0.5, max_tokens=10)
        assert route.call_count == 1


@pytest.mark.anyio
async def test_chat_sends_expected_payload_shape() -> None:
    captured: list[dict[str, Any]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_success_response("ok"))

    with respx.mock(assert_all_called=True) as router:
        router.post(MIMO_URL).mock(side_effect=capture)
        async with OpenAICompatibleClient(["k1"], model="my-model") as client:
            await client.chat("you are helpful", "say hi", temperature=0.3, max_tokens=42)

    assert captured == [
        {
            "model": "my-model",
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "say hi"},
            ],
            "temperature": 0.3,
            "max_tokens": 42,
        },
    ]


@pytest.mark.anyio
async def test_context_manager_closes_underlying_client() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.post(MIMO_URL).mock(return_value=httpx.Response(200, json=_success_response("x")))
        client = OpenAICompatibleClient(["k1"], model="m")
        async with client:
            await client.chat("s", "u", temperature=0.5, max_tokens=10)
        assert client._client.is_closed


def test_from_env_missing_raises() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MIMO_API_KEYS", None)
        with pytest.raises(ConfigurationError, match="MIMO_API_KEYS"):
            OpenAICompatibleClient.from_env(model="m")


def test_from_env_empty_raises() -> None:
    with patch.dict(os.environ, {"MIMO_API_KEYS": "  ,  ,"}), pytest.raises(ConfigurationError, match="MIMO_API_KEYS"):
        OpenAICompatibleClient.from_env(model="m")


def test_from_env_strips_and_splits() -> None:
    with patch.dict(os.environ, {"MIMO_API_KEYS": "a, b ,c"}):
        client = OpenAICompatibleClient.from_env(model="m")
    assert client._api_keys == ["a", "b", "c"]


def test_resolve_provider_base_urls_reads_aligned_env_urls() -> None:
    with patch.dict(
        os.environ,
        {"MIMO_BASE_URLS": ("https://token-plan-ams.xiaomimimo.com/v1,https://token-plan-sgp.xiaomimimo.com/v1")},
        clear=True,
    ):
        assert resolve_provider_base_urls("mimo", 2) == [
            "https://token-plan-ams.xiaomimimo.com/v1",
            "https://token-plan-sgp.xiaomimimo.com/v1",
        ]


def test_resolve_provider_base_urls_rejects_mismatched_url_count() -> None:
    with (
        patch.dict(
            os.environ,
            {"MIMO_BASE_URLS": ("https://token-plan-ams.xiaomimimo.com/v1,https://token-plan-sgp.xiaomimimo.com/v1")},
            clear=True,
        ),
        pytest.raises(ConfigurationError, match="base URL count"),
    ):
        resolve_provider_base_urls("mimo", 3)


@pytest.mark.anyio
async def test_mimo_provider_uses_env_base_url_and_model_override() -> None:
    captured: list[dict[str, Any]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_success_response("ok"))

    with (
        patch.dict(
            os.environ,
            {
                "MIMO_API_KEYS": "dedicated-key",
                "MIMO_BASE_URL": "https://token-plan-sgp.xiaomimimo.com/v1",
                "MIMO_MODEL": "MiMo-V2.5-Pro",
            },
        ),
        respx.mock(assert_all_called=True) as router,
    ):
        router.post(MIMO_DEDICATED_URL).mock(side_effect=capture)
        async with OpenAICompatibleClient.from_env(provider="mimo") as client:
            text = await client.chat("sys", "user", temperature=0.2, max_tokens=99)

    assert text == "ok"
    assert captured[0]["model"] == "MiMo-V2.5-Pro"


@pytest.mark.anyio
async def test_opencode_zen_provider_uses_zen_endpoint_key_and_default_model() -> None:
    captured: list[dict[str, Any]] = []
    auths_seen: list[str] = []

    def capture(request: httpx.Request) -> httpx.Response:
        import json

        auths_seen.append(request.headers.get("authorization", ""))
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_success_response("ok"))

    with (
        patch.dict(
            os.environ,
            {"OPENCODE_ZEN_API_KEYS": "", "OPENCODE_ZEN_API_KEY": "zen-key"},
        ),
        respx.mock(assert_all_called=True) as router,
    ):
        router.post(OPENCODE_ZEN_URL).mock(side_effect=capture)
        async with OpenAICompatibleClient.from_env(provider="opencode_zen") as client:
            text = await client.chat("sys", "user", temperature=0.2, max_tokens=99)

    assert text == "ok"
    assert auths_seen == ["Bearer zen-key"]
    assert captured == [
        {
            "model": "deepseek-v4-flash-free",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
            ],
            "temperature": 0.2,
            "max_tokens": 99,
            "include_reasoning": False,
        },
    ]


@pytest.mark.anyio
async def test_opencode_zen_provider_supports_explicit_model_override() -> None:
    captured: list[dict[str, Any]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_success_response("ok"))

    with respx.mock(assert_all_called=True) as router:
        router.post(OPENCODE_ZEN_URL).mock(side_effect=capture)
        async with OpenAICompatibleClient.for_provider("zen", ["zen-key"], model="minimax-m2.5-free") as client:
            await client.chat("sys", "user", temperature=0.2, max_tokens=99)

    assert captured[0]["model"] == "minimax-m2.5-free"


def test_unknown_provider_raises() -> None:
    with pytest.raises(ConfigurationError, match="Unknown generation provider"):
        OpenAICompatibleClient.from_env(provider="not-a-provider")


def test_resolve_provider_model_uses_provider_specific_env() -> None:
    with patch.dict(
        os.environ,
        {
            "MIMO_MODEL": "MiMo-V2.5-Pro",
            "OPENCODE_ZEN_MODEL": "deepseek-v4-flash-special",
        },
        clear=True,
    ):
        assert resolve_provider_model("opencode_zen") == "deepseek-v4-flash-special"
        assert resolve_provider_model("mimo") == "MiMo-V2.5-Pro"


def test_resolve_provider_model_does_not_leak_mimo_env_to_opencode() -> None:
    with patch.dict(os.environ, {"MIMO_MODEL": "MiMo-V2.5-Pro"}, clear=True):
        assert resolve_provider_model("opencode_zen") == "deepseek-v4-flash-free"
        assert resolve_provider_model("opencode_zen", "explicit-model") == "explicit-model"


def test_explicit_constructor_rejects_empty_keys() -> None:
    with pytest.raises(ConfigurationError, match="No OpenAI-compatible provider keys"):
        OpenAICompatibleClient([])
    with pytest.raises(ConfigurationError, match="No OpenAI-compatible provider keys"):
        OpenAICompatibleClient(["", "  "])


def test_explicit_constructor_rejects_mismatched_base_urls() -> None:
    with pytest.raises(ConfigurationError, match="base URL count"):
        OpenAICompatibleClient(
            ["k1", "k2"],
            base_urls=[
                "https://one.example",
                "https://two.example",
                "https://three.example",
            ],
        )


def test_openrouter_provider_spec_resolves_with_thinking_off() -> None:
    spec = get_provider_spec("openrouter")
    assert spec.base_url == "https://openrouter.ai/api/v1"
    assert spec.default_model == "google/gemma-4-31b-it:free"
    assert spec.key_env_vars == ("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY")
    assert ("reasoning", {"effort": "none"}) in spec.extra_chat_payload


def test_openai_provider_spec_resolves_with_thinking_off() -> None:
    spec = get_provider_spec("openai")
    assert spec.base_url == "https://api.openai.com/v1"
    assert spec.default_model == "gpt-5.4-mini"
    assert spec.key_env_vars == ("OPENAI_API_KEYS", "OPENAI_API_KEY")
    assert ("reasoning_effort", "none") in spec.extra_chat_payload
    assert spec.max_tokens_param == "max_completion_tokens"


def test_daily_token_budget_trips_when_cap_exceeded() -> None:
    budget = DailyTokenBudget({"openrouter": 1000})
    today = date(2026, 6, 11)
    budget.check_and_add("openrouter", 600, today=today)
    with pytest.raises(DailyBudgetExceeded, match="openrouter"):
        budget.check_and_add("openrouter", 600, today=today)


def test_daily_token_budget_resets_across_utc_day_boundary() -> None:
    budget = DailyTokenBudget({"openrouter": 1000})
    budget.check_and_add("openrouter", 600, today=date(2026, 6, 11))
    budget.check_and_add("openrouter", 600, today=date(2026, 6, 12))
    assert budget.spent("openrouter", date(2026, 6, 11)) == 600
    assert budget.spent("openrouter", date(2026, 6, 12)) == 600


def test_daily_token_budget_ignores_uncapped_provider() -> None:
    budget = DailyTokenBudget({"openrouter": 1000})
    budget.check_and_add("openai", 999_999, today=date(2026, 6, 11))


def test_reconcile_replaces_reservation_with_actual() -> None:
    day = date(2026, 6, 13)
    budget = DailyTokenBudget({"openai": 1_000_000})
    budget.check_and_add("openai", 4096, today=day)
    budget.reconcile("openai", reserved=4096, actual=900, today=day)
    assert budget.spent("openai", day) == 900


def test_billable_tokens_discounts_cached_input() -> None:
    usage = {
        "prompt_tokens": 1822,
        "completion_tokens": 23,
        "total_tokens": 1845,
        "prompt_tokens_details": {"cached_tokens": 1280},
    }
    assert billable_tokens(usage) == 885


def test_billable_tokens_equals_total_without_cache() -> None:
    usage = {"prompt_tokens": 120, "completion_tokens": 780, "total_tokens": 900}
    assert billable_tokens(usage) == 900


def test_billable_tokens_none_when_total_missing() -> None:
    assert billable_tokens({"prompt_tokens": 10}) is None


def test_reconcile_ignores_uncapped_provider() -> None:
    budget = DailyTokenBudget({"openrouter": 1000})
    budget.reconcile("openai", reserved=4096, actual=900, today=date(2026, 6, 11))
    assert budget.spent("openai", date(2026, 6, 11)) == 0


@pytest.mark.anyio
async def test_chat_reconciles_budget_to_actual_usage() -> None:
    """After a call the day's tally reflects the response's real total_tokens.

    The inflated max_tokens estimate does not survive, so the cap tracks true spend.

    estimate reserves ~ (200//3 + 8) + 4096 = 4170; reconcile trues to 900.

    """
    from datetime import datetime, timezone

    budget = DailyTokenBudget({"openai": 1_000_000})
    response = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 780, "total_tokens": 900},
    }
    with respx.mock(assert_all_called=True) as router:
        router.post(MIMO_URL).mock(return_value=httpx.Response(200, json=response))
        async with OpenAICompatibleClient(["k1"], model="m", daily_budget=budget, provider_name="openai") as client:
            await client.chat("s" * 100, "u" * 100, temperature=0.5, max_tokens=4096)
    today = datetime.now(timezone.utc).date()
    assert budget.spent("openai", today) == 900


@pytest.mark.anyio
async def test_chat_reconciles_to_billable_discounting_cache() -> None:
    """A response reporting cached prompt tokens is tallied at charged tokens, not raw total_tokens.

    Discounting the cached half keeps a generation workload that reuses one big cached system
    prompt from being stopped about 1.5x early. total=1845, cached=1280 -> billable
    1845 - round(1280*0.75) = 885.
    """
    from datetime import datetime, timezone

    budget = DailyTokenBudget({"openai": 1_000_000})
    response = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": 1822,
            "completion_tokens": 23,
            "total_tokens": 1845,
            "prompt_tokens_details": {"cached_tokens": 1280},
        },
    }
    with respx.mock(assert_all_called=True) as router:
        router.post(MIMO_URL).mock(return_value=httpx.Response(200, json=response))
        async with OpenAICompatibleClient(["k1"], model="m", daily_budget=budget, provider_name="openai") as client:
            await client.chat("s" * 100, "u" * 100, temperature=0.5, max_tokens=4096)
    today = datetime.now(timezone.utc).date()
    assert budget.spent("openai", today) == 885


@pytest.mark.anyio
async def test_chat_trips_budget_before_second_post() -> None:
    """Zero-spend: when the day's cap would be exceeded, the post never fires.

    Estimate per call = ceil((len(system)+len(user))/4) + max_tokens. With a
    200-char system/user pair (100 each) and max_tokens=500, estimate is
    ceil(200/4)+500 = 550. Cap of 1000 admits the first call (550), the second
    projects 1100 > 1000 and raises before posting.
    """
    post_count = 0

    def capture(_request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        post_count += 1
        return httpx.Response(200, json=_success_response("ok"))

    budget = DailyTokenBudget({"openai": 1000})
    system = "s" * 100
    user = "u" * 100
    with respx.mock(assert_all_called=False) as router:
        router.post(MIMO_URL).mock(side_effect=capture)
        async with OpenAICompatibleClient(["k1"], model="m", daily_budget=budget, provider_name="openai") as client:
            text = await client.chat(system, user, temperature=0.5, max_tokens=500)
            assert text == "ok"
            assert post_count == 1

            with pytest.raises(DailyBudgetExceeded, match="openai"):
                await client.chat(system, user, temperature=0.5, max_tokens=500)

    assert post_count == 1


@pytest.mark.anyio
async def test_chat_without_budget_posts_normally() -> None:
    """The guard is opt-in: no budget injected → every call posts as before."""
    post_count = 0

    def capture(_request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        post_count += 1
        return httpx.Response(200, json=_success_response("ok"))

    with respx.mock(assert_all_called=False) as router:
        router.post(MIMO_URL).mock(side_effect=capture)
        async with OpenAICompatibleClient(["k1"], model="m") as client:
            for _ in range(3):
                await client.chat("s", "u", temperature=0.5, max_tokens=99_999)

    assert post_count == 3


@pytest.mark.anyio
async def test_for_provider_threads_budget_into_client() -> None:
    """A budgeted client can be constructed per provider via for_provider."""
    budget = DailyTokenBudget({"openai": 1000})
    client = OpenAICompatibleClient.for_provider("openai", ["k1"], daily_budget=budget, provider_name="openai")
    try:
        assert client._daily_budget is budget
        assert client._provider_name == "openai"
    finally:
        await client.close()


@pytest.mark.anyio
async def test_openai_payload_uses_max_completion_tokens_and_reasoning_none() -> None:
    bodies: list[dict[str, object]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_success_response("[Jane Doe]<human_name>"))

    with respx.mock(assert_all_called=True) as router:
        router.post("https://api.openai.com/v1/chat/completions").mock(side_effect=capture)
        async with OpenAICompatibleClient.for_provider("openai", ["k1"]) as client:
            await client.chat("s", "u", temperature=0.5, max_tokens=2000)

    body = bodies[0]
    assert body["max_completion_tokens"] == 2000
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "none"


def test_is_hard_daily_cap_detects_cloudflare_neuron_exhaustion() -> None:
    body = (
        '{"errors":[{"message":"AiError: you have used up your daily free '
        'allocation of 10,000 neurons","code":4006}],"success":false}'
    )
    assert _is_hard_daily_cap(body) is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "error_body"),
    [
        (401, {"error": {"code": 401, "message": "Unauthorized: invalid API key."}}),
        (
            403,
            {
                "error": {
                    "code": 403,
                    "status": "PERMISSION_DENIED",
                    "message": "Your project has been denied access. Please contact support.",
                },
            },
        ),
        (404, {"error": {"code": 404, "message": "Model not found."}}),
    ],
)
async def test_fatal_auth_error_trips_account_for_the_run(status_code: int, error_body: dict[str, object]) -> None:
    """A permanent per-credential or per-provider failure pulls the account from rotation for the run.

    The failures are 401 (revoked or invalid key), 403 (denied project) and 404 (missing model).
    Re-hitting a dead credential is the auth-flood pattern that risks a free-tier ban, so after
    one such response the account is gone: the next acquire finds none available and the run
    skips the provider through DailyBudgetExceeded instead of hammering it across every segment.
    """
    ledger = AccountLedger({"gemini": AccountLimits(rpm=10)})
    denied = httpx.Response(status_code, json=error_body)
    with respx.mock(assert_all_called=False) as router:
        router.post(MIMO_URL).mock(return_value=denied)
        async with OpenAICompatibleClient(
            ["k1"],
            model="m",
            account_ledger=ledger,
            provider_name="gemini",
            account_ids=["acct1"],
        ) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.chat("s", "u", temperature=0.5, max_tokens=10)
            with pytest.raises(DailyBudgetExceeded):
                await client.chat("s", "u", temperature=0.5, max_tokens=10)


def test_is_hard_daily_cap_detects_openrouter_free_daily() -> None:
    body = (
        '{"error":{"message":"Rate limit exceeded: free-models-per-day. Add 10 '
        'credits to unlock 1000 free model requests per day","code":429}}'
    )
    assert _is_hard_daily_cap(body) is True


def test_is_hard_daily_cap_ignores_transient_per_minute_throttle() -> None:
    body = '{"error":{"message":"Rate limit reached, please slow down."}}'
    assert _is_hard_daily_cap(body) is False


def test_parse_retry_after_reads_delta_seconds() -> None:
    assert _parse_retry_after("30") == 30.0
    assert _parse_retry_after("0") == 0.0


def test_parse_retry_after_none_when_absent_or_http_date() -> None:
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("Wed, 21 Oct 2025 07:28:00 GMT") is None


@pytest.mark.anyio
async def test_hard_daily_cap_429_trips_account_in_ledger() -> None:
    """A hard daily-cap 429 takes the account out of the ledger so the run stops hammering it.

    With one account the retry then finds nothing available, and the cell ends with
    DailyBudgetExceeded rather than a 429 storm.
    """
    ledger = AccountLedger({"groq": AccountLimits(rpm=10, rpd=100)})
    body = '{"errors":[{"message":"used up your daily free allocation of 10,000 neurons","code":4006}]}'
    with respx.mock(assert_all_called=True) as router:
        router.post("https://api.groq.com/openai/v1/chat/completions").mock(return_value=httpx.Response(429, text=body))
        async with OpenAICompatibleClient(
            ["k1"],
            model="m",
            base_urls=["https://api.groq.com/openai/v1"],
            provider_name="groq",
            account_ledger=ledger,
            account_ids=["acct1"],
        ) as client:
            with pytest.raises(DailyBudgetExceeded):
                await client.chat("s", "u", temperature=0.5, max_tokens=10)
    assert ledger.choose("groq", ["acct1"]) == (None, 0.0)
