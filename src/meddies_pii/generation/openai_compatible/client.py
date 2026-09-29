"""Asynchronous transport for OpenAI-compatible chat-completions APIs.

Provider configuration and environment resolution live in :mod:`.providers`.
Token accounting and rate limiting live in :mod:`.quota`. This client owns HTTP
transport, credential rotation, retry classification, and response handling.
"""

from __future__ import annotations

# ruff: file-ignore[private-member-access]
# reason: the seven names this module reads off its `providers` and `quota` siblings are private to
# reason: the `openai_compatible` subpackage, not to their defining module: each is referenced by
# reason: exactly two files, its definer and this one, and nothing outside the subpackage or in any
# reason: test touches one. Dropping the underscore would advertise them as package-public, which
# reason: is the false statement; the access is a sibling reading its own subpackage's internals.
import asyncio
import random
from typing import TYPE_CHECKING, Self, cast

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from meddies_pii import exceptions

from . import providers, quota

if TYPE_CHECKING:
    from types import TracebackType

    from meddies_pii.generation.account_ledger import AccountLedger

RATE_LIMIT_STATUS_CODE = 429
SERVER_ERROR_STATUS_FLOOR = 500

_DEFAULT_RPM_PER_KEY = 100
_DEFAULT_TIMEOUT = 180.0


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient failures only: network errors, 429, and 5xx.

    4xx-not-429 (auth, bad request, not found) is a configuration or
    contract error — burning five attempts on the same key wastes time
    and obscures the actual failure. Let it propagate.

    Returns:
        ``True`` for any transport-level error and for a 429 or 5xx response, ``False`` for
        everything else including every other 4xx. The default is not to retry, so an exception
        type this does not recognize propagates on its first occurrence rather than being
        retried five times under the assumption that it is transient.

    """
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == RATE_LIMIT_STATUS_CODE or status >= SERVER_ERROR_STATUS_FLOOR
    return False


def _parse_retry_after(value: str | None) -> float | None:
    """Read a ``Retry-After`` header in its delta-seconds form.

    The header is either delta-seconds or an HTTP-date; the free
    providers use the seconds form. Returns the seconds to wait, or ``None`` when
    absent or in the date form (the caller applies a default cooldown then).

    Returns:
        The wait in seconds, floored at zero so a negative or past value cannot become a
        negative sleep, or ``None`` when the header is absent or in the HTTP-date form. ``None``
        means "no parseable hint", not "no wait" -- the caller applies its default cooldown.

    """
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _is_hard_daily_cap(body: str) -> bool:
    """Report whether a 429 body signals a daily ceiling rather than a throttle.

    A daily ceiling will not clear until the
    provider's reset (so the account should sit out the rest of the run), vs a
    transient per-minute throttle. Covers Cloudflare's neuron cap (error 4006),
    OpenRouter's unfunded free-models-per-day, and a generic daily-limit message.

    Returns:
        ``True`` when the body matches one of the known daily-ceiling signatures, so the caller
        trips the account for the rest of the run instead of applying a cooldown. ``False`` is
        the safe default: an unrecognized 429 is treated as a per-minute throttle, which costs
        a retry rather than idling an account that was never actually capped.

    """
    b = body.lower()
    return (
        '"code":4006' in b.replace(" ", "")
        or "free-models-per-day" in b
        or ("daily" in b and "limit" in b)
        or "used up your daily" in b
    )


_DEFAULT_429_COOLDOWN = 30.0
"""Transient 429 with no Retry-After header.

How long to sit the account out so we don't immediately re-hit it (a polite default, overridden by Retry-After).

"""

_REQUEST_JITTER_RANGE = (0.5, 2.0)
"""Random pause before each pooled request (seconds).

On top of the ledger's hard rate throttle, this makes traffic organic rather than robotic and spreads the accounts' opening
requests so they don't fire in a single correlated burst — the pattern that flags free-tier multi-account use. Applied only
on the ledger path (the free-tier pool), never the default single-client path.

"""


class OpenAICompatibleClient:
    # reason: Keys, endpoints, quotas, budgets, and account ledger are the client constructor contract.
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        api_keys: list[str],
        *,
        model: str = providers._DEFAULT_MODEL,
        base_url: str = providers._DEFAULT_BASE_URL,
        base_urls: list[str] | tuple[str, ...] | None = None,
        extra_chat_payload: tuple[tuple[str, object], ...] = (),
        max_tokens_param: str = "max_tokens",
        rpm_per_key: int = _DEFAULT_RPM_PER_KEY,
        timeout: float = _DEFAULT_TIMEOUT,
        daily_budget: quota.DailyTokenBudget | None = None,
        provider_name: str | None = None,
        keyless: bool = False,
        account_ledger: AccountLedger | None = None,
        account_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Stable.

        Non-secret per-account ids shared across the run's client instances so the ledger tracks the same account each
        time. A per-account worker passes the account's GLOBAL id (e.g. ["acct3"]) for its single key; otherwise ids are
        generated 1-based for the keys held here.

        Raises:
            ConfigurationError: If no non-blank key was supplied and the provider is not
                declared keyless. Keys are stripped and blanks dropped first, so a variable set
                to whitespace fails here rather than reaching the provider as an empty bearer
                token and coming back as a confusing 401.

        """
        clean_keys = [k.strip() for k in api_keys if k.strip()]
        if not clean_keys:
            if keyless:
                clean_keys = [""]
            else:
                msg = "No OpenAI-compatible provider keys provided."
                raise exceptions.ConfigurationError(msg)

        self._keyless: bool = keyless
        self._api_keys: list[str] = clean_keys
        self._account_ledger: AccountLedger | None = account_ledger
        if account_ids is not None:
            self._account_ids = list(account_ids)
        else:
            self._account_ids = [f"acct{i + 1}" for i in range(len(clean_keys))]
        self._base_urls: list[str] = providers._resolve_base_urls(
            base_url=base_url,
            base_urls=base_urls,
            key_count=len(clean_keys),
        )
        self._key_index: int = 0
        self._key_lock: asyncio.Lock = asyncio.Lock()
        self._model: str = model
        self._daily_budget: quota.DailyTokenBudget | None = daily_budget
        self._provider_name: str | None = provider_name
        self._extra_chat_payload: dict[str, object] = dict(extra_chat_payload)
        self._max_tokens_param: str = max_tokens_param
        self._rate_limiter: quota.AsyncRateLimiter = quota.AsyncRateLimiter(rpm_per_key * len(clean_keys))
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    # reason: OpenAICompatibleClient exposes provider/account as its public contract; bundling would break callers.
    @classmethod
    def from_env(  # ruff: ignore[too-many-arguments]
        cls,
        *,
        provider: str = providers._DEFAULT_PROVIDER,
        model: str | None = None,
        rpm_per_key: int = _DEFAULT_RPM_PER_KEY,
        timeout: float = _DEFAULT_TIMEOUT,
        daily_budget: quota.DailyTokenBudget | None = None,
        account_ledger: AccountLedger | None = None,
    ) -> OpenAICompatibleClient:
        spec = providers.get_provider_spec(provider)
        keys = providers.provider_keys(provider)
        if not keys:
            if spec.keyless:
                keys = [""]
            else:
                joined_env_vars = " or ".join(spec.key_env_vars)
                msg = f"{joined_env_vars} not set or empty in environment."
                raise exceptions.ConfigurationError(msg)
        return cls.for_provider(
            provider,
            keys,
            model=model,
            rpm_per_key=rpm_per_key,
            timeout=timeout,
            daily_budget=daily_budget,
            account_ledger=account_ledger,
            provider_name=providers.normalize_provider_name(provider),
        )

    # reason: Provider keys, endpoints, model resolution, budgets, and account IDs are the factory contract.
    @classmethod
    def for_provider(  # ruff: ignore[too-many-arguments]
        cls,
        provider: str,
        api_keys: list[str],
        *,
        model: str | None = None,
        base_urls: list[str] | tuple[str, ...] | None = None,
        rpm_per_key: int = _DEFAULT_RPM_PER_KEY,
        timeout: float = _DEFAULT_TIMEOUT,
        daily_budget: quota.DailyTokenBudget | None = None,
        account_ledger: AccountLedger | None = None,
        account_ids: list[str] | tuple[str, ...] | None = None,
        provider_name: str | None = None,
    ) -> OpenAICompatibleClient:
        spec = providers.get_provider_spec(provider)
        resolved_base_urls = (
            list(base_urls) if base_urls is not None else providers.resolve_provider_base_urls(provider, len(api_keys))
        )
        return cls(
            api_keys,
            model=providers.resolve_provider_model(provider, model),
            base_url=providers._read_first_env(spec.base_url_env_vars) or spec.base_url,
            base_urls=resolved_base_urls,
            extra_chat_payload=spec.extra_chat_payload,
            max_tokens_param=spec.max_tokens_param,
            rpm_per_key=rpm_per_key,
            timeout=timeout,
            daily_budget=daily_budget,
            provider_name=provider_name or providers.normalize_provider_name(provider),
            keyless=spec.keyless,
            account_ledger=account_ledger,
            account_ids=account_ids,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(_is_retryable),
    )
    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Two estimates, deliberately different.

        The cost budget uses the CONSERVATIVE max_tokens-based count (over-count is safe for a hard spend cap),
        checked/reserved before the post so a raise means zero spend, trued up to *billable* tokens afterward. The ledger's
        TPM/TPD gate uses a REALISTIC count (prompt + typical completion) so a tight-TPM provider isn't false-skipped; the
        ledger records actual raw tokens after the post, keeping the running TPM/TPD exact.

        Organic spacing on the free-tier pool — see _REQUEST_JITTER_RANGE.

        401/403/404 are permanent failures for this credential, never transient: a revoked or invalid key (401), a denied
        project (403, e.g. Google AI Studio "project denied access"), or a missing model (404). Pull the account from
        rotation for the run — re-hitting a dead credential across every segment is the auth-flood pattern that risks a
        free-tier ban. The pool runs one model per provider, so a model-level failure is provider-fatal here; other
        accounts go on. This set matches synthetic.py's fatal_api_error classification, so the two layers agree on what is
        fatal.

        Cost budget (OpenAI grant) trues up to BILLABLE tokens — cached prompt prefixes appear full in total_tokens but
        bill at a discount.

        Returns:
            The assistant message content as a string. Both meters are trued up from the
            response's own usage before it returns, so the budget and the ledger describe what
            the provider actually billed and counted rather than the pre-call estimates.

        Rate-limit ledger records RAW provider tokens (what TPM/TPD meter), not the billable figure. Falls back to the
        estimate when usage is absent.

        Reasoning models put their thinking in ``reasoning_content`` and may omit or null ``content`` (esp. when thinking
        is on and the answer was truncated). Some providers also return structured ``content`` (a list of parts) rather
        than a string. Extract defensively: anything that isn't a plain string becomes empty text, which the generator
        rejects as a failed attempt instead of violating the ``-> str`` contract and crashing downstream. The real fix is
        disabling thinking per provider.

        """
        estimate: int | None = None
        rate_limit_estimate = 0
        if self._provider_name is not None and (self._daily_budget is not None or self._account_ledger is not None):
            estimate = quota._estimate_request_tokens(system, user, max_tokens)
            rate_limit_estimate = quota._estimate_rate_limit_tokens(system, user, max_tokens)
            if self._daily_budget is not None:
                self._daily_budget.check_and_add(self._provider_name, estimate)

        key, base_url, account_id = await self._acquire_credential(rate_limit_estimate)
        if self._account_ledger is not None:
            # reason: this is anti-thundering-herd jitter, not a secret. It desynchronizes concurrent workers so a
            # reason: shared free-tier account is not hit by a burst; an attacker predicting the delay gains
            # reason: nothing, and the ledger, not this sleep, is what enforces the rate limit.
            await asyncio.sleep(random.uniform(*_REQUEST_JITTER_RANGE))  # ruff: ignore[suspicious-non-cryptographic-random-usage]

        payload: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            self._max_tokens_param: max_tokens,
        }
        payload.update(self._extra_chat_payload)
        headers = {} if self._keyless else {"Authorization": f"Bearer {key}"}

        response = await self._client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        if (
            response.status_code == RATE_LIMIT_STATUS_CODE
            and account_id is not None
            and self._account_ledger is not None
            and self._provider_name is not None
        ):
            self._record_rate_limit(response, account_id)
        if (
            response.status_code in {401, 403, 404}
            and account_id is not None
            and self._account_ledger is not None
            and self._provider_name is not None
        ):
            self._account_ledger.trip(self._provider_name, account_id)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else None
        billable = quota.billable_tokens(cast("dict[str, object]", usage)) if usage else None
        raw_total = usage.get("total_tokens") if usage else None
        raw_total = raw_total if isinstance(raw_total, int) else None
        if (
            estimate is not None
            and billable is not None
            and self._daily_budget is not None
            and self._provider_name is not None
        ):
            self._daily_budget.reconcile(self._provider_name, reserved=estimate, actual=billable)
        if account_id is not None and self._account_ledger is not None and self._provider_name is not None:
            self._account_ledger.add_tokens(
                self._provider_name,
                account_id,
                raw_total if raw_total is not None else (estimate or 0),
            )
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""

    def _record_rate_limit(self, response: httpx.Response, account_id: str) -> None:
        """On a 429, take the account out of the ledger's rotation.

        This stops us
        hammering it — the sustained-429 pattern that gets free-tier accounts
        flagged. A hard daily-cap signal blocks it for the run; otherwise honor
        ``Retry-After`` (or a default cooldown) as a transient block. The retry
        decorator then picks a different — or cooled-down — account.
        """
        # reason: narrowing, not validation. The only call site guards on both of these in the same condition
        # reason: before calling (`self._account_ledger is not None and self._provider_name is not None`), so
        # reason: these two lines exist to carry that proof across the call boundary for the type checker.
        assert self._account_ledger is not None  # ruff: ignore[assert]
        assert self._provider_name is not None  # ruff: ignore[assert]
        try:
            body = response.text
        # reason: narrowing was measured and rejected. `httpx.Response.text` realistically raises only
        # reason: ResponseNotRead, but this runs on the 429 path whose job is to trip the account out of
        # reason: rotation; anything escaping here would leave a flagged account being hammered, which is
        # reason: the exact abuse this method exists to stop. An empty body falls through to the cooldown.
        except Exception:  # ruff: ignore[blind-except]
            body = ""
        if _is_hard_daily_cap(body):
            self._account_ledger.trip(self._provider_name, account_id)
            return
        cooldown = _parse_retry_after(response.headers.get("retry-after"))
        self._account_ledger.trip(
            self._provider_name,
            account_id,
            cooldown=cooldown if cooldown is not None else _DEFAULT_429_COOLDOWN,
        )

    async def _acquire_credential(self, est_tokens: int = 0) -> tuple[str, str, str | None]:
        """Pick ``(key, base_url, account_id)`` respecting per-account limits.

        With a ledger: choose an account under all four axes (RPM/RPD/TPM/TPD)
        for a request of ~``est_tokens``, sleep briefly when accounts are
        momentarily at a per-minute cap, and raise ``exceptions.DailyBudgetExceeded`` when
        every account has hit a daily ceiling (RPD or TPD) — so the run skips
        this cell. Without a ledger: fall back to the shared RPM limiter +
        round-robin.

        Returns:
            The key, its aligned base URL, and the account id -- ``None`` for the account id on
            the no-ledger path, which is what tells the caller there is no per-account state to
            true up afterwards. With a ledger the account is reserved before returning, so two
            concurrent callers cannot both spend the same account's last request.

        Raises:
            DailyBudgetExceeded: When every account has hit a daily ceiling, RPD or TPD. A
                per-minute cap is not this case: it sleeps and retries, because that clears on
                its own. Only a daily ceiling ends the cell, since nothing will free it before
                the provider's reset.

        """
        if self._account_ledger is None or self._provider_name is None:
            await self._rate_limiter.wait()
            key, base_url = await self._next_credential()
            return key, base_url, None
        while True:
            async with self._key_lock:
                account_id, wait = self._account_ledger.choose(self._provider_name, self._account_ids, est_tokens)
                if account_id is None:
                    msg = f"All {self._provider_name!r} accounts hit their daily request limit."
                    raise exceptions.DailyBudgetExceeded(
                        msg,
                        context={"provider": self._provider_name},
                    )
                if wait <= 0:
                    self._account_ledger.reserve(self._provider_name, account_id)
                    index = self._account_ids.index(account_id)
                    return self._api_keys[index], self._base_urls[index], account_id
            await asyncio.sleep(wait)

    async def _next_credential(self) -> tuple[str, str]:
        async with self._key_lock:
            key = self._api_keys[self._key_index]
            base_url = self._base_urls[self._key_index]
            self._key_index = (self._key_index + 1) % len(self._api_keys)
            return key, base_url
