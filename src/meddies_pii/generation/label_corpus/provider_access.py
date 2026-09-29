from __future__ import annotations

# ruff: file-ignore[useless-import-alias]
# reason: an explicit `X as X` re-export, which is this module's published surface: ruff's own
# reason: unsafe fix DELETED five of these once and broke every caller, so the alias stays.
# ruff: file-ignore[try-except-in-loop]
# reason: a record-and-continue harness: each item's outcome is the product, so an exception has to
# reason: become that item's recorded result. Hoisting the try would hide every item after the first.
from dataclasses import dataclass

import httpx

from meddies_pii.generation.openai_compatible.client import OpenAICompatibleClient
from meddies_pii.generation.openai_compatible.providers import (
    provider_keys as provider_keys,
)
from meddies_pii.generation.openai_compatible.providers import (
    resolve_provider_base_urls,
)


@dataclass(frozen=True, slots=True)
class KeyPreflightResult:
    key_index: int
    ok: bool
    status: int | None = None
    error_type: str | None = None
    error: str | None = None


async def preflight_keys(
    *,
    provider: str,
    model: str | None,
    keys: list[str],
    base_urls: list[str],
    timeout: float = 60.0,
) -> list[KeyPreflightResult]:
    results: list[KeyPreflightResult] = []
    for index, key in enumerate(keys, start=1):
        try:
            async with OpenAICompatibleClient.for_provider(
                provider,
                [key],
                model=model,
                base_urls=[base_urls[index - 1]],
                rpm_per_key=1,
                timeout=timeout,
            ) as client:
                await client.chat(
                    "You are a health-check endpoint. Return exactly OK.",
                    "Return OK only.",
                    temperature=0,
                    max_tokens=32,
                )
            results.append(KeyPreflightResult(key_index=index, ok=True, status=200))
        # reason: a key preflight sweep must report every key's health, so any failure has to become this
        # reason: key's recorded result rather than abort the sweep and hide the keys after it.
        except Exception as exc:  # ruff: ignore[blind-except]
            status = None
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
            else:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
            results.append(
                KeyPreflightResult(
                    key_index=index,
                    ok=False,
                    status=status,
                    error_type=type(exc).__name__,
                    error=str(exc).splitlines()[0][:200],
                ),
            )
    return results


def configured_base_urls(provider: str, key_count: int) -> list[str]:
    return resolve_provider_base_urls(provider, key_count) if key_count else []
