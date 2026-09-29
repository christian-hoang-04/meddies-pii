"""Pure quota policy shared by generation usage views and pool health checks."""

from __future__ import annotations

OPENROUTER_UNFUNDED_RPD = 50
"""OpenRouter's :free models cap at 50 requests/day for unfunded accounts and 1000 for funded.

>=10 lifetime credits). The funding status is not in a usage record, but >50 requests in a day proves an account is funded
because an unfunded account trips at 50. A funded account at <=50 requests displays against 50; that affects only the
utilization view, never generation.

"""
OPENROUTER_FUNDED_RPD = 1000

NEURON_METERED = frozenset({"cloudflare"})
"""Providers capped by tokens/neurons per day, not by request count.

Their stored RPD is a throttle proxy, so it is not a utilization denominator.

"""


def account_rpd_cap(provider: str, requests: int, rpd_per_account: int | None) -> int | None:
    """Return the real daily request cap for one provider account.

    ``None`` means the provider is token/neuron-metered rather than request-capped.

    Returns:
        The real daily request cap, or ``None`` for a token- or neuron-metered provider.
        OpenRouter is inferred rather than configured: an account that has already made more
        requests than the unfunded ceiling must be funded, so it is read at the funded cap. The
        stored per-account RPD is used for every other request-capped provider.

    """
    if provider == "openrouter":
        return OPENROUTER_FUNDED_RPD if requests > OPENROUTER_UNFUNDED_RPD else OPENROUTER_UNFUNDED_RPD
    if provider in NEURON_METERED:
        return None
    return rpd_per_account
