"""Typed usage records and their persisted-record semantics."""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
from typing import TYPE_CHECKING, NotRequired, TypedDict

from meddies_pii.json_types import is_str_mapping
from meddies_pii.json_types import required_int as _required_int

if TYPE_CHECKING:
    from collections.abc import Mapping


class AccountUsage(TypedDict):
    """One account's persisted request and token counters."""

    requests: int
    tokens: int


class ProviderLimits(TypedDict):
    """Documented, un-headroomed per-account provider ceilings."""

    accounts: NotRequired[int]
    token_cap_per_account: NotRequired[int | None]
    rpd_per_account: NotRequired[int | None]


class ProviderRunStats(TypedDict):
    """Provider sample counts gathered while a weak-label run is active."""

    accepted: int
    rejected: int
    by_language: Mapping[str, int]
    model: NotRequired[str]


class ProviderUsage(TypedDict):
    """One provider's durable daily usage record."""

    model: str
    accepted: int
    rejected: int
    tokens: int
    requests: int
    by_language: dict[str, int]
    accounts: dict[str, AccountUsage]
    limits: ProviderLimits


class UsageTotals(TypedDict):
    accepted: int
    rejected: int
    tokens: int
    requests: int


class UsageRecord(TypedDict):
    """Validated JSON contract written to ``usage/<date>.json``."""

    date: str
    providers: dict[str, ProviderUsage]
    totals: UsageTotals


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, field=field)


def _object_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not is_str_mapping(value):
        msg = f"{field} must be an object with string keys"
        raise ValueError(msg)
    return value


def _decode_account_usage(value: object, *, field: str) -> AccountUsage:
    raw = _object_mapping(value, field=field)
    return {
        "requests": _required_int(raw.get("requests", 0), field=f"{field}.requests"),
        "tokens": _required_int(raw.get("tokens", 0), field=f"{field}.tokens"),
    }


def _decode_limits(value: object, *, field: str) -> ProviderLimits:
    raw = _object_mapping(value, field=field)
    result: ProviderLimits = {}
    for key in ("accounts", "token_cap_per_account", "rpd_per_account"):
        if key not in raw:
            continue
        if key == "accounts":
            result[key] = _required_int(raw[key], field=f"{field}.{key}")
        else:
            result[key] = _optional_int(raw[key], field=f"{field}.{key}")
    return result


def _decode_provider_usage(value: object, *, field: str) -> ProviderUsage:
    raw = _object_mapping(value, field=field)
    model = raw.get("model", "")
    if not isinstance(model, str):
        msg = f"{field}.model must be a string"
        raise ValueError(msg)
    languages_raw = _object_mapping(raw.get("by_language", {}), field=f"{field}.by_language")
    accounts_raw = _object_mapping(raw.get("accounts", {}), field=f"{field}.accounts")
    return {
        "model": model,
        "accepted": _required_int(raw.get("accepted", 0), field=f"{field}.accepted"),
        "rejected": _required_int(raw.get("rejected", 0), field=f"{field}.rejected"),
        "tokens": _required_int(raw.get("tokens", 0), field=f"{field}.tokens"),
        "requests": _required_int(raw.get("requests", 0), field=f"{field}.requests"),
        "by_language": {
            language: _required_int(count, field=f"{field}.by_language.{language}")
            for language, count in languages_raw.items()
        },
        "accounts": {
            account_id: _decode_account_usage(account, field=f"{field}.accounts.{account_id}")
            for account_id, account in accounts_raw.items()
        },
        "limits": _decode_limits(raw.get("limits", {}), field=f"{field}.limits"),
    }


def decode_usage_record(value: object) -> UsageRecord:
    """Validate one persisted usage JSON object at the file boundary.

    Returns:
        The record as the typed shape the dashboard and monitor read, with every provider,
        account, limit and total decoded and coerced. Validating once here means no consumer
        downstream has to re-check a field.

    Raises:
        ValueError: If the date is missing or not a non-empty string, or if any nested block is
            not an object or holds a non-integer where a count belongs. Every message names the
            failing field path, so a malformed day identifies itself. The dashboard catches
            this per file and skips that day rather than failing the whole render.

    """
    raw = _object_mapping(value, field="usage record")
    date = raw.get("date")
    if not isinstance(date, str) or not date:
        msg = "usage record.date must be a non-empty string"
        raise ValueError(msg)
    providers_raw = _object_mapping(raw.get("providers", {}), field="usage record.providers")
    providers = {
        provider: _decode_provider_usage(provider_usage, field=f"usage record.providers.{provider}")
        for provider, provider_usage in providers_raw.items()
    }
    totals_raw = _object_mapping(raw.get("totals", {}), field="usage record.totals")
    totals: UsageTotals = {
        "accepted": _required_int(totals_raw.get("accepted", 0), field="usage record.totals.accepted"),
        "rejected": _required_int(totals_raw.get("rejected", 0), field="usage record.totals.rejected"),
        "tokens": _required_int(totals_raw.get("tokens", 0), field="usage record.totals.tokens"),
        "requests": _required_int(totals_raw.get("requests", 0), field="usage record.totals.requests"),
    }
    return {"date": date, "providers": providers, "totals": totals}


def _copy_limits(limits: ProviderLimits) -> ProviderLimits:
    copied: ProviderLimits = {}
    if "accounts" in limits:
        copied["accounts"] = limits["accounts"]
    if "token_cap_per_account" in limits:
        copied["token_cap_per_account"] = limits["token_cap_per_account"]
    if "rpd_per_account" in limits:
        copied["rpd_per_account"] = limits["rpd_per_account"]
    return copied


def build_usage_record(
    run_date: str,
    per_provider: Mapping[str, ProviderRunStats],
    accounts: Mapping[str, Mapping[str, AccountUsage]],
    limits: Mapping[str, ProviderLimits] | None = None,
) -> UsageRecord:
    """Fold run counters and an account-ledger snapshot into one typed daily record.

    Returns:
        One day's record: per-provider accepted, rejected, token and request counts, the
        per-account ledger snapshot, the provider limits, and the totals summed across all of
        them. Providers are the union of the two inputs, so a provider that appears only in the
        ledger snapshot still gets a row with zero counters rather than being dropped.

    """
    limits = limits or {}
    providers: dict[str, ProviderUsage] = {}
    totals: UsageTotals = {"accepted": 0, "rejected": 0, "tokens": 0, "requests": 0}
    for provider in sorted(set(per_provider) | set(accounts)):
        stats = per_provider.get(provider)
        account_map = dict(accounts.get(provider, {}))
        accepted = stats["accepted"] if stats is not None else 0
        rejected = stats["rejected"] if stats is not None else 0
        tokens = sum(account["tokens"] for account in account_map.values())
        requests = sum(account["requests"] for account in account_map.values())
        providers[provider] = {
            "model": stats.get("model", "") if stats is not None else "",
            "accepted": accepted,
            "rejected": rejected,
            "tokens": tokens,
            "requests": requests,
            "by_language": dict(stats["by_language"]) if stats is not None else {},
            "accounts": account_map,
            "limits": _copy_limits(limits.get(provider, {})),
        }
        totals["accepted"] += accepted
        totals["rejected"] += rejected
        totals["tokens"] += tokens
        totals["requests"] += requests
    return {"date": run_date, "providers": providers, "totals": totals}


def merge_usage(old: UsageRecord, new: UsageRecord) -> UsageRecord:
    providers: dict[str, ProviderUsage] = {}
    for provider in sorted(set(old["providers"]) | set(new["providers"])):
        earlier = old["providers"].get(provider)
        later = new["providers"].get(provider)
        if earlier is None:
            # reason: narrowing, and true by construction. `provider` is drawn from the union of both records'
            # reason: provider keys, so absence from `old` means presence in `new`. The assert states for the
            # reason: type checker what the loop's own iteration set already guarantees.
            assert later is not None  # ruff: ignore[assert]
            providers[provider] = later
            continue
        if later is None:
            providers[provider] = earlier
            continue
        account_ids = sorted(set(earlier["accounts"]) | set(later["accounts"]))
        accounts: dict[str, AccountUsage] = {
            account_id: {
                "requests": earlier["accounts"].get(account_id, {"requests": 0, "tokens": 0})["requests"]
                + later["accounts"].get(account_id, {"requests": 0, "tokens": 0})["requests"],
                "tokens": earlier["accounts"].get(account_id, {"requests": 0, "tokens": 0})["tokens"]
                + later["accounts"].get(account_id, {"requests": 0, "tokens": 0})["tokens"],
            }
            for account_id in account_ids
        }
        languages = {
            language: max(
                earlier["by_language"].get(language, 0),
                later["by_language"].get(language, 0),
            )
            for language in sorted(set(earlier["by_language"]) | set(later["by_language"]))
        }
        providers[provider] = {
            "model": later["model"] or earlier["model"],
            "accepted": max(earlier["accepted"], later["accepted"]),
            "rejected": max(earlier["rejected"], later["rejected"]),
            "tokens": earlier["tokens"] + later["tokens"],
            "requests": earlier["requests"] + later["requests"],
            "by_language": languages,
            "accounts": accounts,
            "limits": later["limits"] or earlier["limits"],
        }
    totals: UsageTotals = {
        "accepted": sum(provider["accepted"] for provider in providers.values()),
        "rejected": sum(provider["rejected"] for provider in providers.values()),
        "tokens": sum(provider["tokens"] for provider in providers.values()),
        "requests": sum(provider["requests"] for provider in providers.values()),
    }
    return {
        "date": new["date"] or old["date"],
        "providers": providers,
        "totals": totals,
    }
