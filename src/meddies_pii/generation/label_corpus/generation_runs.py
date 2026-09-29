from __future__ import annotations

# ruff: file-ignore[try-except-in-loop]
# reason: the try IS this loop's per-item parse decision -- the failing item is skipped, recorded or
# reason: partitioned by name. Hoisting it would discard which item failed and abort the rest.
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.exceptions import DailyBudgetExceeded
from meddies_pii.generation.label_corpus.generation_policy import (
    DEFAULT_LANGUAGE_KEYS,
    OUT_ROOT,
    PROVIDER_CONCURRENCY,
    SYNTHETIC_ROOT,
    WEAK_LABEL_FREE_PROVIDERS,
    _provider_limits,
    build_account_ledger,
    build_budget,
)
from meddies_pii.generation.label_corpus.runner import (
    SyntheticGenerationRequest,
    configured_base_urls,
    provider_keys,
    run_synthetic_generation,
    summary_without_rows,
)
from meddies_pii.generation.label_corpus.synthetic import language_paths
from meddies_pii.generation.openai_compatible.providers import (
    get_provider_spec,
    resolve_provider_model,
)
from meddies_pii.generation.usage import (
    AccountUsage,
    ProviderLimits,
    ProviderRunStats,
    build_usage_record,
    render_usage_dashboard,
    write_usage_record,
)
from meddies_pii.jsonl import count_jsonl
from meddies_pii.languages import normalize_language

if TYPE_CHECKING:
    from meddies_pii.generation.account_ledger import AccountLedger
    from meddies_pii.generation.openai_compatible.quota import DailyTokenBudget

logger = logging.getLogger(__name__)

type GenerateOne = Callable[..., Awaitable[dict[str, object]]]


def neediest_first(out_dir: str | Path, languages: Sequence[str]) -> list[str]:
    def current(lang: str) -> int:
        path = language_paths(out_dir, lang).accepted
        return count_jsonl(path) if path.exists() else 0

    return sorted(languages, key=current)


DEFAULT_LANGUAGE_TARGET = 15_000
"""Per-language corpus targets: the total accepted samples wanted per language across the whole corpus.

Vietnamese (primary market) and English (the eval baseline, historically the most starved language) get the higher target;
the rest share the default. The weak-label scheduler serves the neediest language first and stops one once it reaches its
target, so the corpus fills evenly instead of piling onto whichever languages the dominant provider emits fastest.

"""
LANGUAGE_TARGET_OVERRIDES = {"vietnamese": 20_000, "english": 20_000}


def language_target(language: str) -> int:
    return LANGUAGE_TARGET_OVERRIDES.get(language, DEFAULT_LANGUAGE_TARGET)


def languages_by_deficit(current: Mapping[str, int], languages: Sequence[str]) -> list[str]:
    """Order languages neediest-first by remaining deficit to their target.

    Any language already at or above target is dropped. Ties broken by name so the schedule
    is stable across rounds and concurrent accounts.

    Returns:
        The languages still short of their target, neediest first. A language at or past
        target is dropped rather than ordered last, so an exhausted schedule returns empty and
        the caller stops rather than re-filling what is already complete.

    """
    scored = [(lang, language_target(lang) - current.get(lang, 0)) for lang in languages]
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
    return [lang for lang, deficit in ranked if deficit > 0]


def cumulative_language_counts(out_root: str | Path, languages: Sequence[str]) -> dict[str, int]:
    """Total accepted samples per language across the whole synthetic corpus.

    Counted over every date, provider and account, keyed by language. This is the baseline
    the deficit scheduler measures targets against — counting only today's run
    dir would re-fill languages that are already complete from prior days.

    Returns:
        Accepted-sample totals keyed by the caller's own language spelling, not the normalized
        code, so the result drops straight into the deficit scheduler. A language with no rows
        anywhere maps to 0 rather than being absent, which keeps it in the schedule.

    """
    root = Path(out_root)
    counts: dict[str, int] = {}
    for lang in languages:
        code = normalize_language(lang).code
        counts[lang] = sum(count_jsonl(path) for path in root.glob(f"**/accepted.{code}.jsonl"))
    return counts


# reason: generate one exposes provider/max as its public contract; bundling would break callers.
async def generate_one(  # ruff: ignore[too-many-arguments]
    *,
    provider: str,
    model: str | None,
    language: str,
    target_count: int,
    out_dir: str | Path,
    scenario_names: tuple[str, ...],
    required_label_mode: str,
    budget: DailyTokenBudget,
    split_purpose: str = "train",
    count_mode: str = "total",
    account_ledger: AccountLedger | None = None,
    api_keys: Sequence[str] | None = None,
    base_urls: Sequence[str] | None = None,
    account_ids: Sequence[str] | None = None,
    max_concurrency: int = 4,
) -> dict[str, object]:
    return await run_synthetic_generation(
        SyntheticGenerationRequest(
            provider=provider,
            model=model,
            language=language,
            target_count=target_count,
            output_dir=out_dir,
            domain_profile="medical",
            split_purpose=split_purpose,
            scenario_names=scenario_names,
            required_label_mode=required_label_mode,
            count_mode=count_mode,
            require_vietnamese_marker=True,
            daily_budget=budget,
            account_ledger=account_ledger,
            api_keys=api_keys,
            base_urls=base_urls,
            account_ids=account_ids,
            max_concurrency=max_concurrency,
        ),
    )


def strip_rows(summaries: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {label: summary_without_rows(summary) for label, summary in summaries.items()}


async def run_eval_gold(
    per_lang: int = 200,
    languages: Sequence[str] = DEFAULT_LANGUAGE_KEYS,
    token_cap: int | None = None,
    *,
    out_root: str | Path = OUT_ROOT,
    generate_one: GenerateOne = generate_one,
) -> dict[str, object]:
    budget = build_budget(token_cap)
    out = str(Path(out_root) / "eval_gold")
    # reason: one setup syscall before any concurrent work is scheduled, not a per-item call inside the
    # reason: generation loop. anyio is only a transitive dependency here, so taking trio.Path would add a
    # reason: direct async-filesystem dependency to avoid a single mkdir that blocks for microseconds.
    Path(out).mkdir(exist_ok=True, parents=True)  # ruff: ignore[blocking-path-method-in-async-function]
    summaries: dict[str, dict[str, object]] = {}
    ordered_languages = tuple(languages)
    for lang in neediest_first(out, ordered_languages):
        try:
            summaries[lang] = await generate_one(
                provider="openai",
                model="gpt-5.4-mini",
                language=lang,
                target_count=per_lang,
                out_dir=out,
                scenario_names=("all_labels_dense", "adversarial_obfuscation"),
                required_label_mode="scenario_default",
                budget=budget,
                split_purpose="challenge",
            )
        except DailyBudgetExceeded as exc:
            logger.warning("openai daily budget exhausted at %s — stopping: %s", lang, exc)
            summaries[lang] = {"skipped": "budget_exceeded", "provider": "openai"}
            break
    return {
        "per_lang": per_lang,
        "languages": list(ordered_languages),
        "summaries": strip_rows(summaries),
    }


async def safe_generate(
    label: str,
    *,
    generate_one: GenerateOne = generate_one,
    **kwargs: object,
) -> dict[str, object]:
    provider = kwargs.get("provider")
    try:
        return await generate_one(**kwargs)
    except DailyBudgetExceeded as exc:
        logger.warning("budget exhausted: %s %s (%s) — skipping", provider, label, exc)
        return {"skipped": "budget_exceeded", "provider": provider}
    # reason: one cell of a long corpus sweep; recording the failure and continuing is the contract, and
    # reason: narrowing would abort a multi-hour run on the first provider error nobody anticipated.
    except Exception as exc:  # ruff: ignore[blind-except]
        logger.warning("generation failed: %s %s — %s", provider, label, str(exc)[:200])
        return {"error": str(exc)[:200], "provider": provider}


# reason: run weak labels keeps write usage beside build budget; splitting would fragment diagnostics.
async def run_weak_labels(  # ruff: ignore[complex-structure,too-many-arguments,too-many-locals,too-many-statements]
    private_url_per_lang: int = 40,
    secret_per_lang: int = 20,
    languages: Sequence[str] = DEFAULT_LANGUAGE_KEYS,
    run_date: str | None = None,
    *,
    providers: Sequence[str] = WEAK_LABEL_FREE_PROVIDERS,
    openai_cap: int | None = None,
    count_mode: str = "additional",
    max_rounds: int = 1000,
    max_minutes: float | None = 180.0,
    out_root: str | Path = SYNTHETIC_ROOT,
    usage_dir: str | Path | None = None,
    generate_one: GenerateOne = generate_one,
    account_keys: Callable[[str], Sequence[str]] = provider_keys,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Per-language corpus targets drive the schedule (neediest-first).

    Replacing the old fixed-order + daily rotation that let the dominant provider's fastest languages run away while
    english (last in the list) starved. `baseline` is the on-disk count at run start; `produced` tracks what this run adds,
    so the workers measure deficit without re-globbing the corpus every round.

    out_dir is per-account (set in the worker) so concurrent accounts never share an accepted.{lang}.jsonl — otherwise the
    generator's file-based accepted_count would tally every account's rows. The downstream BIOES assembly globs
    synthetic/**/accepted*.jsonl, so per-account subdirs are collected automatically.

    One concurrent worker per (provider, account). Each account generates batches across every language until it hits one
    of its OWN daily ceilings (RPD/TPD -> safe_generate returns "budget_exceeded"), the wall-clock deadline passes, or
    max_rounds passes complete. Every account runs in parallel, each throttled by the ledger on all four axes
    (RPM/RPD/TPM/TPD), so each account drives toward its full daily quota and a slow account (e.g. llm7 at 1 rpm) never
    blocks the rest.

    Hard wall-clock bound: cancel any account still running at the deadline (a slow llm7 account, a throttled provider) so
    the run always ends on time and writes its usage summary. The between-cell check above lets accounts stop gracefully;
    this cancels the ones stuck mid-request. Accepted samples are appended to disk as they pass, so cancellation loses at
    most the one in-flight sample per account.

    Returns:
        The run date, the output root, and the per-label summaries with their row payloads
        stripped. The summaries describe what this run added; the on-disk corpus is the
        authority for cumulative totals, which is why the deficit scheduler re-reads it rather
        than accumulating from here.

    """
    # reason: the operator's local calendar date is the intent — it names the run's output directory and is
    # reason: never compared across zones or serialized as an instant. Adding tz=UTC would put an evening run
    # reason: in Vietnam (UTC+7) under the next day's folder, which is a behaviour change, not a fix.
    run_date = run_date or datetime.now().strftime("%Y-%m-%d")  # ruff: ignore[call-datetime-now-without-tzinfo]
    language_list = tuple(languages)
    baseline = cumulative_language_counts(out_root, language_list)
    produced: dict[str, int] = {}
    budget = build_budget(openai_cap)
    ledger = build_account_ledger()
    out = str(Path(out_root) / "weak_labels" / run_date)
    # reason: one setup syscall before any concurrent work is scheduled, not a per-item call inside the
    # reason: generation loop. anyio is only a transitive dependency here, so taking trio.Path would add a
    # reason: direct async-filesystem dependency to avoid a single mkdir that blocks for microseconds.
    Path(out).mkdir(exist_ok=True, parents=True)  # ruff: ignore[blocking-path-method-in-async-function]
    pool = tuple(providers)
    common = {
        "scenario_names": ("private_portal_secret_focus",),
        "required_label_mode": "private_url_secret",
        "budget": budget,
        "account_ledger": ledger,
        "split_purpose": "train",
        "count_mode": count_mode,
    }
    summaries: dict[str, dict[str, object]] = {}
    deadline = now() + max_minutes * 60 if max_minutes is not None else None

    # reason: attempt exposes label/account ids as its public contract; bundling would break callers.
    async def attempt(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
        label: str,
        provider: str,
        language: str,
        target: int,
        out_dir: str,
        api_keys: Sequence[str],
        base_urls: Sequence[str],
        account_ids: Sequence[str],
    ) -> dict[str, object]:
        result = await safe_generate(
            label,
            provider=provider,
            model=None,
            language=language,
            target_count=target,
            out_dir=out_dir,
            generate_one=generate_one,
            api_keys=api_keys,
            base_urls=base_urls,
            account_ids=account_ids,
            max_concurrency=PROVIDER_CONCURRENCY.get(provider, 1),
            **common,
        )
        summaries[label] = result
        return result

    async def account_worker(provider: str, account_id: str, api_key: str, base_url: str) -> None:
        keys = (api_key,)
        urls = (base_url,)
        ids = (account_id,)
        account_out = str(Path(out) / provider / account_id)
        for round_index in range(max_rounds):
            if deadline is not None and now() >= deadline:
                return
            current = {lang: baseline.get(lang, 0) + produced.get(lang, 0) for lang in language_list}
            todo = languages_by_deficit(current, language_list)
            if not todo:
                return
            for language in todo:
                if deadline is not None and now() >= deadline:
                    return
                prefix = f"{provider}:{account_id}:{language}"
                private_url = await attempt(
                    f"{prefix}:private_url:r{round_index}",
                    provider,
                    language,
                    private_url_per_lang,
                    account_out,
                    keys,
                    urls,
                    ids,
                )
                if private_url.get("skipped") == "budget_exceeded":
                    return
                produced[language] = produced.get(language, 0) + _summary_count(private_url)
                secret = await attempt(
                    f"{prefix}:secret:r{round_index}",
                    provider,
                    language,
                    secret_per_lang,
                    account_out,
                    keys,
                    urls,
                    ids,
                )
                if secret.get("skipped") == "budget_exceeded":
                    return
                produced[language] = produced.get(language, 0) + _summary_count(secret)

    worker_coros = []
    for provider in pool:
        keys = list(account_keys(provider))
        if not keys and get_provider_spec(provider).keyless:
            keys = [""]
        if not keys:
            logger.warning("weak-labels: no keys configured for %s — skip", provider)
            continue
        base_urls = configured_base_urls(provider, len(keys))
        for index, key in enumerate(keys):
            base_url = base_urls[index] if index < len(base_urls) else base_urls[0]
            worker_coros.append(account_worker(provider, f"acct{index + 1}", key, base_url))

    workers = asyncio.gather(*worker_coros)
    if deadline is None:
        await workers
    else:
        try:
            await asyncio.wait_for(workers, timeout=max(0.0, deadline - now()))
        except TimeoutError:
            logger.info(
                "weak-labels hit the %s-min deadline; cancelled in-flight accounts",
                max_minutes,
            )

    _write_usage(
        run_date=run_date,
        out=out,
        ledger=ledger,
        usage_dir=Path(usage_dir) if usage_dir else Path(out_root) / "usage",
    )
    return {"run_date": run_date, "out": out, "summaries": strip_rows(summaries)}


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _summary_count(summary: Mapping[str, object]) -> int:
    return _int_or_zero(summary.get("accepted_count", 0))


def _account_usage_snapshot(
    snapshot: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, dict[str, AccountUsage]]:
    """Decode the ledger's dynamic snapshot once before persisting it.

    Returns:
        Request and token counts per provider and account, every value coerced to a plain
        ``int``. Decoding once here rather than at each read means the persisted usage record
        has a fixed shape, so a malformed ledger entry becomes a zero at write time instead of
        a type error somewhere in the dashboard.

    """
    return {
        provider: {
            account_id: {
                "requests": _int_or_zero(entry.get("requests", 0)),
                "tokens": _int_or_zero(entry.get("tokens", 0)),
            }
            for account_id, entry in accounts.items()
        }
        for provider, accounts in snapshot.items()
    }


def _decode_provider_limits(raw_limits: Mapping[str, object]) -> ProviderLimits:
    limits: ProviderLimits = {}
    for key in ("accounts", "token_cap_per_account", "rpd_per_account"):
        value = raw_limits.get(key)
        if value is None:
            if key != "accounts":
                if key == "token_cap_per_account":
                    limits["token_cap_per_account"] = None
                else:
                    limits["rpd_per_account"] = None
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            if key == "accounts":
                limits["accounts"] = value
            elif key == "token_cap_per_account":
                limits["token_cap_per_account"] = value
            else:
                limits["rpd_per_account"] = value
    return limits


def _write_usage(
    *,
    run_date: str,
    out: str,
    ledger: AccountLedger,
    usage_dir: Path,
) -> None:
    """Fold the run into daily usage and refresh the HTML dashboard.

    Usage is folded per provider and per account.

    Telemetry is secondary to generation: a failure here must never lose
    generated data, so every error is logged and swallowed. accepted/rejected
    counts are read straight from the on-disk files under ``out`` — the ground
    truth. Each accepted row is appended exactly once, so this is exact even when
    the deadline cancels a cell mid-write or ``count_mode=additional`` grows the
    files across rounds (summing per-cell summaries would over- or under-count).
    Per-account requests + tokens come from the ledger snapshot.
    """
    # reason: write usage's try keeps decode with snapshot; splitting would split cleanup from writes.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        out_path = Path(out)
        raw_snapshot: Mapping[str, Mapping[str, Mapping[str, object]]] = ledger.snapshot()
        snapshot = _account_usage_snapshot(raw_snapshot)
        seen = set(snapshot)
        if out_path.exists():
            seen |= {p.name for p in out_path.iterdir() if p.is_dir()}

        per_provider: dict[str, ProviderRunStats] = {}
        for provider in seen:
            provider_dir = out_path / provider
            accepted = 0
            rejected = 0
            by_language: dict[str, int] = {}
            if provider_dir.exists():
                for path in provider_dir.glob("**/accepted.*.jsonl"):
                    code = path.name[len("accepted.") : -len(".jsonl")]
                    n = count_jsonl(path)
                    accepted += n
                    by_language[code] = by_language.get(code, 0) + n
                for path in provider_dir.glob("**/rejected.*.jsonl"):
                    rejected += count_jsonl(path)
            per_provider[provider] = {
                "model": resolve_provider_model(provider),
                "accepted": accepted,
                "rejected": rejected,
                "by_language": by_language,
            }
        limits = {
            provider: _decode_provider_limits(_provider_limits(provider)) for provider in set(per_provider) | set(snapshot)
        }
        record = build_usage_record(run_date, per_provider, snapshot, limits)
        write_usage_record(record, usage_dir)
        render_usage_dashboard(usage_dir)
    except Exception:
        logger.exception("usage dashboard update failed")


def resolve_languages(selected: Sequence[str] | None) -> tuple[str, ...]:
    if not selected:
        return DEFAULT_LANGUAGE_KEYS
    resolved: list[str] = []
    unknown: list[str] = []
    for lang in selected:
        try:
            resolved.append(normalize_language(lang).config_slug)
        except ValueError:
            unknown.append(lang)
    if unknown:
        msg = f"unknown language(s): {unknown}; valid: {list(DEFAULT_LANGUAGE_KEYS)}"
        raise ValueError(msg)
    return tuple(resolved)
