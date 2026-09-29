from __future__ import annotations

# ruff: file-ignore[useless-import-alias]
# reason: an explicit `X as X` re-export, which is this module's published surface: ruff's own
# reason: unsafe fix DELETED five of these once and broke every caller, so the alias stays.
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol

from meddies_pii.exceptions import (
    ConfigurationError,
    DailyBudgetExceeded,
    MeddiesException,
)
from meddies_pii.generation.label_corpus.provider_access import (
    configured_base_urls as configured_base_urls,
)
from meddies_pii.generation.label_corpus.provider_access import (
    preflight_keys as preflight_keys,
)
from meddies_pii.generation.label_corpus.provider_access import (
    provider_keys as provider_keys,
)
from meddies_pii.generation.label_corpus.synthetic import LabelCorpusGenerator
from meddies_pii.generation.openai_compatible.client import OpenAICompatibleClient
from meddies_pii.generation.openai_compatible.providers import resolve_provider_model
from meddies_pii.json_types import as_object_list
from meddies_pii.jsonl import count_jsonl
from meddies_pii.languages import normalize_language

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.generation.account_ledger import AccountLedger
    from meddies_pii.generation.openai_compatible.quota import DailyTokenBudget
    from meddies_pii.taxonomy import PiiLabel


@dataclass(frozen=True, slots=True)
class SyntheticGenerationRequest:
    provider: str
    model: str | None
    language: str
    target_count: int
    output_dir: str | Path
    domain_profile: str
    split_purpose: str = "train"
    scenario_names: Sequence[str] | None = None
    count_mode: str = "total"
    max_concurrency: int = 4
    rpm_per_key: int = 60
    required_labels: Sequence[PiiLabel] = ()
    required_label_mode: str = "private_url_secret"
    max_attempt_multiplier: float = 3.0
    seed: int = 20260513
    log_every: int = 25
    require_vietnamese_marker: bool = True
    daily_budget: DailyTokenBudget | None = None
    account_ledger: AccountLedger | None = None
    api_keys: Sequence[str] | None = None
    base_urls: Sequence[str] | None = None
    account_ids: Sequence[str] | None = None


@dataclass(frozen=True, slots=True)
class LanguageGenerationPlan:
    """One request template expanded over the requested language set."""

    request_template: SyntheticGenerationRequest
    supported_languages: tuple[str, ...]

    @property
    def languages(self) -> tuple[str, ...]:
        if self.request_template.language == "all":
            return self.supported_languages
        return (self.request_template.language,)


@dataclass(frozen=True, slots=True)
class LanguageGenerationSuccess:
    language: str
    accepted_count: object | None


type LanguageGenerationFailureKind = Literal["configuration", "generation", "unexpected"]


@dataclass(frozen=True, slots=True)
class LanguageGenerationFailure:
    language: str
    kind: LanguageGenerationFailureKind
    message: str


@dataclass(frozen=True, slots=True)
class LanguageGenerationResult:
    successes: tuple[LanguageGenerationSuccess, ...]
    failures: tuple[LanguageGenerationFailure, ...]


class _Generator(Protocol):
    # reason: _Generator exposes language/log every as its public contract; bundling would break callers.
    async def generate_until(  # ruff: ignore[too-many-arguments]
        self,
        *,
        language: str,
        target_count: int,
        output_dir: str | Path,
        domain_profile: str,
        split_purpose: str,
        scenario_names: Sequence[str] | None,
        count_mode: str,
        required_labels: Sequence[PiiLabel],
        required_label_mode: str,
        max_attempt_multiplier: float,
        require_vietnamese_marker: bool,
        log_every: int,
    ) -> Mapping[str, object]: ...


type ClientFactory = Callable[..., AbstractAsyncContextManager[OpenAICompatibleClient]]
type GeneratorClass = Callable[..., _Generator]
type RunRequest = Callable[[SyntheticGenerationRequest], Awaitable[dict[str, object]]]


async def run_synthetic_generation(
    request: SyntheticGenerationRequest,
    *,
    client_factory: ClientFactory = OpenAICompatibleClient.from_env,
    generator_cls: GeneratorClass = LabelCorpusGenerator,
) -> dict[str, object]:
    """Run one provider-backed synthetic-generation request.

    This is the package-owned seam behind the public generation CLI: resolve the
    provider model, open the provider client, construct the Meddies Labels generator,
    and return the generator summary. Callers supply only a request object.

    Returns:
        The generator's own summary mapping, copied into a plain ``dict`` so the caller does
        not hold a reference into generator state after the client context closes. Explicit
        ``api_keys`` on the request bypass ``client_factory`` entirely and open a
        provider-scoped client instead, so an injected factory is not consulted on that path.

    """
    model = resolve_provider_model(request.provider, request.model)
    client_context: AbstractAsyncContextManager[OpenAICompatibleClient]
    if request.api_keys is not None:
        client_context = OpenAICompatibleClient.for_provider(
            request.provider,
            list(request.api_keys),
            model=model,
            base_urls=tuple(request.base_urls) if request.base_urls is not None else None,
            account_ids=tuple(request.account_ids) if request.account_ids is not None else None,
            rpm_per_key=request.rpm_per_key,
            daily_budget=request.daily_budget,
            account_ledger=request.account_ledger,
        )
    else:
        client_context = client_factory(
            provider=request.provider,
            model=model,
            rpm_per_key=request.rpm_per_key,
            daily_budget=request.daily_budget,
            account_ledger=request.account_ledger,
        )
    async with client_context as client:
        generator = generator_cls(
            client,
            provider=request.provider,
            model=model,
            max_concurrency=request.max_concurrency,
            seed=request.seed,
        )
        return dict(
            await generator.generate_until(
                language=request.language,
                target_count=request.target_count,
                output_dir=request.output_dir,
                domain_profile=request.domain_profile,
                split_purpose=request.split_purpose,
                scenario_names=request.scenario_names,
                count_mode=request.count_mode,
                required_labels=request.required_labels,
                required_label_mode=request.required_label_mode,
                max_attempt_multiplier=request.max_attempt_multiplier,
                require_vietnamese_marker=request.require_vietnamese_marker,
                log_every=request.log_every,
            ),
        )


async def run_language_generation(
    plan: LanguageGenerationPlan,
    *,
    run_request: RunRequest | None = None,
) -> LanguageGenerationResult:
    """Execute a language plan while retaining successful and partial artifacts.

    Returns:
        Every language's outcome split into successes and failures, so a run that fails
        partway still reports what it did produce. Failures carry a ``kind`` that separates
        the three cases the loop distinguishes: a ``configuration`` failure STOPS the run and
        the remaining languages are never attempted, while ``generation`` and ``unexpected``
        failures are recorded and the loop continues to the next language.

    """
    execute = run_synthetic_generation if run_request is None else run_request
    successes: list[LanguageGenerationSuccess] = []
    failures: list[LanguageGenerationFailure] = []

    for language in plan.languages:
        request = replace(plan.request_template, language=language)
        try:
            summary = await execute(request)
        except ConfigurationError as error:
            failures.append(
                LanguageGenerationFailure(
                    language=language,
                    kind="configuration",
                    message=error.message,
                ),
            )
            break
        except MeddiesException as error:
            failures.append(
                LanguageGenerationFailure(
                    language=language,
                    kind="generation",
                    message=error.message,
                ),
            )
        # reason: this arm exists for exceptions the code does NOT know — `kind="unexpected"` names that
        # reason: intent — so narrowing it to a known set would empty the very case it records.
        except Exception as error:  # ruff: ignore[blind-except]
            failures.append(
                LanguageGenerationFailure(
                    language=language,
                    kind="unexpected",
                    message=str(error),
                ),
            )
        else:
            successes.append(
                LanguageGenerationSuccess(
                    language=language,
                    accepted_count=summary.get("accepted_count"),
                ),
            )

    return LanguageGenerationResult(
        successes=tuple(successes),
        failures=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class GenerationSegment:
    language: str
    scenario: str
    domain_profile: str
    required_label_mode: str
    target_count: int
    count_mode: str
    planned_count: int | None = None
    required_labels: tuple[PiiLabel, ...] = ()
    split_purpose: str = "train"
    enforce_target: bool = False


@dataclass(frozen=True, slots=True)
class GenerationRunPlan:
    run_id: str
    provider: str
    model: str | None
    output_dir: Path
    max_concurrency: int
    rpm_per_key: int
    max_attempt_multiplier: float
    segments: tuple[GenerationSegment, ...]
    summary_path: Path | None = None
    dry_run: bool = False
    warnings: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    api_keys: tuple[str, ...] | None = None
    base_urls: tuple[str, ...] | None = None
    require_vietnamese_marker: bool = True


class GenerationRunError(RuntimeError):
    def __init__(self, summary: dict[str, object], message: str) -> None:
        super().__init__(message)
        self.summary = summary


@dataclass(frozen=True, slots=True)
class ScenarioTarget:
    scenario: str
    domain_profile: str
    total_weight: int


@dataclass(frozen=True, slots=True)
class WeightedSegmentSpec:
    language: str
    scenario: str
    required_label_mode: str
    weight: int


MIMO_GENERAL_SCENARIOS: tuple[ScenarioTarget, ...] = (
    ScenarioTarget("general_consumer_admin_support", "general", 5_000),
    ScenarioTarget("hr_employer_school_insurance_telecom_admin", "general", 4_000),
    ScenarioTarget("code_log_security_account_recovery", "code_logs", 4_000),
    ScenarioTarget("adversarial_formatting", "code_logs", 3_000),
    ScenarioTarget("one_hop_deferred_clue", "general", 2_000),
    ScenarioTarget("public_negative_url_date_contrast", "general", 2_000),
)
LABEL_MODE_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("private_url_secret", 40),
    ("private_url_only", 20),
    ("secret_only", 20),
    ("none", 20),
)
LANGUAGE_WEIGHTS: tuple[tuple[str, int], ...] = (("Vietnamese", 60), ("English", 40))
OPENCODE_ZEN_SEGMENTS: tuple[WeightedSegmentSpec, ...] = (
    WeightedSegmentSpec("Vietnamese", "private_portal_secret_focus", "private_url_secret", 24),
    WeightedSegmentSpec("Vietnamese", "public_negative_contrast", "private_url_secret", 16),
    WeightedSegmentSpec("Vietnamese", "adversarial_obfuscation", "private_url_secret", 12),
    WeightedSegmentSpec("Vietnamese", "structured_payload", "scenario_default", 8),
    WeightedSegmentSpec("English", "private_portal_secret_focus", "private_url_secret", 16),
    WeightedSegmentSpec("English", "public_negative_contrast", "private_url_secret", 10),
    WeightedSegmentSpec("English", "adversarial_obfuscation", "private_url_secret", 8),
    WeightedSegmentSpec("English", "structured_payload", "scenario_default", 6),
)


async def run_generation_plan(
    plan: GenerationRunPlan,
    *,
    run_request: RunRequest = run_synthetic_generation,
) -> dict[str, object]:
    """Run planned Meddies Labels generation segments with accounting and resume checks.

    Returns:
        The run summary, which is also the file at ``plan.summary_path``. A ``dry_run`` plan
        returns after writing the summary and before any segment executes, so the summary
        describes what WOULD run rather than what did.

    Raises:
        GenerationRunError: When a segment raises, or when an ``enforce_target`` segment
            accepts fewer rows than its target. The summary is written to disk BEFORE the
            raise in both cases, and the exception carries it on ``.summary``, so a failed
            run is resumable rather than lost -- callers should read that attribute rather
            than re-deriving state from the output directory.

    """
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_run_summary(plan)
    _write_summary(plan.summary_path, summary)
    if plan.dry_run:
        return summary

    for index, segment in enumerate(plan.segments, start=1):
        before = accepted_count(plan.output_dir, segment.language)
        request = SyntheticGenerationRequest(
            provider=plan.provider,
            model=plan.model,
            language=segment.language,
            target_count=segment.target_count,
            output_dir=plan.output_dir,
            domain_profile=segment.domain_profile,
            split_purpose=segment.split_purpose,
            scenario_names=(segment.scenario,),
            count_mode=segment.count_mode,
            max_concurrency=plan.max_concurrency,
            rpm_per_key=plan.rpm_per_key,
            required_labels=segment.required_labels,
            required_label_mode=segment.required_label_mode,
            max_attempt_multiplier=plan.max_attempt_multiplier,
            require_vietnamese_marker=plan.require_vietnamese_marker,
            api_keys=plan.api_keys,
            base_urls=plan.base_urls,
        )
        result: dict[str, object] = {
            "index": index,
            "language": segment.language,
            "scenario": segment.scenario,
            "required_label_mode": segment.required_label_mode,
            "target_count": segment.target_count,
            "count_mode": segment.count_mode,
            "accepted_before": before,
            "accepted_after": before,
            "status": "running",
            "error": None,
        }
        try:
            generation_summary = await run_request(request)
        except DailyBudgetExceeded as exc:
            result["status"] = "budget_exceeded"
            result["error"] = str(exc)
            _run_results(summary).append(result)
            _write_summary(plan.summary_path, summary)
            return summary
        except Exception as exc:
            after = accepted_count(plan.output_dir, segment.language)
            result["accepted_after"] = after
            result["status"] = "error"
            result["error"] = str(exc)[:500]
            _run_results(summary).append(result)
            _write_summary(plan.summary_path, summary)
            raise GenerationRunError(summary, str(exc)) from exc

        after = accepted_count(plan.output_dir, segment.language)
        result["accepted_after"] = after
        result["status"] = "ok"
        result["summary"] = summary_without_rows(generation_summary)
        _run_results(summary).append(result)
        _write_summary(plan.summary_path, summary)
        if segment.enforce_target and after < segment.target_count:
            message = f"segment stopped short: {segment.language} accepted={after} target={segment.target_count}"
            result["status"] = "shortfall"
            result["error"] = message
            _write_summary(plan.summary_path, summary)
            raise GenerationRunError(summary, message)
    return summary


def build_mimo_general_segments(total_target: int) -> tuple[GenerationSegment, ...]:
    scenario_allocations = allocate_counts(
        total_target,
        tuple((scenario, scenario.total_weight) for scenario in MIMO_GENERAL_SCENARIOS),
    )
    cumulative_by_language = {language: 0 for language, _ in LANGUAGE_WEIGHTS}
    segments: list[GenerationSegment] = []
    for scenario_target, scenario_count in scenario_allocations:
        for language, language_count in allocate_counts(scenario_count, LANGUAGE_WEIGHTS):
            for label_mode, segment_count in allocate_counts(language_count, LABEL_MODE_WEIGHTS):
                cumulative_by_language[language] += segment_count
                segments.append(
                    GenerationSegment(
                        language=language,
                        scenario=scenario_target.scenario,
                        domain_profile=scenario_target.domain_profile,
                        required_label_mode=label_mode,
                        target_count=cumulative_by_language[language],
                        count_mode="total",
                        planned_count=segment_count,
                        enforce_target=True,
                    ),
                )
    return tuple(segments)


def build_opencode_zen_daily_segments(
    daily_target: int,
    segments: tuple[WeightedSegmentSpec, ...] = OPENCODE_ZEN_SEGMENTS,
) -> tuple[GenerationSegment, ...]:
    return tuple(
        GenerationSegment(
            language=segment.language,
            scenario=segment.scenario,
            domain_profile="medical",
            required_label_mode=segment.required_label_mode,
            target_count=count,
            count_mode="additional",
            planned_count=count,
            enforce_target=False,
        )
        for segment, count in allocate_counts(
            daily_target,
            tuple((segment, segment.weight) for segment in segments),
        )
    )


def resolve_max_concurrency(requested: int, key_count: int) -> int:
    if requested > 0:
        return requested
    return max(1, key_count)


def language_targets(segments: Sequence[GenerationSegment]) -> dict[str, int]:
    return {
        language: sum(
            segment.planned_count or segment.target_count for segment in segments if segment.language == language
        )
        for language in {segment.language for segment in segments}
    }


def allocate_counts[T](total: int, weighted_items: tuple[tuple[T, int], ...]) -> list[tuple[T, int]]:
    if total <= 0:
        return []
    weight_total = sum(weight for _, weight in weighted_items)
    raw = [(item, total * weight / weight_total) for item, weight in weighted_items]
    counts = [(item, int(value)) for item, value in raw]
    remainder = total - sum(count for _, count in counts)
    for idx, _ in sorted(
        enumerate(raw),
        key=lambda entry: entry[1][1] - int(entry[1][1]),
        reverse=True,
    )[:remainder]:
        item, count = counts[idx]
        counts[idx] = (item, count + 1)
    return [(item, count) for item, count in counts if count > 0]


def _run_results(summary: dict[str, object]) -> list[object]:
    """Return the validated mutable results collection in a run-summary record.

    Returns:
        The live ``results`` list inside ``summary``, not a copy -- callers append to it and
        expect the summary they later write to disk to carry the appended entry.

    Raises:
        ValueError: When ``results`` is absent or is not a list. Refused rather than
            defaulted to an empty list, because a summary missing its results collection
            means the caller is holding a record of a different shape, and silently
            substituting one would drop every segment result the run went on to append.

    """
    results = as_object_list(summary.get("results"))
    if results is None:
        msg = "generation run summary.results must be a list"
        raise ValueError(msg)
    return results


def build_run_summary(plan: GenerationRunPlan) -> dict[str, object]:
    return {
        "run_id": plan.run_id,
        "provider": plan.provider,
        "model": plan.model or "provider-default",
        "output_dir": str(plan.output_dir),
        "max_concurrency": plan.max_concurrency,
        "rpm_per_key": plan.rpm_per_key,
        "max_attempt_multiplier": plan.max_attempt_multiplier,
        "dry_run": plan.dry_run,
        "warnings": list(plan.warnings),
        "segments": [asdict(segment) for segment in plan.segments],
        "results": [],
        **plan.metadata,
    }


def accepted_count(output_dir: Path, language: str) -> int:
    profile = normalize_language(language)
    paths = _language_paths(output_dir, profile.code)
    if paths["accepted"].exists():
        return count_jsonl(paths["accepted"])
    if not paths["summary"].exists():
        return 0
    try:
        payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0
    accepted = payload.get("accepted_count", 0)
    return accepted if isinstance(accepted, int) and not isinstance(accepted, bool) else 0


def _language_paths(output_dir: Path, code: str) -> dict[str, Path]:
    return {
        "accepted": output_dir / f"accepted.{code}.jsonl",
        "summary": output_dir / f"summary.{code}.json",
    }


def _write_summary(path: Path | None, summary: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summary_without_rows(summary: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key != "rows"}
