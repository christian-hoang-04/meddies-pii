"""Targeted Meddies Labels synthetic generation engine.

The generation loop, dedup, summary, and file I/O. Reads its catalog of what to
generate from catalog.py, its prompt templates from prompts.py, and its accept/
reject gate from validate.py.

Seed the dedup set from the live `accepted.{code}.jsonl` AND any sibling `pm_accepted.{code}.jsonl` (ADR 0008 §5): rows
migrated under the pm_* prefix are otherwise invisible to dedup, letting a cross-run duplicate of an already-accepted text
re-enter the corpus silently.

"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import httpx

from meddies_pii.exceptions import DailyBudgetExceeded
from meddies_pii.generation.label_corpus.catalog import (
    GenerationProfile,
    Scenario,
    _scenario_by_name,
    generation_profile_for_domain,
    required_labels_for_mode,
    sample_span_target,
    scenario_pool_for_profile,
)
from meddies_pii.generation.label_corpus.edge_cases import sample_edge_cases
from meddies_pii.generation.label_corpus.prompts import (
    targeted_system_prompt,
    targeted_user_prompt,
)
from meddies_pii.generation.label_corpus.validate import (
    ValidationResult,
    accepted_record,
    validate_tagged_document,
)
from meddies_pii.generation.openai_compatible.providers import normalize_provider_name
from meddies_pii.json_types import is_object_dict, is_str_list, is_str_mapping
from meddies_pii.json_types import required_int as _required_int
from meddies_pii.jsonl import append_jsonl, count_jsonl, read_jsonl
from meddies_pii.languages import normalize_language

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from meddies_pii.generation.openai_compatible.client import OpenAICompatibleClient
    from meddies_pii.taxonomy import PiiLabel

logger = logging.getLogger(__name__)

DEFAULT_TARGETED_GENERATION_DIR = Path("data/bioes-v2/synthetic/targeted_generation")
_DEFAULT_GENERATION_MAX_TOKENS = 4096
_OPENCODE_ZEN_DEEPSEEK_GENERATION_MAX_TOKENS = 8192


def _generation_max_tokens(provider: str, model: str) -> int:
    normalized_provider = normalize_provider_name(provider)
    normalized_model = model.strip().lower()
    if normalized_provider == "opencode_zen" and normalized_model == "deepseek-v4-flash-free":
        return _OPENCODE_ZEN_DEEPSEEK_GENERATION_MAX_TOKENS
    return _DEFAULT_GENERATION_MAX_TOKENS


@dataclass(frozen=True, slots=True)
class GenerationPaths:
    accepted: Path
    raw: Path
    rejected: Path
    summary: Path


class GenerationSummary(TypedDict):
    language: str
    accepted_count: int
    raw_attempt_count: int
    rejected_count: int
    label_distribution: dict[str, int]
    scenario_distribution: dict[str, int]
    rejection_distribution: dict[str, int]
    paths: dict[str, str]


def language_paths(output_dir: str | Path, language: str) -> GenerationPaths:
    profile = normalize_language(language)
    root = Path(output_dir)
    return GenerationPaths(
        accepted=root / f"accepted.{profile.code}.jsonl",
        raw=root / f"raw.{profile.code}.jsonl",
        rejected=root / f"rejected.{profile.code}.jsonl",
        summary=root / f"summary.{profile.code}.json",
    )


def _label_counts(records: Iterable[Mapping[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        labels = record.get("label", [])
        if not isinstance(labels, list):
            continue
        for span in labels:
            if not is_object_dict(span):
                continue
            category = span.get("category")
            if isinstance(category, str):
                counts[category] += 1
    return counts


def _existing_text_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for seed_path in (path, path.with_name(f"pm_{path.name}")):
        if not seed_path.exists():
            continue
        for record in read_jsonl(seed_path):
            text = record.get("text")
            if isinstance(text, str):
                hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
    return hashes


def write_summary(paths: GenerationPaths, *, language: str) -> GenerationSummary:
    accepted_records = list(read_jsonl(paths.accepted)) if paths.accepted.exists() else []
    rejected_records = list(read_jsonl(paths.rejected)) if paths.rejected.exists() else []
    scenario_counts: Counter[str] = Counter()
    for record in accepted_records:
        info = record.get("info")
        if is_str_mapping(info):
            scenario_counts[str(info.get("scenario", "unknown"))] += 1
    rejection_counts: Counter[str] = Counter()
    for record in rejected_records:
        errors = record.get("errors")
        if isinstance(errors, list):
            rejection_counts.update(str(error) for error in errors)
    summary: GenerationSummary = {
        "language": normalize_language(language).name,
        "accepted_count": len(accepted_records),
        "raw_attempt_count": count_jsonl(paths.raw),
        "rejected_count": len(rejected_records),
        "label_distribution": dict(sorted(_label_counts(accepted_records).items())),
        "scenario_distribution": dict(sorted(scenario_counts.items())),
        "rejection_distribution": dict(sorted(rejection_counts.items())),
        "paths": {
            "accepted": str(paths.accepted),
            "raw": str(paths.raw),
            "rejected": str(paths.rejected),
            "summary": str(paths.summary),
        },
    }
    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    paths.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class LabelCorpusGenerator:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        provider: str,
        model: str,
        max_concurrency: int,
        seed: int = 20260513,
    ) -> None:
        self.client = client
        self.provider = provider
        self.model = model
        self.max_concurrency = max_concurrency
        # reason: the generator's scenario sampling is seeded from a constructor parameter so a corpus run can be
        # reason: replayed row for row. Reproducibility is the contract; this draws scenario names, never a
        # reason: secret, a token, or a key.
        self.random = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]

    # reason: LabelCorpusGenerator orders generate before write; helper seams would fragment diagnostics.
    async def generate_until(  # ruff: ignore[complex-structure,too-many-arguments,too-many-locals]
        self,
        *,
        language: str,
        target_count: int,
        output_dir: str | Path,
        domain_profile: str = "medical",
        split_purpose: str = "train",
        scenario_names: Sequence[str] | None = None,
        count_mode: str = "total",
        required_labels: Sequence[PiiLabel] = (),
        required_label_mode: str = "private_url_secret",
        max_attempt_multiplier: float = 3.0,
        require_vietnamese_marker: bool = True,
        log_every: int = 25,
    ) -> GenerationSummary:
        profile = normalize_language(language)
        generation_profile = generation_profile_for_domain(domain_profile, split_purpose=split_purpose)
        scenarios = scenario_pool_for_profile(generation_profile, scenario_names)
        paths = language_paths(output_dir, profile.name)
        paths.accepted.parent.mkdir(parents=True, exist_ok=True)
        text_hashes = _existing_text_hashes(paths.accepted)
        accepted = count_jsonl(paths.accepted)
        attempts = count_jsonl(paths.raw)
        normalized_count_mode = count_mode.strip().lower()
        if normalized_count_mode == "total":
            target_accepted = target_count
        elif normalized_count_mode == "additional":
            target_accepted = accepted + target_count
        else:
            msg = "count_mode must be either 'total' or 'additional'"
            raise ValueError(msg)
        max_attempts = attempts + max(1, int((target_accepted - accepted) * max_attempt_multiplier))

        logger.info(
            (
                "Starting targeted Meddies Labels generation: language=%s domain=%s split=%s scenarios=%s "
                "accepted=%s target=%s attempts=%s max_attempts=%s"
            ),
            profile.name,
            generation_profile.domain_profile,
            generation_profile.split_purpose,
            ",".join(s.name for s in scenarios),
            accepted,
            target_accepted,
            attempts,
            max_attempts,
        )

        while accepted < target_accepted and attempts < max_attempts:
            batch_size = min(self.max_concurrency, target_accepted - accepted)
            batch = [
                self._generate_attempt(
                    language=profile.name,
                    generation_profile=generation_profile,
                    scenarios=scenarios,
                    attempt_index=attempts + offset + 1,
                    required_label_mode=required_label_mode,
                    extra_required_labels=required_labels,
                )
                for offset in range(batch_size)
            ]
            attempts += batch_size
            for result in await asyncio.gather(*batch):
                append_jsonl(paths.raw, [result])
                if "error" in result:
                    append_jsonl(paths.rejected, [result | {"errors": ["api_error"]}])
                    if result.get("fatal_api_error"):
                        write_summary(paths, language=profile.name)
                        msg = (
                            "Fatal generation API error; stopping to avoid burning attempts: "
                            f"{result.get('status_code')}: {result.get('error')}"
                        )
                        raise RuntimeError(
                            msg,
                        )
                    continue

                scenario = _scenario_by_name(str(result["scenario"]))
                resolved_required_labels = required_labels_for_mode(
                    scenario,
                    required_label_mode,
                    extra_required_labels=required_labels,
                )
                validation = validate_tagged_document(
                    str(result["content"]),
                    language=profile.name,
                    required_labels=resolved_required_labels,
                    require_vietnamese_marker=require_vietnamese_marker,
                )
                text_hash = hashlib.sha256(validation.raw_text.encode("utf-8")).hexdigest()
                if validation.ok and text_hash in text_hashes:
                    validation = ValidationResult(
                        ok=False,
                        raw_text=validation.raw_text,
                        spans=validation.spans,
                        errors=("duplicate_text",),
                        had_label_repairs=validation.had_label_repairs,
                    )
                if not validation.ok:
                    rejected = {
                        **result,
                        "errors": list(validation.errors),
                        "parsed_label_count": len(validation.spans),
                        "parsed_unique_labels": sorted({span.label for span in validation.spans}),
                    }
                    append_jsonl(paths.rejected, [rejected])
                    continue

                edge_cases = result.get("edge_cases")
                record = accepted_record(
                    validation=validation,
                    language=profile.name,
                    scenario=scenario,
                    document_type=str(result["document_type"]),
                    text_format=str(result["text_format"]),
                    model=self.model,
                    provider=self.provider,
                    attempt_index=_required_int(result["attempt_index"], field="attempt_index"),
                    source_dataset=generation_profile.source_dataset,
                    domain_bucket=generation_profile.domain_bucket,
                    domain_profile=generation_profile.domain_profile,
                    split_purpose=generation_profile.split_purpose,
                    edge_cases=edge_cases if is_str_list(edge_cases) else [],
                )
                append_jsonl(paths.accepted, [dict(record)])
                text_hashes.add(text_hash)
                accepted += 1
                if accepted % log_every == 0 or accepted == target_accepted:
                    logger.info(
                        "Accepted %s/%s %s rows",
                        accepted,
                        target_accepted,
                        profile.name,
                    )

        summary = write_summary(paths, language=profile.name)
        if summary["accepted_count"] < target_accepted:
            logger.warning(
                "Stopped before target for %s: accepted=%s target=%s attempts=%s max_attempts=%s",
                profile.name,
                summary["accepted_count"],
                target_accepted,
                attempts,
                max_attempts,
            )
        return summary

    # reason: LabelCorpusGenerator keeps language/extra at its adapter seam; bundling would hide required inputs.
    async def _generate_attempt(  # ruff: ignore[too-many-arguments]
        self,
        *,
        language: str,
        generation_profile: GenerationProfile,
        scenarios: Sequence[Scenario],
        attempt_index: int,
        required_label_mode: str,
        extra_required_labels: Sequence[PiiLabel],
    ) -> dict[str, object]:
        """Zero-spend guard tripped.

        Propagate as fatal, never swallow into a rejected row (that would keep generating past the daily cap).

        Returns:
            One attempt record. On success it carries the sampled generation parameters --
            language, domain profile, split purpose, scenario, required labels and mode, span
            and unique-label minimums, document type, text format, edge cases -- plus the
            model's ``content``. On an ordinary failure it carries the same parameters with
            ``content`` replaced by ``error``, ``status_code`` and ``fatal_api_error``; a 401,
            403 or 404 sets that last flag, because a bad key or model name will fail every
            remaining attempt rather than being worth a retry.

        Raises:
            DailyBudgetExceeded: Re-raised rather than folded into a failure record. The broad
                handler below would otherwise turn it into a rejected row, and the caller would
                keep requesting past the daily cap because rejected rows look like ordinary
                misses. This is the one exception the ``except Exception`` must not absorb.

        """
        profile = normalize_language(language)
        scenario = self.random.choice(scenarios)
        required_labels = required_labels_for_mode(
            scenario,
            required_label_mode,
            extra_required_labels=extra_required_labels,
        )
        span_target = sample_span_target(self.random, scenario, required_labels)
        edge_cases = sample_edge_cases(self.random, scenario)
        document_type = self.random.choice(generation_profile.document_types)
        text_format = self.random.choice(generation_profile.text_formats)
        system = targeted_system_prompt(
            profile,
            domain_hint=generation_profile.domain_hint,
            public_url_hint=generation_profile.public_url_hint,
        )
        user = targeted_user_prompt(
            profile=profile,
            scenario=scenario,
            document_type=document_type,
            text_format=text_format,
            min_spans=span_target,
            min_unique_labels=scenario.min_unique_labels,
            required_labels=required_labels,
            public_negative_hint=generation_profile.public_negative_hint,
            adversarial_hint=generation_profile.adversarial_hint,
            realism_hint=generation_profile.realism_hint,
            edge_cases=edge_cases,
        )
        try:
            content = await self.client.chat(
                system,
                user,
                temperature=0.85,
                max_tokens=_generation_max_tokens(self.provider, self.model),
            )
            return {
                "attempt_index": attempt_index,
                "language": profile.name,
                "domain_profile": generation_profile.domain_profile,
                "split_purpose": generation_profile.split_purpose,
                "scenario": scenario.name,
                "required_label_mode": required_label_mode,
                "required_labels": list(required_labels),
                "min_spans": span_target,
                "min_unique_labels": scenario.min_unique_labels,
                "document_type": document_type,
                "text_format": text_format,
                "edge_cases": list(edge_cases),
                "content": content,
            }
        except DailyBudgetExceeded:
            raise
        except Exception as exc:
            status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            if status_code in {401, 403, 404}:
                logger.exception(
                    "Provider auth/availability error %s for %s — check its API key and model name.",
                    status_code,
                    self.provider,
                )
            else:
                logger.exception("Generation attempt failed")
            return {
                "attempt_index": attempt_index,
                "language": profile.name,
                "domain_profile": generation_profile.domain_profile,
                "split_purpose": generation_profile.split_purpose,
                "scenario": scenario.name,
                "required_label_mode": required_label_mode,
                "required_labels": list(required_labels),
                "min_spans": span_target,
                "min_unique_labels": scenario.min_unique_labels,
                "document_type": document_type,
                "text_format": text_format,
                "error": str(exc),
                "status_code": status_code,
                "fatal_api_error": status_code in {401, 403, 404},
            }
