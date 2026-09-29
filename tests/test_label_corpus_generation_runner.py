"""Per-language corpus targets drive scheduling.

Vietnamese/english are the priority languages (20k), the rest default to 15k. A language already at or above its target is
dropped; the rest are ordered neediest-first (largest deficit), ties broken by name for determinism.

The deficit baseline must sum a language across every date/provider/account dir, not just today's run — otherwise a
language complete from prior days gets re-filled. french has no files anywhere -> 0.

End-to-end scheduler contract: a language already at its corpus target is never requested; the neediest is served and
filled up to (not past) target; english now gets a secret pass too (the old english-only-private_url skip is gone, since
english is now a 20k priority language).

Cloudflare is neuron-metered (10k neurons/day), not request-capped. The dashboard must not show requests against a bogus
RPD proxy (122) that reads as "idle" when an account is neuron-exhausted — so the DISPLAY cap is None ("/ ∞"). The ledger
still throttles cloudflare by its rpd=110 proxy internally; this only affects the utilization view.

BAN-SAFETY INVARIANT: every configured (headroomed) cap must stay AT OR BELOW the provider's real documented free-tier
ceiling. Caps verified live 2026-06-20 via response headers (groq 1000req/6000tpm; cerebras 5/min, 2400/day, 30k tpm, 1M
tpd; mistral 25/min, 375k tpm) + 429 bodies / docs (cloudflare 10k neurons/day; openrouter 50 unfunded / 1000 funded; nim
40rpm; llm7 30rpm). Raising any value above documented = ban risk → this test fails.

"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from meddies_pii.exceptions import (
    ConfigurationError,
    DailyBudgetExceeded,
    MeddiesException,
)
from meddies_pii.generation.label_corpus.runner import (
    GenerationRunError,
    GenerationRunPlan,
    GenerationSegment,
    LanguageGenerationPlan,
    SyntheticGenerationRequest,
    run_generation_plan,
    run_language_generation,
)
from meddies_pii.json_types import as_object_list, is_str_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping


def _result_at(summary: Mapping[str, object], index: int) -> Mapping[str, object]:
    """Return one segment result out of a run summary, asserting the shape the caller reads.

    `run_generation_plan` returns `dict[str, object]` and appends each segment result to a
    `list[object]`, because the result dict is built empty and then mutated with values of
    several types — a shape a closed TypedDict cannot express.

    Returns:
        The segment result at that index, narrowed to a mapping the caller can read by key.

    """
    results = as_object_list(summary["results"])
    assert results is not None, "results must be a JSON list"
    value = results[index]
    assert is_str_mapping(value), f"result {index} must be a mapping, got {type(value).__name__}"
    return value


def _segment(
    *,
    language: str = "English",
    target_count: int = 1,
    count_mode: str = "additional",
    enforce_target: bool = False,
) -> GenerationSegment:
    return GenerationSegment(
        language=language,
        scenario="general_consumer_admin_support",
        domain_profile="general",
        required_label_mode="private_url_secret",
        target_count=target_count,
        count_mode=count_mode,
        enforce_target=enforce_target,
    )


def _plan(tmp_path: Path, *segments: GenerationSegment) -> GenerationRunPlan:
    return GenerationRunPlan(
        run_id="test-run",
        provider="mimo",
        model="fake-model",
        output_dir=tmp_path,
        max_concurrency=1,
        rpm_per_key=10,
        max_attempt_multiplier=1.0,
        segments=segments,
        summary_path=tmp_path / "run_summaries" / "test-run.summary.json",
    )


def _append_accepted(output_dir: Path, code: str = "en") -> None:
    path = output_dir / f"accepted.{code}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "synthetic row"}, ensure_ascii=False) + "\n")


def _language_plan(tmp_path: Path, *languages: str) -> LanguageGenerationPlan:
    return LanguageGenerationPlan(
        request_template=SyntheticGenerationRequest(
            provider="mimo",
            model="fake-model",
            language="all",
            target_count=10,
            output_dir=tmp_path,
            domain_profile="general",
        ),
        supported_languages=languages,
    )


def test_language_generation_collects_mixed_results_and_keeps_running(
    tmp_path: Path,
) -> None:
    attempted: list[str] = []

    # reason: `runner.py:394` calls this injected hook as `await run_request(request)`, so dropping `async` would
    # reason: leave the plan awaiting a value that is not awaitable.
    async def run_request(  # ruff: ignore[unused-async]
        request: SyntheticGenerationRequest,
    ) -> dict[str, object]:
        attempted.append(request.language)
        if request.language == "English":
            msg = "provider rejected one document"
            raise MeddiesException(msg)
        return {"accepted_count": 2}

    result = asyncio.run(
        run_language_generation(
            _language_plan(tmp_path, "Vietnamese", "English", "French"),
            run_request=run_request,
        ),
    )

    assert attempted == ["Vietnamese", "English", "French"]
    assert [(item.language, item.accepted_count) for item in result.successes] == [
        ("Vietnamese", 2),
        ("French", 2),
    ]
    assert [(item.language, item.kind, item.message) for item in result.failures] == [
        ("English", "generation", "provider rejected one document"),
    ]


def test_language_generation_stops_after_configuration_failure(
    tmp_path: Path,
) -> None:
    attempted: list[str] = []

    # reason: `runner.py:394` calls this injected hook as `await run_request(request)`, so dropping `async` would
    # reason: leave the plan awaiting a value that is not awaitable.
    async def run_request(  # ruff: ignore[unused-async]
        request: SyntheticGenerationRequest,
    ) -> dict[str, object]:
        attempted.append(request.language)
        msg = "MIMO_API_KEY is missing"
        raise ConfigurationError(msg)

    result = asyncio.run(
        run_language_generation(
            _language_plan(tmp_path, "Vietnamese", "English"),
            run_request=run_request,
        ),
    )

    assert attempted == ["Vietnamese"]
    assert result.successes == ()
    assert [(item.language, item.kind, item.message) for item in result.failures] == [
        ("Vietnamese", "configuration", "MIMO_API_KEY is missing"),
    ]


def test_language_generation_records_unexpected_failure_and_continues(
    tmp_path: Path,
) -> None:
    attempted: list[str] = []

    # reason: `runner.py:394` calls this injected hook as `await run_request(request)`, so dropping `async` would
    # reason: leave the plan awaiting a value that is not awaitable.
    async def run_request(  # ruff: ignore[unused-async]
        request: SyntheticGenerationRequest,
    ) -> dict[str, object]:
        attempted.append(request.language)
        if request.language == "Vietnamese":
            msg = "socket closed"
            raise RuntimeError(msg)
        return {"accepted_count": 4}

    result = asyncio.run(
        run_language_generation(
            _language_plan(tmp_path, "Vietnamese", "English"),
            run_request=run_request,
        ),
    )

    assert attempted == ["Vietnamese", "English"]
    assert [(item.language, item.accepted_count) for item in result.successes] == [("English", 4)]
    assert [(item.language, item.kind, item.message) for item in result.failures] == [
        ("Vietnamese", "unexpected", "socket closed"),
    ]


def test_generation_plan_stops_cleanly_on_budget_exceeded(tmp_path: Path) -> None:
    calls = 0

    # reason: `runner.py:394` calls this injected hook as `await run_request(request)`, so dropping `async` would
    # reason: leave the plan awaiting a value that is not awaitable.
    async def budget_stop(_request: SyntheticGenerationRequest) -> dict[str, object]:  # ruff: ignore[unused-async]
        nonlocal calls
        calls += 1
        msg = "daily token cap reached"
        raise DailyBudgetExceeded(msg)

    summary = asyncio.run(
        run_generation_plan(
            _plan(tmp_path, _segment(), _segment(language="Vietnamese")),
            run_request=budget_stop,
        ),
    )

    assert calls == 1
    first = _result_at(summary, 0)
    assert first["status"] == "budget_exceeded"
    assert first["accepted_before"] == 0
    assert first["accepted_after"] == 0
    assert not (tmp_path / "accepted.en.jsonl").exists()


def test_generation_plan_records_fatal_provider_error_and_stops(
    tmp_path: Path,
) -> None:
    calls = 0

    # reason: `runner.py:394` calls this injected hook as `await run_request(request)`, so dropping `async` would
    # reason: leave the plan awaiting a value that is not awaitable.
    async def fatal(_request: SyntheticGenerationRequest) -> dict[str, object]:  # ruff: ignore[unused-async]
        nonlocal calls
        calls += 1
        msg = "Fatal generation API error: 401"
        raise RuntimeError(msg)

    with pytest.raises(GenerationRunError, match="Fatal generation API error"):
        asyncio.run(
            run_generation_plan(
                _plan(tmp_path, _segment(), _segment(language="Vietnamese")),
                run_request=fatal,
            ),
        )

    summary = json.loads((tmp_path / "run_summaries" / "test-run.summary.json").read_text(encoding="utf-8"))
    assert calls == 1
    assert len(summary["results"]) == 1
    assert summary["results"][0]["status"] == "error"
    assert "401" in summary["results"][0]["error"]


def test_generation_plan_resumes_from_partial_accepted_count(tmp_path: Path) -> None:
    _append_accepted(tmp_path)
    seen_requests: list[SyntheticGenerationRequest] = []

    # reason: `runner.py:394` calls this injected hook as `await run_request(request)`, so dropping `async` would
    # reason: leave the plan awaiting a value that is not awaitable.
    async def add_one(request: SyntheticGenerationRequest) -> dict[str, object]:  # ruff: ignore[unused-async]
        seen_requests.append(request)
        _append_accepted(Path(request.output_dir))
        return {"accepted_count": 2, "rows": [{"do": "not embed"}]}

    summary = asyncio.run(
        run_generation_plan(
            _plan(
                tmp_path,
                _segment(target_count=2, count_mode="total", enforce_target=True),
            ),
            run_request=add_one,
        ),
    )

    assert seen_requests[0].target_count == 2
    assert seen_requests[0].count_mode == "total"
    result = _result_at(summary, 0)
    assert result["accepted_before"] == 1
    assert result["accepted_after"] == 2
    assert result["status"] == "ok"


def test_generation_plan_summary_schema_excludes_raw_rows(tmp_path: Path) -> None:
    # reason: `runner.py:394` calls this injected hook as `await run_request(request)`, so dropping `async` would
    # reason: leave the plan awaiting a value that is not awaitable.
    async def add_one(request: SyntheticGenerationRequest) -> dict[str, object]:  # ruff: ignore[unused-async]
        _append_accepted(Path(request.output_dir))
        return {
            "accepted_count": 1,
            "rows": [{"raw": "large"}],
            "paths": {"summary": "s"},
        }

    summary = asyncio.run(run_generation_plan(_plan(tmp_path, _segment()), run_request=add_one))

    assert {
        "run_id",
        "provider",
        "model",
        "output_dir",
        "segments",
        "results",
        "warnings",
    } <= set(summary)
    result = _result_at(summary, 0)
    assert {
        "accepted_before",
        "accepted_after",
        "status",
        "error",
    } <= set(result)
    nested = result["summary"]
    assert is_str_mapping(nested), "the per-segment summary must be a mapping"
    assert "rows" not in nested


def test_languages_by_deficit_serves_neediest_and_skips_satisfied() -> None:
    from meddies_pii.generation.label_corpus.generation_runs import (
        language_target,
        languages_by_deficit,
    )

    assert language_target("vietnamese") == 20_000
    assert language_target("english") == 20_000
    assert language_target("german") == 15_000

    langs = ("german", "english", "vietnamese", "french")
    current = {"german": 15_000, "english": 0, "vietnamese": 19_000, "french": 14_000}
    order = languages_by_deficit(current, langs)
    assert "german" not in order
    assert order == ["english", "french", "vietnamese"]


def test_cumulative_language_counts_sums_across_whole_corpus(tmp_path: Path) -> None:
    from meddies_pii.generation.label_corpus.generation_runs import (
        cumulative_language_counts,
    )

    a = tmp_path / "weak_labels/2026-06-20/mistral/acct1"
    b = tmp_path / "weak_labels/2026-06-21/groq/acct2"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "accepted.de.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (b / "accepted.de.jsonl").write_text("{}\n", encoding="utf-8")
    (b / "accepted.en.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")

    counts = cumulative_language_counts(tmp_path, ("german", "english", "french"))
    assert counts["german"] == 3
    assert counts["english"] == 3
    assert counts["french"] == 0


def test_run_weak_labels_skips_at_target_and_serves_neediest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct from private_url so the test can tell them apart.

    filled to target (2 rounds x url+secret), then stops.

    """
    import meddies_pii.generation.label_corpus.generation_runs as gr
    from meddies_pii.generation.label_corpus.synthetic import language_paths

    monkeypatch.setattr(gr, "DEFAULT_LANGUAGE_TARGET", 4)
    monkeypatch.setattr(gr, "LANGUAGE_TARGET_OVERRIDES", {})

    prior = tmp_path / "weak_labels/2026-06-20/mistral/acct1"
    prior.mkdir(parents=True)
    (prior / "accepted.de.jsonl").write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")

    requested: list[tuple[str, int]] = []

    # reason: `generation_runs.py:172` and `:202` call this injected hook as `await generate_one(...)`, so dropping
    # reason: `async` would leave the run awaiting a value that is not awaitable.
    async def fake_generate_one(**kwargs: object) -> dict[str, object]:  # ruff: ignore[unused-async]
        lang = str(kwargs["language"])
        target_count = kwargs["target_count"]
        assert isinstance(target_count, int)
        assert not isinstance(target_count, bool)
        requested.append((lang, target_count))
        path = language_paths(str(kwargs["out_dir"]), lang).accepted
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{}\n")
        return {"accepted_count": 1}

    asyncio.run(
        gr.run_weak_labels(
            private_url_per_lang=1,
            secret_per_lang=2,
            languages=("german", "english"),
            run_date="2026-06-21",
            providers=("mistral",),
            out_root=tmp_path,
            max_rounds=10,
            max_minutes=None,
            generate_one=fake_generate_one,
            account_keys=lambda _provider: ["k1"],
        ),
    )

    langs = [lang for lang, _ in requested]
    assert "german" not in langs
    assert "english" in langs
    assert any(tc == 2 for lang, tc in requested if lang == "english")
    assert langs.count("english") == 4


def test_cloudflare_provider_limits_has_no_request_cap() -> None:
    """A genuinely request-capped provider still reports its real RPD."""
    from meddies_pii.generation.label_corpus.generation_runs import _provider_limits

    assert _provider_limits("cloudflare")["rpd_per_account"] is None
    assert _provider_limits("groq")["rpd_per_account"] == 1000


def test_account_rate_limits_respect_documented_caps() -> None:
    """(rpm, rpd, tpm, tpd) documented ceilings; None = unbounded/unpublished → skip."""
    from meddies_pii.generation.label_corpus.generation_policy import (
        ACCOUNT_RATE_LIMITS,
    )

    documented = {
        "groq": (60, 1000, 6000, 500_000),
        "cerebras": (5, 2400, 30_000, 1_000_000),
        "mistral": (25, None, 375_000, None),
        "nim": (40, None, None, None),
        "openrouter": (None, 1000, None, None),
        "llm7": (30, None, None, None),
    }
    for provider, ceiling in documented.items():
        limits = ACCOUNT_RATE_LIMITS[provider]
        for axis, cap in zip(("rpm", "rpd", "tpm", "tpd"), ceiling, strict=True):
            configured = getattr(limits, axis)
            if cap is not None and configured is not None:
                assert configured <= cap, f"{provider} {axis}={configured} EXCEEDS documented {cap} — ban risk"


def test_accepted_count_rejects_malformed_persisted_summary_counter(
    tmp_path: Path,
) -> None:
    from meddies_pii.generation.label_corpus.runner import accepted_count

    (tmp_path / "summary.en.json").write_text('{"accepted_count":"2"}', encoding="utf-8")

    assert accepted_count(tmp_path, "English") == 0


def test_usage_telemetry_failure_keeps_generation_artifacts_and_logs_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from meddies_pii.generation.account_ledger import AccountLedger
    from meddies_pii.generation.label_corpus import generation_runs

    accepted = tmp_path / "out" / "groq" / "acct1" / "accepted.en.jsonl"
    accepted.parent.mkdir(parents=True)
    accepted.write_text('{"text":"kept"}\n', encoding="utf-8")

    def fail_dashboard(_usage_dir: Path) -> Path:
        msg = "dashboard write failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(generation_runs, "render_usage_dashboard", fail_dashboard)
    caplog.set_level(logging.ERROR)

    generation_runs._write_usage(
        run_date="2026-07-29",
        out=str(tmp_path / "out"),
        ledger=AccountLedger({}),
        usage_dir=tmp_path / "usage",
    )

    assert accepted.read_text(encoding="utf-8") == '{"text":"kept"}\n'
    failure = next(record for record in caplog.records if record.message == "usage dashboard update failed")
    assert failure.exc_info is not None
