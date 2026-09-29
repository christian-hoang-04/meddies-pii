from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING

from meddies_pii.eval_baseline.opf_benchmark.metrics import (
    TimingRepeat,
    compute_correctness,
    summarize_timing,
)
from meddies_pii.eval_baseline.opf_benchmark.ranking import (
    BenchmarkRecommendation,
    rank_benchmark_results,
)
from meddies_pii.eval_baseline.opf_benchmark.results import (
    BenchmarkResultRow,
    format_results_grid,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from meddies_pii.eval_baseline.opf_benchmark.dataset import (
        BenchmarkDoc,
        PinnedBenchmarkDocSet,
    )
    from meddies_pii.eval_baseline.opf_benchmark.registry import BenchmarkConfig
    from meddies_pii.spans import CharSpan


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    rows: tuple[BenchmarkResultRow, ...]
    doc_set_sha256: str
    grid: str
    recommendation: BenchmarkRecommendation


class MockOpfBackend:
    # reason: stands in for a real OPF backend, which the harness calls as `backend.predict(...)` at
    # reason: :56 and :68; a static form would stop it mirroring the object it substitutes for.
    def predict(  # ruff: ignore[no-self-use]
        self,
        # reason: the real backend reads the config; this stand-in ignores it and keeps the parameter
        # reason: so the signature still mirrors what the harness calls, per the note above.
        config: BenchmarkConfig,  # ruff: ignore[unused-method-argument]
        docs: Sequence[BenchmarkDoc],
    ) -> dict[str, list[CharSpan]]:
        return {doc.doc_id: list(doc.gold_spans) for doc in docs}


def run_mock_benchmark(
    *,
    doc_set: PinnedBenchmarkDocSet,
    registry: Sequence[BenchmarkConfig],
) -> BenchmarkRunResult:
    backend = MockOpfBackend()
    reference_by_bucket = {
        bucket: backend.predict(_canonical_reference_config(registry), doc_set.docs_for_bucket(bucket))
        for bucket in doc_set.bucket_counts
    }
    rows: list[BenchmarkResultRow] = []
    for config in registry:
        if config.skip_reason:
            rows.append(config.to_skipped(stub=True))
            continue
        docs = doc_set.docs_for_bucket(config.length_bucket)
        if not docs:
            rows.append(config.to_skipped(reason="no_docs_for_length_bucket", stub=True))
            continue
        predictions = backend.predict(config, docs)
        gold = {doc.doc_id: list(doc.gold_spans) for doc in docs}
        correctness = compute_correctness(
            predictions,
            _select_reference(reference_by_bucket, config.length_bucket),
            gold,
        )
        timing = summarize_timing(
            repeats=_stub_repeats(config, docs),
            docs=len(docs),
            tokens=sum(doc.token_count for doc in docs),
        )
        rows.append(
            config.to_result(
                docs_per_sec=timing.docs_per_sec,
                tokens_per_sec=timing.tokens_per_sec,
                median_seconds=timing.median_seconds,
                min_seconds=timing.min_seconds,
                max_seconds=timing.max_seconds,
                peak_gpu_memory_mb=timing.peak_gpu_memory_mb,
                exact_agree_f1=correctness.exact_agree_f1,
                containment_agree_f1=correctness.containment_agree_f1,
                exact_gold_f1=correctness.exact_gold_f1,
                containment_gold_f1=correctness.containment_gold_f1,
                stub=True,
            ),
        )
    recommendation = rank_benchmark_results(tuple(rows))
    return BenchmarkRunResult(
        rows=tuple(rows),
        doc_set_sha256=doc_set.sha256,
        grid=format_results_grid(tuple(rows)),
        recommendation=recommendation,
    )


def _canonical_reference_config(registry: Sequence[BenchmarkConfig]) -> BenchmarkConfig:
    for config in registry:
        # reason: canonical keeps path id/batch size in one gate; helper predicates would scatter the rule.
        if (
            config.path_id == "A"  # ruff: ignore[too-many-boolean-expressions]
            and config.batch_size == 1
            and config.compile_mode == "none"
            and config.triton == "on"
            and config.length_bucket == "short"
            and config.skip_reason is None
        ):
            return config
    msg = "registry missing canonical CRF reference config"
    raise ValueError(msg)


def _select_reference(
    reference_by_bucket: Mapping[str, dict[str, list[CharSpan]]],
    bucket: str,
) -> dict[str, list[CharSpan]]:
    return reference_by_bucket.get(bucket, {})


def _stub_repeats(
    config: BenchmarkConfig,
    docs: Sequence[BenchmarkDoc],
) -> tuple[TimingRepeat, TimingRepeat, TimingRepeat]:
    docs_per_sec = _stub_docs_per_sec(config)
    seconds = max(0.001, len(docs) / docs_per_sec)
    peak = _stub_peak_memory(config)
    return (
        TimingRepeat(
            timed_seconds=seconds * 1.05,
            warmup_seconds=seconds * 0.5,
            peak_gpu_memory_mb=peak,
        ),
        TimingRepeat(
            timed_seconds=seconds,
            warmup_seconds=seconds * 0.5,
            peak_gpu_memory_mb=peak + 32,
        ),
        TimingRepeat(
            timed_seconds=seconds * 0.95,
            warmup_seconds=seconds * 0.5,
            peak_gpu_memory_mb=peak + 16,
        ),
    )


def _stub_docs_per_sec(config: BenchmarkConfig) -> float:
    base_by_path: dict[str, float] = {"A": 8.0, "A2": 24.0, "B": 36.0, "C": 44.0}
    length_penalty: dict[str, float] = {"short": 1.0, "medium": 0.55, "long": 0.28}
    compile_bonus: dict[str, float] = {
        "none": 1.0,
        "reduce-overhead": 1.08,
        "max-autotune": 1.12,
    }
    batch_scale = max(1.0, sqrt(min(config.batch_size, 128)))
    triton_bonus = 1.12 if config.triton == "on" else 1.0
    return (
        base_by_path[config.path_id]
        * length_penalty[config.length_bucket]
        * compile_bonus[config.compile_mode]
        * batch_scale
        * triton_bonus
    )


def _stub_peak_memory(config: BenchmarkConfig) -> int:
    base_by_path = {"A": 4200, "A2": 5200, "B": 6400, "C": 3600}
    return base_by_path[config.path_id] + int(config.batch_size * 12)
