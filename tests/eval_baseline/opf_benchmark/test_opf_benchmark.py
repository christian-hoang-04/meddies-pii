from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
from meddies_pii.eval_baseline.baseline.datasets import EvalRow
from meddies_pii.eval_baseline.opf_benchmark.dataset import (
    LENGTH_BUCKETS,
    build_pinned_benchmark_doc_set,
)
from meddies_pii.eval_baseline.opf_benchmark.decoding import (
    bioes_token_entities_to_spans,
    token_logits_to_bioes_spans,
)
from meddies_pii.eval_baseline.opf_benchmark.harness import run_mock_benchmark
from meddies_pii.eval_baseline.opf_benchmark.metrics import (
    TimingRepeat,
    compute_correctness,
    summarize_timing,
)
from meddies_pii.eval_baseline.opf_benchmark.ranking import (
    rank_benchmark_results,
)
from meddies_pii.eval_baseline.opf_benchmark.registry import (
    ATTENTION_KERNELS,
    BATCH_SIZES,
    COMPILE_MODES,
    LENGTH_BUCKET_NAMES,
    ONNX_VARIANTS,
    PATH_IDS,
    TRITON_MODES,
    build_config_registry,
    select_representative_configs,
    validate_registry_completeness,
)
from meddies_pii.spans import CharSpan


def _row(doc_id: str, token_count: int, label: str = "human_name") -> EvalRow:
    words = [f"w{i}" for i in range(token_count)]
    words[1] = "John"
    words[2] = "Smith"
    text = " ".join(words)
    start = text.index("John")
    end = start + len("John Smith")
    return EvalRow(
        doc_id=doc_id,
        dataset="unit",
        shard="benchmark",
        text=text,
        gold_spans=(CharSpan(start=start, end=end, text=text[start:end], label=label),),
        language="en",
        slices=frozenset({"unit"}),
    )


def test_varied_length_doc_set_is_stable_and_requires_all_buckets() -> None:
    rows = [
        _row("short-b", 70),
        _row("short-a", 66),
        _row("medium-a", 250),
        _row("long-a", 530),
    ]

    first = build_pinned_benchmark_doc_set(rows, per_bucket=1)
    second = build_pinned_benchmark_doc_set(list(reversed(rows)), per_bucket=1)

    assert tuple(doc.doc_id for doc in first.docs) == (
        "unit:benchmark:short-a",
        "unit:benchmark:medium-a",
        "unit:benchmark:long-a",
    )
    assert first.sha256 == second.sha256
    assert first.bucket_counts == {"short": 1, "medium": 1, "long": 1}
    assert tuple(first.bucket_counts) == LENGTH_BUCKETS


def test_registry_coverage_enumerates_all_paths_and_sweep_dimensions() -> None:
    registry = build_config_registry()

    assert validate_registry_completeness(registry) == []
    assert {config.path_id for config in registry} == set(PATH_IDS)
    assert {config.batch_size for config in registry} == set(BATCH_SIZES)
    assert {config.length_bucket for config in registry} == set(LENGTH_BUCKET_NAMES)
    assert {config.compile_mode for config in registry if config.path_id in {"A", "A2", "B"}} == set(COMPILE_MODES)
    assert {config.triton for config in registry if config.path_id in {"A", "A2"}} == set(TRITON_MODES)
    assert {config.onnx_variant for config in registry if config.path_id == "C"} == set(ONNX_VARIANTS)
    assert {config.attention_kernel for config in registry if config.path_id == "B"} == set(ATTENTION_KERNELS)
    assert all(config.skip_reason for config in registry if config.path_id == "A" and config.batch_size != 1)


def test_timing_summary_excludes_warmup_and_uses_three_repeat_median() -> None:
    summary = summarize_timing(
        repeats=(
            TimingRepeat(timed_seconds=10.0, warmup_seconds=100.0, peak_gpu_memory_mb=1000),
            TimingRepeat(timed_seconds=8.0, warmup_seconds=100.0, peak_gpu_memory_mb=1200),
            TimingRepeat(timed_seconds=12.0, warmup_seconds=100.0, peak_gpu_memory_mb=1100),
        ),
        docs=40,
        tokens=4000,
    )

    assert summary.median_seconds == 10.0
    assert summary.docs_per_sec == 4.0
    assert summary.tokens_per_sec == 400.0
    assert summary.min_seconds == 8.0
    assert summary.max_seconds == 12.0
    assert summary.peak_gpu_memory_mb == 1200


def test_correctness_reports_agreement_vs_crf_and_f1_vs_gold() -> None:
    predicted = {
        "doc-1": [CharSpan(start=0, end=4, text="John", label="human_name")],
        "doc-2": [CharSpan(start=0, end=14, text="secret-token-1", label="secret")],
    }
    crf_reference = {
        "doc-1": [CharSpan(start=0, end=4, text="John", label="human_name")],
        "doc-2": [CharSpan(start=0, end=20, text="secret-token-123456", label="secret")],
    }
    gold = {
        "doc-1": [CharSpan(start=0, end=4, text="John", label="human_name")],
        "doc-2": [CharSpan(start=0, end=20, text="secret-token-123456", label="secret")],
    }

    summary = compute_correctness(predicted, crf_reference, gold)

    assert summary.exact_agree_f1 == 0.5
    assert summary.containment_agree_f1 == 2 / 3
    assert summary.exact_gold_f1 == 0.5
    assert summary.containment_gold_f1 == 2 / 3


def test_hf_bioes_token_decoder_reassembles_end_and_singleton_tags() -> None:
    text = "Call John Smith at +84 912"
    spans = bioes_token_entities_to_spans(
        text,
        [
            {"entity": "B-private_person", "index": 1, "start": 5, "end": 9},
            {"entity": "E-private_person", "index": 2, "start": 10, "end": 15},
            {"entity": "S-private_phone", "index": 4, "start": 19, "end": 26},
        ],
    )

    assert spans == [
        CharSpan(start=5, end=15, text="John Smith", label="private_person"),
        CharSpan(start=19, end=26, text="+84 912", label="private_phone"),
    ]


def test_raw_onnx_logits_decoder_uses_argmax_labels_and_offsets() -> None:
    text = "Email test@example.com key"
    id2label = {
        0: "O",
        1: "B-private_email",
        2: "I-private_email",
        3: "E-private_email",
        4: "S-secret",
    }
    logits = [
        [10.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 10.0],
    ]
    offsets = [(0, 5), (6, 10), (10, 18), (18, 22), (23, 26)]

    assert token_logits_to_bioes_spans(text, logits, offsets, id2label) == [
        CharSpan(
            start=6,
            end=22,
            text="test@example.com",
            label="private_email",
        ),
        CharSpan(start=23, end=26, text="key", label="secret"),
    ]


def test_auto_ranking_picks_fastest_acceptable_config_and_per_path_bests() -> None:
    registry = build_config_registry()
    a = next(
        config for config in registry if config.path_id == "A" and config.batch_size == 1 and config.skip_reason is None
    )
    b_slow = next(
        config for config in registry if config.path_id == "B" and config.batch_size == 8 and config.skip_reason is None
    )
    b_fast_wrong = next(
        config for config in registry if config.path_id == "B" and config.batch_size == 256 and config.skip_reason is None
    )
    c_fast = next(
        config for config in registry if config.path_id == "C" and config.batch_size == 128 and config.skip_reason is None
    )

    recommendation = rank_benchmark_results(
        [
            a.to_result(
                docs_per_sec=10,
                tokens_per_sec=1000,
                exact_agree_f1=1.0,
                containment_agree_f1=1.0,
                exact_gold_f1=0.9,
                containment_gold_f1=0.95,
            ),
            b_slow.to_result(
                docs_per_sec=20,
                tokens_per_sec=2000,
                exact_agree_f1=0.99,
                containment_agree_f1=1.0,
                exact_gold_f1=0.89,
                containment_gold_f1=0.94,
            ),
            b_fast_wrong.to_result(
                docs_per_sec=200,
                tokens_per_sec=20_000,
                exact_agree_f1=0.80,
                containment_agree_f1=0.82,
                exact_gold_f1=0.5,
                containment_gold_f1=0.6,
            ),
            c_fast.to_result(
                docs_per_sec=50,
                tokens_per_sec=5000,
                exact_agree_f1=0.985,
                containment_agree_f1=0.99,
                exact_gold_f1=0.88,
                containment_gold_f1=0.94,
            ),
        ],
        min_exact_agreement_f1=0.98,
    )

    assert recommendation.winner is not None
    assert recommendation.winner.config_id == c_fast.config_id
    assert recommendation.best_by_path["B"].config_id == b_fast_wrong.config_id
    assert recommendation.acceptable_by_path["B"].config_id == b_slow.config_id


def test_mock_dry_run_exercises_full_registry_and_emits_stub_grid() -> None:
    rows = [_row("short-a", 70), _row("medium-a", 260), _row("long-a", 540)]
    doc_set = build_pinned_benchmark_doc_set(rows, per_bucket=1)

    result = run_mock_benchmark(doc_set=doc_set, registry=build_config_registry())

    assert result.doc_set_sha256 == doc_set.sha256
    assert len(result.rows) == len(build_config_registry())
    assert result.grid.startswith("path\tconfig_id\tstatus\tdocs/s\ttokens/s")
    assert "A\t" in result.grid
    assert "A2\t" in result.grid
    assert "B\t" in result.grid
    assert "C\t" in result.grid
    assert all(row.stub for row in result.rows)
    assert any(row.skip_reason for row in result.rows)


def test_representative_subset_covers_all_paths_without_skipped_configs() -> None:
    sub = select_representative_configs(build_config_registry())
    assert {config.path_id for config in sub} == set(PATH_IDS)
    assert all(config.skip_reason is None for config in sub)
    assert all(config.length_bucket == "medium" for config in sub)
    assert {config.batch_size for config in sub} <= {1, 32, 128}
    assert {config.batch_size for config in sub if config.path_id == "A"} == {1}
    assert len(sub) <= 12
