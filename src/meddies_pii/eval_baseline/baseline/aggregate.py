"""Aggregate eval shards into comparable span-F1 reports.

Rows may arrive as ordinary re-iterable sequences or zero-argument iterator
factories. Aggregation keeps metric counts rather than materializing a second
whole-matrix row/doc/span view; fixture identities remain fail-closed.
"""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: the guard reports a missing optional dependency, not a caller passing the wrong type, so TypeError
# reason: would misdescribe it; the same function raises this type from non-isinstance guards too.
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import starmap
from typing import TypeAlias, TypedDict

from meddies_pii.eval_baseline.baseline.datasets import canonical_language
from meddies_pii.evaluation.identity import (
    DatasetShardIdentity,
    EvaluationContract,
    FixtureRowIdentity,
    canonical_sha256,
    is_sha256,
)
from meddies_pii.evaluation.span_metrics import (
    ContainmentSpanMetricBlock,
    ExactSpanMetricBlock,
    SpanMetricBlock,
    containment_span_counts,
    containment_span_prf_from_counts,
    exact_span_counts,
    exact_span_prf_from_counts,
    typed_span_counts_by_label,
)
from meddies_pii.json_types import JsonObject, JsonValue, is_str_list, is_str_mapping
from meddies_pii.spans import CharSpan, char_span_from_value
from meddies_pii.taxonomy import PII_LABEL_SET, PiiLabel

EvalRowDict = Mapping[str, JsonValue]
EvalRowSource: TypeAlias = Sequence[EvalRowDict] | Callable[[], Iterable[EvalRowDict]]
_Counts = dict[str, int]
Summary: TypeAlias = dict[str, int | float]
MetricBlock: TypeAlias = SpanMetricBlock


class LabelMetrics(TypedDict):
    exact: ExactSpanMetricBlock
    containment: ContainmentSpanMetricBlock


FixtureConfig: TypeAlias = dict[str, int | str]


class FixtureManifest(TypedDict):
    sha256: str
    rows: int
    full9_gold_spans: int
    configs: dict[str, FixtureConfig]


class ResultProvenance(TypedDict):
    """Raw-byte identities for exactly the result artifacts that were scored."""

    result_set_sha256: str
    result_sha256_by_config: dict[str, str]


class EvaluationContractProvenance(TypedDict):
    """Canonical evaluation contract bound to this aggregate's metrics."""

    payload: JsonObject
    sha256: str


class AggregateReport(TypedDict):
    evaluation_contract: EvaluationContractProvenance
    fixture: FixtureManifest
    provenance: ResultProvenance
    overall: Summary
    per_config: dict[str, Summary]
    per_language: dict[str, Summary]
    per_label: dict[str, LabelMetrics]
    per_label_full9: dict[str, LabelMetrics]


def spans_from_values(raw: object) -> list[CharSpan]:
    """Convert well-formed persisted span values, ignoring malformed entries.

    Returns:
        The valid spans in their input order.

    """
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        return []
    return [span for value in raw if (span := char_span_from_value(value)) is not None]


def _iter_rows(source: EvalRowSource) -> Iterable[EvalRowDict]:
    """Yield one pass over a re-iterable source without copying its rows.

    Returns:
        The source itself when it is re-iterable, or a fresh iterator from calling it, so the caller can walk the rows
        without holding them all in memory.

    """
    # reason: `EvalRowSource` is a sequence-or-factory union, and nothing observable at runtime
    # reason: separates the arms for the checker: it treats a callable as possibly also a
    # reason: sequence, so `callable()` yields an intersection with no provable signature and
    # reason: `isinstance(..., Sequence)` yields one whose items degrade to `object`. Both were
    # reason: tried. Splitting the parameter in two is the real fix and is a caller-wide change.
    if callable(source):
        return source()  # ty: ignore[call-top-callable, invalid-return-type]
    return source


def _empty_exact_counts() -> _Counts:
    return {"tp": 0, "pred_total": 0, "gold_total": 0}


def _empty_containment_counts() -> _Counts:
    return {"precision_tp": 0, "recall_tp": 0, "pred_total": 0, "gold_total": 0}


def _add_counts(total: _Counts, observed: Mapping[str, int]) -> None:
    for key, value in observed.items():
        total[key] += value


@dataclass(frozen=True, slots=True)
class _MetricObservation:
    full_exact: _Counts
    supported_exact: _Counts
    full_containment: _Counts
    supported_containment: _Counts
    label_exact: dict[PiiLabel, _Counts]
    label_containment: dict[PiiLabel, _Counts]


class _ScoreAccumulator:
    """Merge precomputed per-row metric observations into one report slice."""

    def __init__(self, *, include_labels: bool = False) -> None:
        self.rows = 0
        self._full_exact = _empty_exact_counts()
        self._supported_exact = _empty_exact_counts()
        self._full_containment = _empty_containment_counts()
        self._supported_containment = _empty_containment_counts()
        self._label_exact = {label: _empty_exact_counts() for label in PII_LABEL_SET} if include_labels else None
        self._label_containment = (
            {label: _empty_containment_counts() for label in PII_LABEL_SET} if include_labels else None
        )

    def merge(self, observation: _MetricObservation) -> None:
        self.rows += 1
        _add_counts(self._full_exact, observation.full_exact)
        _add_counts(self._supported_exact, observation.supported_exact)
        _add_counts(self._full_containment, observation.full_containment)
        _add_counts(self._supported_containment, observation.supported_containment)
        if self._label_exact is None or self._label_containment is None:
            return
        for label in PII_LABEL_SET:
            _add_counts(self._label_exact[label], observation.label_exact[label])
            _add_counts(self._label_containment[label], observation.label_containment[label])

    def summary(self) -> Summary:
        return _summary_from_counts(
            self._supported_exact,
            self._supported_containment,
            self._full_exact,
            self._full_containment,
            rows=self.rows,
        )

    def label_report(self, labels: Iterable[PiiLabel]) -> dict[str, LabelMetrics]:
        if self._label_exact is None or self._label_containment is None:
            msg = "label metrics were not requested"
            raise RuntimeError(msg)
        return {
            label: LabelMetrics(
                exact=exact_span_prf_from_counts(self._label_exact[label]),
                containment=containment_span_prf_from_counts(self._label_containment[label]),
            )
            for label in sorted(labels)
        }


def _observe_metrics(
    predicted: Sequence[CharSpan],
    gold: Sequence[CharSpan],
    *,
    supported_labels: frozenset[PiiLabel],
) -> _MetricObservation:
    """Compute the four general metric observations once for one persisted row.

    Returns:
        One row's four count families: full and supported exact counts, and full and supported containment counts.

    """
    supported_gold = tuple(span for span in gold if span.label in supported_labels)
    label_exact, label_containment = typed_span_counts_by_label(predicted, gold)
    return _MetricObservation(
        full_exact=exact_span_counts(predicted, gold),
        supported_exact=exact_span_counts(predicted, supported_gold),
        full_containment=containment_span_counts(predicted, gold),
        supported_containment=containment_span_counts(predicted, supported_gold),
        label_exact=label_exact,
        label_containment=label_containment,
    )


def _summary_from_counts(
    supported_exact_counts: Mapping[str, int],
    supported_containment_counts: Mapping[str, int],
    full_exact_counts: Mapping[str, int],
    full_containment_counts: Mapping[str, int],
    *,
    rows: int,
) -> Summary:
    exact = exact_span_prf_from_counts(supported_exact_counts)
    containment = containment_span_prf_from_counts(supported_containment_counts)
    full_exact = exact_span_prf_from_counts(full_exact_counts)
    full_containment = containment_span_prf_from_counts(full_containment_counts)
    return {
        "rows": rows,
        "gold_spans": int(exact["gold_total"]),
        "pred_spans": int(exact["pred_total"]),
        "exact_f1": round(float(exact["f1"]), 4),
        "exact_precision": round(float(exact["precision"]), 4),
        "exact_recall": round(float(exact["recall"]), 4),
        "containment_f1": round(float(containment["f1"]), 4),
        "containment_precision": round(float(containment["precision"]), 4),
        "containment_recall": round(float(containment["recall"]), 4),
        "full9_gold_spans": int(full_exact["gold_total"]),
        "full9_pred_spans": int(full_exact["pred_total"]),
        "full9_exact_f1": round(float(full_exact["f1"]), 4),
        "full9_exact_precision": round(float(full_exact["precision"]), 4),
        "full9_exact_recall": round(float(full_exact["recall"]), 4),
        "full9_containment_f1": round(float(full_containment["f1"]), 4),
        "full9_containment_precision": round(float(full_containment["precision"]), 4),
        "full9_containment_recall": round(float(full_containment["recall"]), 4),
    }


def _macro_f1(by_label: Mapping[str, MetricBlock]) -> float:
    values = [float(metrics["f1"]) for metrics in by_label.values()]
    return round(sum(values) / len(values), 4) if values else 0.0


def _rows_per_second(rows: int, seconds: float) -> float:
    return round(rows / seconds, 1) if seconds > 0 else 0.0


def _fixture_manifest(
    identities_by_config: Mapping[str, DatasetShardIdentity],
) -> FixtureManifest:
    configs: dict[str, FixtureConfig] = {}
    total_rows = 0
    total_gold = 0
    for config, identity in sorted(identities_by_config.items()):
        config_gold = sum(len(row.gold_spans) for row in identity.fixture)
        configs[config] = {
            "rows": identity.row_count,
            "full9_gold_spans": config_gold,
            "fixture_sha256": identity.fixture_identity.digest,
        }
        total_rows += identity.row_count
        total_gold += config_gold
    return {
        "sha256": canonical_sha256({
            "dataset_shard_fixture_identities": {
                config: identity.fixture_identity.to_payload() for config, identity in sorted(identities_by_config.items())
            },
        }),
        "rows": total_rows,
        "full9_gold_spans": total_gold,
        "configs": configs,
    }


def _assert_full9_gold_integrity(
    fixture: FixtureManifest,
    overall: Summary,
    per_config: Mapping[str, Summary],
) -> None:
    fixture_total = int(fixture["full9_gold_spans"])
    scored_total = int(overall["full9_gold_spans"])
    if fixture_total != scored_total:
        msg = f"full-9 gold span count changed during aggregation: fixture={fixture_total}, scored={scored_total}"
        raise AssertionError(
            msg,
        )

    fixture_configs = fixture["configs"]
    if not isinstance(fixture_configs, Mapping):
        msg = "fixture configs must be a mapping"
        raise AssertionError(msg)
    for config, fixture_summary in fixture_configs.items():
        if not isinstance(fixture_summary, Mapping):
            msg = f"fixture config {config!r} must be a mapping"
            raise AssertionError(msg)
        fixture_config_total = int(fixture_summary["full9_gold_spans"])
        scored_config_total = int(per_config[str(config)]["full9_gold_spans"])
        if fixture_config_total != scored_config_total:
            msg = (
                "full-9 gold span count changed during config aggregation: "
                f"config={config!r}, fixture={fixture_config_total}, "
                f"scored={scored_config_total}"
            )
            raise AssertionError(
                msg,
            )


# reason: aggregate results coordinates merge with assert; extra seams would fragment diagnostics.
def aggregate_results(  # ruff: ignore[too-many-arguments,too-many-locals]
    rows_by_config: Mapping[str, EvalRowSource],
    *,
    supported_labels: frozenset[PiiLabel],
    expected_evaluation_contract: EvaluationContract,
    expected_dataset_shard_identities: Mapping[str, DatasetShardIdentity],
    result_sha256_by_config: Mapping[str, str],
    timing_by_config: Mapping[str, Mapping[str, float]] | None = None,
) -> AggregateReport:
    """Validate and score every persisted result row in one source pass.

    The report is assembled in memory but is never returned until each row has
    matched the complete expected fixture, including order and row count.

    Returns:
        The scored report, carrying the evaluation contract's identity, the fixture identity, the provenance of every
        source digest, and the metrics.

    Raises:
        ValueError: if a result cell does not match its required fixture identity, if the artifact digests and dataset
            shard identities disagree, if a digest is not SHA-256 hex, or if the sources carry mixed evaluation contracts.

    """
    _assert_identity_headers(
        rows_by_config,
        expected_evaluation_contract,
        expected_dataset_shard_identities,
    )
    provenance = _result_provenance(
        result_sha256_by_config,
        expected_dataset_shard_identities,
    )
    _assert_source_result_digests(rows_by_config, provenance)
    overall_accumulator = _ScoreAccumulator(include_labels=True)
    per_config: dict[str, Summary] = {}
    per_language_accumulators: dict[str, _ScoreAccumulator] = {}
    for config, source in rows_by_config.items():
        identity = expected_dataset_shard_identities[config]
        config_accumulator = _ScoreAccumulator()
        expected_rows = iter(identity.fixture)
        for row in _iter_rows(source):
            try:
                expected_row = next(expected_rows)
            except StopIteration:
                msg = "result cell fixture does not match its required identity"
                raise ValueError(msg) from None
            if not _result_row_matches_fixture(row, expected_row):
                msg = "result cell fixture does not match its required identity"
                raise ValueError(msg)
            observation = _observe_metrics(
                spans_from_values(row.get("pred_spans")),
                spans_from_values(row.get("gold_spans")),
                supported_labels=supported_labels,
            )
            language = canonical_language(str(row.get("language", "unknown")))
            language_accumulator = per_language_accumulators.setdefault(language, _ScoreAccumulator())
            for accumulator in (
                config_accumulator,
                language_accumulator,
                overall_accumulator,
            ):
                accumulator.merge(observation)
        try:
            next(expected_rows)
        except StopIteration:
            pass
        else:
            msg = "result cell fixture does not match its required identity"
            raise ValueError(msg)
        summary = config_accumulator.summary()
        if timing_by_config and config in timing_by_config:
            seconds = float(timing_by_config[config].get("elapsed_seconds", 0.0))
            summary["elapsed_seconds"] = round(seconds, 2)
            summary["rows_per_second"] = _rows_per_second(config_accumulator.rows, seconds)
        per_config[config] = summary

    per_language = {language: accumulator.summary() for language, accumulator in sorted(per_language_accumulators.items())}
    per_label = overall_accumulator.label_report(supported_labels)
    per_label_full9 = overall_accumulator.label_report(PII_LABEL_SET)
    overall = overall_accumulator.summary()
    overall["macro_exact_f1"] = _macro_f1({label: blocks["exact"] for label, blocks in per_label.items()})
    overall["macro_containment_f1"] = _macro_f1({label: blocks["containment"] for label, blocks in per_label.items()})
    overall["full9_macro_exact_f1"] = _macro_f1({label: blocks["exact"] for label, blocks in per_label_full9.items()})
    overall["full9_macro_containment_f1"] = _macro_f1({
        label: blocks["containment"] for label, blocks in per_label_full9.items()
    })
    if timing_by_config:
        total_seconds = sum(float(entry.get("elapsed_seconds", 0.0)) for entry in timing_by_config.values())
        overall["elapsed_seconds"] = round(total_seconds, 2)
        overall["rows_per_second"] = _rows_per_second(overall_accumulator.rows, total_seconds)
    fixture = _fixture_manifest(expected_dataset_shard_identities)
    _assert_full9_gold_integrity(fixture, overall, per_config)
    return {
        "evaluation_contract": _evaluation_contract_provenance(expected_evaluation_contract),
        "fixture": fixture,
        "provenance": provenance,
        "overall": overall,
        "per_config": per_config,
        "per_language": per_language,
        "per_label": per_label,
        "per_label_full9": per_label_full9,
    }


def _evaluation_contract_provenance(
    evaluation_contract: EvaluationContract,
) -> EvaluationContractProvenance:
    """Return the canonical contract identity that governs scored result rows.

    Returns:
        The contract's full payload together with its digest, which is what pins the scored rows to one contract.

    """
    return {
        "payload": evaluation_contract.to_payload(),
        "sha256": evaluation_contract.digest,
    }


def _result_provenance(
    result_sha256_by_config: Mapping[str, str],
    expected_dataset_shard_identities: Mapping[str, DatasetShardIdentity],
) -> ResultProvenance:
    if set(result_sha256_by_config) != set(expected_dataset_shard_identities):
        msg = "result artifact digests and dataset shard identities differ"
        raise ValueError(msg)
    normalized = {config: digest for config, digest in sorted(result_sha256_by_config.items()) if is_sha256(digest)}
    if len(normalized) != len(result_sha256_by_config):
        msg = "result artifact digests must be SHA-256 hex strings"
        raise ValueError(msg)
    return {
        "result_sha256_by_config": normalized,
        "result_set_sha256": canonical_sha256({"result_sha256_by_config": normalized}),
    }


def _assert_source_result_digests(rows_by_config: Mapping[str, EvalRowSource], provenance: ResultProvenance) -> None:
    """Keep a MatrixResults source bound to the raw bytes it will re-validate.

    Raises:
        ValueError: if the source's digests do not match the raw bytes it will re-validate.

    """
    for config, source in rows_by_config.items():
        source_digest = getattr(source, "expected_result_sha256", None)
        if source_digest is not None and source_digest != provenance["result_sha256_by_config"][config]:
            msg = "result source digest does not match aggregate provenance"
            raise ValueError(msg)


def _assert_identity_headers(
    rows_by_config: Mapping[str, EvalRowSource],
    expected_evaluation_contract: EvaluationContract,
    expected_dataset_shard_identities: Mapping[str, DatasetShardIdentity],
) -> None:
    if set(rows_by_config) != set(expected_dataset_shard_identities):
        msg = "requested result cells and dataset shard identities differ"
        raise ValueError(msg)
    for dataset, identity in expected_dataset_shard_identities.items():
        if identity.evaluation_contract != expected_evaluation_contract:
            msg = "mixed evaluation contracts cannot be aggregated"
            raise ValueError(msg)
        if identity.dataset != dataset or identity.shard != "full":
            msg = "dataset shard identity does not match requested result cell"
            raise ValueError(msg)


def _result_row_matches_fixture(row: EvalRowDict, expected: FixtureRowIdentity) -> bool:
    """Validate exactly the result-row fields encoded in ``FixtureRowIdentity``.

    Returns:
        Whether every field the fixture identity encodes matches, including each gold span in order.

    """
    raw_spans = row.get("gold_spans")
    raw_slice = row.get("slice")
    raw_slices = tuple(raw_slice) if is_str_list(raw_slice) else None
    if not isinstance(raw_spans, list) or raw_slices is None:
        return False
    stable_id = row.get("doc_id")
    text_sha256 = row.get("id")
    language = row.get("language")
    # reason: result row keeps stable id/text in one gate; helper predicates would scatter the rule.
    if (
        not isinstance(stable_id, str)  # ruff: ignore[too-many-boolean-expressions]
        or not isinstance(text_sha256, str)
        or not isinstance(language, str)
        or stable_id != expected.stable_id
        or text_sha256 != expected.text_sha256
        or language != expected.language
        or tuple(sorted(raw_slices)) != expected.slices
        or len(raw_spans) != len(expected.gold_spans)
    ):
        return False
    return all(
        starmap(
            _result_span_matches_fixture,
            zip(raw_spans, expected.gold_spans, strict=True),
        ),
    )


def _result_span_matches_fixture(raw_span: object, expected: CharSpan) -> bool:
    if not is_str_mapping(raw_span):
        return False
    start = raw_span.get("start")
    end = raw_span.get("end")
    text = raw_span.get("text")
    label = raw_span.get("label")
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and isinstance(text, str)
        and isinstance(label, str)
        and start == expected.start
        and end == expected.end
        and text == expected.text
        and label == expected.label
    )


def _fmt_row(name: str, summary: Summary, width: int) -> str:
    return (
        f"{name:<{width}} rows={summary['rows']:>5} "
        f"gold={summary['gold_spans']:>6} pred={summary['pred_spans']:>6} "
        f"exactF1={summary['exact_f1']:.3f} "
        f"(P={summary['exact_precision']:.3f} R={summary['exact_recall']:.3f}) "
        f"containF1={summary['containment_f1']:.3f}"
    )


def format_aggregate_report(report: AggregateReport, *, model: str) -> str:
    """Compact text grid for the Modal log (stays well under the 64KB blob cap).

    Returns:
        The report as a text grid, sized to stay under Modal's 64KB single-log-line cap.

    """
    lines: list[str] = [f"=== {model} baseline ==="]
    overall = report["overall"]
    speed = (
        f" | {float(overall['rows_per_second']):.1f} rows/s ({float(overall['elapsed_seconds']):.0f}s)"
        if "rows_per_second" in overall
        else ""
    )
    lines.append(
        f"OVERALL rows={overall['rows']} gold={overall['gold_spans']} "
        f"pred={overall['pred_spans']} | "
        f"exact micro-F1={overall['exact_f1']:.3f} macro-F1={overall['macro_exact_f1']:.3f} "
        f"| containment micro-F1={overall['containment_f1']:.3f} "
        f"macro-F1={overall['macro_containment_f1']:.3f} "
        f"| full9 exact-F1={overall['full9_exact_f1']:.3f}{speed}",
    )
    config_width = max((len(name) for name in report["per_config"]), default=10)
    lines.append("--- per config (source) ---")
    lines.extend(_fmt_row(name, report["per_config"][name], config_width) for name in sorted(report["per_config"]))
    lang_width = max((len(name) for name in report["per_language"]), default=6)
    lines.append("--- per language ---")
    lines.extend(_fmt_row(name, report["per_language"][name], lang_width) for name in sorted(report["per_language"]))
    label_width = max((len(name) for name in report["per_label"]), default=12)
    lines.append("--- per label (pooled, exact P/R/F1 + containment F1) ---")
    for label in sorted(report["per_label"]):
        exact = report["per_label"][label]["exact"]
        contain = report["per_label"][label]["containment"]
        lines.append(
            f"{label:<{label_width}} "
            f"gold={int(exact['gold_total']):>6} pred={int(exact['pred_total']):>6} "
            f"exact P={float(exact['precision']):.3f} R={float(exact['recall']):.3f} "
            f"F1={float(exact['f1']):.3f} | containF1={float(contain['f1']):.3f}",
        )
    return "\n".join(lines)
