"""Evidence aggregation and budget projection for the regex release gate."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from meddies_pii.eval_baseline.baseline.views import MODEL_CORE_VIEW, MODEL_PLUS_REGEX_VIEW
from meddies_pii.eval_baseline.regex_release.regex_bootstrap import (
    PairedBootstrapResult,
    PairedDocumentCounts,
    paired_document_bootstrap,
    percentile,
)
from meddies_pii.eval_baseline.regex_release.regex_report import (
    ELIGIBILITY_BOOTSTRAP_REPLICATES,
    RegexCorpusManifest,
    build_regex_quality_report,
)
from meddies_pii.evaluation.identity import canonical_sha256, is_sha256
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import PII_LABEL_SET, PiiLabel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from meddies_pii.eval_baseline.regex_release.regex_release_contract import ReleaseGateContract


@dataclass(frozen=True, slots=True)
class LabelMetricCounts:
    label: str
    model_tp: int
    model_fp: int
    model_fn: int
    regex_tp: int
    regex_fp: int
    regex_fn: int

    def __post_init__(self) -> None:
        values = (
            self.model_tp,
            self.model_fp,
            self.model_fn,
            self.regex_tp,
            self.regex_fp,
            self.regex_fn,
        )
        if not self.label or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            msg = "label counts require a label and non-negative integers"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ChildRegexEvidence:
    config: str
    corpus_manifest: RegexCorpusManifest
    documents: tuple[PairedDocumentCounts, ...]
    label_counts: tuple[LabelMetricCounts, ...]
    contract_sha256: str
    evaluation_contract_sha256: str
    runtime_sha256: str
    evidence_sha256: str
    ignore_list_sha256: str | None = None
    """Digest of the ignore-list active when this child was scored. None means no ignore-list contract was declared, so no
    false positives were excluded.
    """

    def __post_init__(self) -> None:
        if self.config not in {"eval", "eval-challenge"}:
            msg = "child evidence config is not part of the release gate"
            raise ValueError(msg)
        if len({counts.label for counts in self.label_counts}) != len(self.label_counts):
            msg = "child evidence labels must be unique"
            raise ValueError(msg)
        if any(
            not is_sha256(digest)
            for digest in (
                self.contract_sha256,
                self.evaluation_contract_sha256,
                self.runtime_sha256,
                self.evidence_sha256,
            )
        ):
            msg = "child evidence requires SHA-256 identities"
            raise ValueError(msg)
        if self.ignore_list_sha256 is not None and not is_sha256(self.ignore_list_sha256):
            msg = "child evidence ignore-list digest must be a SHA-256"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SliceMetrics:
    name: str
    documents: int | None
    model_f1: float
    regex_f1: float
    model_precision: float
    regex_precision: float
    model_recall: float
    regex_recall: float


@dataclass(frozen=True, slots=True)
class ChallengeGateEvidence:
    """The challenge-only block the three gate conditions are evaluated on."""

    documents: int
    model_f1: float
    regex_f1: float
    model_precision: float
    regex_precision: float
    bootstrap: PairedBootstrapResult
    locked_negative_new_false_positives: int
    sample_adequate_for_inference: bool
    quality_gate_passed: bool
    model_fp_ignored: int = 0
    """How many false positives this precision already excludes via the ignore-list, so a verdict can answer "how much
    delta came from exclusions" on its face. Zero when no ignore-list is active.
    """
    regex_fp_ignored: int = 0


@dataclass(frozen=True, slots=True)
class LeakMetrics:
    """Redaction safety: gold-PII characters no predicted span of any label covered.

    REPORT-ONLY. Promoting leak into the gate conditions is a founder decision, so
    `aggregate_exposed_children` never reads these values when deciding a verdict.
    The distribution is reported rather than a mean because a mean hides the tail,
    and the tail is the row where a patient identifier survived redaction.
    """

    scope: str
    view: str
    rows_with_gold: int
    any_leak_rows: int
    any_leak_row_rate: float
    p95_row_leak_fraction: float
    max_row_leak_fraction: float
    char_weighted_leak_fraction: float


@dataclass(frozen=True, slots=True)
class OverRedactionMetrics:
    """Redaction cost: masked characters no gold span of any label covers.

    The mirror of `LeakMetrics` — leak is "PII that survived," over-redaction
    is "healthy prose that didn't." REPORT-ONLY. Promoting it into the gate
    conditions is a founder decision, so `aggregate_exposed_children` never
    reads these values when deciding a verdict. The distribution is reported
    rather than a mean for the same reason as leak's: a mean hides the tail,
    and the tail is the row where an entire readable document got masked.

    Rows predating the `document_chars` schema addition (2026-08-11) cannot
    supply a non-gold denominator, so `aggregate_exposed_children` omits this
    block entirely whenever any row in scope lacks it — never a partial count
    over only the rows that happen to carry it. From this schema addition
    onward, `document_chars` enters the persisted evidence digest.
    """

    scope: str
    view: str
    rows_with_text: int
    any_overredaction_rows: int
    any_overredaction_row_rate: float
    p95_row_overredaction_fraction: float
    max_row_overredaction_fraction: float
    char_weighted_overredaction_fraction: float


@dataclass(frozen=True, slots=True)
class IgnoredFalsePositives:
    """Per-child ignore-list exclusion totals, for gate-verdict transparency."""

    config: str
    model_fp_ignored: int
    regex_fp_ignored: int


@dataclass(frozen=True, slots=True)
class RegexAggregateReport:
    """Challenge-only gate evidence beside a descriptive pooled 5,100-row block.

    Every field named `descriptive_*` or reported at the pooled 5,100-row level is
    development narrative only. `quality_gate_passed` mirrors `challenge`.
    """

    documents: int
    model_f1: float
    regex_f1: float
    model_precision: float
    regex_precision: float
    locked_negative_new_false_positives: int
    descriptive_pooled_bootstrap: PairedBootstrapResult
    challenge: ChallengeGateEvidence
    descriptive_pooled_by_language: tuple[SliceMetrics, ...]
    descriptive_pooled_by_label: tuple[SliceMetrics, ...]
    child_evidence_sha256s: tuple[str, ...]
    evidence_sha256: str
    quality_gate_passed: bool
    development_release_gate_passed: bool
    shipping_confirmation_passed: bool
    model_fp_ignored: int = 0
    """Pooled ignore-list exclusion totals across all 5,100 rows, plus the same totals broken out per child (eval,
    eval-challenge) for transparency.
    """
    regex_fp_ignored: int = 0
    child_ignored_false_positives: tuple[IgnoredFalsePositives, ...] = ()
    descriptive_leak_metrics: tuple[LeakMetrics, ...] = ()
    """One entry per (scope, view) over eval, eval-challenge and pooled. Kept off ChallengeGateEvidence on purpose: that
    block holds the gate conditions, and leak is report-only until a founder decision promotes it.
    """
    descriptive_overredaction_metrics: tuple[OverRedactionMetrics, ...] = ()
    """Same shape as descriptive_leak_metrics, mirrored for over-redaction. Empty whenever any document in scope predates
    the document_chars field.
    """


@dataclass(frozen=True, slots=True)
class CostProjection:
    sample_rows: int
    sample_seconds: float
    projected_total_rows: int
    projected_total_seconds: float
    projected_cost_usd: float


class BudgetProjectionError(RuntimeError):
    def __init__(self, projection: CostProjection) -> None:
        self.projection = projection
        super().__init__(
            "100-row projection exceeds the frozen time or cost ceiling: "
            f"{projection.projected_total_seconds:.3f}s, "
            f"${projection.projected_cost_usd:.6f}",
        )


class _SpanDetector(Protocol):
    def detect(self, texts: Sequence[str]) -> Sequence[object]: ...


class ProjectedInferenceAdapter:
    """Run a fixed sample first, then infer every remaining row exactly once."""

    name = "meddies-pii-q8-ort-cpu"

    # reason: each keyword is an independently measured inference or budget control; a bundle would weaken test injection.
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        detector: _SpanDetector,
        supported_labels: frozenset[PiiLabel],
        sample_rows: int,
        projected_total_rows: int,
        reserve_seconds: float,
        timeout_seconds: float,
        cost_rate_usd_per_second: float,
        cost_ceiling_usd: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if sample_rows <= 0 or projected_total_rows < sample_rows:
            msg = "projection row counts are invalid"
            raise ValueError(msg)
        if (
            min(
                reserve_seconds,
                timeout_seconds,
                cost_rate_usd_per_second,
                cost_ceiling_usd,
            )
            < 0
        ):
            msg = "projection limits cannot be negative"
            raise ValueError(msg)
        self.detector = detector
        self.supported_labels = supported_labels
        self.sample_rows = sample_rows
        self.projected_total_rows = projected_total_rows
        self.reserve_seconds = reserve_seconds
        self.timeout_seconds = timeout_seconds
        self.cost_rate_usd_per_second = cost_rate_usd_per_second
        self.cost_ceiling_usd = cost_ceiling_usd
        self.clock = clock
        self.projection: CostProjection | None = None

    def load(self) -> None:
        loader = getattr(self.detector, "load", None)
        if callable(loader):
            loader()

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        if len(texts) < self.sample_rows:
            msg = "inference shard is smaller than projection sample"
            raise ValueError(msg)
        started = self.clock()
        sampled = self.detector.detect(texts[: self.sample_rows])
        sample_seconds = self.clock() - started
        projected_seconds = sample_seconds / self.sample_rows * self.projected_total_rows + self.reserve_seconds
        projection = CostProjection(
            sample_rows=self.sample_rows,
            sample_seconds=sample_seconds,
            projected_total_rows=self.projected_total_rows,
            projected_total_seconds=projected_seconds,
            projected_cost_usd=projected_seconds * self.cost_rate_usd_per_second,
        )
        self.projection = projection
        if (
            projection.projected_total_seconds > self.timeout_seconds
            or projection.projected_cost_usd > self.cost_ceiling_usd
        ):
            raise BudgetProjectionError(projection)
        remaining = self.detector.detect(texts[self.sample_rows :])
        return [_detection_spans(detection) for detection in (*tuple(sampled), *tuple(remaining))]


# reason: the release gate must validate both child identities and their shared corpus contract before aggregation.
def aggregate_exposed_children(  # ruff: ignore[complex-structure,too-many-statements,too-many-locals]
    *,
    contract: ReleaseGateContract,
    children: tuple[ChildRegexEvidence, ChildRegexEvidence],
    hydrated_fixture_sha256s: Mapping[str, str],
) -> RegexAggregateReport:
    """Gate on the challenge shard alone; describe the pooled 5,100 rows beside it.

    `hydrated_fixture_sha256s` must come from the hydration receipt, which records
    each shard digest independently of the child manifests being verified.

    Bumped because character-safety fields now use gold spans union exact audited ignore-list spans. Field names stay
    stable, but the persisted numeric semantics and therefore the digest changed.

    Returns:
        The challenge-gated and pooled descriptive regex aggregate report.

    Raises:
        ValueError: If child evidence, hydration pins, corpus identities, or paired rows violate the release contract.

    """
    if contract.bootstrap_replicates != ELIGIBILITY_BOOTSTRAP_REPLICATES:
        msg = "contract replicates diverge from the eligibility bootstrap"
        raise ValueError(msg)
    by_config = {child.config: child for child in children}
    if set(by_config) != {"eval", "eval-challenge"} or len(by_config) != len(children):
        msg = "aggregate requires exactly one child for each evaluation config"
        raise ValueError(msg)
    expected_rows = contract.dataset_rows
    for config, child in by_config.items():
        if len(child.documents) != expected_rows[config]:
            msg = f"{config} child row count does not match contract"
            raise ValueError(msg)
        # reason: all fields jointly bind one child artifact to the release contract before any aggregation.
        if (
            child.contract_sha256 != contract.sha256  # ruff: ignore[too-many-boolean-expressions]
            or child.corpus_manifest.dataset != contract.dataset_repo
            or child.corpus_manifest.dataset_revision != contract.dataset_revision
            or child.corpus_manifest.shard != config
            or not child.corpus_manifest.historically_exposed
            or child.corpus_manifest.blinded
        ):
            msg = f"{config} child identity does not match contract"
            raise ValueError(msg)
        if child.ignore_list_sha256 != contract.ignore_list_sha256:
            msg = f"{config} child ignore-list digest does not match the contract"
            raise ValueError(msg)
        if child.corpus_manifest.document_count != len(child.documents) or any(
            document.corpus_manifest_sha256 != child.corpus_manifest.sha256 for document in child.documents
        ):
            msg = f"{config} child does not bind its corpus manifest"
            raise ValueError(msg)
        hydrated_fixture_sha256 = hydrated_fixture_sha256s.get(config)
        if hydrated_fixture_sha256 is None:
            msg = f"hydration receipt has no fixture digest for {config}"
            raise ValueError(msg)
        child.corpus_manifest.validate_persisted_fixture(
            dataset=contract.dataset_repo,
            shard=config,
            fixture_sha256=hydrated_fixture_sha256,
            source_row_sha256s=tuple(document.source_row_sha256 for document in child.documents),
        )
    identity_tuples = {(child.evaluation_contract_sha256, child.runtime_sha256) for child in children}
    if len(identity_tuples) != 1:
        msg = "child evaluation and runtime pins must be identical"
        raise ValueError(msg)

    documents = tuple(document for config in ("eval", "eval-challenge") for document in by_config[config].documents)
    document_ids = [document.document_id for document in documents]
    source_rows = [document.source_row_sha256 for document in documents]
    if len(set(document_ids)) != len(document_ids) or len(set(source_rows)) != len(source_rows):
        msg = "aggregate child rows must be disjoint"
        raise ValueError(msg)

    challenge_child = by_config["eval-challenge"]
    challenge_report = build_regex_quality_report(
        corpus_manifest=challenge_child.corpus_manifest,
        documents=challenge_child.documents,
        seed=contract.seed,
    )
    challenge_model_fp_ignored, challenge_regex_fp_ignored = _document_ignored_totals(challenge_child.documents)
    challenge = ChallengeGateEvidence(
        documents=challenge_report.documents,
        model_f1=challenge_report.model_f1,
        regex_f1=challenge_report.regex_f1,
        model_precision=challenge_report.model_precision,
        regex_precision=challenge_report.regex_precision,
        bootstrap=challenge_report.bootstrap,
        locked_negative_new_false_positives=(challenge_report.locked_negative_new_false_positives),
        sample_adequate_for_inference=challenge_report.sample_adequate_for_inference,
        quality_gate_passed=challenge_report.quality_gate_passed,
        model_fp_ignored=challenge_model_fp_ignored,
        regex_fp_ignored=challenge_regex_fp_ignored,
    )
    descriptive_pooled_bootstrap = paired_document_bootstrap(
        documents,
        replicates=contract.bootstrap_replicates,
        seed=contract.seed,
    )
    totals = _document_totals(documents)
    pooled_model_fp_ignored, pooled_regex_fp_ignored = _document_ignored_totals(documents)
    child_ignored_false_positives = tuple(
        IgnoredFalsePositives(config, *_document_ignored_totals(by_config[config].documents))
        for config in ("eval", "eval-challenge")
    )
    descriptive_leak_metrics = (
        *(
            metric
            for config in ("eval", "eval-challenge")
            for metric in _leak_metrics(config, by_config[config].documents)
        ),
        *_leak_metrics("pooled", documents),
    )
    descriptive_overredaction_metrics = (
        *(
            metric
            for config in ("eval", "eval-challenge")
            for metric in _overredaction_metrics(config, by_config[config].documents)
        ),
        *_overredaction_metrics("pooled", documents),
    )
    descriptive_pooled_by_language = tuple(
        _slice_metrics(
            language,
            tuple(document for document in documents if document.language == language),
        )
        for language in sorted({document.language for document in documents})
    )
    labels = _combine_label_counts(children)
    descriptive_pooled_by_label = tuple(_label_slice_metrics(counts) for counts in labels)
    child_hashes = tuple(by_config[config].evidence_sha256 for config in ("eval", "eval-challenge"))
    evidence_sha256 = canonical_sha256({
        "schema_version": 6,
        "contract_sha256": contract.sha256,
        "child_evidence_sha256s": list(child_hashes),
        "document_count": len(documents),
        "source_row_sha256s": sorted(source_rows),
        "challenge_gate": {
            "documents": challenge.documents,
            "evidence_sha256": challenge_report.evidence_sha256,
            "observed_delta": challenge.bootstrap.observed_delta,
            "lower_95": challenge.bootstrap.lower_95,
            "upper_95": challenge.bootstrap.upper_95,
            "replicates": challenge.bootstrap.replicates,
            "seed": challenge.bootstrap.seed,
            "quality_gate_passed": challenge.quality_gate_passed,
            "model_fp_ignored": challenge.model_fp_ignored,
            "regex_fp_ignored": challenge.regex_fp_ignored,
        },
        "descriptive_pooled_bootstrap": {
            "observed_delta": descriptive_pooled_bootstrap.observed_delta,
            "lower_95": descriptive_pooled_bootstrap.lower_95,
            "upper_95": descriptive_pooled_bootstrap.upper_95,
            "replicates": descriptive_pooled_bootstrap.replicates,
            "seed": descriptive_pooled_bootstrap.seed,
        },
        "pooled_model_fp_ignored": pooled_model_fp_ignored,
        "pooled_regex_fp_ignored": pooled_regex_fp_ignored,
        "child_ignored_false_positives": [
            {
                "config": item.config,
                "model_fp_ignored": item.model_fp_ignored,
                "regex_fp_ignored": item.regex_fp_ignored,
            }
            for item in child_ignored_false_positives
        ],
        "descriptive_leak_metrics": [
            {
                "scope": item.scope,
                "view": item.view,
                "rows_with_gold": item.rows_with_gold,
                "any_leak_rows": item.any_leak_rows,
                "any_leak_row_rate": item.any_leak_row_rate,
                "p95_row_leak_fraction": item.p95_row_leak_fraction,
                "max_row_leak_fraction": item.max_row_leak_fraction,
                "char_weighted_leak_fraction": item.char_weighted_leak_fraction,
            }
            for item in descriptive_leak_metrics
        ],
        "descriptive_overredaction_metrics": [
            {
                "scope": item.scope,
                "view": item.view,
                "rows_with_text": item.rows_with_text,
                "any_overredaction_rows": item.any_overredaction_rows,
                "any_overredaction_row_rate": item.any_overredaction_row_rate,
                "p95_row_overredaction_fraction": (item.p95_row_overredaction_fraction),
                "max_row_overredaction_fraction": (item.max_row_overredaction_fraction),
                "char_weighted_overredaction_fraction": (item.char_weighted_overredaction_fraction),
            }
            for item in descriptive_overredaction_metrics
        ],
    })
    return RegexAggregateReport(
        documents=len(documents),
        model_f1=_f1(*totals[:3]),
        regex_f1=_f1(*totals[3:]),
        model_precision=_precision(totals[0], totals[1]),
        regex_precision=_precision(totals[3], totals[4]),
        locked_negative_new_false_positives=(challenge.locked_negative_new_false_positives),
        descriptive_pooled_bootstrap=descriptive_pooled_bootstrap,
        challenge=challenge,
        descriptive_pooled_by_language=descriptive_pooled_by_language,
        descriptive_pooled_by_label=descriptive_pooled_by_label,
        model_fp_ignored=pooled_model_fp_ignored,
        regex_fp_ignored=pooled_regex_fp_ignored,
        child_ignored_false_positives=child_ignored_false_positives,
        descriptive_leak_metrics=descriptive_leak_metrics,
        descriptive_overredaction_metrics=descriptive_overredaction_metrics,
        child_evidence_sha256s=child_hashes,
        evidence_sha256=evidence_sha256,
        quality_gate_passed=challenge.quality_gate_passed,
        development_release_gate_passed=False,
        shipping_confirmation_passed=False,
    )


def _document_totals(
    documents: tuple[PairedDocumentCounts, ...],
) -> tuple[int, int, int, int, int, int]:
    return (
        sum(document.model_tp for document in documents),
        sum(document.model_fp for document in documents),
        sum(document.model_fn for document in documents),
        sum(document.regex_tp for document in documents),
        sum(document.regex_fp for document in documents),
        sum(document.regex_fn for document in documents),
    )


def _leak_metrics(scope: str, documents: tuple[PairedDocumentCounts, ...]) -> tuple[LeakMetrics, ...]:
    """Summarize per-row leak fractions for both views of one scope.

    Rows carrying no gold PII are excluded: they cannot leak, and counting them
    as clean would dilute the percentile that exists to expose the worst rows.

    Returns:
        Leak metrics for each requested scope and evaluation view.

    """
    scored = [document for document in documents if document.gold_pii_chars > 0]
    gold_chars = sum(document.gold_pii_chars for document in scored)
    metrics: list[LeakMetrics] = []
    views: tuple[tuple[str, Callable[[PairedDocumentCounts], int]], ...] = (
        (MODEL_CORE_VIEW, lambda item: item.model_leaked_chars),
        (MODEL_PLUS_REGEX_VIEW, lambda item: item.regex_leaked_chars),
    )
    for view, leaked_of in views:
        fractions = sorted(leaked_of(document) / document.gold_pii_chars for document in scored)
        leaked_chars = sum(leaked_of(document) for document in scored)
        any_leak_rows = sum(1 for fraction in fractions if fraction > 0)
        metrics.append(
            LeakMetrics(
                scope=scope,
                view=view,
                rows_with_gold=len(scored),
                any_leak_rows=any_leak_rows,
                any_leak_row_rate=any_leak_rows / len(scored) if scored else 0.0,
                p95_row_leak_fraction=percentile(fractions, 0.95) if fractions else 0.0,
                max_row_leak_fraction=fractions[-1] if fractions else 0.0,
                char_weighted_leak_fraction=(leaked_chars / gold_chars if gold_chars else 0.0),
            ),
        )
    return tuple(metrics)


def _overredaction_metrics(scope: str, documents: tuple[PairedDocumentCounts, ...]) -> tuple[OverRedactionMetrics, ...]:
    """Summarize per-row over-redaction fractions for both views of one scope.

    Mirrors `_leak_metrics`. Rows carrying no non-gold text are excluded: they
    have nothing to over-redact, and counting them as clean would dilute the
    percentile that exists to expose the worst rows. Whenever any document in
    scope predates the `document_chars` field (document_chars is None), the
    whole scope is unscoreable and this returns an empty tuple rather than a
    count computed over only the rows that happen to carry the field.

    Returns:
        Over-redaction metrics for each requested scope and evaluation view.

    """
    if any(document.document_chars is None for document in documents):
        return ()
    non_gold_totals = {
        document.document_id: document.document_chars - document.gold_pii_chars
        for document in documents
        if document.document_chars is not None
    }
    scored = [document for document in documents if non_gold_totals[document.document_id] > 0]
    non_gold_chars = sum(non_gold_totals[document.document_id] for document in scored)
    metrics: list[OverRedactionMetrics] = []
    views: tuple[tuple[str, Callable[[PairedDocumentCounts], int]], ...] = (
        (MODEL_CORE_VIEW, lambda item: item.model_overredacted_chars),
        (MODEL_PLUS_REGEX_VIEW, lambda item: item.regex_overredacted_chars),
    )
    for view, overredacted_of in views:
        fractions = sorted(overredacted_of(document) / non_gold_totals[document.document_id] for document in scored)
        overredacted_chars = sum(overredacted_of(document) for document in scored)
        any_overredaction_rows = sum(1 for fraction in fractions if fraction > 0)
        metrics.append(
            OverRedactionMetrics(
                scope=scope,
                view=view,
                rows_with_text=len(scored),
                any_overredaction_rows=any_overredaction_rows,
                any_overredaction_row_rate=(any_overredaction_rows / len(scored) if scored else 0.0),
                p95_row_overredaction_fraction=(percentile(fractions, 0.95) if fractions else 0.0),
                max_row_overredaction_fraction=fractions[-1] if fractions else 0.0,
                char_weighted_overredaction_fraction=(overredacted_chars / non_gold_chars if non_gold_chars else 0.0),
            ),
        )
    return tuple(metrics)


def _document_ignored_totals(
    documents: tuple[PairedDocumentCounts, ...],
) -> tuple[int, int]:
    return (
        sum(document.model_fp_ignored for document in documents),
        sum(document.regex_fp_ignored for document in documents),
    )


def _combine_label_counts(
    children: tuple[ChildRegexEvidence, ChildRegexEvidence],
) -> tuple[LabelMetricCounts, ...]:
    by_label: dict[str, list[int]] = {label: [0, 0, 0, 0, 0, 0] for label in PII_LABEL_SET}
    for child in children:
        for counts in child.label_counts:
            total = by_label.setdefault(counts.label, [0, 0, 0, 0, 0, 0])
            for index, value in enumerate((
                counts.model_tp,
                counts.model_fp,
                counts.model_fn,
                counts.regex_tp,
                counts.regex_fp,
                counts.regex_fn,
            )):
                total[index] += value
    return tuple(LabelMetricCounts(label, *by_label[label]) for label in sorted(by_label))


def _slice_metrics(name: str, documents: tuple[PairedDocumentCounts, ...]) -> SliceMetrics:
    totals = _document_totals(documents)
    return SliceMetrics(
        name=name,
        documents=len(documents),
        model_f1=_f1(*totals[:3]),
        regex_f1=_f1(*totals[3:]),
        model_precision=_precision(totals[0], totals[1]),
        regex_precision=_precision(totals[3], totals[4]),
        model_recall=_recall(totals[0], totals[2]),
        regex_recall=_recall(totals[3], totals[5]),
    )


def _label_slice_metrics(counts: LabelMetricCounts) -> SliceMetrics:
    return SliceMetrics(
        name=counts.label,
        documents=None,
        model_f1=_f1(counts.model_tp, counts.model_fp, counts.model_fn),
        regex_f1=_f1(counts.regex_tp, counts.regex_fp, counts.regex_fn),
        model_precision=_precision(counts.model_tp, counts.model_fp),
        regex_precision=_precision(counts.regex_tp, counts.regex_fp),
        model_recall=_recall(counts.model_tp, counts.model_fn),
        regex_recall=_recall(counts.regex_tp, counts.regex_fn),
    )


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp else 0.0


def _recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _detection_spans(detection: object) -> list[CharSpan]:
    raw_spans = getattr(detection, "spans", detection)
    if not isinstance(raw_spans, (list, tuple)) or not all(isinstance(span, CharSpan) for span in raw_spans):
        msg = "detector output must expose CharSpan values"
        raise TypeError(msg)
    return list(raw_spans)
