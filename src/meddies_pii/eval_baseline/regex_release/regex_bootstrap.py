"""Deterministic stratified paired bootstrap for the regex quality lane."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from meddies_pii.evaluation.identity import canonical_sha256, is_sha256
from meddies_pii.spans import CharSpan, char_span_to_dict

LengthBucket = Literal["short", "medium", "long"]
SHORT_DOCUMENT_CODEPOINT_LIMIT = 512
MEDIUM_DOCUMENT_CODEPOINT_LIMIT = 2_048


@dataclass(frozen=True, slots=True)
class PairedDocumentCounts:
    """Hold paired document-level evidence for bootstrap aggregation.

    False positives are already netted out of `model_fp` and `regex_fp` because
    they matched an audited ignore-list coordinate. Zero when no ignore-list
    contract is active, so existing constructors stay valid.

    Redaction-safety evidence, independent of the exact-span scoring above: how many gold-PII characters no predicted
    span of ANY label covered. One gold_pii_chars serves both views because the paired lane already refuses rows whose
    gold_spans differ, and a second copy could only ever disagree.

    Over-redaction evidence: the document's total character count, so a non-gold denominator (document_chars -
    gold_pii_chars) can be derived. None means this row predates the field (a persisted evidence artifact from before
    the schema addition on 2026-08-11); callers must treat None as "over-redaction is unknown for this row", never as
    zero. How many non-gold characters each view's predicted spans covered anyway (masking healthy prose) lives in
    model_overredacted_chars/regex_overredacted_chars, which stay 0 whenever document_chars is None.
    """

    document_id: str
    language: str
    length_bucket: LengthBucket
    source_row_sha256: str
    corpus_manifest_sha256: str
    model_tp: int
    model_fp: int
    model_fn: int
    regex_tp: int
    regex_fp: int
    regex_fn: int
    model_fp_ignored: int = 0
    regex_fp_ignored: int = 0
    gold_pii_chars: int = 0
    model_leaked_chars: int = 0
    regex_leaked_chars: int = 0
    document_chars: int | None = None
    model_overredacted_chars: int = 0
    regex_overredacted_chars: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.model_tp,
            self.model_fp,
            self.model_fn,
            self.regex_tp,
            self.regex_fp,
            self.regex_fn,
            self.model_fp_ignored,
            self.regex_fp_ignored,
            self.gold_pii_chars,
            self.model_leaked_chars,
            self.regex_leaked_chars,
            self.model_overredacted_chars,
            self.regex_overredacted_chars,
        )
        if not self.document_id or not self.language:
            msg = "document_id and canonical language are required"
            raise ValueError(msg)
        if not is_sha256(self.source_row_sha256) or not is_sha256(self.corpus_manifest_sha256):
            msg = "paired document evidence requires SHA-256 identities"
            raise ValueError(msg)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
            msg = "paired TP/FP/FN counts must be non-negative integers"
            raise ValueError(msg)
        if self.model_leaked_chars > self.gold_pii_chars or self.regex_leaked_chars > self.gold_pii_chars:
            msg = "leaked gold characters cannot exceed the gold total"
            raise ValueError(msg)
        if self.document_chars is not None:
            if (
                not isinstance(self.document_chars, int)
                or isinstance(self.document_chars, bool)
                or self.document_chars < 0
            ):
                msg = "document_chars must be a non-negative integer"
                raise ValueError(msg)
            if self.document_chars < self.gold_pii_chars:
                msg = "document_chars cannot be smaller than gold_pii_chars"
                raise ValueError(msg)
            non_gold_chars = self.document_chars - self.gold_pii_chars
            if self.model_overredacted_chars > non_gold_chars or self.regex_overredacted_chars > non_gold_chars:
                msg = "over-redacted characters cannot exceed the non-gold total"
                raise ValueError(msg)
        elif self.model_overredacted_chars or self.regex_overredacted_chars:
            msg = "over-redacted characters require a known document_chars"
            raise ValueError(msg)


def source_row_sha256(
    *,
    stable_id: str,
    text_sha256: str,
    gold_spans: tuple[CharSpan, ...] | list[CharSpan],
    language: str,
    slices: tuple[str, ...] | list[str],
) -> str:
    """Hash the persisted fixture fields that identify one evaluation row.

    Returns:
        The canonical SHA-256 digest of the persisted row identity fields.

    Raises:
        ValueError: If the row identity fields cannot form the required canonical payload.

    """
    if not stable_id or not language or not is_sha256(text_sha256):
        msg = "source row identity fields are malformed"
        raise ValueError(msg)
    return canonical_sha256({
        "schema_version": 1,
        "stable_id": stable_id,
        "text_sha256": text_sha256,
        "gold_spans": [char_span_to_dict(span) for span in gold_spans],
        "language": language,
        "slices": sorted(slices),
    })


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    observed_delta: float
    lower_95: float
    upper_95: float
    replicates: int
    seed: int
    documents: int
    strata: int


def length_bucket(codepoints: int) -> LengthBucket:
    if codepoints < 0:
        msg = "document length cannot be negative"
        raise ValueError(msg)
    if codepoints < SHORT_DOCUMENT_CODEPOINT_LIMIT:
        return "short"
    if codepoints < MEDIUM_DOCUMENT_CODEPOINT_LIMIT:
        return "medium"
    return "long"


def paired_document_bootstrap(
    documents: tuple[PairedDocumentCounts, ...],
    *,
    replicates: int = 10_000,
    seed: int,
) -> PairedBootstrapResult:
    """Resample paired documents within language-length strata.

    Returns:
        The paired bootstrap estimates, confidence intervals, and adequacy evidence.

    Raises:
        ValueError: If documents, strata, replicate count, or sample weights cannot support the paired bootstrap.

    """
    if not documents:
        msg = "paired bootstrap requires at least one document"
        raise ValueError(msg)
    if replicates <= 0:
        msg = "paired bootstrap replicates must be positive"
        raise ValueError(msg)
    identifiers = [document.document_id for document in documents]
    if len(set(identifiers)) != len(identifiers):
        msg = "paired bootstrap document_id values must be unique"
        raise ValueError(msg)

    by_stratum: dict[tuple[str, LengthBucket], list[PairedDocumentCounts]] = defaultdict(list)
    for document in documents:
        by_stratum[document.language, document.length_bucket].append(document)

    # reason: bootstrap resampling must be reproducible from the preregistered seed; it has no security purpose.
    random_source = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    deltas: list[float] = []
    for _ in range(replicates):
        sampled: list[PairedDocumentCounts] = []
        for stratum in sorted(by_stratum):
            members = by_stratum[stratum]
            sampled.extend(members[random_source.randrange(len(members))] for _ in range(len(members)))
        deltas.append(_f1_delta(sampled))
    deltas.sort()

    return PairedBootstrapResult(
        observed_delta=_f1_delta(documents),
        lower_95=percentile(deltas, 0.025),
        upper_95=percentile(deltas, 0.975),
        replicates=replicates,
        seed=seed,
        documents=len(documents),
        strata=len(by_stratum),
    )


def _f1_delta(
    documents: tuple[PairedDocumentCounts, ...] | list[PairedDocumentCounts],
) -> float:
    model = _sum_counts(documents, prefix="model")
    regex = _sum_counts(documents, prefix="regex")
    return _micro_f1(*regex) - _micro_f1(*model)


def _sum_counts(
    documents: tuple[PairedDocumentCounts, ...] | list[PairedDocumentCounts],
    *,
    prefix: Literal["model", "regex"],
) -> tuple[int, int, int]:
    if prefix == "model":
        return (
            sum(document.model_tp for document in documents),
            sum(document.model_fp for document in documents),
            sum(document.model_fn for document in documents),
        )
    return (
        sum(document.regex_tp for document in documents),
        sum(document.regex_fp for document in documents),
        sum(document.regex_fn for document in documents),
    )


def _micro_f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def percentile(sorted_values: list[float], probability: float) -> float:
    """Linear-interpolated percentile of an ascending-sorted sample.

    Returns:
        The linearly interpolated percentile value.

    """
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction
