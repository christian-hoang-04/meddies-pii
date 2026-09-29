from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: adapters load their model stack inside `load()`, so listing an adapter costs nothing.
# ruff: file-ignore[useless-import-alias]
# reason: an explicit `X as X` re-export, which is this module's published surface: ruff's own
# reason: unsafe fix DELETED five of these once and broke every caller, so the alias stays.
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from meddies_pii.eval_baseline.baseline.models import EvalRow as EvalRow
from meddies_pii.eval_baseline.baseline.subset import (
    PinnedSubset,
    build_external_stratified_subset,
    build_smoke_subset,
)
from meddies_pii.languages import is_supported_meddies_language, normalize_language
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import is_pii_label

V2_REPO_ID = "Meddies/meddies-pii-v2"
EXTERNAL_REPO_ID = "Meddies/meddies-pii-external"
V2_DATASET_REVISION = "28aaef5dffd36aabead650c74658a6f814eb4db0"
EXTERNAL_DATASET_REVISION = "02ab5af55607a77267bc575377f6a20c9629af68"
V2_EVAL_CONFIGS = ("eval", "eval-challenge")
AI4PRIVACY_EXTERNAL_CONFIGS = (
    "ai4privacy_de",
    "ai4privacy_en",
    "ai4privacy_es",
    "ai4privacy_fil",
    "ai4privacy_fr",
    "ai4privacy_id",
    "ai4privacy_ja",
    "ai4privacy_ko",
    "ai4privacy_ms",
    "ai4privacy_pt",
    "ai4privacy_vi",
    "ai4privacy_zh",
)
EXTERNAL_CONFIGS = (
    *AI4PRIVACY_EXTERNAL_CONFIGS,
    "creddata_en",
    "gretel_en",
    "nemotron_en",
)
EVAL_DATASETS = ("v2-eval", "v2-eval-challenge", *EXTERNAL_CONFIGS)
EVAL_EXPECTED_ROWS = {
    "v2-eval": 1_700,
    "v2-eval-challenge": 3_400,
    "ai4privacy_de": 23_616,
    "ai4privacy_en": 40_580,
    "ai4privacy_es": 17_877,
    "ai4privacy_fil": 5_087,
    "ai4privacy_fr": 25_690,
    "ai4privacy_id": 5_363,
    "ai4privacy_ja": 5_027,
    "ai4privacy_ko": 5_230,
    "ai4privacy_ms": 5_089,
    "ai4privacy_pt": 8_665,
    "ai4privacy_vi": 5_195,
    "ai4privacy_zh": 4_971,
    "creddata_en": 1_403,
    "gretel_en": 5_000,
    "nemotron_en": 99_892,
}
EVAL_FIXTURE_SHA256 = "1ccd86a42833d45ce20c9d222e8891c6cd334e07ca774ba5ab64debfd5378cb5"


def select_eval_datasets(datasets: str | None) -> tuple[str, ...]:
    """Parse a runner's optional comma-list, preserving order and rejecting duplicates.

    Returns:
        The selected dataset names in the order given, or the full canonical tuple when the
        argument is absent or blank.

    Raises:
        ValueError: If a name repeats, or if any name is not one of the canonical datasets.
            Both are reported with the offending names rather than a bare rejection.

    """
    selected = tuple(part.strip() for part in datasets.split(",") if part.strip()) if datasets else EVAL_DATASETS
    duplicates = sorted({dataset for dataset in selected if selected.count(dataset) > 1})
    if duplicates:
        msg = f"duplicate datasets: {duplicates}"
        raise ValueError(msg)
    unknown = [dataset for dataset in selected if dataset not in EVAL_DATASETS]
    if unknown:
        msg = f"unknown datasets: {unknown}"
        raise ValueError(msg)
    return selected


def row_from_record(
    record: Mapping[str, Any],
    *,
    dataset: str,
    shard: str | None = None,
    index: int,
) -> EvalRow:
    text = _record_text(record)
    info = _record_info(record)
    raw_language = str(info.get("language") or record.get("language") or "unknown")
    language = canonical_language(raw_language)
    resolved_shard = shard if shard is not None else language
    doc_id = str(
        info.get("id")
        or info.get("uid")
        or record.get("uid")
        or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        or f"row-{index}",
    )
    spans = tuple(span for raw_span in _raw_span_rows(record) if (span := _parse_span(text, raw_span)) is not None)
    return EvalRow(
        doc_id=doc_id,
        dataset=dataset,
        shard=resolved_shard,
        text=text,
        gold_spans=spans,
        language=language,
        slices=_slices(info),
    )


# reason: load rows exposes repo id/revision as its public contract; bundling would break callers.
def load_rows(  # ruff: ignore[too-many-arguments]
    repo_id: str,
    config: str,
    *,
    split: str,
    dataset: str,
    shard: str | None = None,
    limit: int | None = None,
    revision: str | None = None,
) -> list[EvalRow]:
    from datasets import load_dataset

    split_expr = f"{split}[:{limit}]" if limit is not None else split
    if revision is None:
        table = load_dataset(repo_id, config, split=split_expr)
    else:
        table = load_dataset(repo_id, config, split=split_expr, revision=revision)
    rows: list[EvalRow] = []
    for index, row in enumerate(table):
        if isinstance(row, Mapping):
            rows.append(row_from_record(dict(row), dataset=dataset, shard=shard, index=index))
    return rows


def load_v2_eval_rows(
    config: str,
    *,
    limit: int | None = None,
    revision: str = V2_DATASET_REVISION,
) -> list[EvalRow]:
    if config not in V2_EVAL_CONFIGS:
        msg = f"unsupported v2 eval config: {config!r}"
        raise ValueError(msg)
    dataset = "v2-eval" if config == "eval" else "v2-eval-challenge"
    return load_rows(
        V2_REPO_ID,
        config,
        split="train",
        dataset=dataset,
        limit=limit,
        revision=revision,
    )


def load_external_rows(
    config: str,
    *,
    limit: int | None = None,
    revision: str = EXTERNAL_DATASET_REVISION,
) -> list[EvalRow]:
    if config not in EXTERNAL_CONFIGS:
        msg = f"unsupported external config: {config!r}"
        raise ValueError(msg)
    return load_rows(
        EXTERNAL_REPO_ID,
        config,
        split="eval",
        dataset="external",
        shard=config,
        limit=limit,
        revision=revision,
    )


def load_eval_cell(
    dataset: str,
    *,
    v2_limit: int | None,
    external_limit: int | None,
    v2_revision: str = V2_DATASET_REVISION,
    external_revision: str = EXTERNAL_DATASET_REVISION,
) -> list[EvalRow]:
    """Load one canonical matrix cell with limits chosen by its runner.

    Returns:
        The rows for that cell. The two v2 cells take the v2 limit and revision; an external
        config takes the external pair, so one call site can load either without branching.

    Raises:
        ValueError: If the dataset is neither a v2 cell nor a known external config.

    """
    if dataset == "v2-eval":
        return load_v2_eval_rows("eval", limit=v2_limit, revision=v2_revision)
    if dataset == "v2-eval-challenge":
        return load_v2_eval_rows(
            "eval-challenge",
            limit=v2_limit,
            revision=v2_revision,
        )
    if dataset in EXTERNAL_CONFIGS:
        return load_external_rows(
            dataset,
            limit=external_limit,
            revision=external_revision,
        )
    msg = f"unknown dataset: {dataset!r}"
    raise ValueError(msg)


def load_all_external_rows(*, limit_per_config: int | None = None) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for config in EXTERNAL_CONFIGS:
        rows.extend(load_external_rows(config, limit=limit_per_config))
    return rows


def load_external_stratified_subset(
    *,
    target_spans: int = 4_000,
    max_rows_per_config: int = 250,
    limit_per_config: int | None = None,
) -> PinnedSubset:
    return build_external_stratified_subset(
        load_all_external_rows(limit_per_config=limit_per_config),
        target_spans=target_spans,
        max_rows_per_config=max_rows_per_config,
    )


def load_first_light_rows(*, limit: int = 32, per_source_limit: int = 8) -> list[EvalRow]:
    return list(
        load_first_light_subset(
            limit=limit,
            per_source_limit=per_source_limit,
        ).rows,
    )


def load_first_light_subset(*, limit: int = 32, per_source_limit: int = 8) -> PinnedSubset:
    rows: list[EvalRow] = []
    rows.extend(load_v2_eval_rows("eval", limit=per_source_limit))
    rows.extend(load_v2_eval_rows("eval-challenge", limit=per_source_limit))
    for config in EXTERNAL_CONFIGS[: max(1, min(len(EXTERNAL_CONFIGS), 4))]:
        rows.extend(load_external_rows(config, limit=per_source_limit))
        if len(rows) >= limit * 2:
            break
    return build_smoke_subset(rows, target_rows=limit)


def _record_text(record: Mapping[str, Any]) -> str:
    text = record.get("text") or record.get("raw")
    if not isinstance(text, str) or not text:
        msg = "eval row is missing non-empty text/raw"
        raise ValueError(msg)
    return text


def _record_info(record: Mapping[str, Any]) -> Mapping[str, Any]:
    info = record.get("info")
    return info if isinstance(info, Mapping) else {}


def _raw_span_rows(record: Mapping[str, Any]) -> Sequence[object]:
    spans = record.get("label") or record.get("spans") or ()
    return spans if isinstance(spans, Sequence) and not isinstance(spans, str) else ()


def _parse_span(text: str, raw_span: object) -> CharSpan | None:
    """Degenerate/out-of-bounds gold span (e.g.

    zero-width start==end) — drop rather than crash the eval; it's unmatchable anyway. Magnitude shows up in the
    aggregate's gold-span counts if it's ever systematic.

    Misaligned gold span (recorded text != text[start:end]) — drop, same reason.

    Returns:
        The parsed span, or None for any of the drop cases above: a non-mapping, a missing or
        non-PII label, non-integer or out-of-bounds offsets, and a recorded surface that
        disagrees with the text the offsets select.

    """
    if not isinstance(raw_span, Mapping):
        return None
    label = raw_span.get("category") or raw_span.get("label")
    if not isinstance(label, str) or not is_pii_label(label):
        return None
    start = raw_span.get("start")
    end = raw_span.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end <= start or end > len(text):
        return None
    surface = text[start:end]
    raw_text = raw_span.get("text")
    if isinstance(raw_text, str) and raw_text and raw_text != surface:
        return None
    return CharSpan(start=start, end=end, text=surface, label=label)


def canonical_language(language: str) -> str:
    """Canonical ISO code for any language alias the eval datasets emit.

    v2 emits full names ("English"), external emits ISO codes ("en"), and some
    rows leak locales ("us" for nemotron, "tl"/"fil" for Filipino). All collapse
    to one code via the shared `normalize_language` alias table, so per-language
    rollups don't fragment into parallel buckets. Non-17 languages -> "unknown".

    Returns:
        The canonical ISO code for a supported language, else ``"unknown"``.

    """
    if is_supported_meddies_language(language):
        return normalize_language(language).code
    return "unknown"


def _slices(info: Mapping[str, Any]) -> frozenset[str]:
    values: set[str] = set()
    for key in (
        "source_dataset",
        "source",
        "domain_bucket",
        "domain_profile",
        "document_type",
        "text_format",
        "scenario",
        "split_purpose",
    ):
        value = info.get(key)
        if isinstance(value, str) and value:
            values.add(f"{key}={value}")
    raw_edge_cases = info.get("edge_cases")
    if isinstance(raw_edge_cases, Sequence) and not isinstance(raw_edge_cases, str):
        values.update(f"edge_case={item}" for item in raw_edge_cases if item)
    return frozenset(values)
