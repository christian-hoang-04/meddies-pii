"""External source dispatch and deterministic local-first loading for mixed BIOES data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, cast

from datasets import load_dataset

from meddies_pii.training.bioes.data.external_records import read_jsonl_records
from meddies_pii.training.bioes.data.mixed import (
    convert_ai4privacy_row,
    convert_gretel_row,
    convert_nemotron_row,
    is_supported_meddies_language,
)
from meddies_pii.training.bioes.data.record_schema import NormalizedRecord, Record, record_from_object

if TYPE_CHECKING:
    from pathlib import Path

EXTERNAL_DATASETS: tuple[str, ...] = (
    "nvidia/Nemotron-PII",
    "ai4privacy/pii-masking-openpii-1m",
    "ai4privacy/open-pii-masking-500k-ai4privacy",
    "gretelai/gretel-pii-masking-en-v1",
)

_LOCAL_STEM_PREFIX_TO_DATASET: tuple[tuple[str, str], ...] = (
    ("nvidia-health", "nvidia/Nemotron-PII"),
    ("nemotron", "nvidia/Nemotron-PII"),
    ("gretel", "gretelai/gretel-pii-masking-en-v1"),
    ("ai4privacy-1m", "ai4privacy/pii-masking-openpii-1m"),
    ("ai4privacy-500k", "ai4privacy/open-pii-masking-500k-ai4privacy"),
    ("ai4privacy", "ai4privacy/open-pii-masking-500k-ai4privacy"),
)


def _iter_external(dataset_id: str, split: str) -> Iterator[Record]:
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    if isinstance(dataset, (str, bytes)) or not isinstance(dataset, Iterable):
        msg = f"{dataset_id} {split!r} split must be iterable"
        raise TypeError(msg)
    for row_index, row in enumerate(dataset):
        yield record_from_object(row, source=f"{dataset_id}:{split}[{row_index}]")


def convert_external(
    dataset_id: str,
    row: Record,
    row_index: int,
) -> tuple[NormalizedRecord | None, Counter[str]]:
    default_uid = f"{dataset_id}:{row_index}"
    if dataset_id == "nvidia/Nemotron-PII":
        return convert_nemotron_row(row, dataset_id=dataset_id, default_uid=default_uid)
    if dataset_id == "gretelai/gretel-pii-masking-en-v1":
        return convert_gretel_row(row, dataset_id=dataset_id, default_uid=default_uid)
    return convert_ai4privacy_row(row, dataset_id=dataset_id, default_uid=default_uid)


def dataset_id_for_stem(stem: str) -> str | None:
    for prefix, dataset_id in _LOCAL_STEM_PREFIX_TO_DATASET:
        if stem == prefix or stem.startswith(f"{prefix}-"):
            return dataset_id
    return None


def load_local_external_rows(
    external_dir: Path,
) -> tuple[list[dict[str, object]], Counter[str]]:
    """Read sorted local external JSONL files through their canonical converter.

    Returns:
        The converted rows and a counter of what was dropped, keyed by dataset and reason. A
        file whose stem maps to no known dataset is counted under ``unknown_source`` and
        skipped rather than guessed at, so an unrecognized drop shows up in the manifest
        instead of silently shrinking the corpus.

    """
    rows: list[dict[str, object]] = []
    dropped: Counter[str] = Counter()
    for path in sorted(external_dir.glob("*.jsonl")):
        dataset_id = dataset_id_for_stem(path.stem)
        if dataset_id is None:
            dropped[f"unknown_source:{path.stem}"] += 1
            continue
        for row_index, record in enumerate(read_jsonl_records(path)):
            converted, row_dropped = convert_external(dataset_id, dict(record), row_index)
            dropped.update({f"{dataset_id}:{key}": value for key, value in row_dropped.items()})
            if converted is not None:
                # reason: a TypedDict is a dict at runtime; no consumer mutates these rows destructively.
                rows.append(cast("dict[str, object]", converted))
    return rows, dropped


def target_external_count(medical_count: int, medical_ratio: float) -> int:
    if not 0 < medical_ratio < 1:
        msg = "--medical-ratio must be between 0 and 1"
        raise ValueError(msg)
    return round(medical_count * (1 - medical_ratio) / medical_ratio)


def _external_quotas(target_external: int) -> dict[str, int]:
    base = target_external // len(EXTERNAL_DATASETS)
    quotas = dict.fromkeys(EXTERNAL_DATASETS, base)
    for dataset_id in EXTERNAL_DATASETS[: target_external - base * len(EXTERNAL_DATASETS)]:
        quotas[dataset_id] += 1
    return quotas


def load_remote_external_rows(
    *,
    target_external: int,
    split: str,
    max_scan_per_dataset: int,
    supported_languages_only: bool,
) -> tuple[list[dict[str, object]], Counter[str]]:
    rows: list[dict[str, object]] = []
    dropped: Counter[str] = Counter()
    for dataset_id, quota in _external_quotas(target_external).items():
        accepted = 0
        for scanned, row in enumerate(_iter_external(dataset_id, split), start=1):
            if supported_languages_only:
                language = str(row.get("locale") or row.get("language") or "UNKNOWN")
                if not is_supported_meddies_language(language):
                    dropped[f"{dataset_id}:unsupported_language:{language.lower()}"] += 1
                    if scanned >= max_scan_per_dataset:
                        break
                    continue
            converted, row_dropped = convert_external(dataset_id, dict(row), scanned - 1)
            dropped.update({f"{dataset_id}:{key}": value for key, value in row_dropped.items()})
            if converted is not None:
                # reason: a TypedDict is a dict at runtime; no consumer mutates these rows destructively.
                rows.append(cast("dict[str, object]", converted))
                accepted += 1
                if accepted >= quota:
                    break
            if scanned >= max_scan_per_dataset:
                break
        if accepted < quota:
            dropped[f"{dataset_id}:row_shortfall"] += quota - accepted
    return rows, dropped
