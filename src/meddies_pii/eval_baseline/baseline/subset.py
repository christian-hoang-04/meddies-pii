from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from meddies_pii.eval_baseline.baseline.models import EvalRow


@dataclass(frozen=True, slots=True)
class StratifiedSubsetSpec:
    name: str
    target_rows: int
    strata: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PinnedSubset:
    name: str
    rows: tuple[EvalRow, ...]
    sha256: str
    stratum_counts: dict[str, int]


def build_smoke_subset(rows: list[EvalRow], *, target_rows: int = 200) -> PinnedSubset:
    return build_pinned_subset(
        rows,
        StratifiedSubsetSpec(
            name="smoke-200",
            target_rows=target_rows,
            strata=("dataset", "language"),
        ),
    )


# reason: build external combines stratum key and values; splitting would detach artifact evidence.
def build_external_stratified_subset(  # ruff: ignore[complex-structure]
    rows: list[EvalRow],
    *,
    target_spans: int = 4_000,
    max_rows_per_config: int = 250,
) -> PinnedSubset:
    if target_spans <= 0:
        msg = "target_spans must be positive"
        raise ValueError(msg)
    if max_rows_per_config <= 0:
        msg = "max_rows_per_config must be positive"
        raise ValueError(msg)

    external_rows = [row for row in rows if row.dataset == "external"]
    buckets: dict[str, list[EvalRow]] = {}
    for row in external_rows:
        buckets.setdefault(_stratum_key(row, ("dataset", "shard")), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row.stable_id)

    selected: list[EvalRow] = []
    selected_by_key: Counter[str] = Counter()
    span_total = 0
    ordered_keys = sorted(buckets)
    while span_total < target_spans:
        progressed = False
        for key in ordered_keys:
            if selected_by_key[key] >= max_rows_per_config:
                continue
            bucket = buckets[key]
            if not bucket:
                continue
            row = bucket.pop(0)
            selected.append(row)
            selected_by_key[key] += 1
            span_total += len(row.gold_spans)
            progressed = True
            if span_total >= target_spans:
                break
        if not progressed:
            break

    counts = Counter(_stratum_key(row, ("dataset", "shard")) for row in selected)
    return PinnedSubset(
        name="external-stratified",
        rows=tuple(selected),
        sha256=subset_sha256(row.stable_id for row in selected),
        stratum_counts=dict(sorted(counts.items())),
    )


def build_pinned_subset(
    rows: list[EvalRow],
    spec: StratifiedSubsetSpec,
) -> PinnedSubset:
    if spec.target_rows <= 0:
        msg = "target_rows must be positive"
        raise ValueError(msg)
    buckets: dict[str, list[EvalRow]] = {}
    for row in rows:
        buckets.setdefault(_stratum_key(row, spec.strata), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row.stable_id)

    selected: list[EvalRow] = []
    ordered_keys = sorted(buckets)
    while len(selected) < spec.target_rows:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) == spec.target_rows:
                break
        if not progressed:
            break

    counts = Counter(_stratum_key(row, spec.strata) for row in selected)
    return PinnedSubset(
        name=spec.name,
        rows=tuple(selected),
        sha256=subset_sha256(row.stable_id for row in selected),
        stratum_counts=dict(sorted(counts.items())),
    )


def subset_sha256(stable_ids: Iterable[str]) -> str:
    payload = {"ids": list(stable_ids), "version": 1}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_subset_manifest(path: str | Path, subset: PinnedSubset) -> None:
    payload = {
        "name": subset.name,
        "sha256": subset.sha256,
        "rows": [row.stable_id for row in subset.rows],
        "stratum_counts": subset.stratum_counts,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _stratum_key(row: EvalRow, strata: tuple[str, ...]) -> str:
    parts: list[str] = []
    for name in strata:
        value = getattr(row, name)
        parts.append(f"{name}={value}")
    return "|".join(parts)
