"""Pre-GPU quality+diversity assembly gate (ADR 0008 §2 + §3).

The gate is a pure function over PII-label records. It decides whether an
assembled corpus may proceed to training by checking three things that a raw
row count cannot see on its own:

1. **Per-label span floors (§2).** Each label must carry at least its floor of
   spans. A starved label hard-blocks assembly — Ha's deliberate choice over a
   warn-and-proceed flag, because a starved label silently degrades the
   retrained model.
2. **The language x label coverage matrix (§3).** Entropy hides zero cells: a
   distribution can look balanced on each axis while a specific
   (language, label) pair has no documents at all. The gate walks every cell of
   the 17-language x 9-label grid and blocks on any empty (`zero_cells`) or
   under-filled (`sparse_cells`) cell — unless the cell carries a recorded
   exhaustion waiver (§3 escape valve).
3. **Shannon entropy over the language and label document distributions (§3).**
   A distributional squeeze (one language or label dominating) blocks ship even
   when every cell is non-empty.

The grid axes are derived from the codebase, never hardcoded: the 9 labels come
from :data:`PII_LABELS` and the 17 languages from the shared language
registry. Row languages are normalized onto those canonical codes via
:func:`is_supported_meddies_language` (membership gate) plus
:func:`normalize_language` (alias -> canonical code); a row whose language is not
in the supported set is excluded from the grid rather than spawning a phantom
cell.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import cast

from meddies_pii.languages import (
    GRID_LANGUAGE_CODES,
    is_supported_meddies_language,
    normalize_language,
)
from meddies_pii.taxonomy import PII_LABELS

GRID_LANGUAGES: tuple[str, ...] = tuple(GRID_LANGUAGE_CODES)
""": Canonical 17-language grid axis — the ISO code of each generation profile."""

GRID_LABELS: tuple[str, ...] = PII_LABELS

Cell = tuple[str, str]

_GRID_LABEL_SET: frozenset[str] = frozenset(GRID_LABELS)


@dataclass(frozen=True)
class GateReport:
    """The verdict of one assembly-gate evaluation.

    ``passed`` is the AND of every gate: no floor shortfall, no non-waived
    zero/sparse cell, and every entropy axis at or above its target. The other
    fields are the evidence behind the verdict — each blocking detail also gets
    a human-readable line in ``blocking_reasons``.
    """

    passed: bool
    label_floor_failures: dict[str, int]
    zero_cells: list[Cell]
    sparse_cells: list[Cell]
    waived_cells: list[Cell]
    entropy: dict[str, float]
    entropy_failures: dict[str, float]
    blocking_reasons: list[str] = field(default_factory=list)


def _get(value: object, key: str) -> object:
    """Read ``key`` from ``value`` when it is a mapping, else return ``None``.

    Taking ``object`` (not a narrowed ``Mapping``) avoids the ``ty`` quirk where
    ``isinstance(x, Mapping)`` over an ``object`` narrows the value to
    ``Mapping[Never, Never]`` and then rejects a literal-string ``.get`` key.

    Returns:
        The value stored under ``key``, or ``None`` when the container is not a mapping or has no such key.

    """
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value).get(key)
    return None


def _resolve_language(raw_language: str) -> str | None:
    """Map a row's raw language onto a canonical grid code, or ``None``.

    Returns ``None`` when the language is outside Meddies' supported set so the
    caller can exclude the row from the grid instead of inventing a cell.

    Returns:
        The canonical grid code for the row's language, or ``None`` when that language is outside the supported set.

    """
    if not is_supported_meddies_language(raw_language):
        return None
    return normalize_language(raw_language).code


def _row_language(row: Mapping[str, object]) -> str | None:
    raw = _get(row.get("info"), "language")
    if not isinstance(raw, str):
        return None
    return _resolve_language(raw)


def _row_labels(row: Mapping[str, object]) -> list[str]:
    """Extract canonical labels from a row's spans.

    Accepts both span-list keys (``label`` and ``spans``) and both per-span
    label keys (``category`` and ``label``) — the two record shapes that flow
    through the PII-label pipeline. Non-grid labels are dropped silently; the
    grid only measures the 9 labels it gates on.

    Returns:
        The canonical labels found on the row's spans, or an empty list when the row carries none.

    """
    spans = row.get("label")
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        spans = row.get("spans")
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        return []

    labels: list[str] = []
    for span in spans:
        raw_label = _get(span, "category")
        if not isinstance(raw_label, str):
            raw_label = _get(span, "label")
        if not isinstance(raw_label, str):
            continue
        label = raw_label.strip().lower()
        if label in _GRID_LABEL_SET:
            labels.append(label)
    return labels


def _shannon_entropy_bits(counts: Iterable[int]) -> float:
    """Shannon entropy in bits of a count distribution.

    Empty or single-outcome distributions have zero entropy. A uniform
    distribution over ``k`` outcomes reaches the maximum ``log2(k)``.

    Returns:
        The distribution's entropy in bits, which is zero for an empty or single-outcome distribution.

    """
    weights = [count for count in counts if count > 0]
    total = sum(weights)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for weight in weights:
        probability = weight / total
        entropy -= probability * math.log2(probability)
    return entropy


def evaluate_corpus(
    rows: Iterable[Mapping[str, object]],
    *,
    label_floors: Mapping[str, int],
    cell_min: int,
    entropy_targets: dict[str, float],
    waivers: AbstractSet[Cell] = frozenset(),
) -> GateReport:
    """Score an assembled Meddies Labels corpus against the §2/§3 assembly gate.

    Args:
        rows: Meddies Labels records (``{"info": {"language": ...}, "label"/"spans",
            "text"}``). Consumed once; rows whose language is outside the
            supported set are excluded from the grid.
        label_floors: Minimum span count per label. A label below its floor is a
            blocking shortfall (floor minus observed) in ``label_floor_failures``.
        cell_min: Minimum document count for a (language, label) cell. A cell
            with ``0`` docs is a zero cell; ``0 < docs < cell_min`` is sparse.
        entropy_targets: Minimum Shannon entropy (bits) per axis. Keys
            ``"language"`` and ``"label"``; a missing key means no floor on that
            axis.
        waivers: (language, label) cells with a recorded exhaustion waiver.
            A waived cell is excluded from the zero/sparse blocking sets and
            listed in ``waived_cells``.

    Returns:
        A :class:`GateReport`. ``passed`` is ``True`` only when there are no
        floor shortfalls, no non-waived zero/sparse cells, and every entropy
        axis meets its target.

    """
    return evaluate_counts(
        count_corpus(rows),
        label_floors=label_floors,
        cell_min=cell_min,
        entropy_targets=entropy_targets,
        waivers=waivers,
    )


@dataclass(frozen=True)
class CorpusCounts:
    """Pre-aggregated gate counts.

    Summable, so per-source counts merge by addition — the basis for an
    incremental, per-source-parallel gate.
    """

    label_span_counts: Counter[str] = field(default_factory=Counter)
    cell_doc_counts: Counter[Cell] = field(default_factory=Counter)
    language_doc_counts: Counter[str] = field(default_factory=Counter)
    label_doc_counts: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, object]:
        return {
            "label_span_counts": dict(self.label_span_counts),
            "cell_doc_counts": {f"{lang}|{lab}": n for (lang, lab), n in self.cell_doc_counts.items()},
            "language_doc_counts": dict(self.language_doc_counts),
            "label_doc_counts": dict(self.label_doc_counts),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> CorpusCounts:
        cells: Counter[Cell] = Counter()
        raw_cells = data.get("cell_doc_counts") or {}
        if isinstance(raw_cells, Mapping):
            for key, n in raw_cells.items():
                lang, _, lab = str(key).partition("|")
                cells[lang, lab] = int(cast("int", n))

        def _counter(key: str) -> Counter[str]:
            raw = data.get(key) or {}
            return (
                Counter({str(k): int(cast("int", v)) for k, v in raw.items()}) if isinstance(raw, Mapping) else Counter()
            )

        return cls(
            _counter("label_span_counts"),
            cells,
            _counter("language_doc_counts"),
            _counter("label_doc_counts"),
        )


def count_corpus(rows: Iterable[Mapping[str, object]]) -> CorpusCounts:
    """Single pass over one source's rows -> CorpusCounts (the expensive step).

    Returns:
        The counts for one source, gathered in a single pass over its rows.

    """
    counts = CorpusCounts()
    for row in rows:
        language = _row_language(row)
        if language is None:
            continue
        counts.language_doc_counts[language] += 1
        row_labels = _row_labels(row)
        for label in row_labels:
            counts.label_span_counts[label] += 1
        for label in set(row_labels):
            counts.cell_doc_counts[language, label] += 1
            counts.label_doc_counts[label] += 1
    return counts


def merge_counts(parts: Iterable[CorpusCounts]) -> CorpusCounts:
    """Sum per-source counts (Counter addition) — the incremental merge.

    Returns:
        The per-source counts summed together, which is what makes the gate incremental.

    """
    merged = CorpusCounts()
    for part in parts:
        merged.label_span_counts.update(part.label_span_counts)
        merged.cell_doc_counts.update(part.cell_doc_counts)
        merged.language_doc_counts.update(part.language_doc_counts)
        merged.label_doc_counts.update(part.label_doc_counts)
    return merged


def evaluate_counts(
    counts: CorpusCounts,
    *,
    label_floors: Mapping[str, int],
    cell_min: int,
    entropy_targets: dict[str, float],
    waivers: AbstractSet[Cell] = frozenset(),
) -> GateReport:
    """Score pre-aggregated CorpusCounts (no row iteration) -> GateReport.

    Returns:
        The gate verdict, naming which label floors, cells, and entropy targets failed and which were waived.

    """
    label_floor_failures = _label_floor_failures(label_floors, counts.label_span_counts)
    zero_cells, sparse_cells, waived_cells = _coverage_cells(counts.cell_doc_counts, cell_min=cell_min, waivers=waivers)
    entropy = {
        "language": _shannon_entropy_bits(counts.language_doc_counts[code] for code in GRID_LANGUAGES),
        "label": _shannon_entropy_bits(counts.label_doc_counts[label] for label in GRID_LABELS),
    }
    entropy_failures = {
        axis: entropy[axis] for axis, target in entropy_targets.items() if axis in entropy and entropy[axis] < target
    }
    blocking_reasons = _blocking_reasons(
        label_floor_failures=label_floor_failures,
        zero_cells=zero_cells,
        sparse_cells=sparse_cells,
        entropy_failures=entropy_failures,
        entropy_targets=entropy_targets,
    )
    return GateReport(
        passed=not blocking_reasons,
        label_floor_failures=label_floor_failures,
        zero_cells=zero_cells,
        sparse_cells=sparse_cells,
        waived_cells=waived_cells,
        entropy=entropy,
        entropy_failures=entropy_failures,
        blocking_reasons=blocking_reasons,
    )


def _label_floor_failures(label_floors: Mapping[str, int], label_span_counts: Mapping[str, int]) -> dict[str, int]:
    failures: dict[str, int] = {}
    for label, floor in label_floors.items():
        shortfall = floor - label_span_counts.get(label, 0)
        if shortfall > 0:
            failures[label] = shortfall
    return failures


def _coverage_cells(
    cell_doc_counts: Mapping[Cell, int],
    *,
    cell_min: int,
    waivers: AbstractSet[Cell],
) -> tuple[list[Cell], list[Cell], list[Cell]]:
    zero_cells: list[Cell] = []
    sparse_cells: list[Cell] = []
    waived_cells: list[Cell] = []
    for language in GRID_LANGUAGES:
        for label in GRID_LABELS:
            cell = (language, label)
            docs = cell_doc_counts.get(cell, 0)
            if docs >= cell_min:
                continue
            if cell in waivers:
                waived_cells.append(cell)
            elif docs == 0:
                zero_cells.append(cell)
            else:
                sparse_cells.append(cell)
    return zero_cells, sparse_cells, waived_cells


def _blocking_reasons(
    *,
    label_floor_failures: Mapping[str, int],
    zero_cells: Sequence[Cell],
    sparse_cells: Sequence[Cell],
    entropy_failures: Mapping[str, float],
    entropy_targets: Mapping[str, float],
) -> list[str]:
    reasons: list[str] = [
        f"label floor: {label} short by {label_floor_failures[label]} spans" for label in sorted(label_floor_failures)
    ]
    for language, label in zero_cells:
        reasons.append(f"zero cell: ({language}, {label}) has 0 docs")
    for language, label in sparse_cells:
        reasons.append(f"sparse cell: ({language}, {label}) below cell_min")
    reasons.extend(
        f"entropy: {axis} {entropy_failures[axis]:.4f} < target {entropy_targets[axis]:.4f}"
        for axis in sorted(entropy_failures)
    )
    return reasons
