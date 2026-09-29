"""Assemble one 17-cell PII350 aggregate from per-cell score summaries.

Cells for a single checkpoint can be paid for in different workspaces and under
different evaluator generations. This module joins those `cell_score_summary`
payloads offline. It recomputes every micro metric from raw span counts, so a
cell contributes in proportion to the spans it actually scored; averaging the
per-cell F1 values the aggregate already rounded would reweight the matrix by
cell count instead.

Mixed-generation assembly is explicitly authorized, so the per-cell evaluator
generation is recorded in `provenance` rather than collapsed into one contract.

``scripts/ops/assemble_pii350_checkpoint_aggregate.py`` is the command-line
wrapper over this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

from meddies_pii.eval_baseline.baseline.datasets import EVAL_DATASETS, EVAL_EXPECTED_ROWS
from meddies_pii.evaluation.identity import canonical_sha256, is_sha256
from meddies_pii.evaluation.span_metrics import (
    ContainmentSpanMetricBlock,
    ExactSpanMetricBlock,
    SpanMetricBlock,
    containment_span_prf_from_counts,
    exact_span_prf_from_counts,
)
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from pathlib import Path


class LabelMetricPair(TypedDict):
    exact: ExactSpanMetricBlock
    containment: ContainmentSpanMetricBlock


ASSEMBLED_SCHEMA_VERSION = 1
CELL_SCORE_SUMMARY_SCHEMA_VERSION = 1
EXPECTED_TOTAL_ROWS = sum(EVAL_EXPECTED_ROWS[dataset] for dataset in EVAL_DATASETS)
EXACT_COUNT_KEYS = ("tp", "pred_total", "gold_total")
CONTAINMENT_COUNT_KEYS = ("precision_tp", "recall_tp", "pred_total", "gold_total")
COUNT_BLOCKS = {
    "exact": EXACT_COUNT_KEYS,
    "containment": CONTAINMENT_COUNT_KEYS,
    "full9_exact": EXACT_COUNT_KEYS,
    "full9_containment": CONTAINMENT_COUNT_KEYS,
}
ABSENT_FIELDS = {
    "contract": "a fill/partial run contract covers only its own cells",
    "cost_ledger": "cost is per source run, not per assembled matrix",
    "estimated_all_in_cost_usd": "cost is per source run",
    "evaluation_contract": "cells span more than one evaluator generation; see provenance",
    "fixture.sha256": "the canonical shard-identity hash needs the fixture payloads, not their digests",
    "modal_app_id": "one assembly has no single Modal app",
    "peak_vram_bytes": "measured per source run",
    "throughput": "measured per source run",
    "truncated_documents": "measured per source run",
}
"""Fields a native single-run aggregate carries that a mixed-generation assembly cannot honestly produce.

They are named in the output rather than faked.

"""


def _empty_counts(keys: Sequence[str]) -> dict[str, int]:
    return dict.fromkeys(keys, 0)


def _require_counts(value: object, keys: Sequence[str], *, role: str) -> dict[str, int]:
    if not isinstance(value, dict):
        msg = f"{role} count block is missing or has unexpected keys"
        raise SystemExit(msg)
    counts: dict[str, int] = {}
    for key, number in value.items():
        if not isinstance(key, str) or type(number) is not int or number < 0:
            msg = f"{role} count {key} must be a nonnegative integer"
            raise SystemExit(msg)
        counts[key] = number
    if set(counts) != set(keys):
        msg = f"{role} count block is missing or has unexpected keys"
        raise SystemExit(msg)
    return counts


def _add(total: dict[str, int], observed: Mapping[str, int]) -> None:
    for key, value in observed.items():
        total[key] += int(value)


def _summary_from_counts(
    exact: Mapping[str, int],
    containment: Mapping[str, int],
    full9_exact: Mapping[str, int],
    full9_containment: Mapping[str, int],
    *,
    rows: int,
) -> dict[str, int | float]:
    """Mirror the shape aggregate_results emits so both reports compare directly.

    Returns:
        Row and span totals plus exact and containment precision, recall and F1,
        each rounded to four places, and the same six figures again over the
        full-9 label set. The key names match ``aggregate_results`` exactly, which
        is what lets a checkpoint report and a baseline report be read side by
        side without a translation step.

    """
    exact_metrics = exact_span_prf_from_counts(exact)
    containment_metrics = containment_span_prf_from_counts(containment)
    full9_exact_metrics = exact_span_prf_from_counts(full9_exact)
    full9_containment_metrics = containment_span_prf_from_counts(full9_containment)
    return {
        "rows": rows,
        "gold_spans": int(exact_metrics["gold_total"]),
        "pred_spans": int(exact_metrics["pred_total"]),
        "exact_f1": round(float(exact_metrics["f1"]), 4),
        "exact_precision": round(float(exact_metrics["precision"]), 4),
        "exact_recall": round(float(exact_metrics["recall"]), 4),
        "containment_f1": round(float(containment_metrics["f1"]), 4),
        "containment_precision": round(float(containment_metrics["precision"]), 4),
        "containment_recall": round(float(containment_metrics["recall"]), 4),
        "full9_gold_spans": int(full9_exact_metrics["gold_total"]),
        "full9_pred_spans": int(full9_exact_metrics["pred_total"]),
        "full9_exact_f1": round(float(full9_exact_metrics["f1"]), 4),
        "full9_exact_precision": round(float(full9_exact_metrics["precision"]), 4),
        "full9_exact_recall": round(float(full9_exact_metrics["recall"]), 4),
        "full9_containment_f1": round(float(full9_containment_metrics["f1"]), 4),
        "full9_containment_precision": round(float(full9_containment_metrics["precision"]), 4),
        "full9_containment_recall": round(float(full9_containment_metrics["recall"]), 4),
    }


def _macro_f1(by_label: Mapping[str, SpanMetricBlock]) -> float:
    values = [float(metrics["f1"]) for metrics in by_label.values()]
    return round(sum(values) / len(values), 4) if values else 0.0


def load_summary(path: Path) -> dict[str, Any]:
    """Read one cell_score_summary payload and prove it describes itself.

    Returns:
        The payload, once its own ``summary_digest`` has been recomputed over the
        rest of the body and matched.

    Raises:
        SystemExit: when the file is unreadable, is not an object, carries a
            field set other than the exact required one, states an unsupported
            schema version, fails its own digest, carries a malformed identity
            digest, or holds no cells. Every one of these means the summary
            cannot be trusted to describe the run it claims, so the assembly
            refuses rather than aggregating a payload it cannot authenticate.

    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"cell score summary is unreadable: {path}"
        raise SystemExit(msg) from error
    if not isinstance(payload, Mapping):
        msg = f"cell score summary must be an object: {path}"
        raise SystemExit(msg)
    required = {
        "schema_version",
        "checkpoint_digest",
        "trajectory_digest",
        "evaluation_generation_digest",
        "model",
        "profile",
        "cells",
        "summary_digest",
    }
    if set(payload) != required:
        msg = f"cell score summary has missing or unexpected fields: {path}"
        raise SystemExit(msg)
    if payload["schema_version"] != CELL_SCORE_SUMMARY_SCHEMA_VERSION:
        msg = f"cell score summary schema is unsupported: {path}"
        raise SystemExit(msg)
    body = {key: value for key, value in payload.items() if key != "summary_digest"}
    if payload["summary_digest"] != canonical_sha256(body):
        msg = f"cell score summary digest does not match its body: {path}"
        raise SystemExit(msg)
    if not is_sha256(payload["evaluation_generation_digest"]) or not is_sha256(payload["checkpoint_digest"]):
        msg = f"cell score summary identity digest is invalid: {path}"
        raise SystemExit(msg)
    if not isinstance(payload["cells"], Mapping) or not payload["cells"]:
        msg = f"cell score summary carries no cells: {path}"
        raise SystemExit(msg)
    return dict(payload)


def collect_cells(summaries: Sequence[Mapping[str, Any]], *, checkpoint_digest: str) -> dict[str, dict[str, Any]]:
    """Join every summary's cells, refusing a duplicate or foreign-checkpoint cell.

    Returns:
        One entry per dataset, carrying the cell's own payload plus the profile,
        generation digest and model of the summary it came from, so each cell
        stays attributable after the join.

    Raises:
        SystemExit: when a summary belongs to a different checkpoint, names a
            dataset outside the evaluation set, supplies a cell some other
            summary already supplied, or holds a cell payload that is not a
            mapping. A silently resolved duplicate would let one cell be counted
            twice, so both origins are named in the refusal.

    """
    collected: dict[str, dict[str, Any]] = {}
    origin: dict[str, str] = {}
    for summary in summaries:
        if summary["checkpoint_digest"] != checkpoint_digest:
            msg = f"cell score summary belongs to a different checkpoint: {summary['checkpoint_digest']}"
            raise SystemExit(msg)
        for dataset, cell in summary["cells"].items():
            if dataset not in EVAL_DATASETS:
                msg = f"cell score summary has an unknown cell: {dataset}"
                raise SystemExit(msg)
            if dataset in collected:
                msg = (
                    f"cell {dataset} appears in more than one summary: "
                    f"{origin[dataset]} and {summary['evaluation_generation_digest']}"
                )
                raise SystemExit(msg)
            if not isinstance(cell, Mapping):
                msg = f"cell {dataset} payload is invalid"
                raise SystemExit(msg)
            collected[dataset] = {
                **cell,
                "source_profile": summary["profile"],
                "evaluation_generation_digest": summary["evaluation_generation_digest"],
                "model": summary["model"],
            }
            origin[dataset] = summary["evaluation_generation_digest"]
    return collected


def require_complete_coverage(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compute coverage from what is present; never accept an asserted count.

    Returns:
        The coverage fraction, the total row count, the per-cell rows counted
        here, an always-empty excluded list, and the flag that lets the full
        matrix take a safety veto.

    Raises:
        SystemExit: when a cell is absent, when a cell's row count is not exactly
            the frozen expectation for that dataset, or when the assembled total
            is not the expected total. The row check uses ``type(rows) is not
            int`` so a bool cannot pass as a count.

    """
    missing = [dataset for dataset in EVAL_DATASETS if dataset not in cells]
    if missing:
        msg = f"assembly is missing cells: {missing}"
        raise SystemExit(msg)
    rows_by_cell: dict[str, int] = {}
    for dataset in EVAL_DATASETS:
        rows = cells[dataset].get("rows")
        if type(rows) is not int or rows != EVAL_EXPECTED_ROWS[dataset]:
            msg = f"cell {dataset} row count is {rows}, expected {EVAL_EXPECTED_ROWS[dataset]}"
            raise SystemExit(msg)
        rows_by_cell[dataset] = rows
    total = sum(rows_by_cell.values())
    if total != EXPECTED_TOTAL_ROWS:
        msg = f"assembled rows total {total}, expected {EXPECTED_TOTAL_ROWS}"
        raise SystemExit(msg)
    return {
        "coverage_cells": f"{len(EVAL_DATASETS)}/{len(EVAL_DATASETS)}",
        "rows": total,
        "rows_by_cell": rows_by_cell,
        "excluded_cells": [],
        "eligible_for_full_matrix_safety_veto": True,
    }


def _parse_exclusion(value: str) -> tuple[str, str]:
    generation_digest, separator, cell = value.partition(":")
    if not separator or not is_sha256(generation_digest) or not cell:
        msg = f"exclusion must be <generation_digest>:<cell> with a lowercase SHA-256 generation digest: {value}"
        raise SystemExit(
            msg,
        )
    return generation_digest, cell


def apply_cell_exclusions(
    summaries: Sequence[Mapping[str, Any]],
    exclusions: Sequence[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop named (generation, cell) pairs so an intended duplicate resolves.

    Only an explicitly named pair is dropped, and each drop is returned for the
    record. An excluded cell stays auditable in the output rather than
    disappearing; any duplicate nobody named still reaches the refusal.

    Returns:
        The summaries with the named cells removed, and one record per drop
        carrying the generation digest, the cell, and the result digest that was
        dropped, so the exclusion is reconstructable from the output alone.

    Raises:
        SystemExit: when an exclusion names no cell in any supplied summary,
            which means the caller's intent does not match the inputs and
            applying the rest would quietly aggregate something else.

    """
    remaining = [dict(summary) for summary in summaries]
    for summary in remaining:
        summary["cells"] = dict(summary["cells"])
    applied: list[dict[str, Any]] = []
    for generation_digest, cell in exclusions:
        holders = [
            summary
            for summary in remaining
            if summary["evaluation_generation_digest"] == generation_digest and cell in summary["cells"]
        ]
        if not holders:
            msg = f"exclusion names no cell in any supplied summary: {generation_digest}:{cell}"
            raise SystemExit(msg)
        for summary in holders:
            dropped = summary["cells"].pop(cell)
            applied.append({
                "evaluation_generation_digest": generation_digest,
                "cell": cell,
                "result_sha256": dropped.get("result_sha256"),
            })
    return remaining, applied


# reason: assemble owns from counts and macro f1 together; splitting would fragment diagnostics.
def assemble(  # ruff: ignore[complex-structure,too-many-locals,too-many-statements]
    summaries: Sequence[Mapping[str, Any]],
    *,
    checkpoint_digest: str,
    trajectory_digest: str,
    exclusions: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    summaries, applied_exclusions = apply_cell_exclusions(summaries, exclusions)
    cells = collect_cells(summaries, checkpoint_digest=checkpoint_digest)
    coverage = require_complete_coverage(cells)
    overall = {name: _empty_counts(keys) for name, keys in COUNT_BLOCKS.items()}
    by_label = {
        label: {
            "exact": _empty_counts(EXACT_COUNT_KEYS),
            "containment": _empty_counts(CONTAINMENT_COUNT_KEYS),
        }
        for label in sorted(PII_LABEL_SET)
    }
    by_language: dict[str, dict[str, Any]] = {}
    per_config: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    result_sha256_by_cell: dict[str, str] = {}
    fixture_configs: dict[str, Any] = {}

    for dataset in EVAL_DATASETS:
        cell = cells[dataset]
        counts = cell.get("counts")
        if not isinstance(counts, Mapping):
            msg = f"cell {dataset} carries no counts block"
            raise SystemExit(msg)
        blocks = {
            name: _require_counts(counts.get(name), keys, role=f"{dataset} {name}") for name, keys in COUNT_BLOCKS.items()
        }
        for name in COUNT_BLOCKS:
            _add(overall[name], blocks[name])
        per_config[dataset] = _summary_from_counts(
            blocks["exact"],
            blocks["containment"],
            blocks["full9_exact"],
            blocks["full9_containment"],
            rows=int(cell["rows"]),
        )
        label_counts = cell.get("counts_by_label")
        if not isinstance(label_counts, Mapping):
            msg = f"cell {dataset} carries no per-label counts"
            raise SystemExit(msg)
        for label in sorted(PII_LABEL_SET):
            entry = label_counts.get(label)
            if not isinstance(entry, Mapping):
                msg = f"cell {dataset} is missing label counts for {label}"
                raise SystemExit(msg)
            _add(
                by_label[label]["exact"],
                _require_counts(
                    entry.get("exact"),
                    EXACT_COUNT_KEYS,
                    role=f"{dataset} {label} exact",
                ),
            )
            _add(
                by_label[label]["containment"],
                _require_counts(
                    entry.get("containment"),
                    CONTAINMENT_COUNT_KEYS,
                    role=f"{dataset} {label} containment",
                ),
            )
        language_counts = cell.get("counts_by_language")
        if not isinstance(language_counts, Mapping):
            msg = f"cell {dataset} carries no per-language counts"
            raise SystemExit(msg)
        for language, entry in language_counts.items():
            if not isinstance(entry, Mapping):
                msg = f"cell {dataset} language block {language} is invalid"
                raise SystemExit(msg)
            bucket = by_language.setdefault(
                str(language),
                {
                    "rows": 0,
                    "exact": _empty_counts(EXACT_COUNT_KEYS),
                    "containment": _empty_counts(CONTAINMENT_COUNT_KEYS),
                },
            )
            rows = entry.get("rows")
            if type(rows) is not int or rows < 0:
                msg = f"cell {dataset} language {language} row count is invalid"
                raise SystemExit(msg)
            bucket["rows"] = int(bucket["rows"]) + rows
            _add(
                bucket["exact"],
                _require_counts(
                    entry.get("exact"),
                    EXACT_COUNT_KEYS,
                    role=f"{dataset} {language} exact",
                ),
            )
            _add(
                bucket["containment"],
                _require_counts(
                    entry.get("containment"),
                    CONTAINMENT_COUNT_KEYS,
                    role=f"{dataset} {language} containment",
                ),
            )
        result_digest = cell.get("result_sha256")
        if not is_sha256(result_digest):
            msg = f"cell {dataset} result digest is invalid"
            raise SystemExit(msg)
        result_sha256_by_cell[dataset] = str(result_digest)
        provenance[dataset] = {
            "source_profile": cell["source_profile"],
            "evaluation_generation_digest": cell["evaluation_generation_digest"],
            "result_sha256": result_digest,
            "fixture_sha256": cell.get("fixture_sha256"),
            "dataset_shard_identity_sha256": cell.get("dataset_shard_identity_sha256"),
            "model": cell["model"],
        }
        fixture_configs[dataset] = {
            "rows": int(cell["rows"]),
            "full9_gold_spans": blocks["full9_exact"]["gold_total"],
            "fixture_sha256": cell.get("fixture_sha256"),
        }

    per_label: dict[str, LabelMetricPair] = {
        label: LabelMetricPair(
            exact=exact_span_prf_from_counts(by_label[label]["exact"]),
            containment=containment_span_prf_from_counts(by_label[label]["containment"]),
        )
        for label in sorted(PII_LABEL_SET)
    }
    overall_summary = _summary_from_counts(
        overall["exact"],
        overall["containment"],
        overall["full9_exact"],
        overall["full9_containment"],
        rows=int(coverage["rows"]),
    )
    overall_summary["macro_exact_f1"] = _macro_f1({label: blocks["exact"] for label, blocks in per_label.items()})
    overall_summary["macro_containment_f1"] = _macro_f1({
        label: blocks["containment"] for label, blocks in per_label.items()
    })
    per_language = {
        language: _summary_from_counts(
            by_language[language]["exact"],
            by_language[language]["containment"],
            by_language[language]["exact"],
            by_language[language]["containment"],
            rows=int(by_language[language]["rows"]),
        )
        for language in sorted(by_language)
    }
    generations = sorted({cell["evaluation_generation_digest"] for cell in cells.values()})
    body = {
        "schema_version": ASSEMBLED_SCHEMA_VERSION,
        "status": "assembled_terminal",
        "assembly": {
            "kind": "mixed_evaluator_generation_assembly",
            "evaluation_generation_digests": generations,
            "source_profiles": sorted({cell["source_profile"] for cell in cells.values()}),
            "micro_aggregation": "recomputed_from_per_cell_span_counts",
            "absent_fields": ABSENT_FIELDS,
        },
        "checkpoint_digest": checkpoint_digest,
        "trajectory_digest": trajectory_digest,
        "coverage": coverage,
        "result_sha256_by_cell": result_sha256_by_cell,
        "provenance": provenance,
        "provenance_exclusions": applied_exclusions,
        "report": {
            "fixture": {
                "rows": int(coverage["rows"]),
                "full9_gold_spans": overall["full9_exact"]["gold_total"],
                "configs": fixture_configs,
            },
            "overall": overall_summary,
            "per_config": per_config,
            "per_language": per_language,
            "per_label": per_label,
            "counts": overall,
        },
    }
    return {**body, "assembled_digest": canonical_sha256(body)}
