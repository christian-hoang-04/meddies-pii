#!/usr/bin/env python
"""Consolidate per-model eval aggregates into one side-by-side baseline report.

Each model's aggregate (the JSON written by ``aggregate_results`` -> overall /
per_config / per_language / per_label) lives on its Modal Volume at
``reports/<model>-aggregate.json``. Download each into one local dir, then:

    uv run python scripts/reports/build_baseline_eval_report.py \
        --input-dir <dir-of-model-aggregates> \
        --out ../meddies-pii-context/reports/2026-06-30-baseline-eval.md \
        --generated 2026-06-30

The report compares every model across the 17 eval configs + per-language +
per-label. Exact and containment F1 render in separate fixed-nine matrices.
Supported-subset label scores stay in a secondary capability table, where labels
a model can't emit render as ``n/s`` (not supported).
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import argparse
import contextlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from meddies_pii.evaluation.identity import (
    EvaluationContract,
    canonical_sha256,
    is_sha256,
    parse_evaluation_contract,
)
from meddies_pii.json_types import is_str_mapping

_MISSING = "—"
_NOT_SUPPORTED = "n/s"
_FIXTURE_CONFIG_FIELDS = ("fixture_sha256", "rows", "full9_gold_spans")


def _f1(value: object) -> str:
    return f"{float(value):.3f}" if isinstance(value, int | float) else _MISSING


def _fixture_config_summary(value: object) -> dict[str, dict[str, object]]:
    """Keep only data identity when deciding whether model reports compare."""
    if not isinstance(value, Mapping):
        msg = "aggregates are missing fixture config summaries"
        raise ValueError(msg)
    summary: dict[str, dict[str, object]] = {}
    for name, config in value.items():
        if not isinstance(name, str) or not is_str_mapping(config):
            msg = "fixture config summaries must map names to objects"
            raise ValueError(msg)
        missing = [field for field in _FIXTURE_CONFIG_FIELDS if field not in config]
        if missing:
            msg = f"fixture config {name!r} is missing data identity fields: {missing}"
            raise ValueError(msg)
        summary[name] = {field: config[field] for field in _FIXTURE_CONFIG_FIELDS}
    return summary


def _result_provenance(value: object, expected_configs: Mapping[str, object]) -> tuple[str, dict[str, str]]:
    """Validate one aggregate's scored-result identities without comparing models."""
    if not isinstance(value, Mapping):
        msg = "aggregates are missing result provenance"
        raise ValueError(msg)
    result_set_sha256 = value.get("result_set_sha256")
    result_sha256_by_config = value.get("result_sha256_by_config")
    if not is_sha256(result_set_sha256) or not isinstance(result_sha256_by_config, Mapping):
        msg = "aggregates have invalid result provenance"
        raise ValueError(msg)
    normalized = {
        cast("str", config): digest
        for config, digest in result_sha256_by_config.items()
        if isinstance(config, str) and is_sha256(digest)
    }
    normalized = dict(sorted(normalized.items()))
    if len(normalized) != len(result_sha256_by_config):
        msg = "result provenance must map configs to SHA-256 digests"
        raise ValueError(msg)
    if set(normalized) != set(expected_configs):
        msg = "result provenance config keys do not match fixture configs"
        raise ValueError(msg)
    expected = canonical_sha256({"result_sha256_by_config": normalized})
    if result_set_sha256 != expected:
        msg = "result provenance set digest does not match config digests"
        raise ValueError(msg)
    return result_set_sha256, normalized


def _evaluation_contract_provenance(value: object) -> tuple[str, EvaluationContract]:
    """Parse one aggregate's canonical evaluation contract and verify its digest."""
    if not isinstance(value, Mapping):
        msg = "aggregates are missing evaluation contract provenance"
        raise ValueError(msg)
    payload = value.get("payload")
    digest = value.get("sha256")
    if not is_sha256(digest):
        msg = "evaluation contract provenance has an invalid SHA-256"
        raise ValueError(msg)
    try:
        contract = parse_evaluation_contract(payload)
    except ValueError as error:
        msg = "evaluation contract payload is invalid"
        raise ValueError(msg) from error
    if payload != contract.to_payload():
        msg = "evaluation contract payload must be canonical"
        raise ValueError(msg)
    if digest != contract.digest:
        msg = "evaluation contract digest does not match its payload"
        raise ValueError(msg)
    return digest, contract


def _matrix(
    title: str,
    row_keys: Sequence[str],
    models: Sequence[str],
    cell: Callable[[str, str], str],
) -> list[str]:
    """— (missing) or n/s (unsupported) — excluded from the mean.

    Unweighted mean of each model's numeric cells across the matrix rows — a quick model-vs-model compare that weights
    every config/language/label equally (contrast the row-weighted micro-F1 in Overall).

    """
    header = f"| {title} | " + " | ".join(models) + " |"
    divider = "|" + "---|" * (len(models) + 1)
    lines = [header, divider]
    numeric: dict[str, list[float]] = {model: [] for model in models}
    for key in row_keys:
        rendered = [cell(model, key) for model in models]
        for model, value in zip(models, rendered, strict=True):
            with contextlib.suppress(ValueError):
                numeric[model].append(float(value))
        lines.append(f"| {key} | {' | '.join(rendered)} |")
    avg_cells = [
        f"**{sum(numeric[model]) / len(numeric[model]):.3f}**" if numeric[model] else _MISSING for model in models
    ]
    lines.append(f"| **avg** | {' | '.join(avg_cells)} |")
    return lines


# reason: build report keeps sorted union beside matrix; splitting would duplicate totals or escaping.
def build_report(  # ruff: ignore[complex-structure,too-many-locals,too-many-statements]
    aggregates: Mapping[str, Mapping[str, Any]],
    *,
    generated: str,
    inference_profile: Mapping[str, Any] | None = None,
) -> str:
    models = sorted(aggregates)
    fixture_digests = {
        model: str(aggregate.get("fixture", {}).get("sha256", "")) for model, aggregate in aggregates.items()
    }
    missing_fixture = [model for model, digest in fixture_digests.items() if not digest]
    if missing_fixture:
        msg = f"aggregates are missing fixture digests: {missing_fixture}"
        raise ValueError(msg)
    if len(set(fixture_digests.values())) > 1:
        msg = f"cannot compare mixed fixtures: {fixture_digests}"
        raise ValueError(msg)
    fixture_configs = {
        model: _fixture_config_summary(aggregate.get("fixture", {}).get("configs"))
        for model, aggregate in aggregates.items()
    }
    if len({json.dumps(configs, sort_keys=True) for configs in fixture_configs.values()}) > 1:
        msg = f"cannot compare mixed fixture config summaries: {fixture_configs}"
        raise ValueError(msg)
    result_provenance = {
        model: _result_provenance(aggregate.get("provenance"), fixture_configs[model])
        for model, aggregate in aggregates.items()
    }
    evaluation_contracts = {
        model: _evaluation_contract_provenance(aggregate.get("evaluation_contract"))
        for model, aggregate in aggregates.items()
    }

    lines: list[str] = [
        f"# Baseline eval — {len(models)} PII models x Meddies Labels",
        "",
        (
            f"Generated {generated}. The primary comparison uses one fixed nine-label taxonomy, "
            "so a model is penalized for labels it cannot emit. Supported-subset scores are "
            "reported separately as a capability diagnostic. "
            f"Models present: {', '.join(models) or 'none yet'}."
        ),
        f"Fixture SHA-256: `{next(iter(fixture_digests.values()), '')}`.",
        "",
        "## Overall — fixed nine-label taxonomy",
        "",
        (
            "| model | rows | gold spans | exact micro-F1 | exact macro-F1 | "
            "contain micro-F1 | contain macro-F1 | supported exact | supported contain |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model in models:
        overall = aggregates[model].get("overall", {})
        lines.append(
            f"| {model} | {overall.get('rows', _MISSING)} | "
            f"{overall.get('full9_gold_spans', _MISSING)} | "
            f"{_f1(overall.get('full9_exact_f1'))} | "
            f"{_f1(overall.get('full9_macro_exact_f1'))} | "
            f"{_f1(overall.get('full9_containment_f1'))} | "
            f"{_f1(overall.get('full9_macro_containment_f1'))} | "
            f"{_f1(overall.get('exact_f1'))} | "
            f"{_f1(overall.get('containment_f1'))} |",
        )

    lines += [
        "",
        "## Evaluation contract provenance",
        "",
        (
            "Each row binds its scores to a validated evaluation contract. The model "
            "label is presentation only; the model artifact below is the evaluated "
            "identity."
        ),
        "",
        "| model label | contract SHA-256 | model artifact reference | revision | artifact SHA-256 |",
        "|---|---|---|---|---|",
    ]
    for model in models:
        contract_digest, contract = evaluation_contracts[model]
        artifact = contract.model
        lines.append(
            f"| {model} | `{contract_digest}` | {artifact.reference} | `{artifact.revision}` | `{artifact.sha256}` |",
        )

    lines += [
        "",
        "## Result artifact provenance",
        "",
        (
            "Each digest identifies the raw result JSONL bytes that produced one model's "
            "scores. Result digests are model outputs, so they are preserved per model and "
            "must not be compared across models."
        ),
    ]
    for model in models:
        result_set_sha256, result_sha256_by_config = result_provenance[model]
        lines += [
            "",
            f"### {model}",
            "",
            f"Result-set SHA-256: `{result_set_sha256}`.",
            "",
            "| config | result JSONL SHA-256 |",
            "|---|---|",
        ]
        lines += [f"| {config} | `{digest}` |" for config, digest in result_sha256_by_config.items()]

    lines += [
        "",
        "## Inference speed",
        "",
        (
            "rows/s = predict() throughput under each model's recorded runtime "
            "(excluding model load and compile cold-start). Hardware and batching can differ "
            "across models, so speed is not a normalized ranking. A model timed on a subset "
            "of cells still reports its measured rate. **lfm-bioes-spike ran unbatched "
            "(batch=1)** as a correctness-first spike, so its rate is a floor rather than "
            "the architecture's ceiling."
        ),
        "",
        "| model | rows/s |",
        "|---|---|",
    ]
    for model in models:
        overall = aggregates[model].get("overall", {})
        rps = overall.get("rows_per_second")
        lines.append(f"| {model} | {f'{float(rps):.1f}' if rps is not None else _MISSING} |")

    def config_exact_cell(model: str, key: str) -> str:
        block = aggregates[model].get("per_config", {}).get(key)
        return _f1(block.get("full9_exact_f1")) if block else _MISSING

    def config_containment_cell(model: str, key: str) -> str:
        block = aggregates[model].get("per_config", {}).get(key)
        return _f1(block.get("full9_containment_f1")) if block else _MISSING

    def language_exact_cell(model: str, key: str) -> str:
        block = aggregates[model].get("per_language", {}).get(key)
        return _f1(block.get("full9_exact_f1")) if block else _MISSING

    def language_containment_cell(model: str, key: str) -> str:
        block = aggregates[model].get("per_language", {}).get(key)
        return _f1(block.get("full9_containment_f1")) if block else _MISSING

    def label_exact_cell(model: str, key: str) -> str:
        block = aggregates[model].get("per_label_full9", {}).get(key)
        return _f1(block.get("exact", {}).get("f1")) if block else _MISSING

    def label_containment_cell(model: str, key: str) -> str:
        block = aggregates[model].get("per_label_full9", {}).get(key)
        return _f1(block.get("containment", {}).get("f1")) if block else _MISSING

    def supported_label_cell(model: str, key: str) -> str:
        block = aggregates[model].get("per_label", {}).get(key)
        if block is None:
            return _NOT_SUPPORTED
        return _f1(block.get("exact", {}).get("f1"))

    configs = _sorted_union(aggregates, "per_config")
    languages = _sorted_union(aggregates, "per_language")
    labels = _sorted_union(aggregates, "per_label_full9")
    supported_labels = _sorted_union(aggregates, "per_label")

    lines += [
        "",
        (
            "The **avg** row (bottom of each matrix) is the unweighted mean across the "
            "rows — every config/language/label weighted equally, so it rewards "
            "consistency; contrast the row-weighted fixed-taxonomy micro-F1 in Overall."
        ),
        "",
        "## Per config (source x language) — fixed nine-label exact F1",
        "",
    ]
    lines += _matrix("config", configs, models, config_exact_cell)
    lines += [
        "",
        "## Per config (source x language) — fixed nine-label containment F1",
        "",
    ]
    lines += _matrix("config", configs, models, config_containment_cell)
    lines += ["", "## Per language (ISO) — fixed nine-label exact F1", ""]
    lines += _matrix("lang", languages, models, language_exact_cell)
    lines += [
        "",
        "## Per language (ISO) — fixed nine-label containment F1",
        "",
    ]
    lines += _matrix("lang", languages, models, language_containment_cell)
    lines += ["", "## Per label — fixed nine-label exact F1", ""]
    lines += _matrix("label", labels, models, label_exact_cell)
    lines += ["", "## Per label — fixed nine-label containment F1", ""]
    lines += _matrix("label", labels, models, label_containment_cell)
    lines += [
        "",
        "## Supported-subset per label — exact F1 capability diagnostic",
        "",
        (
            "`n/s` means the adapter declares that the model cannot emit the label. "
            "This table is not the primary model comparison."
        ),
        "",
    ]
    lines += _matrix("label", supported_labels, models, supported_label_cell)
    if inference_profile:
        lines += _inference_section(inference_profile)
    lines.append("")
    return "\n".join(lines)


def _inference_section(profile: Mapping[str, Any]) -> list[str]:
    """Render the opf / v2 serving inference profile: speed and accuracy.

    This is throughput on a fixed benchmark doc set, distinct from the eval F1 above.
    """
    lines = ["", "## Inference profile — serving speed (opf = meddies-pii-v2 arch)", ""]
    note = profile.get("note")
    if note:
        lines += [str(note), ""]
    lines += [
        "| path | docs/s | gold-F1 | agree-vs-CRF | verdict |",
        "|---|---|---|---|---|",
    ]
    for row in profile.get("paths", []):
        lines.append(
            f"| {row.get('path', '?')} | {row.get('docs_per_sec', _MISSING)} | "
            f"{_f1(row.get('gold_f1'))} | {_f1(row.get('agree_vs_crf'))} | "
            f"{row.get('verdict', '')} |",
        )
    compile_rows = profile.get("compile") or []
    if compile_rows:
        lines += ["", "**torch.compile (A2 bs32):**", ""]
        lines += ["| mode | docs/s | gold-F1 |", "|---|---|---|"]
        for row in compile_rows:
            lines.append(f"| {row.get('mode', '?')} | {row.get('docs_per_sec', _MISSING)} | {_f1(row.get('gold_f1'))} |")
    recipe = profile.get("recipe")
    if recipe:
        lines += ["", f"**Recipe:** {recipe}"]
    return lines


def _sorted_union(aggregates: Mapping[str, Mapping[str, Any]], section: str) -> list[str]:
    keys: set[str] = set()
    for agg in aggregates.values():
        block = agg.get(section)
        if isinstance(block, Mapping):
            keys.update(str(key) for key in block)
    return sorted(keys)


def _load_aggregates(input_dir: Path) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            aggregates[path.stem] = payload
    return aggregates


def _load_inference_profile(input_dir: Path) -> dict[str, Any] | None:
    path = input_dir / "_inference_profile.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--generated", required=True)
    args = parser.parse_args()

    aggregates = _load_aggregates(args.input_dir)
    report = build_report(
        aggregates,
        generated=args.generated,
        inference_profile=_load_inference_profile(args.input_dir),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(aggregates)} models)")


if __name__ == "__main__":
    main()
