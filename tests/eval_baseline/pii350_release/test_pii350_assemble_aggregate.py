from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from meddies_pii.eval_baseline.baseline.datasets import EVAL_DATASETS, EVAL_EXPECTED_ROWS
from meddies_pii.evaluation.identity import canonical_sha256
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from types import ModuleType

CHECKPOINT_DIGEST = "a" * 64
TRAJECTORY_DIGEST = "d" * 64
GENERATION_A = "1" * 64
GENERATION_B = "2" * 64


def _assembler() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "ops" / "assemble_pii350_checkpoint_aggregate.py"
    spec = importlib.util.spec_from_file_location("pii350_assemble_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _exact(tp: int, pred: int, gold: int) -> dict[str, int]:
    return {"tp": tp, "pred_total": pred, "gold_total": gold}


def _containment(tp: int, pred: int, gold: int) -> dict[str, int]:
    return {
        "precision_tp": tp,
        "recall_tp": tp,
        "pred_total": pred,
        "gold_total": gold,
    }


def _cell(dataset: str, *, tp: int, pred: int, gold: int, rows: int | None = None) -> dict[str, Any]:
    row_count = EVAL_EXPECTED_ROWS[dataset] if rows is None else rows
    return {
        "dataset": dataset,
        "result_sha256": hashlib.sha256(dataset.encode()).hexdigest(),
        "fixture_sha256": hashlib.sha256(f"fixture:{dataset}".encode()).hexdigest(),
        "dataset_shard_identity_sha256": hashlib.sha256(f"identity:{dataset}".encode()).hexdigest(),
        "elapsed_seconds": 12.5,
        "rows": row_count,
        "counts": {
            "exact": _exact(tp, pred, gold),
            "containment": _containment(tp, pred, gold),
            "full9_exact": _exact(tp, pred, gold),
            "full9_containment": _containment(tp, pred, gold),
        },
        "counts_by_label": {
            label: {"exact": _exact(0, 0, 0), "containment": _containment(0, 0, 0)} for label in sorted(PII_LABEL_SET)
        },
        "counts_by_language": {
            "en": {
                "rows": row_count,
                "exact": _exact(tp, pred, gold),
                "containment": _containment(tp, pred, gold),
            },
        },
    }


def _summary(
    cells: dict[str, Any],
    *,
    profile: str,
    generation: str,
    checkpoint: str = CHECKPOINT_DIGEST,
) -> dict[str, Any]:
    for cell in cells.values():
        cell["evaluation_contract_sha256"] = generation
    body = {
        "schema_version": 1,
        "checkpoint_digest": checkpoint,
        "trajectory_digest": TRAJECTORY_DIGEST,
        "evaluation_generation_digest": generation,
        "model": "pii350-checkpoint-" + checkpoint[:16],
        "profile": profile,
        "cells": cells,
    }
    return {**body, "summary_digest": canonical_sha256(body)}


def _cell_counts_for(dataset: str, index: int) -> dict[str, int]:
    """Give each cell a different precision and recall so micro != macro."""
    rows = EVAL_EXPECTED_ROWS[dataset]
    return {"tp": rows, "pred": rows + index + 1, "gold": rows + 2 * (index + 1)}


def _split_summaries(*, first_count: int = 15) -> tuple[dict[str, Any], dict[str, Any]]:
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    for index, dataset in enumerate(EVAL_DATASETS):
        numbers = _cell_counts_for(dataset, index)
        cell = _cell(dataset, tp=numbers["tp"], pred=numbers["pred"], gold=numbers["gold"])
        if index < first_count:
            first[dataset] = cell
        else:
            second[dataset] = cell
    return (
        _summary(first, profile="diffusionllm", generation=GENERATION_A),
        _summary(second, profile="huyhoang0411ha", generation=GENERATION_B),
    )


def _expected_micro() -> dict[str, float]:
    """Sum the same counts by plain arithmetic, independent of the assembler."""
    tp = pred = gold = 0
    for index, dataset in enumerate(EVAL_DATASETS):
        numbers = _cell_counts_for(dataset, index)
        tp += numbers["tp"]
        pred += numbers["pred"]
        gold += numbers["gold"]
    precision = tp / pred
    recall = tp / gold
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pred": pred,
        "gold": gold,
    }


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_assembly_recomputes_micro_totals_from_per_cell_counts() -> None:
    """A mean of the per-cell F1 values is a different number.

    So this proves the micro totals came from counts rather than from averaging rounded metrics.

    """
    assembler = _assembler()
    first, second = _split_summaries()

    assembled = assembler.assemble(
        [first, second],
        checkpoint_digest=CHECKPOINT_DIGEST,
        trajectory_digest=TRAJECTORY_DIGEST,
    )

    expected = _expected_micro()
    overall = assembled["report"]["overall"]
    assert overall["exact_precision"] == expected["precision"]
    assert overall["exact_recall"] == expected["recall"]
    assert overall["exact_f1"] == expected["f1"]
    assert overall["pred_spans"] == expected["pred"]
    assert overall["gold_spans"] == expected["gold"]
    assert overall["rows"] == 263_785
    assert assembled["coverage"]["coverage_cells"] == "17/17"
    assert assembled["coverage"]["rows"] == 263_785
    assert len(assembled["report"]["per_config"]) == 17

    per_cell_f1 = [assembled["report"]["per_config"][dataset]["exact_f1"] for dataset in EVAL_DATASETS]
    assert round(sum(per_cell_f1) / len(per_cell_f1), 4) != overall["exact_f1"]


def test_assembly_records_per_cell_source_provenance() -> None:
    assembler = _assembler()
    first, second = _split_summaries()

    assembled = assembler.assemble(
        [first, second],
        checkpoint_digest=CHECKPOINT_DIGEST,
        trajectory_digest=TRAJECTORY_DIGEST,
    )

    provenance = assembled["provenance"]
    assert set(provenance) == set(EVAL_DATASETS)
    generations = {entry["evaluation_generation_digest"] for entry in provenance.values()}
    assert generations == {GENERATION_A, GENERATION_B}
    profiles = {entry["source_profile"] for entry in provenance.values()}
    assert profiles == {"diffusionllm", "huyhoang0411ha"}
    for dataset, entry in provenance.items():
        assert entry["result_sha256"] == hashlib.sha256(dataset.encode()).hexdigest()
    assert assembled["assembly"]["evaluation_generation_digests"] == sorted([GENERATION_A, GENERATION_B])
    assert "evaluation_contract" in assembled["assembly"]["absent_fields"]


def test_assembly_refuses_a_missing_cell() -> None:
    assembler = _assembler()
    first, second = _split_summaries()
    second["cells"].pop(EVAL_DATASETS[-1])
    second = _summary(second["cells"], profile="huyhoang0411ha", generation=GENERATION_B)

    with pytest.raises(SystemExit, match="missing cells"):
        assembler.assemble(
            [first, second],
            checkpoint_digest=CHECKPOINT_DIGEST,
            trajectory_digest=TRAJECTORY_DIGEST,
        )


def _duplicated_v2_eval_summaries() -> tuple[dict[str, Any], dict[str, Any]]:
    """Mirror the step-146 join: the fill generation repeats the old v2-eval.

    The fill generation also carries v2-eval, which the 16-cell generation has.

    """
    first, second = _split_summaries(first_count=16)
    fill_cells = dict(second["cells"])
    fill_cells["v2-eval"] = _cell("v2-eval", tp=1, pred=2, gold=3)
    return first, _summary(fill_cells, profile="huyhoang0411ha", generation=GENERATION_B)


def test_exclusion_resolves_a_duplicate_and_keeps_the_other_generation() -> None:
    """The surviving v2-eval is the one from the generation that was not excluded."""
    assembler = _assembler()
    first, second = _duplicated_v2_eval_summaries()

    assembled = assembler.assemble(
        [first, second],
        checkpoint_digest=CHECKPOINT_DIGEST,
        trajectory_digest=TRAJECTORY_DIGEST,
        exclusions=[(GENERATION_B, "v2-eval")],
    )

    assert assembled["coverage"]["coverage_cells"] == "17/17"
    assert assembled["provenance"]["v2-eval"]["evaluation_generation_digest"] == GENERATION_A
    assert assembled["provenance"]["v2-eval"]["source_profile"] == "diffusionllm"
    assert assembled["report"]["per_config"]["v2-eval"]["rows"] == (EVAL_EXPECTED_ROWS["v2-eval"])


def test_exclusion_is_recorded_in_the_assembled_provenance() -> None:
    """An assembly with nothing excluded still carries the empty record."""
    assembler = _assembler()
    first, second = _duplicated_v2_eval_summaries()
    dropped_digest = second["cells"]["v2-eval"]["result_sha256"]

    assembled = assembler.assemble(
        [first, second],
        checkpoint_digest=CHECKPOINT_DIGEST,
        trajectory_digest=TRAJECTORY_DIGEST,
        exclusions=[(GENERATION_B, "v2-eval")],
    )

    assert assembled["provenance_exclusions"] == [
        {
            "evaluation_generation_digest": GENERATION_B,
            "cell": "v2-eval",
            "result_sha256": dropped_digest,
        },
    ]
    plain_first, plain_second = _split_summaries()
    plain = assembler.assemble(
        [plain_first, plain_second],
        checkpoint_digest=CHECKPOINT_DIGEST,
        trajectory_digest=TRAJECTORY_DIGEST,
    )
    assert plain["provenance_exclusions"] == []


def test_exclusion_of_an_absent_pair_is_refused() -> None:
    """gretel_en lives only in the first generation.

    So naming it here is a typo the assembler has to refuse rather than silently apply nothing.

    """
    assembler = _assembler()
    first, second = _duplicated_v2_eval_summaries()

    assert "gretel_en" not in second["cells"]
    with pytest.raises(SystemExit, match="names no cell in any supplied summary"):
        assembler.assemble(
            [first, second],
            checkpoint_digest=CHECKPOINT_DIGEST,
            trajectory_digest=TRAJECTORY_DIGEST,
            exclusions=[(GENERATION_B, "gretel_en")],
        )
    with pytest.raises(SystemExit, match="names no cell in any supplied summary"):
        assembler.assemble(
            [first, second],
            checkpoint_digest=CHECKPOINT_DIGEST,
            trajectory_digest=TRAJECTORY_DIGEST,
            exclusions=[("9" * 64, "v2-eval")],
        )


def test_an_unexcluded_duplicate_is_still_refused() -> None:
    assembler = _assembler()
    first, second = _duplicated_v2_eval_summaries()

    with pytest.raises(SystemExit, match="appears in more than one summary"):
        assembler.assemble(
            [first, second],
            checkpoint_digest=CHECKPOINT_DIGEST,
            trajectory_digest=TRAJECTORY_DIGEST,
        )


def test_cli_rejects_a_malformed_exclusion(tmp_path: Path) -> None:
    assembler = _assembler()
    first, second = _duplicated_v2_eval_summaries()
    first_path = _write(tmp_path, "first.json", first)
    second_path = _write(tmp_path, "second.json", second)

    def run(exclusion: str) -> int:
        result: object = assembler.main([
            "--summary",
            str(first_path),
            "--summary",
            str(second_path),
            "--checkpoint-digest",
            CHECKPOINT_DIGEST,
            "--trajectory-digest",
            TRAJECTORY_DIGEST,
            "--exclude",
            exclusion,
            "--output",
            str(tmp_path / "out.json"),
        ])
        assert isinstance(result, int)
        return result

    with pytest.raises(SystemExit, match="must be <generation_digest>:<cell>"):
        run("v2-eval")
    with pytest.raises(SystemExit, match="must be <generation_digest>:<cell>"):
        run("not-a-digest:v2-eval")
    assert run(f"{GENERATION_B}:v2-eval") == 0


def test_assembly_refuses_a_cell_supplied_by_two_summaries() -> None:
    assembler = _assembler()
    first, second = _split_summaries()
    duplicated = EVAL_DATASETS[0]
    cells = dict(second["cells"])
    cells[duplicated] = _cell(duplicated, tp=1, pred=2, gold=3)
    second = _summary(cells, profile="huyhoang0411ha", generation=GENERATION_B)

    with pytest.raises(SystemExit, match="appears in more than one summary"):
        assembler.assemble(
            [first, second],
            checkpoint_digest=CHECKPOINT_DIGEST,
            trajectory_digest=TRAJECTORY_DIGEST,
        )


def test_assembly_refuses_a_cell_whose_row_count_drifted() -> None:
    assembler = _assembler()
    first, second = _split_summaries()
    drifted = EVAL_DATASETS[0]
    cells = dict(first["cells"])
    cells[drifted] = _cell(drifted, tp=1, pred=2, gold=3, rows=7)
    first = _summary(cells, profile="diffusionllm", generation=GENERATION_A)

    with pytest.raises(SystemExit, match="row count is 7"):
        assembler.assemble(
            [first, second],
            checkpoint_digest=CHECKPOINT_DIGEST,
            trajectory_digest=TRAJECTORY_DIGEST,
        )


def test_assembly_refuses_a_summary_from_another_checkpoint() -> None:
    assembler = _assembler()
    first, second = _split_summaries()
    foreign = _summary(
        second["cells"],
        profile="huyhoang0411ha",
        generation=GENERATION_B,
        checkpoint="b" * 64,
    )

    with pytest.raises(SystemExit, match="different checkpoint"):
        assembler.assemble(
            [first, foreign],
            checkpoint_digest=CHECKPOINT_DIGEST,
            trajectory_digest=TRAJECTORY_DIGEST,
        )


def test_summary_loader_refuses_a_tampered_digest(tmp_path: Path) -> None:
    assembler = _assembler()
    first, _second = _split_summaries()
    tampered = {**first, "profile": "meddies-run"}
    path = _write(tmp_path, "tampered.json", tampered)

    with pytest.raises(SystemExit, match="digest does not match"):
        assembler.load_summary(path)


def test_cli_writes_an_assembled_aggregate(tmp_path: Path) -> None:
    assembler = _assembler()
    first, second = _split_summaries()
    first_path = _write(tmp_path, "first.json", first)
    second_path = _write(tmp_path, "second.json", second)
    output = tmp_path / "assembled.json"

    exit_code = assembler.main([
        "--summary",
        str(first_path),
        "--summary",
        str(second_path),
        "--checkpoint-digest",
        CHECKPOINT_DIGEST,
        "--trajectory-digest",
        TRAJECTORY_DIGEST,
        "--output",
        str(output),
    ])

    assert exit_code == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["coverage"]["coverage_cells"] == "17/17"
    assert written["checkpoint_digest"] == CHECKPOINT_DIGEST
    body = {key: value for key, value in written.items() if key != "assembled_digest"}
    assert written["assembled_digest"] == canonical_sha256(body)


def test_cli_requires_at_least_two_summaries(tmp_path: Path) -> None:
    assembler = _assembler()
    first, _second = _split_summaries()
    first_path = _write(tmp_path, "only.json", first)

    with pytest.raises(SystemExit, match="at least two"):
        assembler.main([
            "--summary",
            str(first_path),
            "--checkpoint-digest",
            CHECKPOINT_DIGEST,
            "--trajectory-digest",
            TRAJECTORY_DIGEST,
        ])
