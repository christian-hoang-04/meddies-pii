from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meddies_pii.eval_baseline.opf_benchmark.results import BenchmarkResultRow


@dataclass(frozen=True, slots=True)
class BenchmarkRecommendation:
    winner: BenchmarkResultRow | None
    best_by_path: dict[str, BenchmarkResultRow]
    acceptable_by_path: dict[str, BenchmarkResultRow]
    reason: str


def rank_benchmark_results(
    rows: list[BenchmarkResultRow] | tuple[BenchmarkResultRow, ...],
    *,
    min_exact_agreement_f1: float = 0.98,
    min_exact_gold_f1: float = 0.0,
) -> BenchmarkRecommendation:
    ok_rows = [row for row in rows if row.status == "ok"]
    best_by_path = _best_by_path(ok_rows)
    acceptable = [
        row for row in ok_rows if row.exact_agree_f1 >= min_exact_agreement_f1 and row.exact_gold_f1 >= min_exact_gold_f1
    ]
    acceptable_by_path = _best_by_path(acceptable)
    winner = max(acceptable, key=lambda row: row.docs_per_sec, default=None)
    if winner is None:
        reason = "No config met the correctness threshold. Inspect error/skip rows before choosing speed."
    else:
        reason = (
            f"{winner.path_id} wins by median docs/sec among configs with "
            f"exact agreement F1 >= {min_exact_agreement_f1:.3f}."
        )
    return BenchmarkRecommendation(
        winner=winner,
        best_by_path=best_by_path,
        acceptable_by_path=acceptable_by_path,
        reason=reason,
    )


def _best_by_path(rows: list[BenchmarkResultRow]) -> dict[str, BenchmarkResultRow]:
    best: dict[str, BenchmarkResultRow] = {}
    for row in rows:
        current = best.get(row.path_id)
        if current is None or row.docs_per_sec > current.docs_per_sec:
            best[row.path_id] = row
    return best
