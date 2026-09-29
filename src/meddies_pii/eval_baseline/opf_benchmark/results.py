from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RunStatus = Literal["ok", "skipped", "error"]


MIN_ACCEPTABLE_EXACT_F1 = 0.98


@dataclass(frozen=True, slots=True)
class BenchmarkResultRow:
    config_id: str
    path_id: str
    path_name: str
    batch_size: int
    length_bucket: str
    dtype: str
    compile_mode: str
    triton: str
    onnx_variant: str
    attention_kernel: str
    status: RunStatus
    docs_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    median_seconds: float = 0.0
    min_seconds: float = 0.0
    max_seconds: float = 0.0
    peak_gpu_memory_mb: int = 0
    exact_agree_f1: float = 0.0
    containment_agree_f1: float = 0.0
    exact_gold_f1: float = 0.0
    containment_gold_f1: float = 0.0
    skip_reason: str | None = None
    error: str | None = None
    stub: bool = False

    @property
    def is_acceptable_default(self) -> bool:
        return self.status == "ok" and self.exact_agree_f1 >= MIN_ACCEPTABLE_EXACT_F1


def format_results_grid(
    rows: list[BenchmarkResultRow] | tuple[BenchmarkResultRow, ...],
    *,
    max_rows: int | None = None,
) -> str:
    selected = tuple(rows if max_rows is None else rows[:max_rows])
    lines = [
        "path\tconfig_id\tstatus\tdocs/s\ttokens/s\tpeak_mb\texact_agree\tcontain_agree\texact_gold\tcontain_gold\tskip/error",
    ]
    for row in selected:
        reason = row.skip_reason or row.error or ""
        lines.append(
            "\t".join((
                row.path_id,
                row.config_id,
                row.status,
                f"{row.docs_per_sec:.2f}",
                f"{row.tokens_per_sec:.2f}",
                str(row.peak_gpu_memory_mb),
                f"{row.exact_agree_f1:.4f}",
                f"{row.containment_agree_f1:.4f}",
                f"{row.exact_gold_f1:.4f}",
                f"{row.containment_gold_f1:.4f}",
                reason,
            )),
        )
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"... truncated {len(rows) - max_rows} rows ...")
    return "\n".join(lines)
