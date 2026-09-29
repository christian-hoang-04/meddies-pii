from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.pdf_redaction.benchmark.spatial import OracleRegion, PageQuad


@dataclass(frozen=True, slots=True)
class FullMatrixInputs:
    source: Path
    oracle_regions: tuple[OracleRegion, ...]
    forbidden_values: tuple[str, ...] = field(repr=False)
    negative_controls: tuple[PageQuad, ...]

    def __post_init__(self) -> None:
        if not self.source.is_file():
            msg = "full matrix source PDF is missing"
            raise FileNotFoundError(msg)
        if not self.oracle_regions:
            msg = "full matrix requires oracle regions"
            raise ValueError(msg)
        if not self.forbidden_values or any(not value for value in self.forbidden_values):
            msg = "full matrix requires non-empty synthetic values"
            raise ValueError(msg)
        if any(oracle.value_index >= len(self.forbidden_values) for oracle in self.oracle_regions):
            msg = "oracle value index is out of range"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class FullMatrixExecution:
    result: dict[str, object]
    promoted_output: Path
    promoted_output_sha256: str
