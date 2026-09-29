"""Full benchmark execution facade."""

from meddies_pii.pdf_redaction.benchmark.execution_matrix import execute_full_matrix
from meddies_pii.pdf_redaction.benchmark.execution_types import (
    FullMatrixExecution,
    FullMatrixInputs,
)

__all__ = ["FullMatrixExecution", "FullMatrixInputs", "execute_full_matrix"]
