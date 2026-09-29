from __future__ import annotations

import os

# reason: a fresh interpreter is the only way to observe what an import does NOT load — in-process the
# reason: module under test is already imported, so the probe has to run somewhere else.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[3]


def _run_import_probe(script: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    # reason: argv is a list, so no shell parses it, and both elements are fixed — `sys.executable` is
    # reason: this interpreter and `script` comes from the literals each test passes in.
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_subset_import_does_not_load_dataset_module() -> None:
    result = _run_import_probe(
        """
import sys
import meddies_pii.eval_baseline.baseline.subset

assert "meddies_pii.eval_baseline.baseline.datasets" not in sys.modules
""",
    )

    assert result.returncode == 0, result.stderr


def test_dataset_import_reexports_neutral_eval_row_owner() -> None:
    result = _run_import_probe(
        """
from meddies_pii.eval_baseline.baseline.datasets import EvalRow as compatibility_row
from meddies_pii.eval_baseline.baseline.models import EvalRow as model_row

assert compatibility_row is model_row
""",
    )

    assert result.returncode == 0, result.stderr
