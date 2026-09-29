from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from meddies_pii.generation.gemini.inference import (
    prepare_hf_review,
    submit_batch_job,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_prepare_hf_review_requires_external_provider_opt_in_for_default_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets_module = ModuleType("datasets")

    def fail_load_dataset(*_args: object, **_kwargs: object) -> None:
        pytest.fail("prepare_hf_review loaded a dataset before privacy gate")

    # reason: the code under test imports `datasets` by name out of `sys.modules`, so the fake must be a
    # reason: real module; `ModuleType` declares no `load_dataset`, and no annotation admits the write.
    datasets_module.load_dataset = fail_load_dataset  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    with pytest.raises(ValueError, match="external provider opt-in"):
        prepare_hf_review("pii-bioes", str(tmp_path / "requests.jsonl"))


def test_submit_batch_job_rejects_local_file_without_data_classification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    source.write_text('{"request": {}}\n', encoding="utf-8")
    client = SimpleNamespace()

    with pytest.raises(ValueError, match="data classification"):
        submit_batch_job(
            client,
            str(source),
            model="gemini-test",
            allow_external_provider=True,
        )
