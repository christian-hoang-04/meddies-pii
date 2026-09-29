from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_script_module() -> ModuleType:
    script_path = Path("scripts/archive/process_all_subsets.py")
    spec = importlib.util.spec_from_file_location("process_all_subsets", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_exits_nonzero_when_any_config_fails_without_allow_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()

    def fail_process_config(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(module, "process_config", fail_process_config)
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_all_subsets.py",
            "--configs",
            "english",
            "--output_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
