from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import NoReturn

import pytest


def _load_script_module() -> types.ModuleType:
    script_path = Path("scripts/archive/reformat_cleaned.py")
    spec = importlib.util.spec_from_file_location("reformat_cleaned", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reformat_cleaned_exits_nonzero_after_partial_push_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()

    def fail_process_config(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "push failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(module, "process_config", fail_process_config)
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        # reason: the code under test calls `login(token=...)` by keyword, so this parameter's
        # reason: name is part of the contract and cannot be underscore-prefixed.
        types.SimpleNamespace(login=lambda token: None),  # ruff: ignore[unused-lambda-argument]
    )
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr(sys, "argv", ["reformat_cleaned.py", "--configs", "english", "--publish"])

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
