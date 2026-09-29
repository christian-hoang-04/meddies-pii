from __future__ import annotations

import os

# reason: a fresh interpreter is the only way to observe what an import does NOT load — in-process the
# reason: module under test is already imported, so the probe has to run somewhere else.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import override

import pytest
import torch
import transformers
from torch import nn

from meddies_pii.training.bioes.data import artifacts


def test_disable_unsloth_statistics_timeout_sets_remote_import_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNSLOTH_DISABLE_STATISTICS", raising=False)

    artifacts._disable_unsloth_statistics_timeout()

    assert os.environ["UNSLOTH_DISABLE_STATISTICS"] == "1"


def test_load_pretrained_adapts_third_party_factories_at_one_runtime_boundary() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Factory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls.append((model_id, kwargs))
            return {"loaded": model_id}

    assert artifacts.load_pretrained(Factory, "model-id", {"local_files_only": True}) == {"loaded": "model-id"}
    assert calls == [("model-id", {"local_files_only": True})]


def test_load_pretrained_refuses_objects_without_the_factory_method() -> None:
    with pytest.raises(ImportError, match="does not expose from_pretrained"):
        artifacts.load_pretrained(object(), "model-id", {})


def test_artifact_module_imports_without_optional_torch_runtime() -> None:
    project_root = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch blocked by test")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_without_torch

        from meddies_pii.training.bioes.data.artifacts import _require_torch
        from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger

        for operation in (
            _require_torch,
            lambda: HiddenStateTokenTagger(
                backbone=object(),
                hidden_size=4,
                num_labels=3,
                request_hidden_states=False,
            ),
        ):
            try:
                operation()
            except ImportError as error:
                assert "torch is required" in str(error)
            else:
                raise AssertionError("optional Torch operation did not fail explicitly")
        """,
    )

    # reason: argv is a list, so no shell parses it, and every element is fixed here — `sys.executable`
    # reason: is this interpreter and `script` is the literal built above from `textwrap.dedent`.
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_hf_builder_preserves_pinned_model_and_tokenizer_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordingFactory:
        def __init__(self, product: object) -> None:
            self.product = product
            self.calls: list[tuple[str, dict[str, object]]] = []

        def from_pretrained(self, model_id: str, **kwargs: object) -> object:
            self.calls.append((model_id, kwargs))
            return self.product

    class FakeTokenizer:
        pad_token: str | None = None
        # reason: these are the tokenizer's sentinel strings, not credentials. The rule keys on a name
        # reason: containing `token`, and a stand-in for a tokenizer has to carry the names the real one does.
        eos_token: str | None = "<eos>"  # ruff: ignore[hardcoded-password-string]
        # reason: these are the tokenizer's sentinel strings, not credentials. The rule keys on a name
        # reason: containing `token`, and a stand-in for a tokenizer has to carry the names the real one does.
        unk_token: str | None = "<unk>"  # ruff: ignore[hardcoded-password-string]

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = nn.Embedding(8, 4)

        @override
        def forward(self, **kwargs: object) -> dict[str, torch.Tensor]:
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            return {"last_hidden_state": self.embedding(input_ids)}

    tokenizer = FakeTokenizer()
    backbone = FakeBackbone()
    tokenizer_factory = RecordingFactory(tokenizer)
    model_factory = RecordingFactory(backbone)
    monkeypatch.setattr(transformers, "AutoTokenizer", tokenizer_factory)
    monkeypatch.setattr(transformers, "AutoModel", model_factory)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = artifacts.build_hf_artifacts(
        "model-id",
        model_revision="model-revision",
        tokenizer_id="tokenizer-id",
        tokenizer_revision="tokenizer-revision",
        trust_remote_code=True,
    )

    assert tokenizer_factory.calls == [
        (
            "tokenizer-id",
            {"revision": "tokenizer-revision", "trust_remote_code": True},
        ),
    ]
    assert model_factory.calls == [
        (
            "model-id",
            {
                "dtype": torch.float32,
                "revision": "model-revision",
                "trust_remote_code": True,
            },
        ),
    ]
    # reason: these are the tokenizer's sentinel strings, not credentials. The rule keys on a name
    # reason: containing `token`, and a stand-in for a tokenizer has to carry the names the real one does.
    assert result.tokenizer.pad_token == "<eos>"  # ruff: ignore[hardcoded-password-string]
    assert result.tagger.backbone is backbone
