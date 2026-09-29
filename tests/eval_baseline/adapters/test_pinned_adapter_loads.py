from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from meddies_pii.eval_baseline.adapters import gliner2, openmed, opf_backend

if TYPE_CHECKING:
    import pytest


def test_gliner2_load_uses_pinned_snapshot_before_loading_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class Model:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            calls["model"] = (path, kwargs)
            return object()

    def snapshot_download(model_id: str, **kwargs: object) -> str:
        calls["snapshot"] = (model_id, kwargs)
        return "/pinned/gliner2"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls["tokenizer"] = (model_id, kwargs)
            return object()

    modules = {
        "gliner2": SimpleNamespace(GLiNER2=Model),
        "huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download),
    }
    monkeypatch.setattr(gliner2.importlib, "import_module", modules.__getitem__)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))

    adapter = gliner2.Gliner2Adapter()
    adapter.load()

    assert calls["snapshot"] == (
        gliner2.MODEL_ID,
        {"revision": gliner2.MODEL_REVISION},
    )
    assert calls["model"] == (
        "/pinned/gliner2",
        {"map_location": "cuda", "quantize": True, "compile": False},
    )
    assert calls["tokenizer"] == (
        gliner2.WINDOW_TOKENIZER_ID,
        {"revision": gliner2.WINDOW_TOKENIZER_REVISION, "use_fast": True},
    )


def test_openmed_load_passes_pinned_revision_to_tokenizer_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class Tokenizer:
        model_max_length = 0

    class Model:
        def to(self, device: str) -> Model:
            calls["to"] = device
            return self

    def tokenizer_from_pretrained(model_id: str, **kwargs: object) -> Tokenizer:
        calls["tokenizer"] = (model_id, kwargs)
        return Tokenizer()

    def model_from_pretrained(model_id: str, **kwargs: object) -> Model:
        calls["model"] = (model_id, kwargs)
        return Model()

    def pipeline(*args: object, **kwargs: object) -> object:
        calls["pipeline"] = (args, kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), bfloat16="bf16"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from_pretrained),
            AutoModelForTokenClassification=SimpleNamespace(from_pretrained=model_from_pretrained),
            pipeline=pipeline,
        ),
    )

    openmed.OpenMedAdapter().load()

    assert calls["tokenizer"] == (
        openmed.MODEL_ID,
        {"revision": openmed.MODEL_REVISION, "use_fast": True},
    )
    assert calls["model"] == (
        openmed.MODEL_ID,
        {"revision": openmed.MODEL_REVISION, "torch_dtype": "bf16"},
    )


def test_opf_checkpoint_download_uses_pinned_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def snapshot_download(repo_id: str, **kwargs: object) -> None:
        calls.append((repo_id, kwargs))
        original = Path(str(kwargs["local_dir"])) / "original"
        original.mkdir(parents=True)
        (original / "config.json").write_text(json.dumps({"model_type": "openai_privacy_filter"}), encoding="utf-8")

    monkeypatch.setattr(
        opf_backend.importlib,
        "import_module",
        lambda _name: SimpleNamespace(snapshot_download=snapshot_download),
    )

    target = opf_backend.prepare_native_checkpoint(tmp_path / "checkpoint")

    assert target == tmp_path / "checkpoint"
    assert len(calls) == 1
    repository, request = calls[0]
    assert repository == opf_backend.MODEL_ID
    assert request["revision"] == opf_backend.MODEL_REVISION
    assert request["allow_patterns"] == ["original/*"]
    assert request["local_dir"] != str(target)
