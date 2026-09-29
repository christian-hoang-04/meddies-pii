from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    build_label_to_id,
)
from meddies_pii.eval_baseline.adapters import opf_backend
from meddies_pii.eval_baseline.adapters.lfm_bioes import LfmBioesAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator
from meddies_pii.eval_baseline.adapters.openmed import (
    MODEL_MAX_LENGTH,
    OpenMedAdapter,
)
from meddies_pii.eval_baseline.adapters.openmed import (
    MODEL_REVISION as OPENMED_MODEL_REVISION,
)
from meddies_pii.eval_baseline.adapters.opf_backend import (
    MODEL_REVISION as OPF_MODEL_REVISION,
)
from meddies_pii.eval_baseline.adapters.opf_backend import (
    apply_native_env,
    build_native_batched_predictor,
    detected_span_to_char_span,
    prepare_native_checkpoint,
)
from meddies_pii.spans import CharSpan


class _Tokenizer:
    model_max_length = 0


class _OpenMedModel:
    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str) -> _OpenMedModel:
        self.moves.append(device)
        return self


def _install_openmed_packages(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gpu: bool,
    output: object,
) -> tuple[_Tokenizer, _OpenMedModel, list[tuple[str, dict[str, object]]]]:
    tokenizer = _Tokenizer()
    model = _OpenMedModel()
    pipeline_calls: list[tuple[str, dict[str, object]]] = []

    def tokenizer_load(*args: object, **kwargs: object) -> _Tokenizer:
        assert args
        assert kwargs == {
            "revision": OPENMED_MODEL_REVISION,
            "use_fast": True,
        }
        return tokenizer

    def model_load(*args: object, **kwargs: object) -> _OpenMedModel:
        assert args
        assert kwargs == {
            "revision": OPENMED_MODEL_REVISION,
            "torch_dtype": "bf16",
        }
        return model

    def pipeline(task: str, **kwargs: object) -> object:
        pipeline_calls.append((task, kwargs))
        return lambda _texts, **_call_kwargs: output

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: gpu), bfloat16="bf16"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
            AutoModelForTokenClassification=SimpleNamespace(from_pretrained=model_load),
            pipeline=pipeline,
        ),
    )
    return tokenizer, model, pipeline_calls


@pytest.mark.parametrize(("gpu", "expected_device"), [(False, -1), (True, 0)])
# reason: pytest binds this parameter from the parametrize argnames tuple, so the boolean is labelled at every call site.
def test_openmed_load_configures_pipeline_once_and_predicts_clean_spans(
    monkeypatch: pytest.MonkeyPatch,
    gpu: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    expected_device: int,
) -> None:
    tokenizer, model, calls = _install_openmed_packages(
        monkeypatch,
        gpu=gpu,
        output=[[{"entity_group": "FIRST_NAME", "start": 0, "end": 3}]],
    )
    adapter = OpenMedAdapter(batch_size=7, stride=19)

    adapter.load()
    adapter.load()

    assert tokenizer.model_max_length == MODEL_MAX_LENGTH
    assert model.moves == (["cuda"] if gpu else [])
    assert calls == [
        (
            "token-classification",
            {
                "model": model,
                "tokenizer": tokenizer,
                "aggregation_strategy": "simple",
                "batch_size": 7,
                "device": expected_device,
            },
        ),
    ]
    assert adapter.predict(["Ana"]) == [[CharSpan(start=0, end=3, text="Ana", label="human_name")]]


@pytest.mark.parametrize(
    ("output", "error"),
    [
        ("bad-output", "unexpected openmed pipeline output"),
        ([[], []], "returned 2 outputs for 1 texts"),
    ],
)
def test_openmed_predict_rejects_bad_pipeline_shapes(monkeypatch: pytest.MonkeyPatch, output: object, error: str) -> None:
    _install_openmed_packages(monkeypatch, gpu=False, output=output)
    adapter = OpenMedAdapter()
    adapter.load()
    with pytest.raises(RuntimeError, match=error):
        adapter.predict(["Ana"])


def test_openmed_predict_requires_load() -> None:
    with pytest.raises(RuntimeError, match=r"load\(\) must be called"):
        OpenMedAdapter().predict(["Ana"])


class _Classifier:
    def __init__(self, out_features: int) -> None:
        self.out_features = out_features
        self.weight = torch.zeros(1)
        self.loaded: dict[str, torch.Tensor] | None = None

    def load_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool) -> None:
        assert strict is True
        self.loaded = state


class _LfmTagger:
    def __init__(self, label_count: int, target_label_id: int) -> None:
        self.classifier = _Classifier(label_count)
        self._parameter = torch.nn.Parameter(torch.zeros(1))
        self.was_evaluated = False
        self.label_count = label_count
        self.target_label_id = target_label_id

    def eval(self) -> None:
        self.was_evaluated = True

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._parameter

    # reason: `lfm_bioes.py:185` calls `tagger(input_ids=..., attention_mask=...)`, so both names are written at the
    # reason: call site and this double has to answer to them.
    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:  # ruff: ignore[unused-method-argument]
        logits = torch.full((1, input_ids.shape[1], self.label_count), -10.0)
        logits[0, 1, self.target_label_id] = 10.0
        return {"logits": logits}


class _LfmTokenizer:
    def __call__(self, raw: str, **kwargs: object) -> dict[str, object]:
        assert kwargs["max_length"] == 13
        return {
            "input_ids": [101, 11, 102],
            "attention_mask": [1, 1, 1],
            "offset_mapping": [(0, 0), (0, len(raw)), (0, 0)],
            "num_truncated_tokens": 1,
        }


def _install_lfm_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    label_count: int,
) -> tuple[_LfmTagger, tuple[str, ...]]:
    from meddies_pii.training.bioes.data import artifacts as artifacts_module

    vocabulary = build_bioes_label_space(ENTITY_LABELS)
    label_to_id = build_label_to_id(vocabulary)
    tagger = _LfmTagger(label_count, label_to_id["S-human_name"])

    def factory(*args: object, **kwargs: object) -> object:
        assert args
        assert "model_revision" in kwargs
        return SimpleNamespace(
            tagger=tagger,
            label_vocab=vocabulary,
            label_to_id=label_to_id,
            tokenizer=_LfmTokenizer(),
        )

    def fake_load(*_args: object, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return {"num_labels": label_count, "classifier": {"weight": torch.ones(1)}}

    monkeypatch.setattr(artifacts_module, "build_native_checkpoint_artifacts", factory)
    monkeypatch.setattr(torch, "load", fake_load)
    return tagger, vocabulary


def test_lfm_adapter_loads_fake_checkpoint_predicts_and_logs_truncation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "backbone_adapter").mkdir(parents=True)
    (checkpoint / "classifier.pt").write_bytes(b"fixture")
    tagger, vocabulary = _install_lfm_loader(monkeypatch, label_count=37)
    adapter = LfmBioesAdapter(str(checkpoint), max_seq_length=13)

    adapter.load()
    adapter.load()
    assert tagger.was_evaluated is True
    assert tagger.classifier.loaded == {"weight": torch.ones(1)}
    assert adapter.predict(["Ana"]) == [[CharSpan(start=0, end=3, text="Ana", label="human_name")]]
    assert "LFM_BIOES_TRUNCATED::docs=1 truncated=1 rate=1.000" in capsys.readouterr().out
    assert len(vocabulary) == 37


def test_lfm_adapter_load_rejects_missing_or_mismatched_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Missing adapter dir"):
        LfmBioesAdapter(str(tmp_path / "missing")).load()

    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "backbone_adapter").mkdir(parents=True)
    (checkpoint / "classifier.pt").write_bytes(b"fixture")
    _install_lfm_loader(monkeypatch, label_count=36)
    with pytest.raises(RuntimeError, match="num_labels mismatch"):
        LfmBioesAdapter(str(checkpoint)).load()
    with pytest.raises(RuntimeError, match=r"load\(\) must be called"):
        LfmBioesAdapter(str(checkpoint)).predict(["Ana"])


def test_opf_backend_applies_env_and_converts_spans() -> None:
    apply_native_env(SimpleNamespace(triton="off", compile_mode="reduce-overhead", batch_size=2))
    assert os.environ["OPF_MOE_TRITON"] == "0"
    assert os.environ["OPF_TORCH_COMPILE"] == "1"
    assert os.environ["OPF_TORCH_COMPILE_MODE"] == "reduce-overhead"
    apply_native_env(SimpleNamespace(triton="on", compile_mode="none", batch_size=2))
    assert os.environ["OPF_MOE_TRITON"] == "1"
    assert "OPF_TORCH_COMPILE" not in os.environ
    assert detected_span_to_char_span(SimpleNamespace(start="1", end="4", text=123, label="private_person")) == CharSpan(
        1,
        4,
        "123",
        "private_person",
    )


def test_opf_backend_prepares_fake_download_and_rejects_non_object_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "downloaded"

    def snapshot_download(
        *,
        repo_id: str,
        revision: str,
        local_dir: str,
        allow_patterns: list[str],
    ) -> None:
        assert repo_id == "fake/opfilter"
        assert revision == OPF_MODEL_REVISION
        assert allow_patterns == ["original/*"]
        original = Path(local_dir) / "original"
        original.mkdir()
        (original / "config.json").write_text("{}", encoding="utf-8")
        (original / "weights.bin").write_bytes(b"fixture")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    assert prepare_native_checkpoint(target, model_id="fake/opfilter") == target
    assert (target / "config.json").is_file()
    assert (target / "weights.bin").read_bytes() == b"fixture"
    assert not (target / "original").exists()

    unverified = tmp_path / "unverified"
    unverified.mkdir()
    (unverified / "config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint cache is unverified"):
        prepare_native_checkpoint(unverified)


def test_opf_backend_builds_cpu_predictor_with_fake_vendor_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    class Encoding:
        @staticmethod
        def encode(text: str, *, allowed_special: str) -> list[int]:
            assert allowed_special == "all"
            return [ord(char) for char in text]

    class Runtime:
        n_ctx = 2
        pad_token_id = 0
        device = "cpu"
        encoding = Encoding()
        label_info = SimpleNamespace(span_class_names=("O", "private_person"))

        @staticmethod
        # reason: `lfm_bioes.py:185` calls the tagger with `attention_mask=...`, so the name is written at the call site.
        def model(input_ids: torch.Tensor, *, attention_mask: torch.Tensor) -> torch.Tensor:  # ruff: ignore[unused-static-method-argument]
            logits = torch.zeros((*input_ids.shape, 2))
            logits[..., 1] = 5.0
            return logits

    def labels_to_spans(labels: dict[int, int], label_info: object) -> list[tuple[int, int, int]]:
        assert label_info is Runtime.label_info
        return [(1, min(labels), max(labels) + 1), (1, -1, 0)] if labels else []

    def decode_text_with_offsets(token_ids: tuple[int, ...], _encoding: Encoding) -> tuple[str, list[int], list[int]]:
        text = "".join(chr(token_id) for token_id in token_ids)
        return text, list(range(len(text))), list(range(1, len(text) + 1))

    runtime_module = SimpleNamespace(load_inference_runtime=lambda **_kwargs: Runtime())
    decoding_module = SimpleNamespace(build_sequence_decoder=lambda **_kwargs: (None, {}))
    spans_module = SimpleNamespace(
        labels_to_spans=labels_to_spans,
        decode_text_with_offsets=decode_text_with_offsets,
        token_spans_to_char_spans=lambda spans, starts, ends: [
            (label, start, end) if start < 0 else (label, starts[start], ends[end - 1]) for label, start, end in spans
        ],
        trim_char_spans_whitespace=lambda spans, _text: spans,
    )
    modules = {
        "opf._core.runtime": runtime_module,
        "opf._core.decoding": decoding_module,
        "opf._core.spans": spans_module,
    }
    original_import = importlib.import_module

    def fake_import(name: str) -> object:
        return modules[name] if name in modules else original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    monkeypatch.setattr(opf_backend, "prepare_native_checkpoint", lambda _: tmp_path)
    predictor = build_native_batched_predictor(
        SimpleNamespace(batch_size=2, triton="off", compile_mode="none"),
        checkpoint_dir=tmp_path,
    )
    output = predictor([
        SimpleNamespace(doc_id="one", text="Ana"),
        SimpleNamespace(doc_id="empty", text=""),
    ])
    assert output == {
        "one": [CharSpan(start=0, end=3, text="Ana", label="private_person")],
        "empty": [],
    }
