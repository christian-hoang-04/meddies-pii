from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from meddies_pii import translation

if TYPE_CHECKING:
    from pathlib import Path


class _Tokenizer:
    def __init__(self) -> None:
        self.conversations: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        self.conversations.append(conversation)
        return f"prompt:{conversation[-1]['content']}"


class _Model:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.prompts: list[str] = []

    def get_tokenizer(self) -> _Tokenizer:
        return self.tokenizer

    def generate(self, prompts: list[str], *, sampling_params: object, use_tqdm: bool) -> list[SimpleNamespace]:
        assert sampling_params == {
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 0.8,
        }
        assert use_tqdm
        self.prompts = prompts
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=f"<think>hidden</think> vi:{prompt}")]) for prompt in prompts
        ]


def test_translate_texts_uses_the_validated_vllm_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _Tokenizer()
    model = _Model(tokenizer)
    model_options: list[dict[str, object]] = []

    def make_model(**options: object) -> _Model:
        model_options.append(options)
        return model

    fake_vllm = SimpleNamespace(
        LLM=make_model,
        SamplingParams=lambda **options: options,
    )
    monkeypatch.setattr(translation.importlib, "import_module", lambda _name: fake_vllm)

    outputs = translation.translate_texts(
        ["first", "second"],
        model_id="local/model",
        max_model_len=2048,
        temperature=0.2,
        max_tokens=512,
        top_p=0.8,
    )

    assert model_options == [
        {
            "model": "local/model",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.95,
            "max_model_len": 2048,
            "enable_prefix_caching": True,
        },
    ]
    assert model.prompts == ["prompt:first", "prompt:second"]
    assert [conversation[-1]["content"] for conversation in tokenizer.conversations] == [
        "first",
        "second",
    ]
    assert outputs == ["vi:prompt:first", "vi:prompt:second"]


def test_translate_texts_rejects_an_incompatible_vllm_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        translation.importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="expected LLM API"):
        translation.translate_texts(["text"])


def test_translate_file_preserves_source_records_at_the_file_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join([
            json.dumps({"text_tagged": "[Ada]<human_name>"}),
            json.dumps(["not", "an", "object"]),
            json.dumps({"text_tagged": 3}),
            "",
        ]),
        encoding="utf-8",
    )
    received: list[list[str]] = []

    def translate(texts: list[str], *, model_id: str) -> list[str]:
        assert model_id == "local/model"
        received.append(texts)
        return [f"vi:{text}" for text in texts]

    monkeypatch.setattr(translation, "translate_texts", translate)
    output_dir = tmp_path / "translated"

    translation.translate_file(input_path, output_dir, model_id="local/model")

    assert received == [["[Ada]<human_name>", "", ""]]
    assert [json.loads(path.read_text(encoding="utf-8")) for path in sorted(output_dir.glob("*.json"))] == [
        {"text_tagged": "[Ada]<human_name>", "translated": "vi:[Ada]<human_name>"},
        {"text_tagged": "", "translated": "vi:"},
        {"text_tagged": "", "translated": "vi:"},
    ]


def test_translate_file_accepts_an_empty_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("\n\n", encoding="utf-8")
    output_dir = tmp_path / "translated"

    def unexpected_call(*_args: object, **_kwargs: object) -> list[str]:
        msg = "empty input must not initialize the model"
        raise AssertionError(msg)

    monkeypatch.setattr(translation, "translate_texts", unexpected_call)

    translation.translate_file(input_path, output_dir)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
