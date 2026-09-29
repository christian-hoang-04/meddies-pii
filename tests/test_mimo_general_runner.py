from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from meddies_pii.generation.label_corpus import runner as generation_runner

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Self

    import pytest


def _load_runner() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts/generation/run_mimo_general_20k.py"
    spec = importlib.util.spec_from_file_location("run_mimo_general_20k_for_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provider_keys_reads_first_non_empty_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    monkeypatch.setenv("MIMO_API_KEYS", " key-a, key-b ,, key-c ")
    monkeypatch.setenv("MIMO_API_KEY", "fallback-key")

    assert runner.provider_keys("mimo") == ["key-a", "key-b", "key-c"]


def test_configured_base_urls_reads_provider_urls_without_key_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_runner()
    monkeypatch.setenv("MIMO_BASE_URLS", "https://ams.example/v1,https://sgp.example/v1")

    assert generation_runner.configured_base_urls("mimo", 2) == [
        "https://ams.example/v1",
        "https://sgp.example/v1",
    ]


def test_accepted_count_prefers_jsonl_over_stale_summary(tmp_path: Path) -> None:
    _load_runner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "accepted.vi.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    (output_dir / "summary.vi.json").write_text('{"accepted_count": 1}\n', encoding="utf-8")

    assert generation_runner.accepted_count(output_dir, "Vietnamese") == 2


def test_preflight_keys_reports_bad_index_without_key_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _load_runner()

    class FakeClient:
        def __init__(self, key: str) -> None:
            self.key = key

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def chat(self, *_args: object, **_kwargs: object) -> str:
            if self.key == "bad-secret-value":
                request = httpx.Request("POST", "https://token-plan-ams.xiaomimimo.com/v1/chat/completions")
                response = httpx.Response(401, request=request)
                msg = "401 Unauthorized"
                raise httpx.HTTPStatusError(msg, request=request, response=response)
            return "OK"

    def fake_for_provider(_provider: str, api_keys: list[str], **_kwargs: object) -> FakeClient:
        return FakeClient(api_keys[0])

    monkeypatch.setattr(
        generation_runner.OpenAICompatibleClient,
        "for_provider",
        staticmethod(fake_for_provider),
    )

    results = asyncio.run(
        generation_runner.preflight_keys(
            provider="mimo",
            model="mimo-v2.5-pro",
            keys=["good-secret-value", "bad-secret-value"],
            base_urls=["https://ams.example/v1", "https://sgp.example/v1"],
        ),
    )

    assert [(result.key_index, result.ok, result.status) for result in results] == [
        (1, True, 200),
        (2, False, 401),
    ]
    failed_error = results[1].error
    assert failed_error is not None
    assert "bad-secret-value" not in failed_error
