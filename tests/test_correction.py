from __future__ import annotations

import asyncio

import pytest

from meddies_pii.generation import correction
from meddies_pii.generation.correction import CorrectionRequest, DataCorrector


def test_extract_text_from_provider_candidate() -> None:
    text = DataCorrector.extract_text_from_response({
        "response": {"candidates": [{"content": {"parts": [{"text": "Tagged note"}]}}]},
    })

    assert text == "Tagged note"


def test_extract_text_rejects_non_string_provider_text() -> None:
    text = DataCorrector.extract_text_from_response({"response": {"candidates": [{"content": {"parts": [{"text": 42}]}}]}})

    # reason: the provider handed back the integer 42, and the test name says the extractor REJECTS
    # reason: it. The rule's `not text` would pass on the `0`, `None` or `42` a broken extractor could
    # reason: return, so the string comparison is the whole assertion.
    assert text == ""  # ruff: ignore[compare-to-empty-string]


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    # reason: `correction.py:146` calls `chat(SYSTEM_PROMPT, text, temperature=0.7, max_tokens=8192)`, so these two
    # reason: names are written at the call site.
    async def chat(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:  # ruff: ignore[unused-method-argument]
        self.calls.append((system, user))
        return "Corrected note"


def test_stream_skips_non_object_row_without_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RecordingClient()
    corrector = DataCorrector.__new__(DataCorrector)
    monkeypatch.setattr(corrector, "_client", client, raising=False)
    monkeypatch.setattr(
        correction,
        "load_streaming_train",
        lambda _repo: [object(), {"response": "Source note"}],
    )

    outcome = asyncio.run(corrector.stream_and_correct("test-repo", limit=2))

    assert outcome.skipped_count == 1
    assert outcome.successful_count == 1
    assert client.calls == [(correction.SYSTEM_PROMPT, "Source note")]


@pytest.mark.parametrize(
    ("repo", "limit", "output", "message"),
    [
        ("", 1, "corrected.jsonl", "repo must not be blank"),
        ("Meddies/vie-pii", 0, "corrected.jsonl", "limit must be positive"),
        ("Meddies/vie-pii", 1, "", "output must not be blank"),
    ],
)
def test_correction_request_rejects_invalid_cli_values(repo: str, limit: int, output: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CorrectionRequest(repo=repo, limit=limit, output=output)


def test_correction_request_is_frozen_value_object() -> None:
    request = CorrectionRequest(repo="Meddies/vie-pii", limit=3, output="corrected.jsonl")

    with pytest.raises(AttributeError):
        # reason: the rejected write is the behaviour under test — `limit` is a read-only property and
        # reason: the assertion proves it stays one; admitting the write would delete the test.
        request.limit = 4  # ty: ignore[invalid-assignment]
