"""Test inference without leaking logging changes across the test process.

Module-level logging.disable would execute at collection time and silence logging for every test in the process; scope
the silencing to this module.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from meddies_pii.generation.gemini import inference
from meddies_pii.generation.gemini.inference import (
    download_batch_results,
    merge_results,
    prepare_jsonl,
)


@pytest.fixture(autouse=True)
def _silence_logging() -> object:
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def test_prepare_jsonl(tmp_path: Path) -> None:
    mock_df = pd.DataFrame({"translated": ["Hello World", "Xin chào"]})
    output_path = tmp_path / "output.jsonl"

    with patch("pandas.read_csv", return_value=mock_df):
        prepare_jsonl("dummy.csv", str(output_path))

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["key"] == "0"
    assert "Hello World" in first["request"]["contents"][0]["parts"][0]["text"]

    second = json.loads(lines[1])
    assert second["key"] == "1"
    assert "Xin chào" in second["request"]["contents"][0]["parts"][0]["text"]


def test_prepare_jsonl_missing_column() -> None:
    mock_df = pd.DataFrame({"other_col": [1, 2]})

    with (
        patch("pandas.read_csv", return_value=mock_df),
        pytest.raises(ValueError, match="Column 'translated' not found"),
    ):
        prepare_jsonl("dummy.csv", "output.jsonl")


def test_download_batch_results_success() -> None:
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    mock_job.dest.file_name = "results.jsonl"
    mock_client.batches.get.return_value = mock_job

    sample_jsonl = (
        b'{"key": "0", "response": {"candidates": [{"content": {"parts": [{"text": "Redacted Text"}]}}]}}\n'
        b'{"key": "1", "error": {"code": 400, "message": "Bad Request"}}\n'
    )

    mock_client.files.download.return_value = sample_jsonl

    results = download_batch_results(mock_client, "dumm_job_name")

    assert len(results) == 1
    assert results["0"] == "Redacted Text"


def test_download_batch_results_not_succeeded() -> None:
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_RUNNING"
    mock_client.batches.get.return_value = mock_job

    with pytest.raises(ValueError, match="Job not succeeded"):
        download_batch_results(mock_client, "job_name")
    mock_client.files.download.assert_not_called()


def test_merge_results_to_csv() -> None:
    mock_df = pd.DataFrame({"translated": ["Original 1", "Original 2"], "gemini_fix": ["", ""]})

    results = {"0": "Fixed 1"}

    with (
        patch("os.path.exists", return_value=True),
        patch("pandas.read_csv", return_value=mock_df),
        patch.object(mock_df, "to_csv") as mock_to_csv,
    ):
        merge_results(results, "dummy.csv")

        assert mock_df.loc[0, "gemini_fix"] == "Fixed 1"
        # reason: a DataFrame cell with no result can be `NaN`, `None` or `""`, and the claim here
        # reason: is specifically the empty string. The rule's `not ...` would accept `None`, and
        # reason: would REJECT the `NaN` pandas actually uses for a missing cell.
        assert mock_df.loc[1, "gemini_fix"] == ""  # ruff: ignore[compare-to-empty-string]

        mock_to_csv.assert_called_once()
        args, _ = mock_to_csv.call_args
        assert args[0] == "dummy_processed.csv"


def test_prepare_hf_review_counts_only_accepted_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output_path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(
        inference,
        "load_train_rows",
        lambda _repo, _config: [
            {"raw": "Source note", "label": "[A]<human_name>"},
            object(),
        ],
    )

    with caplog.at_level(logging.INFO):
        count = inference.prepare_hf_review(
            "config",
            str(output_path),
            allow_external_provider=True,
            data_classification="synthetic_public",
        )

    assert count == 1
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1
    assert "Wrote 1 requests" in caplog.text


def test_get_client_constructs_ai_studio_client_without_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-ai-studio-key")
    factory = MagicMock()
    monkeypatch.setattr(inference.genai, "Client", factory)

    client = inference.get_client()

    assert client is factory.return_value
    factory.assert_called_once_with(api_key="test-ai-studio-key")


def test_get_client_constructs_vertex_client_without_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "meddies-test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")
    factory = MagicMock()
    monkeypatch.setattr(inference.genai, "Client", factory)

    client = inference.get_client(use_vertex_ai=True)

    assert client is factory.return_value
    factory.assert_called_once()
    kwargs = factory.call_args.kwargs
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "meddies-test-project"
    assert kwargs["location"] == "asia-southeast1"
    assert kwargs["http_options"].api_version == "v1"
