from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from google.genai import types

from meddies_pii.generation.gemini import inference

if TYPE_CHECKING:
    from pathlib import Path

    from gemini.conftest import BatchJobFactory, ClientFactory


def test_download_results_parses_valid_rows_and_skips_error_rows(
    caplog: pytest.LogCaptureFixture,
    fake_client_factory: ClientFactory,
    batch_job_factory: BatchJobFactory,
) -> None:
    caplog.set_level("INFO", logger=inference.__name__)
    sensitive_result = "result-payload-that-must-not-be-logged"
    content = "\n".join([
        json.dumps({
            "key": "0",
            "response": {"candidates": [{"content": {"parts": [{"text": sensitive_result}]}}]},
        }),
        json.dumps({"key": "1", "error": {"message": "provider rejected row"}}),
        "malformed-json",
        json.dumps(["not-an-object"]),
    ])
    destination = types.BatchJobDestination(file_name="files/results")
    client, batches, files = fake_client_factory()
    batches.get_events = [batch_job_factory(dest=destination)]
    files.download_result = content.encode()

    results = inference.download_batch_results(client, "batches/test")

    assert results == {"0": sensitive_result}
    assert files.download_calls == ["files/results"]
    assert sensitive_result not in caplog.text


def test_download_results_uses_gcs_destination_when_provider_file_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    fake_client_factory: ClientFactory,
    batch_job_factory: BatchJobFactory,
) -> None:
    content = json.dumps({
        "key": "0",
        "response": {"candidates": [{"content": {"parts": [{"text": "fix"}]}}]},
    })
    destination = types.BatchJobDestination(gcs_uri="gs://test-bucket/results")
    monkeypatch.setattr(inference, "download_from_gcs", lambda _: content)
    client, batches, _ = fake_client_factory()
    batches.get_events = [batch_job_factory(dest=destination)]

    results = inference.download_batch_results(client, "batches/test")

    assert results == {"0": "fix"}


@pytest.mark.parametrize(
    ("job", "expected_refusal"),
    [
        (types.BatchJob(), "Job state is unknown"),
        # reason: the parentheses around the state name are regex grouping unless escaped, so the
        # reason: unescaped form would match a message that never names the state.
        (types.BatchJob(state=types.JobState.JOB_STATE_FAILED), r"Job not succeeded \(State: JOB_STATE_FAILED\)"),
        (types.BatchJob(state=types.JobState.JOB_STATE_SUCCEEDED), "Job destination is missing"),
        (
            types.BatchJob(
                state=types.JobState.JOB_STATE_SUCCEEDED,
                dest=types.BatchJobDestination(),
            ),
            "No result destination found",
        ),
    ],
)
def test_download_results_rejects_invalid_job_lifecycle(
    job: types.BatchJob,
    expected_refusal: str,
    fake_client_factory: ClientFactory,
) -> None:
    client, batches, _ = fake_client_factory()
    batches.get_events = [job]

    with pytest.raises(ValueError, match=expected_refusal):
        inference.download_batch_results(client, "batches/test")


def test_download_results_propagates_download_failure_and_rejects_empty_results(
    fake_client_factory: ClientFactory,
    batch_job_factory: BatchJobFactory,
) -> None:
    destination = types.BatchJobDestination(file_name="files/results")
    failing_client, failing_batches, failing_files = fake_client_factory()
    failing_batches.get_events = [batch_job_factory(dest=destination)]
    failing_files.download_error = RuntimeError("download unavailable")
    with pytest.raises(RuntimeError, match="download unavailable"):
        inference.download_batch_results(failing_client, "batches/test")

    empty_client, empty_batches, empty_files = fake_client_factory()
    empty_batches.get_events = [batch_job_factory(dest=destination)]
    empty_files.download_result = b'{"key":"1","error":"failed"}\n'
    with pytest.raises(ValueError, match="no valid responses"):
        inference.download_batch_results(empty_client, "batches/test")


def test_merge_results_preserves_bad_jsonl_rows_and_merges_json_and_csv(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "input.jsonl"
    jsonl_path.write_text('{"text":"first"}\nnot-json\n{"text":"third"}\n', encoding="utf-8")

    inference.merge_results({"0": "fix one", "2": "fix three"}, str(jsonl_path))

    merged_jsonl = (tmp_path / "input_processed.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(merged_jsonl[0])["gemini_fix"] == "fix one"
    assert merged_jsonl[1] == "not-json"
    assert json.loads(merged_jsonl[2])["gemini_fix"] == "fix three"

    json_path = tmp_path / "input.json"
    json_path.write_text('[{"text":"one"},{"text":"two"}]', encoding="utf-8")
    inference.merge_results({"1": "fix two"}, str(json_path))
    assert json.loads((tmp_path / "input_processed.json").read_text(encoding="utf-8"))[1]["gemini_fix"] == "fix two"

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("text\none\ntwo\n", encoding="utf-8")
    inference.merge_results({"0": "fix one", "not-an-index": "ignored"}, str(csv_path))
    assert "fix one" in (tmp_path / "input_processed.csv").read_text(encoding="utf-8")


def test_merge_results_rejects_missing_or_non_object_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input path not found"):
        inference.merge_results({}, str(tmp_path / "missing.jsonl"))

    input_path = tmp_path / "invalid.json"
    input_path.write_text('["not-an-object"]', encoding="utf-8")
    with pytest.raises(ValueError, match="list of objects"):
        inference.merge_results({}, str(input_path))


def test_save_results_to_json_writes_job_scoped_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    inference.save_results_to_json({"0": "fix"}, "batches/test-job")

    assert json.loads((tmp_path / "results_test-job.json").read_text(encoding="utf-8")) == {"0": "fix"}
