from __future__ import annotations

# reason: this module never spawns anything. It imports subprocess for `CalledProcessError`, which it
# reason: raises from a double, and to reach `inference.subprocess` as a monkeypatch target.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from typing import TYPE_CHECKING

import pytest
from google.genai import types

from meddies_pii.generation.gemini import inference

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from gemini.conftest import BatchJobFactory, ClientFactory
    from meddies_pii.generation.gemini.inference import GeminiClient


def test_submit_batch_uses_gcs_for_vertex_local_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_client_factory: ClientFactory,
) -> None:
    source = tmp_path / "requests.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    client, batches, files = fake_client_factory()
    monkeypatch.setattr(inference, "upload_to_gcs", lambda _: "gs://bucket/requests.jsonl")

    inference.submit_batch_job(
        client,
        str(source),
        is_vertex=True,
        allow_external_provider=True,
        data_classification="synthetic_public",
    )

    assert files.upload_calls == []
    assert batches.create_calls[0]["src"] == "gs://bucket/requests.jsonl"


def test_submit_batch_propagates_upload_and_creation_failures(
    tmp_path: Path,
    fake_client_factory: ClientFactory,
) -> None:
    source = tmp_path / "requests.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    upload_client, _, upload_files = fake_client_factory()
    upload_files.upload_error = RuntimeError("upload unavailable")

    with pytest.raises(RuntimeError, match="upload unavailable"):
        inference.submit_batch_job(
            upload_client,
            str(source),
            allow_external_provider=True,
            data_classification="synthetic_public",
        )

    create_client, create_batches, _ = fake_client_factory()
    create_batches.create_error = RuntimeError("create unavailable")
    with pytest.raises(RuntimeError, match="create unavailable"):
        inference.submit_batch_job(
            create_client,
            "files/already-uploaded",
            allow_external_provider=True,
            data_classification="synthetic_public",
        )


def test_submit_batch_rejects_missing_uploaded_file_or_job_name(
    tmp_path: Path,
    fake_client_factory: ClientFactory,
) -> None:
    source = tmp_path / "requests.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    missing_file_client, _, missing_file_files = fake_client_factory()
    missing_file_files.upload_result = types.File()

    with pytest.raises(ValueError, match="file name is missing"):
        inference.submit_batch_job(
            missing_file_client,
            str(source),
            allow_external_provider=True,
            data_classification="synthetic_public",
        )

    missing_job_client, missing_job_batches, _ = fake_client_factory()
    missing_job_batches.create_result = types.BatchJob()
    with pytest.raises(ValueError, match="Job name is missing"):
        inference.submit_batch_job(
            missing_job_client,
            "files/already-uploaded",
            allow_external_provider=True,
            data_classification="synthetic_public",
        )


def test_gcs_helpers_use_local_subprocess_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(inference.time, "time", lambda: 123)
    monkeypatch.setattr(
        inference.subprocess,
        "check_call",
        commands.append,
    )

    assert inference.upload_to_gcs("requests.jsonl", "test-bucket") == ("gs://test-bucket/batch_inputs/123_requests.jsonl")
    assert commands == [
        [
            "gcloud",
            "storage",
            "cp",
            "requests.jsonl",
            "gs://test-bucket/batch_inputs/123_requests.jsonl",
        ],
    ]

    # reason: the code under test calls `subprocess.check_output(cmd, text=True)`, so `text` is written at the call
    # reason: site and is part of the signature this stand-in replaces.
    def check_output(command: list[str], *, text: bool) -> str:  # ruff: ignore[unused-function-argument]
        commands.append(command)
        if command[-1].endswith("/**.jsonl"):
            return "gs://test-bucket/results/a.jsonl\n"
        if command[-2] == "cat":
            return '{"key":"0"}\n'
        msg = f"unexpected command: {command}"
        raise AssertionError(msg)

    monkeypatch.setattr(inference.subprocess, "check_output", check_output)
    assert inference.download_from_gcs("gs://test-bucket/results") == '{"key":"0"}\n\n'
    assert inference.download_from_gcs("gs://test-bucket/results/a.jsonl") == '{"key":"0"}\n'


def test_gcs_helpers_propagate_subprocess_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = subprocess.CalledProcessError(1, ["gcloud"])
    monkeypatch.setattr(
        inference.subprocess,
        "check_call",
        lambda _: (_ for _ in ()).throw(error),
    )
    with pytest.raises(subprocess.CalledProcessError):
        inference.upload_to_gcs("requests.jsonl")

    monkeypatch.setattr(
        inference.subprocess,
        "check_output",
        lambda *_, **__: (_ for _ in ()).throw(error),
    )
    with pytest.raises(subprocess.CalledProcessError):
        inference.download_from_gcs("gs://test-bucket/results.jsonl")


def test_list_jobs_prints_limit_and_propagates_provider_failure(
    capsys: pytest.CaptureFixture[str],
    fake_client_factory: ClientFactory,
    batch_job_factory: BatchJobFactory,
) -> None:
    client, batches, _ = fake_client_factory()
    batches.listed_jobs = [
        batch_job_factory(name="batches/one"),
        batch_job_factory(name="batches/two", state_name="JOB_STATE_FAILED"),
    ]

    inference.list_jobs(client, limit=1)

    output = capsys.readouterr().out
    assert "batches/one" in output
    assert "batches/two" not in output
    list_config = batches.list_calls[0]
    assert isinstance(list_config, types.ListBatchJobsConfig)
    assert list_config.page_size == 1

    failing_client, failing_batches, _ = fake_client_factory()
    failing_batches.list_error = RuntimeError("list unavailable")
    with pytest.raises(RuntimeError, match="list unavailable"):
        inference.list_jobs(failing_client)


def test_monitor_job_retries_then_returns_succeeded_terminal_state(
    fake_client_factory: ClientFactory,
    batch_job_factory: BatchJobFactory,
) -> None:
    succeeded = batch_job_factory()
    client, batches, _ = fake_client_factory()
    batches.get_events = [
        RuntimeError("temporary provider failure"),
        batch_job_factory(state_name="JOB_STATE_RUNNING"),
        succeeded,
    ]

    result = inference.monitor_job(client, "batches/test", poll_interval=0)

    assert result is succeeded
    assert batches.get_calls == ["batches/test", "batches/test", "batches/test"]


def test_monitor_job_raises_for_terminal_failure_and_retry_exhaustion(
    fake_client_factory: ClientFactory,
    batch_job_factory: BatchJobFactory,
) -> None:
    failing_client, failing_batches, _ = fake_client_factory()
    failing_batches.get_events = [
        batch_job_factory(
            state_name="JOB_STATE_FAILED",
            error=types.JobError(message="provider rejected batch"),
        ),
    ]
    with pytest.raises(RuntimeError, match="provider rejected batch"):
        inference.monitor_job(failing_client, "batches/test", poll_interval=0)

    exhausted_client, exhausted_batches, _ = fake_client_factory()
    exhausted_batches.get_events = [
        RuntimeError("unavailable"),
        RuntimeError("unavailable"),
    ]
    with pytest.raises(RuntimeError, match="unavailable"):
        inference.monitor_job(
            exhausted_client,
            "batches/test",
            poll_interval=0,
            max_errors=2,
        )


@pytest.mark.parametrize(
    ("operation", "error_message", "calls_attribute"),
    [
        (inference.cancel_job, "cancel unavailable", "cancel_calls"),
        (inference.delete_job, "delete unavailable", "delete_calls"),
    ],
)
def test_cancel_and_delete_call_provider_and_propagate_failure(
    operation: Callable[[GeminiClient, str], None],
    error_message: str,
    calls_attribute: str,
    fake_client_factory: ClientFactory,
) -> None:
    client, batches, _ = fake_client_factory()

    operation(client, "batches/test")
    assert getattr(batches, calls_attribute) == ["batches/test"]

    failing_client, failing_batches, _ = fake_client_factory()
    if calls_attribute == "cancel_calls":
        failing_batches.cancel_error = RuntimeError(error_message)
    else:
        failing_batches.delete_error = RuntimeError(error_message)
    with pytest.raises(RuntimeError, match=error_message):
        operation(failing_client, "batches/test")
