from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from google.genai import types

from meddies_pii.generation.gemini import inference

if TYPE_CHECKING:
    from pathlib import Path

    from gemini.conftest import ClientFactory


def test_get_client_requires_credentials_and_keeps_api_key_out_of_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger=inference.__name__)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        inference.get_client()

    api_key = "api-key-that-must-not-be-logged"
    created: list[dict[str, object]] = []
    sentinel = object()

    def client_factory(**kwargs: object) -> object:
        created.append(kwargs)
        return sentinel

    monkeypatch.setattr(inference.genai, "Client", client_factory)
    monkeypatch.setenv("GEMINI_API_KEY", api_key)

    assert inference.get_client() is sentinel
    assert created == [{"api_key": api_key}]
    assert api_key not in caplog.text


def test_get_client_requires_vertex_project_and_configures_vertex_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        inference.get_client(use_vertex_ai=True)

    created: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> object:
        created.append(kwargs)
        return object()

    monkeypatch.setattr(inference.genai, "Client", client_factory)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")

    inference.get_client(use_vertex_ai=True)

    assert created[0]["vertexai"] is True
    assert created[0]["project"] == "test-project"
    assert created[0]["location"] == "europe-west4"


def test_external_provider_policy_rejects_missing_or_unknown_classification() -> None:
    with pytest.raises(ValueError, match="external provider opt-in"):
        inference.require_external_provider_opt_in(
            allow_external_provider=False,
            data_classification="synthetic_public",
            operation="test",
        )
    with pytest.raises(ValueError, match="data classification"):
        inference.require_external_provider_opt_in(
            allow_external_provider=True,
            data_classification=" ",
            operation="test",
        )
    with pytest.raises(ValueError, match="unsupported data classification"):
        inference.require_external_provider_opt_in(
            allow_external_provider=True,
            data_classification="restricted",
            operation="test",
        )

    inference.require_external_provider_opt_in(
        allow_external_provider=True,
        data_classification=" Synthetic_Internal ",
        operation="test",
    )


def test_prepare_jsonl_skips_malformed_rows_without_logging_payload(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger=inference.__name__)
    source = tmp_path / "input.jsonl"
    sensitive_text = "patient-payload-that-must-not-be-logged"
    source.write_text(
        "\n".join([
            json.dumps({"text_tagged": sensitive_text}),
            "not-json",
            json.dumps(["not-an-object"]),
            json.dumps({"text_tagged": "   "}),
        ]),
        encoding="utf-8",
    )
    output = tmp_path / "requests.jsonl"

    inference.prepare_jsonl(str(source), str(output))

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["key"] for row in rows] == ["0"]
    request_text = rows[0]["request"]["contents"][0]["parts"][0]["text"]
    assert sensitive_text in request_text
    assert sensitive_text not in caplog.text


def test_prepare_jsonl_reports_missing_file_and_csv_column(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        inference.prepare_jsonl(str(tmp_path / "missing.jsonl"), str(tmp_path / "out.jsonl"))

    source = tmp_path / "input.csv"
    source.write_text("other\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Column 'translated' not found"):
        inference.prepare_jsonl(str(source), str(tmp_path / "out.jsonl"))


def test_prepare_hf_review_writes_object_rows_after_privacy_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"raw": "raw one", "label": "draft one"},
        ["not-an-object"],
        {"raw": "raw two", "label": "draft two"},
    ]
    monkeypatch.setattr(inference, "load_train_rows", lambda *_: rows)
    output = tmp_path / "review.jsonl"

    count = inference.prepare_hf_review(
        "config",
        str(output),
        allow_external_provider=True,
        data_classification="synthetic_public",
    )

    requests = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    assert [request["key"] for request in requests] == ["0", "2"]
    assert "raw one" in requests[0]["request"]["contents"][0]["parts"][0]["text"]


def test_submit_batch_uploads_local_input_and_creates_batch_without_payload_logging(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    fake_client_factory: ClientFactory,
) -> None:
    caplog.set_level("INFO", logger=inference.__name__)
    source = tmp_path / "requests.jsonl"
    sensitive_text = "payload-not-for-logs"
    source.write_text(json.dumps({"request": sensitive_text}) + "\n", encoding="utf-8")
    client, batches, files = fake_client_factory()
    files.upload_result = types.File(name="files/uploaded-input")
    batches.create_result = types.BatchJob(name="batches/created")

    job_name = inference.submit_batch_job(
        client,
        str(source),
        model="gemini-test",
        display_name_prefix="offline-test",
        allow_external_provider=True,
        data_classification="synthetic_internal",
    )

    assert job_name == "batches/created"
    assert files.upload_calls[0]["file"] == str(source)
    upload_config = files.upload_calls[0]["config"]
    assert isinstance(upload_config, types.UploadFileConfig)
    assert upload_config.mime_type == "jsonl"
    assert batches.create_calls[0]["model"] == "gemini-test"
    assert batches.create_calls[0]["src"] == "files/uploaded-input"
    assert sensitive_text not in caplog.text


def test_submit_batch_rejects_policy_before_any_client_call(
    tmp_path: Path,
    fake_client_factory: ClientFactory,
) -> None:
    source = tmp_path / "requests.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    client, batches, files = fake_client_factory()

    with pytest.raises(ValueError, match="unsupported data classification"):
        inference.submit_batch_job(
            client,
            str(source),
            allow_external_provider=True,
            data_classification="restricted",
        )

    assert files.upload_calls == []
    assert batches.create_calls == []
