from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import asyncio
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Self, cast
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from meddies_pii.generation.label_corpus.runner import SyntheticGenerationRequest
    from meddies_pii.generation.openai_compatible.client import OpenAICompatibleClient


def test_pii_labels_is_public_constant() -> None:
    from meddies_pii.constants import PII_LABELS

    assert PII_LABELS == (
        "address",
        "company_name",
        "date",
        "email_address",
        "human_name",
        "id_number",
        "phone_number",
        "private_url",
        "secret",
    )


def test_labels_command_prints_current_labels(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from meddies_pii.cli import main
    from meddies_pii.constants import PII_LABELS

    monkeypatch.setattr(sys, "argv", ["meddies-pii", "labels"])

    main()

    output = capsys.readouterr().out
    assert "Meddies Labels (9)" in output
    for label in PII_LABELS:
        assert f"- {label}" in output


def test_validate_command_accepts_bundled_inline_sample(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from meddies_pii.cli import main

    sample_path = Path("examples/sample.inline.jsonl")
    assert sample_path.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        ["meddies-pii", "validate", str(sample_path)],
    )

    main()

    output = capsys.readouterr().out
    assert "OK: 3 records" in output
    assert "9 spans" in output
    assert "9 labels" in output


def test_demo_command_writes_local_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from meddies_pii.cli import main

    output_dir = tmp_path / "demo-output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["meddies-pii", "demo", "--output-dir", str(output_dir)],
    )

    main()

    output = capsys.readouterr().out
    assert "OK: 3 records" in output
    assert (output_dir / "plain.jsonl").exists()
    assert (output_dir / "spans.jsonl").exists()
    assert (output_dir / "report.json").exists()


def test_validate_command_rejects_unknown_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from meddies_pii.cli import main

    sample_path = tmp_path / "bad.inline.jsonl"
    sample_path.write_text(
        '{"content":"Patient [Ada]<not_a_label>"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["meddies-pii", "validate", str(sample_path)],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert "unknown PII label" in capsys.readouterr().err


def _assert_operation_failure(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> SystemExit:
    from meddies_pii.cli import main

    monkeypatch.setattr(sys, "argv", ["meddies-pii", *argv])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1
    return raised.value


def test_generate_unknown_provider_exits_nonzero_without_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output_dir = tmp_path / "synthetic"
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(
        monkeypatch,
        [
            "generate",
            "--provider",
            "missing-provider",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert "Unknown generation provider: missing-provider" in caplog.text
    assert not output_dir.exists()


def test_generate_missing_provider_configuration_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii.exceptions import ConfigurationError
    from meddies_pii.generation.label_corpus import runner

    # reason: this stands in for `runner.run_synthetic_generation`, which `cli.py` drives through `asyncio.run` and
    # reason: `generation_runs.py:130` awaits, so dropping `async` would hand both a value that cannot be awaited.
    async def fail_missing_key(*_args: object, **_kwargs: object) -> None:  # ruff: ignore[unused-async]
        msg = "No OpenAI-compatible provider keys configured"
        raise ConfigurationError(msg)

    output_dir = tmp_path / "synthetic"
    monkeypatch.setattr(runner, "run_synthetic_generation", fail_missing_key)
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(
        monkeypatch,
        ["generate", "--output-dir", str(output_dir)],
    )

    assert "No OpenAI-compatible provider keys configured" in caplog.text
    assert not output_dir.exists()


def test_generate_aggregates_per_language_failures_before_exiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii import cli
    from meddies_pii.exceptions import MeddiesException
    from meddies_pii.generation.label_corpus import runner

    attempted_languages: list[str] = []

    # reason: this stands in for `runner.run_synthetic_generation`, which `cli.py` drives through `asyncio.run` and
    # reason: `generation_runs.py:130` awaits, so dropping `async` would hand both a value that cannot be awaited.
    async def fail_one_language(request: SyntheticGenerationRequest) -> dict[str, int]:  # ruff: ignore[unused-async]
        language = request.language
        attempted_languages.append(language)
        if language == "Vietnamese":
            msg = "provider refused Vietnamese"
            raise MeddiesException(msg)
        return {"accepted_count": 1}

    output_dir = tmp_path / "synthetic"
    monkeypatch.setattr(cli, "SUPPORTED_LANGUAGES", ("Vietnamese", "English"))
    monkeypatch.setattr(runner, "run_synthetic_generation", fail_one_language)
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(
        monkeypatch,
        ["generate", "--language", "all", "--output-dir", str(output_dir)],
    )

    assert attempted_languages == ["Vietnamese", "English"]
    assert "1 succeeded (English)" in caplog.text
    assert "1 failed (Vietnamese)" in caplog.text
    assert not output_dir.exists()


def test_generate_partial_failure_retains_successful_language_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii import cli
    from meddies_pii.exceptions import MeddiesException
    from meddies_pii.generation.label_corpus import runner

    output_dir = tmp_path / "synthetic"
    output_dir.mkdir()
    existing_marker = output_dir / "keep-me.txt"
    existing_marker.write_text("pre-existing", encoding="utf-8")

    # reason: this stands in for `runner.run_synthetic_generation`, which `cli.py` drives through `asyncio.run` and
    # reason: `generation_runs.py:130` awaits, so dropping `async` would hand both a value that cannot be awaited.
    async def write_then_fail(request: SyntheticGenerationRequest) -> dict[str, int]:  # ruff: ignore[unused-async]
        # reason: the two writes below are what this double exists to do — leave a partial artifact on
        # reason: disk so the test can assert the CLI's cleanup. They are two small files under
        # reason: `tmp_path` on a single-test event loop, so there is no concurrency for them to block.
        if request.language == "Vietnamese":
            Path(request.output_dir, "accepted.vi.jsonl").write_text('{"generated": true}\n', encoding="utf-8")  # ruff: ignore[blocking-path-method-in-async-function]
            return {"accepted_count": 1}
        Path(request.output_dir, "accepted.en.jsonl").write_text('{"partial": true}\n', encoding="utf-8")  # ruff: ignore[blocking-path-method-in-async-function]
        msg = "provider refused English"
        raise MeddiesException(msg)

    monkeypatch.setattr(cli, "SUPPORTED_LANGUAGES", ("Vietnamese", "English"))
    monkeypatch.setattr(runner, "run_synthetic_generation", write_then_fail)
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(
        monkeypatch,
        ["generate", "--language", "all", "--output-dir", str(output_dir)],
    )

    assert existing_marker.read_text(encoding="utf-8") == "pre-existing"
    assert (output_dir / "accepted.vi.jsonl").read_text(encoding="utf-8") == ('{"generated": true}\n')
    assert (output_dir / "accepted.en.jsonl").read_text(encoding="utf-8") == ('{"partial": true}\n')
    assert "1 succeeded (Vietnamese)" in caplog.text
    assert "1 failed (English)" in caplog.text
    assert "partial failed-language progress" in caplog.text


def test_inference_prepare_missing_input_exits_nonzero_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing_input = tmp_path / "missing.jsonl"
    output_path = tmp_path / "requests.jsonl"
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(
        monkeypatch,
        [
            "inference",
            "prepare",
            "--input",
            str(missing_input),
            "--output",
            str(output_path),
        ],
    )

    assert f"Input file not found: {missing_input}" in caplog.text
    assert not output_path.exists()


# reason: Fixtures plus argv, operation, and message are case inputs for provider failures and the no-artifact assertion.
@pytest.mark.parametrize(
    ("argv", "operation", "message"),
    [
        (["inference", "list"], "list", "cannot list jobs"),
        (
            ["inference", "cancel", "--job-name", "jobs/123"],
            "cancel",
            "cannot cancel job",
        ),
        (
            ["inference", "delete", "--job-name", "jobs/123"],
            "delete",
            "cannot delete job",
        ),
        (
            ["inference", "download", "--job-name", "jobs/123"],
            "download",
            "cannot download results",
        ),
    ],
)
def test_inference_provider_operation_failure_exits_nonzero(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    argv: list[str],
    operation: str,
    message: str,
) -> None:
    from meddies_pii import cli

    client = MagicMock()
    if operation == "list":
        client.batches.list.side_effect = RuntimeError(message)
    elif operation == "cancel":
        client.batches.cancel.side_effect = RuntimeError(message)
    elif operation == "delete":
        client.batches.delete.side_effect = RuntimeError(message)
    else:
        client.batches.get.side_effect = RuntimeError(message)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "get_client", lambda **_kwargs: client)
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(monkeypatch, argv)

    assert message in caplog.text
    assert not list(tmp_path.iterdir())


def test_inference_download_refuses_missing_merge_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii import cli

    client = MagicMock()
    client.batches.get.return_value.state.name = "JOB_STATE_SUCCEEDED"
    client.batches.get.return_value.dest.file_name = "results.jsonl"
    client.files.download.return_value = (
        b'{"key":"0","response":{"candidates":[{"content":{"parts":[{"text":"result"}]}}]}}\n'
    )
    missing_input = tmp_path / "missing.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "get_client", lambda **_kwargs: client)
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(
        monkeypatch,
        [
            "inference",
            "download",
            "--job-name",
            "jobs/123",
            "--merge-input",
            str(missing_input),
        ],
    )

    assert f"Input path not found: {missing_input}" in caplog.text
    assert not list(tmp_path.iterdir())


def test_inference_download_all_error_rows_exits_nonzero_without_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii import cli

    client = MagicMock()
    client.batches.get.return_value.state.name = "JOB_STATE_SUCCEEDED"
    client.batches.get.return_value.dest.file_name = "results.jsonl"
    client.files.download.return_value = b'{"key":"0","error":{"message":"bad request"}}\n'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "get_client", lambda **_kwargs: client)
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(
        monkeypatch,
        ["inference", "download", "--job-name", "jobs/123"],
    )

    assert "Batch results contained no valid responses" in caplog.text
    assert not list(tmp_path.iterdir())


def test_inference_monitor_failed_job_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii import cli

    client = MagicMock()
    client.batches.get.return_value.state.name = "JOB_STATE_FAILED"
    client.batches.get.return_value.error = "provider refused input"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "get_client", lambda **_kwargs: client)
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(
        monkeypatch,
        ["inference", "monitor", "--job-name", "jobs/123"],
    )

    assert "Batch job did not succeed (state: JOB_STATE_FAILED)" in caplog.text
    assert "provider refused input" in caplog.text
    assert not list(tmp_path.iterdir())


def test_inference_missing_provider_configuration_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii import cli

    monkeypatch.setattr(
        cli,
        "get_client",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("GEMINI_API_KEY is required")),
    )
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(monkeypatch, ["inference", "list"])

    assert "GEMINI_API_KEY is required" in caplog.text


def test_correct_command_passes_validated_request_to_correction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from meddies_pii.generation import correction
    from meddies_pii.generation.correction import CorrectionRequest

    received: list[CorrectionRequest] = []

    # reason: this stands in for `correction.correct_hf_data`, which `cli.py:225` drives through `asyncio.run`, so
    # reason: dropping `async` would hand it a value that cannot be awaited.
    async def record_request(request: CorrectionRequest) -> None:  # ruff: ignore[unused-async]
        received.append(request)

    output_path = tmp_path / "corrected.jsonl"
    monkeypatch.setattr(correction, "correct_hf_data", record_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "meddies-pii",
            "correct",
            "--repo",
            "Meddies/custom",
            "--limit",
            "3",
            "--output",
            str(output_path),
        ],
    )

    from meddies_pii.cli import main

    main()

    assert received == [CorrectionRequest(repo="Meddies/custom", limit=3, output=str(output_path))]


def test_correct_command_rejects_nonpositive_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output_path = tmp_path / "corrected.jsonl"
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(
        monkeypatch,
        ["correct", "--limit", "0", "--output", str(output_path)],
    )

    assert "limit must be positive" in caplog.text
    assert not output_path.exists()


def test_correction_failure_exits_nonzero_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii.generation import correction

    # reason: this stands in for `correction.correct_hf_data`, which `cli.py:225` drives through `asyncio.run`, so
    # reason: dropping `async` would hand it a value that cannot be awaited.
    async def fail_correction(*_args: object, **_kwargs: object) -> None:  # ruff: ignore[unused-async]
        msg = "provider rejected correction"
        raise RuntimeError(msg)

    output_path = tmp_path / "corrected.jsonl"
    monkeypatch.setattr(correction, "correct_hf_data", fail_correction)
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(monkeypatch, ["correct", "--output", str(output_path)])

    assert "provider rejected correction" in caplog.text
    assert not output_path.exists()


class _FakeCorrector:
    def __init__(self, outcome: object) -> None:
        self._outcome = outcome

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def stream_and_correct(self, **_kwargs: object) -> object:
        return self._outcome

    @staticmethod
    def save_results(records: list[dict[str, object]], output_path: str) -> None:
        Path(output_path).write_text(json.dumps(records), encoding="utf-8")


def test_correction_all_row_failures_persists_diagnostics_then_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii.generation import correction

    output_path = tmp_path / "corrected.jsonl"
    outcome = SimpleNamespace(
        records=[{"error": "provider refused row"}],
        used_real_data=True,
        successful_count=0,
        failed_count=1,
        skipped_count=0,
    )
    monkeypatch.setattr(correction, "DataCorrector", lambda: _FakeCorrector(outcome))
    caplog.set_level(logging.ERROR)

    _assert_operation_failure(monkeypatch, ["correct", "--output", str(output_path)])

    assert json.loads(output_path.read_text(encoding="utf-8")) == outcome.records
    assert "0 succeeded, 1 failed, 0 skipped" in caplog.text


def test_correction_mixed_row_results_succeeds_with_failure_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii import cli
    from meddies_pii.generation import correction

    output_path = tmp_path / "corrected.jsonl"
    outcome = SimpleNamespace(
        records=[{"corrected_text": "fixed"}, {"error": "provider refused row"}],
        used_real_data=True,
        successful_count=1,
        failed_count=1,
        skipped_count=0,
    )
    monkeypatch.setattr(correction, "DataCorrector", lambda: _FakeCorrector(outcome))
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(sys, "argv", ["meddies-pii", "correct", "--output", str(output_path)])

    cli.main()

    assert json.loads(output_path.read_text(encoding="utf-8")) == outcome.records
    assert "1 succeeded, 1 failed, 0 skipped" in caplog.text


def test_correction_outcome_retains_mixed_row_failure_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meddies_pii.generation.correction import DataCorrector

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, *_args: object, **_kwargs: object) -> str:
            self.calls += 1
            if self.calls == 1:
                return "corrected first row"
            msg = "provider refused second row"
            raise RuntimeError(msg)

    datasets_module = ModuleType("datasets")
    # reason: the code under test imports `datasets` by name out of `sys.modules`, so the fake must be a
    # reason: real module; `ModuleType` declares no `load_dataset`, and no annotation admits the write.
    datasets_module.load_dataset = lambda *_args, **_kwargs: {  # ty: ignore[unresolved-attribute]
        "train": [{"response": "first row"}, {"response": "second row"}],
    }
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    corrector = object.__new__(DataCorrector)
    # reason: the double exposes only the `chat` member `stream_and_correct` calls; test-side
    # reason: narrowing is the owner-ruled shape for a stand-in that cannot be a real client.
    corrector._client = cast("OpenAICompatibleClient", FakeClient())
    outcome = asyncio.run(corrector.stream_and_correct(repo="test", limit=2))

    assert outcome.successful_count == 1
    assert outcome.failed_count == 1
    assert outcome.skipped_count == 0
    assert outcome.records[0]["corrected_text"] == "corrected first row"
    assert outcome.records[1]["error"] == "provider refused second row"


def test_successful_inference_operation_keeps_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meddies_pii import cli

    called = False

    def list_success(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "get_client", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "list_jobs", list_success)
    monkeypatch.setattr(sys, "argv", ["meddies-pii", "inference", "list"])

    cli.main()

    assert called
