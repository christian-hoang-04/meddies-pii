from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import logging
import runpy
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from meddies_pii import cli

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.generation.label_corpus.runner import SyntheticGenerationRequest


def _run_cli(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["meddies-pii", *arguments])
    cli.main()


def _run_cli_for_exit(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int | str | None:
    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments)
    return raised.value.code


def test_inference_prepare_interrupt_returns_shell_interrupt_status(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "prepare_jsonl", interrupt)
    caplog.set_level(logging.WARNING)

    exit_code = _run_cli_for_exit(
        monkeypatch,
        "inference",
        "prepare",
        "--input",
        "input.jsonl",
    )

    assert exit_code == cli.INTERRUPTED_EXIT_CODE
    assert "Inference inf-prepare interrupted by user" in caplog.text


def test_generate_success_passes_only_named_scenarios(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from meddies_pii.generation.label_corpus import runner

    received: list[SyntheticGenerationRequest] = []

    # reason: this stands in for `runner.run_synthetic_generation`, which `cli.py` drives through `asyncio.run` and
    # reason: `generation_runs.py:130` awaits, so dropping `async` would hand both a value that cannot be awaited.
    async def generate(request: SyntheticGenerationRequest) -> dict[str, int]:  # ruff: ignore[unused-async]
        received.append(request)
        return {"accepted_count": 2}

    monkeypatch.setattr(runner, "run_synthetic_generation", generate)

    _run_cli(
        monkeypatch,
        "generate",
        "--count",
        "2",
        "--scenarios",
        " discharge, ,follow-up ",
        "--output-dir",
        str(tmp_path),
    )

    assert len(received) == 1
    assert received[0].scenario_names == ("discharge", "follow-up")


def test_generate_aggregates_an_unexpected_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii.generation.label_corpus import runner

    # reason: this stands in for `runner.run_synthetic_generation`, which `cli.py` drives through `asyncio.run` and
    # reason: `generation_runs.py:130` awaits, so dropping `async` would hand both a value that cannot be awaited.
    async def fail(*_args: object, **_kwargs: object) -> None:  # ruff: ignore[unused-async]
        msg = "provider response was malformed"
        raise RuntimeError(msg)

    monkeypatch.setattr(runner, "run_synthetic_generation", fail)
    caplog.set_level(logging.ERROR)

    exit_code = _run_cli_for_exit(
        monkeypatch,
        "generate",
        "--output-dir",
        str(tmp_path),
    )

    assert exit_code == cli.COMMAND_FAILURE_EXIT_CODE
    assert "Unexpected error during generation" in caplog.text
    assert "provider response was malformed" in caplog.text


def test_generate_interrupt_returns_shell_interrupt_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii.generation.label_corpus import runner

    # reason: this stands in for `runner.run_synthetic_generation`, which `cli.py` drives through `asyncio.run` and
    # reason: `generation_runs.py:130` awaits, so dropping `async` would hand both a value that cannot be awaited.
    async def interrupt(*_args: object, **_kwargs: object) -> None:  # ruff: ignore[unused-async]
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_synthetic_generation", interrupt)
    caplog.set_level(logging.INFO)

    exit_code = _run_cli_for_exit(
        monkeypatch,
        "generate",
        "--output-dir",
        str(tmp_path),
    )

    assert exit_code == cli.INTERRUPTED_EXIT_CODE
    assert "Generation interrupted by user" in caplog.text


def test_sample_command_passes_the_requested_file_and_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, int]] = []
    monkeypatch.setattr(
        cli,
        "sample_and_check",
        lambda path, count: received.append((path, count)),
    )

    _run_cli(monkeypatch, "sample", "rows.jsonl", "--count", "17")

    assert received == [("rows.jsonl", 17)]


def test_prepare_hf_runs_without_creating_a_provider_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "requests.jsonl"
    received: list[tuple[object, ...]] = []

    def prepare(*args: object, **kwargs: object) -> None:
        received.append((*args, kwargs))

    monkeypatch.setattr(cli, "prepare_hf_review", prepare)
    monkeypatch.setattr(
        cli,
        "get_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("prepare-hf must not create a provider client")),
    )

    _run_cli(
        monkeypatch,
        "inference",
        "prepare-hf",
        "--config",
        "vi",
        "--output",
        str(output_path),
        "--repo-id",
        "Meddies/source",
        "--raw-col",
        "text",
        "--label-col",
        "spans",
        "--allow-external-provider",
        "--data-classification",
        "synthetic",
    )

    assert received == [
        (
            "vi",
            str(output_path),
            "Meddies/source",
            "text",
            "spans",
            {
                "allow_external_provider": True,
                "data_classification": "synthetic",
            },
        ),
    ]


@pytest.mark.parametrize("explicit_vertex", [False, True])
# reason: pytest binds this parameter from the parametrize argnames tuple, so the boolean is labelled at every call site.
def test_project_job_uses_vertex_for_inference(
    monkeypatch: pytest.MonkeyPatch,
    explicit_vertex: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
) -> None:
    client = object()
    client_options: list[bool] = []
    monitored: list[tuple[object, str]] = []
    monkeypatch.setattr(
        cli,
        "get_client",
        lambda *, use_vertex_ai: (
            client_options.append(use_vertex_ai),
            client,
        )[1],
    )
    monkeypatch.setattr(
        cli,
        "monitor_job",
        lambda actual_client, job_name: monitored.append((actual_client, job_name)),
    )
    arguments = ["inference"]
    if explicit_vertex:
        arguments.append("--vertex-ai")
    arguments.extend(["monitor", "--job-name", "projects/p/locations/l/batchJobs/1"])

    _run_cli(monkeypatch, *arguments)

    assert client_options == [True]
    assert monitored == [(client, "projects/p/locations/l/batchJobs/1")]


@pytest.mark.parametrize("results", [[], [{"response": "ok"}]])
def test_download_writes_only_nonempty_results(monkeypatch: pytest.MonkeyPatch, results: list[dict[str, str]]) -> None:
    saved: list[tuple[list[dict[str, str]], str]] = []
    monkeypatch.setattr(cli, "get_client", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "download_batch_results", lambda _client, _job_name: results)
    monkeypatch.setattr(
        cli,
        "save_results_to_json",
        lambda actual_results, job_name: saved.append((actual_results, job_name)),
    )

    _run_cli(
        monkeypatch,
        "inference",
        "download",
        "--job-name",
        "jobs/123",
    )

    assert saved == ([(results, "jobs/123")] if results else [])


def test_correction_interrupt_returns_shell_interrupt_status(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from meddies_pii.generation import correction

    # reason: this stands in for `correction.correct_hf_data`, which `cli.py:225` drives through `asyncio.run`, so
    # reason: dropping `async` would hand it a value that cannot be awaited.
    async def interrupt(*_args: object, **_kwargs: object) -> None:  # ruff: ignore[unused-async]
        raise KeyboardInterrupt

    monkeypatch.setattr(correction, "correct_hf_data", interrupt)
    caplog.set_level(logging.WARNING)

    exit_code = _run_cli_for_exit(monkeypatch, "correct")

    assert exit_code == cli.INTERRUPTED_EXIT_CODE
    assert "Correction interrupted by user" in caplog.text


def test_demo_rejects_an_invalid_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.jsonl"

    exit_code = _run_cli_for_exit(
        monkeypatch,
        "demo",
        "--sample",
        str(missing),
        "--output-dir",
        str(tmp_path / "output"),
    )

    assert exit_code == cli.COMMAND_FAILURE_EXIT_CODE
    assert str(missing) in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("requested_date", [None, "2026-07-29"])
def test_pool_status_reports_a_missing_usage_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    requested_date: str | None,
) -> None:
    from meddies_pii.generation import pool_monitor

    monkeypatch.setattr(pool_monitor, "load_usage_record", lambda *_args: None)
    arguments = ["pool-status", "--usage-dir", str(tmp_path)]
    if requested_date is not None:
        arguments.extend(["--date", requested_date])

    exit_code = _run_cli_for_exit(monkeypatch, *arguments)

    assert exit_code == 2
    error = capsys.readouterr().err
    assert f"in {tmp_path}" in error
    if requested_date is not None:
        assert f"for {requested_date}" in error
    else:
        assert " for " not in error


def test_pool_status_propagates_the_health_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from meddies_pii.generation import pool_monitor

    record = SimpleNamespace(accounts=2)
    monkeypatch.setattr(pool_monitor, "load_usage_record", lambda *_args: record)
    monkeypatch.setattr(
        pool_monitor,
        "format_pool_status",
        lambda _actual_record: ("2 accounts over cap", 3),
    )

    exit_code = _run_cli_for_exit(
        monkeypatch,
        "pool-status",
        "--usage-dir",
        str(tmp_path),
    )

    assert exit_code == 3
    assert capsys.readouterr().out == "2 accounts over cap\n"


def test_module_entrypoint_runs_the_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["meddies-pii", "labels"])
    monkeypatch.delitem(sys.modules, "meddies_pii.cli")

    runpy.run_module("meddies_pii.cli", run_name="__main__")

    assert "Meddies Labels (9)" in capsys.readouterr().out
