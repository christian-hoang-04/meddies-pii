from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[no-self-use]
# reason: stateless test doubles retain the bound method shape of the process interfaces they replace.
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

    import modal


import io
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from meddies_pii.training.bioes.modal import base230_unsloth_probe as modal_probe
from meddies_pii.training.bioes.trainers import base230_unsloth_probe as probe


def _spec(batch: int = 224) -> modal_probe.ChildSpec:
    return modal_probe.ChildSpec(
        probe.render_probe(batch),
        "/cache/manifest.json",
        ("/cache/train-000.parquet",),
        "/artifacts/probe",
        999.0,
    )


def test_isolated_unsloth_endpoint_has_h100_only_image_and_exact_child_cells() -> None:
    assert modal_probe.FUNCTION_OPTIONS["gpu"] == "H100!"
    assert modal_probe.FUNCTION_OPTIONS["timeout"] == probe.PROBE_HARD_TIMEOUT_SECONDS
    assert modal_probe.FUNCTION_OPTIONS["timeout"] * probe.PROBE_RATE_USD_PER_SECOND <= probe.PROBE_MAX_LIVE_ESTIMATE_USD
    modal_probe._validate_child_spec(_spec(224))
    modal_probe._validate_child_spec(_spec(240))


def test_child_validation_rejects_wrong_backend_batch_candidate_steps_and_multi_cell() -> None:
    for contract, message in (
        ({**probe.render_probe(224), "backend": "transformers_peft"}, "mismatch: backend"),
        ({**probe.render_probe(224), "batch_size": 192}, "invalid batch"),
        ({**probe.render_probe(224), "candidate_key": "encoder230"}, "mismatch: candidate_key"),
        ({**probe.render_probe(224), "optimizer_steps": 9}, "mismatch: optimizer_steps"),
    ):
        with pytest.raises(ValueError, match=message):
            modal_probe._validate_child_spec(replace(_spec(), contract=contract))
    with pytest.raises(ValueError, match="shards"):
        modal_probe._validate_child_spec(replace(_spec(), shard_paths=()))
    with pytest.raises(ValueError, match="absolute"):
        modal_probe._validate_child_spec(replace(_spec(), artifact_dir="relative"))


def test_child_entrypoint_validates_serialized_spec_before_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec_path, result_path = tmp_path / "spec.json", tmp_path / "result.json"
    spec_path.write_text(
        json.dumps({
            "contract": _spec(240).contract,
            "manifest_path": "/cache/manifest.json",
            "shard_paths": ["/cache/train-000.parquet"],
            "artifact_dir": "/artifacts/probe",
            "deadline_monotonic": 999.0,
        }),
    )
    monkeypatch.setattr(
        modal_probe,
        "_run_child",
        lambda spec: {"status": "ok", "batch_size": spec.contract["batch_size"]},
    )
    monkeypatch.setattr(modal_probe.artifacts, "commit", lambda: None)
    assert modal_probe._child_main(str(spec_path), str(result_path)) == 0
    assert json.loads(result_path.read_text())["batch_size"] == 240


def test_isolated_child_refuses_a_non_object_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = replace(_spec(), artifact_dir=str(tmp_path / "probe"))

    def launch_child(_module: str, _spec_path: Path, result_path: Path) -> SimpleNamespace:
        result_path.write_text("[]", encoding="utf-8")
        return SimpleNamespace(pid=17)

    monkeypatch.setattr(modal_probe, "launch_child_process", launch_child)
    monkeypatch.setattr(modal_probe, "_stream_child_output", lambda *_args, **_kwargs: (0, ""))
    monkeypatch.setattr(modal_probe.artifacts, "commit", lambda: None)

    with pytest.raises(RuntimeError, match="JSON object"):
        modal_probe._run_isolated_child(spec)


def test_local_profile_guard_refuses_unset_or_wrong_profile_before_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = probe.render_probe(224)
    remote_called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal remote_called
        remote_called = True
        return {}

    monkeypatch.setattr(modal_probe.run_base230_unsloth_probe, "remote", fail_if_called)
    for profile in (None, "someone-else"):
        if profile is None:
            monkeypatch.delenv("MODAL_PROFILE", raising=False)
        else:
            monkeypatch.setenv("MODAL_PROFILE", profile)
        with pytest.raises(RuntimeError, match="MODAL_PROFILE"):
            modal_probe._local_remote_call(
                contract,
                execute=True,
                confirmation=probe.PROBE_CONFIRMATION,
            )
    assert remote_called is False


def test_append_event_fsyncs_then_commits_each_progress_event(tmp_path: Path) -> None:
    commits: list[str] = []
    modal_probe._append_event(
        tmp_path,
        {"event": "optimizer_step", "step": 1},
        commit=lambda: commits.append("committed"),
    )
    assert commits == ["committed"]
    assert json.loads((tmp_path / "events.jsonl").read_text()) == {
        "event": "optimizer_step",
        "step": 1,
    }


def test_live_child_stdout_is_forwarded_before_wait_returns() -> None:

    forwarded: list[str] = []
    progress_seen = threading.Event()

    class _Process:
        pid = 4242
        # reason: `ChildProcess` declares these as mutable `IO[str] | None`, and a protocol attribute is
        # reason: invariant, so a bare `StringIO` inferred here does not satisfy it.
        stdout: IO[str] | None = io.StringIO('{"event":"optimizer_step"}\n')
        stderr: IO[str] | None = io.StringIO("")

        def wait(self) -> int:
            assert progress_seen.wait(timeout=1.0), "stdout was buffered until exit"
            return 0

    def stdout_sink(line: str) -> None:
        forwarded.append(line)
        progress_seen.set()

    code, stderr = modal_probe._stream_child_output(
        _Process(),
        stdout_sink=stdout_sink,
        stderr_sink=lambda line: pytest.fail(f"unexpected stderr: {line}"),
    )
    assert code == 0
    assert not stderr
    assert forwarded == ['{"event":"optimizer_step"}\n']


def test_unsloth_image_uses_offline_hf_cache_before_source_mount() -> None:
    assert modal_probe.H100_CACHE_ENVIRONMENT == {
        "HF_HOME": "/cache/hf",
        "HF_HUB_CACHE": "/cache/hf",
        "HF_DATASETS_CACHE": "/cache/hf/datasets",
        "TRANSFORMERS_CACHE": "/cache/hf",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert modal_probe.IMAGE_PACKAGES[-2:] == (
        "unsloth==2026.5.2",
        "unsloth_zoo==2026.5.1",
    )


def test_cpu_preflight_resolves_pinned_snapshot_then_loads_local_tokenizer_and_processor(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "models--LiquidAI--LFM2.5-230M-Base" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json"):
        (snapshot / name).write_text("{}")
    snapshot_calls: list[dict[str, object]] = []
    local_load_calls: list[tuple[str, dict[str, object]]] = []

    def snapshot_download(**kwargs: object) -> str:
        snapshot_calls.append(kwargs)
        return str(snapshot)

    class _AutoConfig:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            local_load_calls.append((path, kwargs))
            return type("Config", (), {"model_type": "lfm2"})()

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            local_load_calls.append((path, kwargs))
            return type("Processor", (), {})()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            local_load_calls.append((path, kwargs))
            return type("Tokenizer", (), {"vocab_size": 32000})()

    result = modal_probe._preflight_base230_tokenizer_from_snapshot(
        snapshot_download=snapshot_download,
        auto_config=_AutoConfig,
        auto_processor=_AutoProcessor,
        auto_tokenizer=_AutoTokenizer,
    )
    assert snapshot_calls == [
        {
            "repo_id": "LiquidAI/LFM2.5-230M-Base",
            "revision": "9d2be5519834990d30996f878b6771cccbd24f2c",
            "cache_dir": "/cache/hf",
            "local_files_only": True,
        },
    ]
    assert local_load_calls == [(str(snapshot), {"local_files_only": True})] * 3
    assert result == {
        "snapshot_path": str(snapshot),
        "model_type": "lfm2",
        "processor_class": "Processor",
        "tokenizer_class": "Tokenizer",
        "vocab_size": 32000,
        "required_files": ["config.json", "tokenizer_config.json"],
    }
    assert modal_probe.CPU_PREFLIGHT_OPTIONS["gpu"] is None
    assert modal_probe.CPU_PREFLIGHT_OPTIONS["timeout"] < 600


def test_cpu_preflight_rejects_missing_tokenizer_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    unused_factory = object()
    with pytest.raises(RuntimeError, match=r"tokenizer_config\.json"):
        modal_probe._preflight_base230_tokenizer_from_snapshot(
            snapshot_download=lambda **_: str(snapshot),
            auto_config=unused_factory,
            auto_processor=unused_factory,
            auto_tokenizer=unused_factory,
        )


def test_probe_image_installs_dependencies_then_sets_offline_cache_environment() -> None:
    calls: list[tuple[object, ...]] = []

    class _Image:
        def pip_install(self, *packages: str) -> _Image:
            calls.append(("pip_install", *packages))
            return self

        def env(self, values: dict[str, str]) -> _Image:
            calls.append(("env", values))
            return self

    assert modal_probe._build_probe_image(cast("modal.Image", _Image()))
    assert calls == [
        ("pip_install", *modal_probe.IMAGE_PACKAGES),
        ("env", modal_probe.H100_CACHE_ENVIRONMENT),
    ]


def test_nvml_sampler_reports_latest_step_sample_and_peak_device_used() -> None:
    sampler = modal_probe._NvmlSampler()
    sampler.samples = [
        {"device_vram_used_bytes": 12.0, "gpu_sm_utilization": 40.0},
        {"device_vram_used_bytes": 24.0, "gpu_sm_utilization": 80.0},
    ]
    assert sampler.latest()["gpu_sm_utilization"] == 80.0
    assert sampler.stop() == {
        "nvml_sample_count": 2,
        "peak_device_vram_used_bytes": 24.0,
    }
