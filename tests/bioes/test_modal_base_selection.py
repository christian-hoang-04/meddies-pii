from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import modal
import pytest

from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.modal import base_selection

if TYPE_CHECKING:
    from collections.abc import Mapping


def test_h100_image_keeps_local_source_mount_after_modal_build_steps() -> None:
    """Modal rejects an env/build layer that depends on a local-source mount."""
    dependencies = list(base_selection.image.deps())
    assert len(dependencies) == 2
    assert isinstance(dependencies[0], modal.Image)
    assert type(dependencies[1]).__module__ == "modal.mount"
    assert type(dependencies[1]).__name__ == "Mount"


def test_local_render_has_two_strict_h100_lanes_without_remote_dispatch() -> None:
    rendered = json.loads(base_selection.render_dry_run())
    assert set(rendered["probe_lanes"]) == {"230", "350"}
    assert base_selection.LANE_OPTIONS["gpu"] == "H100!"
    assert base_selection.LANE_OPTIONS["max_containers"] == 1
    assert base_selection.H100_CACHE_ENVIRONMENT == {
        "HF_HOME": "/cache/hf",
        "HF_HUB_CACHE": "/cache/hf/hub",
        "HF_DATASETS_CACHE": "/cache/hf/datasets",
        "TRANSFORMERS_CACHE": "/cache/hf/hub",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_remote_guard_requires_double_confirmation_and_hard_budget() -> None:
    with pytest.raises(RuntimeError, match="requires"):
        base_selection._probe_lane_guard("230", execute=False, confirmation="", estimated_cost_usd=3.0)
    with pytest.raises(RuntimeError, match="budget"):
        base_selection._probe_lane_guard(
            "230",
            execute=True,
            confirmation="LAUNCH_H100_BASE_SELECTION",
            estimated_cost_usd=3.01,
        )
    result = base_selection._probe_lane_guard(
        "230",
        execute=True,
        confirmation="LAUNCH_H100_BASE_SELECTION",
        estimated_cost_usd=3.0,
    )
    assert result["models"] == ["base230", "encoder230"]


def test_h100_dispatch_profile_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    with pytest.raises(RuntimeError, match="MODAL_PROFILE"):
        base_selection.require_modal_profile()
    monkeypatch.setenv("MODAL_PROFILE", "hahuyhoang411")
    base_selection.require_modal_profile()


def _artifact(tmp_path: Path) -> base_selection.VerifiedPackedArtifact:
    return base_selection.VerifiedPackedArtifact(
        revision="a" * 40,
        manifest={"packed_config": "packed", "complete": True, "shards": [{}]},
        manifest_path=str(tmp_path / "manifest.json"),
        shard_paths=(str(tmp_path / "train-000.parquet"),),
    )


def _writer(tmp_path: Path) -> base_selection.ArtifactWriter:
    return base_selection.ArtifactWriter(tmp_path, "230", commit=None, execution_id="test-execution")


def _success(spec: base_selection.ChildSpec) -> Mapping[str, object]:
    return {
        "status": "ok",
        "candidate": spec.candidate_key,
        "batch_size": spec.batch_size,
        "median_tokens_per_second": 100.0 + spec.batch_size,
        "static_bytes": 10,
        "peak_bytes": 60,
        "total_bytes": 100,
        "events": [{"event": "step", "gpu_sm_utilization": 99.0}],
        "nvml": [{"gpu_sm_utilization": 99.0}],
    }


def test_lane_runs_children_sequentially_and_predicts_larger_batch(tmp_path: Path) -> None:
    seen = []

    def child(spec: base_selection.ChildSpec) -> Mapping[str, object]:
        seen.append(spec)
        return _success(spec)

    writer = _writer(tmp_path)
    result = base_selection.run_lane_plan(
        lane="230",
        artifact=_artifact(tmp_path),
        child_runner=child,
        writer=writer,
    )
    assert [item.candidate_key for item in seen] == sorted(
        [item.candidate_key for item in seen],
        key=lambda key: (key != "base230", key),
    )
    assert [item.batch_size for item in seen] == [224, 224] * 2
    assert [item.step_count for item in seen] == [2, 10] * 2
    assert all(item.fused_adamw is False for item in seen)
    assert result["children"]
    assert result["execution_id"] == "test-execution"
    events = writer.events_path.read_text()
    assert "child_started" in events
    assert "gpu_sm_utilization" in events
    records = [json.loads(line) for line in events.splitlines()]
    assert records[0]["event"] == "lane_plan_started"
    assert records[0]["budget_usd"] == base_selection.PROBE_BUDGET_USD
    assert all(record["execution_id"] == "test-execution" for record in records)


def test_child_started_progress_compacts_shard_paths(tmp_path: Path) -> None:
    artifact = base_selection.VerifiedPackedArtifact(
        revision="a" * 40,
        manifest={"packed_config": "packed", "complete": True, "shards": [{}]},
        manifest_path=str(tmp_path / "manifest.json"),
        shard_paths=tuple(f"/cache/hf/packed/data/train-{index:05}.parquet" for index in range(469)),
    )
    writer = _writer(tmp_path)
    base_selection.run_lane_plan(
        lane="230",
        artifact=artifact,
        child_runner=_success,
        writer=writer,
    )
    records = [json.loads(line) for line in writer.events_path.read_text().splitlines()]
    child_started = next(record for record in records if record["event"] == "child_started")
    assert "shard_paths" not in child_started
    assert child_started["shard_count"] == 469
    assert len(child_started["shard_paths_sha256"]) == 64
    assert child_started["shard_path_sample"] == {
        "first": "/cache/hf/packed/data/train-00000.parquet",
        "last": "/cache/hf/packed/data/train-00468.parquet",
    }


def test_oom_224_descends_the_coarse_ladder_without_fixed_32_child(tmp_path: Path) -> None:
    seen = []

    def child(spec: base_selection.ChildSpec) -> Mapping[str, object]:
        seen.append((spec.candidate_key, spec.batch_size, spec.role))
        if spec.batch_size in {224, 192, 160, 128, 96}:
            return {
                "status": "oom",
                "candidate": spec.candidate_key,
                "batch_size": spec.batch_size,
            }
        return _success(spec)

    writer = _writer(tmp_path)
    base_selection.run_lane_plan(
        lane="230",
        artifact=_artifact(tmp_path),
        child_runner=child,
        writer=writer,
    )
    assert ("base230", 224, "discovery_b224") in seen
    assert ("base230", 192, "oom_fallback_b192") in seen
    assert ("base230", 160, "oom_fallback_b160") in seen
    assert ("base230", 128, "oom_fallback_b128") in seen
    assert ("base230", 96, "oom_fallback_b96") in seen
    assert ("base230", 64, "oom_fallback_b64") in seen
    assert ("base230", 64, "stability_10") in seen
    assert all(batch != 32 for _candidate, batch, _role in seen)


def test_candidate_probe_runs_only_base230_at_240_for_exactly_10_steps(tmp_path: Path) -> None:
    seen = []

    def child(spec: base_selection.ChildSpec) -> Mapping[str, object]:
        seen.append(spec)
        return _success(spec)

    result = base_selection.run_candidate_probe_plan(
        lane="230",
        candidate_key="base230",
        batch_size=240,
        step_count=10,
        artifact=_artifact(tmp_path),
        child_runner=child,
        writer=_writer(tmp_path),
    )
    assert [(spec.candidate_key, spec.batch_size, spec.role, spec.step_count) for spec in seen] == [
        ("base230", 240, "explicit_b240_s10", 10),
    ]
    assert result["mode"] == "candidate_probe"
    assert result["candidate"] == "base230"
    assert result["batch_size"] == 240
    assert result["step_count"] == 10
    assert result["next_action"] == "review_10_step_result"


def test_candidate_probe_runs_only_encoder350_at_200_for_exactly_10_steps(tmp_path: Path) -> None:
    seen = []

    def child(spec: base_selection.ChildSpec) -> Mapping[str, object]:
        seen.append(spec)
        return _success(spec)

    result = base_selection.run_candidate_probe_plan(
        lane="350",
        candidate_key="encoder350",
        batch_size=200,
        step_count=10,
        artifact=_artifact(tmp_path),
        child_runner=child,
        writer=base_selection.ArtifactWriter(tmp_path, "350", commit=None, execution_id="encoder350-test"),
    )
    assert [(spec.candidate_key, spec.batch_size, spec.role, spec.step_count) for spec in seen] == [
        ("encoder350", 200, "explicit_b200_s10", 10),
    ]
    assert all(spec.candidate_key != "pii350" for spec in seen)
    assert result["candidate"] == "encoder350"
    assert result["next_action"] == "review_10_step_result"


def test_candidate_probe_never_falls_back_after_an_explicit_240_oom(tmp_path: Path) -> None:
    seen = []

    def child(spec: base_selection.ChildSpec) -> Mapping[str, object]:
        seen.append((spec.batch_size, spec.role, spec.step_count))
        return {
            "status": "oom",
            "candidate": spec.candidate_key,
            "batch_size": spec.batch_size,
        }

    result = base_selection.run_candidate_probe_plan(
        lane="230",
        candidate_key="encoder230",
        batch_size=240,
        step_count=10,
        artifact=_artifact(tmp_path),
        child_runner=child,
        writer=_writer(tmp_path),
    )
    assert seen == [(240, "explicit_b240_s10", 10)]
    assert result["status"] == "oom"
    assert result["next_action"] == "review_oom_artifacts_before_any_lower_batch"


def test_candidate_probe_guard_rejects_wrong_lane_candidate_or_parameters() -> None:
    with pytest.raises(ValueError, match="lane"):
        base_selection.require_candidate_probe("350", "base230", 240, 10)
    with pytest.raises(ValueError, match="candidate"):
        base_selection.require_candidate_probe("230", "encoder350", 240, 10)
    with pytest.raises(ValueError, match="batch_size"):
        base_selection.require_candidate_probe("230", "base230", 192, 10)
    with pytest.raises(ValueError, match="step_count"):
        base_selection.require_candidate_probe("230", "base230", 240, 3)
    base_selection.require_candidate_probe("230", "encoder230", 240, 10)
    base_selection.require_candidate_probe("350", "encoder350", 200, 10)
    with pytest.raises(ValueError, match="batch_size"):
        base_selection.require_candidate_probe("350", "encoder350", 201, 10)
    with pytest.raises(ValueError, match="step_count"):
        base_selection.require_candidate_probe("350", "encoder350", 200, 2)


def test_child_validation_accepts_requested_probe_batches_and_rejects_201() -> None:
    requested = base_selection.ChildSpec(
        lane="230",
        candidate_key="base230",
        batch_size=240,
        fused_adamw=False,
        dataset_revision="a" * 40,
        manifest_path="manifest",
        shard_paths=("shard",),
        artifact_dir="/artifacts/test",
        deadline_monotonic=1.0,
        role="explicit_b240_s10",
        step_count=10,
    )
    base_selection._require_child_spec(requested)
    base_selection._require_child_spec(
        base_selection.ChildSpec(
            lane="350",
            candidate_key="encoder350",
            batch_size=200,
            fused_adamw=False,
            dataset_revision="a" * 40,
            manifest_path="manifest",
            shard_paths=("shard",),
            artifact_dir="/artifacts/test",
            deadline_monotonic=1.0,
            role="explicit_b200_s10",
            step_count=10,
        ),
    )
    with pytest.raises(ValueError, match="batch size"):
        base_selection._require_child_spec(
            base_selection.ChildSpec(
                lane="350",
                candidate_key="encoder350",
                batch_size=201,
                fused_adamw=False,
                dataset_revision="a" * 40,
                manifest_path="manifest",
                shard_paths=("shard",),
                artifact_dir="/artifacts/test",
                deadline_monotonic=1.0,
                role="explicit_b201_s10",
                step_count=10,
            ),
        )


def test_candidate_probe_mode_requires_exactly_one_230_candidate() -> None:
    base_selection.require_probe_mode("230", "candidate_probe", "base230", 240, 10)
    with pytest.raises(ValueError, match="candidate"):
        base_selection.require_probe_mode("230", "candidate_probe", "", 240, 10)
    with pytest.raises(ValueError, match="candidate probe parameters"):
        base_selection.require_probe_mode("230", "full_lane", "base230", 0, 0)
    with pytest.raises(ValueError, match="mode"):
        base_selection.require_probe_mode("230", "unexpected", "", 0, 0)


def test_budget_blocks_next_child_and_records_event(tmp_path: Path) -> None:
    budget = base_selection._budget_seconds()
    values = iter((0.0, 0.0, budget + 1.0, budget + 1.0))

    def clock() -> float:
        return next(values)

    writer = _writer(tmp_path)
    with pytest.raises(RuntimeError, match="budget cannot finish"):
        base_selection.run_lane_plan(
            lane="230",
            artifact=_artifact(tmp_path),
            child_runner=_success,
            writer=writer,
            clock=clock,
        )
    assert "budget_blocked" in writer.events_path.read_text()


def test_child_failure_surfaces_and_is_persisted(tmp_path: Path) -> None:
    def failed(_spec: base_selection.ChildSpec) -> Mapping[str, object]:
        msg = "child exploded"
        raise RuntimeError(msg)

    writer = _writer(tmp_path)
    with pytest.raises(RuntimeError, match="child exploded"):
        base_selection.run_lane_plan(
            lane="230",
            artifact=_artifact(tmp_path),
            child_runner=failed,
            writer=writer,
        )
    assert "child_failed" in writer.events_path.read_text()


def test_two_execution_artifacts_are_isolated_and_preserve_first_attempt(tmp_path: Path) -> None:
    def run_attempt(writer: base_selection.ArtifactWriter) -> list[base_selection.ChildSpec]:
        child_specs = []

        def child(spec: base_selection.ChildSpec) -> Mapping[str, object]:
            child_specs.append(spec)
            Path(spec.artifact_dir, f"{spec.candidate_key}-{spec.role}.spec.json").write_text(
                json.dumps({"shard_paths": spec.shard_paths}),
                encoding="utf-8",
            )
            return _success(spec)

        base_selection.run_lane_plan(
            lane="230",
            artifact=_artifact(tmp_path),
            child_runner=child,
            writer=writer,
        )
        writer.finalize({"status": "ok"})
        return child_specs

    first = base_selection.ArtifactWriter(tmp_path, "230", commit=None)
    first_specs = run_attempt(first)
    first_events = first.events_path.read_bytes()
    first_result = (first.root / "result.json").read_bytes()
    first_spec_files = {path.name: path.read_bytes() for path in sorted(first.root.glob("*.spec.json"))}

    second = base_selection.ArtifactWriter(tmp_path, "230", commit=None)
    second_specs = run_attempt(second)

    assert first.execution_id != second.execution_id
    assert first.root != second.root
    assert first.root == tmp_path / "230" / first.execution_id
    assert second.root == tmp_path / "230" / second.execution_id
    assert all(Path(spec.artifact_dir) == first.root for spec in first_specs)
    assert all(Path(spec.artifact_dir) == second.root for spec in second_specs)
    assert first.events_path.read_bytes() == first_events
    assert (first.root / "result.json").read_bytes() == first_result
    assert {path.name: path.read_bytes() for path in sorted(first.root.glob("*.spec.json"))} == first_spec_files
    assert all(json.loads(line)["execution_id"] == first.execution_id for line in first_events.decode().splitlines())
    assert json.loads(first_result)["execution_id"] == first.execution_id
    assert json.loads((second.root / "result.json").read_text())["execution_id"] == second.execution_id


def test_packed_artifact_accepts_only_packed_and_checks_hashes(tmp_path: Path) -> None:
    shard = tmp_path / "train-000.parquet"
    shard.write_bytes(b"parquet")
    manifest = {
        "packed_config": "packed",
        "complete": True,
        "resume_cursor": 1_000_000,
        "shards": [
            {
                "path": "packed/data/train-000.parquet",
                "bytes": shard.stat().st_size,
                "sha256": base_selection._sha_file(shard),
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    calls = []

    def download(*, filename: str, **kwargs: object) -> str:
        calls.append(kwargs)
        return str(manifest_path if filename.endswith("manifest.json") else shard)

    verified = base_selection.verify_packed_artifact(
        revision=base_selection.PACKED_DATASET_REVISION,
        download=download,
        expected_manifest_sha256=base_selection._sha_file(manifest_path),
        expected_shard_count=1,
    )
    assert verified.shard_paths == (str(shard),)
    assert all(call["local_files_only"] is True for call in calls)
    manifest["packed_config"] = "packed-encoder"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="packed config"):
        base_selection.verify_packed_artifact(
            revision=base_selection.PACKED_DATASET_REVISION,
            download=download,
            expected_manifest_sha256=base_selection._sha_file(manifest_path),
            expected_shard_count=1,
        )


def test_artifact_writer_mirrors_json_event_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    writer = _writer(tmp_path)
    event = {"event": "parent_started", "elapsed_seconds": 0.0}
    writer.append(event)
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["event"] == "parent_started"
    assert emitted["execution_id"] == "test-execution"
    assert event == {"event": "parent_started", "elapsed_seconds": 0.0}


def test_subprocess_child_inherits_stdout_and_keeps_missing_result_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = base_selection.ChildSpec(
        lane="230",
        candidate_key="base230",
        batch_size=64,
        fused_adamw=False,
        dataset_revision="a" * 40,
        manifest_path="manifest",
        shard_paths=("shard",),
        artifact_dir=str(tmp_path),
        deadline_monotonic=1.0,
        role="test",
    )
    captured: dict[str, object] = {}

    def run(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(base_selection.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="did not write a result"):
        base_selection._subprocess_child(spec)
    assert "capture_output" not in captured
    environment = captured["env"]
    assert is_str_mapping(environment)
    pythonpath = environment["PYTHONPATH"]
    assert isinstance(pythonpath, str)
    assert pythonpath.split(":")[0] == "/root/src"


def test_child_events_print_json_and_persist(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = base_selection.ChildSpec(
        lane="230",
        candidate_key="base230",
        batch_size=64,
        fused_adamw=False,
        dataset_revision="a" * 40,
        manifest_path="manifest",
        shard_paths=("shard",),
        artifact_dir=str(tmp_path),
        deadline_monotonic=1.0,
        role="test",
    )
    base_selection._append_child_event(spec, {"event": "step", "loss": 1.0})
    assert json.loads(capsys.readouterr().out)["event"] == "step"
    event_path = tmp_path / "base230-b64-test.events.jsonl"
    assert json.loads(event_path.read_text())["loss"] == 1.0


def test_packed_row_schema_requires_uid_and_accepts_source_metadata() -> None:
    row = {
        "input_ids": [1, 2, 3, 4],
        "labels": [0, 0, 0, 0],
        "seq_lengths": [2, 2],
        "position_ids": [0, 1, 0, 1],
        "row_uids": ["row-a", "row-b"],
        "row_ranges": [
            {
                "uid": "row-a",
                "start": 0,
                "end": 2,
                "source_uid": "source-a",
                "source_index": 0,
            },
            {
                "uid": "row-b",
                "start": 2,
                "end": 4,
                "source_uid": "source-b",
                "source_index": 1,
            },
        ],
        "real_token_count": 4,
        "padded_token_count": 4,
        "boundary_token_count": 0,
    }
    unit, ranges = base_selection._packed_training_unit_from_row(row)
    assert unit.row_uids == ("row-a", "row-b")
    assert tuple(row_range.uid for row_range in ranges) == ("row-a", "row-b")
    invalid_row = {**row, "row_ranges": [{"source_uid": "source-a"}]}
    with pytest.raises(RuntimeError, match="lacks uid"):
        base_selection._packed_training_unit_from_row(invalid_row)


def test_child_result_includes_nvml_on_handled_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = base_selection.ChildSpec(
        lane="230",
        candidate_key="base230",
        batch_size=64,
        fused_adamw=False,
        dataset_revision="a" * 40,
        manifest_path="manifest",
        shard_paths=("shard",),
        artifact_dir=str(tmp_path),
        deadline_monotonic=10.0,
        role="test",
    )
    spec_path, result_path = tmp_path / "spec.json", tmp_path / "result.json"
    spec_path.write_text(json.dumps(__import__("dataclasses").asdict(spec)))
    monkeypatch.setattr(
        base_selection,
        "_run_child",
        lambda _spec: {"status": "failed", "nvml": [], "events": []},
    )
    monkeypatch.setattr(base_selection.artifacts, "commit", lambda: None)
    assert base_selection._child_main(str(spec_path), str(result_path)) == 1
    payload = json.loads(result_path.read_text())
    assert "nvml" in payload
    assert "events" in payload
