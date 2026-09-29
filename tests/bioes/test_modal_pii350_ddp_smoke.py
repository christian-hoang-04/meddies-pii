from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[no-self-use]
# reason: stateless test doubles retain the bound method shape of the CUDA interfaces they replace.
import json

# reason: this module constructs CompletedProcess doubles; it does not spawn a process itself.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import pytest
import torch
from torch import distributed, nn
from torch.multiprocessing.spawn import spawn as torch_spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from meddies_pii.training.bioes.modal import pii350_ddp_smoke as modal_smoke
from meddies_pii.training.bioes.trainers import pii350_ddp_smoke

if TYPE_CHECKING:
    from collections.abc import Mapping


class _RepeatedReentrantCheckpointModule(nn.Module):
    """The packed shape: the same checkpointed encoder runs per document."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Linear(8, 8, bias=False)

    @override
    def forward(self, values: torch.Tensor, repeats: int) -> torch.Tensor:
        pieces: list[torch.Tensor] = [
            cast("torch.Tensor", checkpoint(self.block, values + index, use_reentrant=True)) for index in range(repeats)
        ]
        return torch.stack(pieces).sum(0)


def _reentrant_ddp_worker(rank: int, initialization_path: str, result_path: str, reduction: str) -> None:
    # reason: torch declares the torch.distributed members behind an availability gate, so no static
    # reason: reader can prove they are present; this path runs only inside the multi-rank container.
    distributed.init_process_group(  # ty: ignore[possibly-missing-attribute]
        "gloo",
        init_method=f"file://{initialization_path}",
        world_size=2,
        rank=rank,
    )
    try:
        torch.manual_seed(3407)
        tagger = DistributedDataParallel(_RepeatedReentrantCheckpointModule())
        optimizer = torch.optim.AdamW(tagger.parameters(), lr=3e-4)
        torch.manual_seed(3407 + rank)
        values = torch.randn(4, 8, requires_grad=True)
        optimizer.zero_grad(set_to_none=True)
        if reduction == "stock_ddp":
            loss = tagger(values, 2 + rank).square().mean()
            loss.backward()
            pre_reduction = None
            post_reduction = None
            buckets = None
        elif reduction == "explicit_mean":
            with tagger.no_sync():
                loss = tagger(values, 2 + rank).square().mean()
                loss.backward()
            pre_reduction = modal_smoke._parameter_stage_diagnostics(
                distributed,
                tagger,
                rank=rank,
                stage="gradients",
            )
            buckets = modal_smoke._explicit_mean_trainable_gradients(torch, distributed, tagger, rank=rank)
            post_reduction = modal_smoke._parameter_stage_diagnostics(
                distributed,
                tagger,
                rank=rank,
                stage="gradients",
            )
        else:
            msg = f"unknown reduction mode: {reduction}"
            raise ValueError(msg)
        optimizer.step()
        post_step = modal_smoke._parameter_stage_diagnostics(distributed, tagger, rank=rank, stage="parameters")
        if rank == 0:
            Path(result_path).write_text(
                json.dumps({
                    "pre_reduction": pre_reduction,
                    "post_reduction": post_reduction,
                    "buckets": buckets,
                    "post_step": post_step,
                }),
                encoding="utf-8",
            )
    finally:
        # reason: torch declares the torch.distributed members behind an availability gate, so no static
        # reason: reader can prove they are present; this path runs only inside the multi-rank container.
        distributed.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def _run_reentrant_ddp_case(tmp_path: Path, reduction: str) -> dict[str, Any]:
    initialization_path = tmp_path / f"{reduction}-init"
    result_path = tmp_path / f"{reduction}.json"
    torch_spawn(
        _reentrant_ddp_worker,
        args=(str(initialization_path), str(result_path), reduction),
        nprocs=2,
        join=True,
    )
    return cast("dict[str, Any]", json.loads(result_path.read_text(encoding="utf-8")))


def test_modal_lane_requests_one_two_a100_container_with_reserved_persistence_time() -> None:
    assert modal_smoke.DDP_OPTIONS == {
        "gpu": "A100-40GB:2",
        "cpu": 4.0,
        "memory": 64 * 1024,
        "timeout": 1800,
        "max_containers": 1,
    }
    assert modal_smoke.DDP_OPTIONS["timeout"] > modal_smoke.TRAINING_DEADLINE_SECONDS
    assert modal_smoke.TRAINING_DEADLINE_SECONDS == 1500


def test_variable_reentrant_packed_forwards_require_explicit_gradient_mean(
    tmp_path: Path,
) -> None:
    """RED: stock DDP drifts when checkpointed work repeats by packed document."""
    stock = _run_reentrant_ddp_case(tmp_path, "stock_ddp")
    repaired = _run_reentrant_ddp_case(tmp_path, "explicit_mean")

    stock_post_step = cast("Mapping[str, Any]", stock["post_step"])
    repaired_pre_reduction = cast("Mapping[str, Any]", repaired["pre_reduction"])
    repaired_post_reduction = cast("Mapping[str, Any]", repaired["post_reduction"])
    repaired_post_step = cast("Mapping[str, Any]", repaired["post_step"])
    assert float(stock_post_step["max_abs_difference"]) > 0.0
    assert float(repaired_pre_reduction["max_abs_difference"]) > 0.0
    assert float(repaired_post_reduction["max_abs_difference"]) == 0.0
    assert float(repaired_post_step["max_abs_difference"]) == 0.0
    buckets = repaired["buckets"]
    assert isinstance(buckets, list)
    assert len(buckets) == 1
    bucket = cast("Mapping[str, Any]", buckets[0])
    assert bucket["device_type"] == "cpu"
    assert bucket["dtype"] == "torch.float32"
    assert bucket["parameter_names"] == ["block.weight"]


def _valid_pre_torchrun_evidence() -> dict[str, object]:
    uuid_rows = [
        "0, NVIDIA A100-SXM4-40GB, GPU-11111111-1111-1111-1111-111111111111, 40960",
        "1, NVIDIA A100-SXM4-40GB, GPU-22222222-2222-2222-2222-222222222222, 40960",
    ]
    return {
        "availability": "validated",
        "topology_command": ["nvidia-smi", "topo", "-m"],
        "topology_returncode": 255,
        "topology_stdout": "",
        "topology_stderr": "",
        "topology_unavailable_reason": "nvidia-smi topo -m returned 255",
        "uuid_inventory_command": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        "uuid_inventory_returncode": 0,
        "uuid_inventory_raw": uuid_rows,
        "uuid_inventory_stderr": "",
        "uuid_devices": [
            {
                "index": 0,
                "name": "NVIDIA A100-SXM4-40GB",
                "uuid": "GPU-11111111-1111-1111-1111-111111111111",
                "memory_total_mib": 40960,
            },
            {
                "index": 1,
                "name": "NVIDIA A100-SXM4-40GB",
                "uuid": "GPU-22222222-2222-2222-2222-222222222222",
                "memory_total_mib": 40960,
            },
        ],
        "nvlink_status_command": ["nvidia-smi", "nvlink", "--status"],
        "nvlink_status_returncode": 0,
        "nvlink_status_raw": "GPU 0: Link 0: 25 GB/s\nGPU 1: Link 0: 25 GB/s\n",
        "nvlink_status_stderr": "",
    }


def test_child_spec_and_launch_gate_fail_closed() -> None:
    contract = pii350_ddp_smoke.render_ddp_smoke_contract()
    spec = modal_smoke.ChildSpec(
        contract=contract,
        matched_constant3_contract=modal_smoke.matched_constant3_contract(),
        shard_paths=("/cache/hf/packed/data/train-00000.parquet",),
        expected_encoder_checkpoint_attestation={"body_tensor_count": 1},
        artifact_dir="/artifacts/pii350-ddp-smoke/test",
        outer_modal_started_monotonic=100.0,
        absolute_training_deadline_monotonic=1_600.0,
        pre_torchrun_interconnect=_valid_pre_torchrun_evidence(),
    )
    modal_smoke.validate_child_spec(spec)

    with pytest.raises(RuntimeError, match="confirmation"):
        pii350_ddp_smoke.require_ddp_smoke_execute(
            contract,
            execute=True,
            confirmation="wrong",
            primary_action=pii350_ddp_smoke.DDP_SMOKE_PRIMARY_ACTION,
        )


def test_child_spec_revalidates_uuid_and_nvlink_provenance_before_torchrun() -> None:
    contract = pii350_ddp_smoke.render_ddp_smoke_contract()
    evidence = _valid_pre_torchrun_evidence()
    spec = modal_smoke.ChildSpec(
        contract=contract,
        matched_constant3_contract=modal_smoke.matched_constant3_contract(),
        shard_paths=("/cache/hf/packed/data/train-00000.parquet",),
        expected_encoder_checkpoint_attestation={"body_tensor_count": 1},
        artifact_dir="/artifacts/pii350-ddp-smoke/test",
        outer_modal_started_monotonic=100.0,
        absolute_training_deadline_monotonic=1_600.0,
        pre_torchrun_interconnect=evidence,
    )
    modal_smoke.validate_child_spec(spec)
    rejected = replace(spec, pre_torchrun_interconnect={**evidence, "nvlink_status_returncode": 1})
    with pytest.raises(RuntimeError, match="NVLink status must return zero"):
        modal_smoke.validate_child_spec(rejected)


def test_step_one_diagnostics_create_their_nested_directory_on_a_fresh_artifact_root(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "fresh-artifact-root"
    spec = modal_smoke.ChildSpec(
        contract={"config_digest": "diagnostic-test"},
        matched_constant3_contract={},
        shard_paths=(),
        expected_encoder_checkpoint_attestation={},
        artifact_dir=str(artifact_root),
        outer_modal_started_monotonic=0.0,
        absolute_training_deadline_monotonic=0.0,
        pre_torchrun_interconnect={},
    )
    assert not artifact_root.exists()

    for phase in modal_smoke.STEP_ONE_DIAGNOSTIC_PHASES:
        modal_smoke._persist_step_one_diagnostics(
            spec,
            rank=0,
            phase=phase,
            diagnostics={"max_abs_difference": 0.0},
        )

    assert artifact_root.joinpath("diagnostics").is_dir()
    assert {
        path.relative_to(artifact_root).as_posix() for path in artifact_root.joinpath("diagnostics").glob("*.json")
    } == set(modal_smoke._step_one_diagnostic_paths())


def test_outer_modal_entry_deadline_is_transported_unchanged_into_torchrun_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = pii350_ddp_smoke.render_ddp_smoke_contract()
    captured: list[modal_smoke.ChildSpec] = []
    monkeypatch.setattr(modal_smoke.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        modal_smoke,
        "_validated_receipt_inventory",
        lambda _baseline: (
            ("/cache/hf/packed/data/train-00000.parquet",),
            {"body_tensor_count": 1},
        ),
    )
    monkeypatch.setattr(modal_smoke.time, "time_ns", lambda: 7)
    monkeypatch.setattr(modal_smoke, "capture_pre_torchrun_interconnect", _valid_pre_torchrun_evidence)
    monkeypatch.setattr(
        modal_smoke,
        "_run_isolated_torchrun",
        lambda spec: captured.append(spec) or {"status": "ok"},
    )

    result = modal_smoke.run_pii350_ddp_smoke.get_raw_f()(
        contract,
        execute=True,
        confirmation=pii350_ddp_smoke.DDP_SMOKE_CONFIRMATION,
        primary_action=pii350_ddp_smoke.DDP_SMOKE_PRIMARY_ACTION,
    )

    assert result == {"status": "ok"}
    assert len(captured) == 1
    spec = captured[0]
    assert spec.outer_modal_started_monotonic == 100.0
    assert spec.absolute_training_deadline_monotonic == 1_600.0
    assert spec.pre_torchrun_interconnect == _valid_pre_torchrun_evidence()
    assert pii350_ddp_smoke.training_deadline_reached(
        absolute_deadline_monotonic=spec.absolute_training_deadline_monotonic,
        now_monotonic=1_600.0,
    )


def test_pre_torchrun_evidence_uses_uuid_inventory_and_raw_nvlink_status() -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(command))
        if command[1:3] == ["topo", "-m"]:
            return subprocess.CompletedProcess(command, 255, stdout="", stderr="")
        if "--query-gpu=index,name,uuid,memory.total" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "0, NVIDIA A100-SXM4-40GB, GPU-11111111-1111-1111-1111-111111111111, 40960\n"
                    "1, NVIDIA A100-SXM4-40GB, GPU-22222222-2222-2222-2222-222222222222, 40960\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GPU 0: Link 0: 25 GB/s\nGPU 1: Link 11: 25 GB/s\n",
            stderr="",
        )

    evidence = modal_smoke.capture_pre_torchrun_interconnect(run=run)

    assert commands == [
        ("nvidia-smi", "topo", "-m"),
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ),
        ("nvidia-smi", "nvlink", "--status"),
    ]
    assert evidence["availability"] == "validated"
    assert evidence["topology_unavailable_reason"] == "nvidia-smi topo -m returned 255"
    assert evidence["uuid_devices"] == _valid_pre_torchrun_evidence()["uuid_devices"]
    assert evidence["nvlink_status_raw"] == "GPU 0: Link 0: 25 GB/s\nGPU 1: Link 11: 25 GB/s\n"


@pytest.mark.parametrize(
    ("uuid_returncode", "uuid_stdout", "nvlink_returncode", "nvlink_stdout", "match"),
    [
        pytest.param(
            1,
            "",
            0,
            "Link 0: 25 GB/s\n",
            "UUID inventory returned 1",
            id="uuid-nonzero",
        ),
        pytest.param(0, "", 0, "Link 0: 25 GB/s\n", "exactly two non-empty rows", id="uuid-empty"),
        pytest.param(
            0,
            (
                "0, NVIDIA A100-SXM4-40GB, not-a-gpu-uuid, 40960\n1, NVIDIA A100-SXM4-40GB, "
                "GPU-22222222-2222-2222-2222-222222222222, 40960\n"
            ),
            0,
            "Link 0: 25 GB/s\n",
            "GPU UUID is malformed",
            id="uuid-malformed",
        ),
        pytest.param(
            0,
            (
                "0, NVIDIA A100-SXM4-40GB, GPU-11111111-1111-1111-1111-111111111111, 40960\n1, NVIDIA "
                "A100-SXM4-40GB, GPU-11111111-1111-1111-1111-111111111111, 40960\n"
            ),
            0,
            "Link 0: 25 GB/s\n",
            "GPU UUIDs must be unique",
            id="uuid-duplicate",
        ),
        pytest.param(
            0,
            (
                "0, NVIDIA A100-SXM4-40GB, GPU-11111111-1111-1111-1111-111111111111, 40960\n1, NVIDIA "
                "A100-SXM4-40GB, GPU-22222222-2222-2222-2222-222222222222, 40960\n"
            ),
            1,
            "",
            "NVLink status returned 1",
            id="nvlink-nonzero",
        ),
        pytest.param(
            0,
            (
                "0, NVIDIA A100-SXM4-40GB, GPU-11111111-1111-1111-1111-111111111111, 40960\n1, NVIDIA "
                "A100-SXM4-40GB, GPU-22222222-2222-2222-2222-222222222222, 40960\n"
            ),
            0,
            "",
            "NVLink status output is empty",
            id="nvlink-empty",
        ),
        pytest.param(
            0,
            (
                "0, NVIDIA A100-SXM4-80GB, GPU-11111111-1111-1111-1111-111111111111, 81920\n1, NVIDIA "
                "A100-SXM4-80GB, GPU-22222222-2222-2222-2222-222222222222, 81920\n"
            ),
            0,
            "Link 0: 25 GB/s\n",
            "UUID inventory must prove two A100-40GB devices",
            id="scheduler-upgraded-to-80gb",
        ),
    ],
)
def test_pre_torchrun_evidence_rejects_invalid_uuid_or_nvlink_output(
    uuid_returncode: int,
    uuid_stdout: str,
    nvlink_returncode: int,
    nvlink_stdout: str,
    match: str,
) -> None:
    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[1:3] == ["topo", "-m"]:
            return subprocess.CompletedProcess(command, 255, stdout="", stderr="topo unavailable")
        if "--query-gpu=index,name,uuid,memory.total" in command:
            return subprocess.CompletedProcess(command, uuid_returncode, stdout=uuid_stdout, stderr="uuid stderr")
        return subprocess.CompletedProcess(command, nvlink_returncode, stdout=nvlink_stdout, stderr="nvlink stderr")

    with pytest.raises(RuntimeError, match=match):
        modal_smoke.capture_pre_torchrun_interconnect(run=run)


def test_rejected_pre_torchrun_evidence_is_durable_with_raw_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = pii350_ddp_smoke.render_ddp_smoke_contract()
    raw_evidence = {
        "availability": "validated",
        "topology_returncode": 255,
        "topology_stdout": "",
        "uuid_inventory_raw": ["0, NVIDIA A100-SXM4-40GB, malformed, 40960"],
        "nvlink_status_raw": "GPU 0: Link 0: 25 GB/s\n",
    }
    monkeypatch.setattr(modal_smoke, "ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        modal_smoke,
        "_validated_receipt_inventory",
        lambda _baseline: (
            ("/cache/hf/packed/data/train-00000.parquet",),
            {"body_tensor_count": 1},
        ),
    )
    monkeypatch.setattr(modal_smoke.time, "time_ns", lambda: 7)
    commits: list[str] = []
    monkeypatch.setattr(modal_smoke.artifacts, "commit", lambda: commits.append("commit"))

    def reject() -> dict[str, object]:
        msg = "UUID inventory evidence is invalid"
        raise modal_smoke.PreTorchrunInterconnectError(msg, evidence=raw_evidence)

    monkeypatch.setattr(modal_smoke, "capture_pre_torchrun_interconnect", reject)

    with pytest.raises(modal_smoke.PreTorchrunInterconnectError):
        modal_smoke.run_pii350_ddp_smoke.get_raw_f()(
            contract,
            execute=True,
            confirmation=pii350_ddp_smoke.DDP_SMOKE_CONFIRMATION,
            primary_action=pii350_ddp_smoke.DDP_SMOKE_PRIMARY_ACTION,
        )

    result = (tmp_path / f"{contract['config_digest']}-7" / "result.json").read_text(encoding="utf-8")
    assert commits == ["commit"]
    assert "malformed" in result
    assert "GPU 0: Link 0: 25 GB/s" in result


def test_collective_readiness_probe_is_gpu_free_testable_and_rejects_bad_sum() -> None:
    class Tensor:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    class Cuda:
        @staticmethod
        def can_device_access_peer(local_rank: int, peer_rank: int) -> bool:
            return (local_rank, peer_rank) == (0, 1)

    class Torch:
        cuda = Cuda()
        int64 = object()

        @staticmethod
        def tensor(value: int, *, device: int, dtype: object) -> Tensor:
            assert (value, device, dtype) == (1, 0, Torch.int64)
            return Tensor(value)

    class Distributed:
        class ReduceOp:
            SUM = object()

        def __init__(self, reduced_sum: int) -> None:
            self.reduced_sum = reduced_sum
            self.all_reduce_calls = 0

        def all_gather_object(self, gathered: list[object], value: dict[str, object]) -> None:
            if "can_access_peer" in value:
                gathered[:] = [
                    {"rank": 0, "peer_rank": 1, "can_access_peer": True},
                    {"rank": 1, "peer_rank": 0, "can_access_peer": True},
                ]
            else:
                gathered[:] = [
                    {"rank": 0, "observed_all_reduce_sum": self.reduced_sum},
                    {"rank": 1, "observed_all_reduce_sum": self.reduced_sum},
                ]

        def all_reduce(self, tensor: Tensor, *, op: object) -> None:
            assert op is self.ReduceOp.SUM
            self.all_reduce_calls += 1
            tensor.value = self.reduced_sum

    passing = Distributed(reduced_sum=3)
    assert (
        modal_smoke._require_collective_ready(Torch(), passing, local_rank=0, rank=0)["nccl_all_reduce"]["observed_sum"]
        == 3
    )
    assert passing.all_reduce_calls == 1

    with pytest.raises(RuntimeError, match="returned 2, expected 3"):
        modal_smoke._require_collective_ready(Torch(), Distributed(reduced_sum=2), local_rank=0, rank=0)


def test_torchrun_failure_result_and_event_commit_before_exception(tmp_path: Path) -> None:
    calls: list[str] = []
    root = tmp_path / "artifacts"
    root.mkdir()
    result_path = root / "result.json"

    outcome = modal_smoke.record_torchrun_outcome(
        root=root,
        result_path=result_path,
        result={"status": "failed", "error": "rank zero topology failure"},
        returncode=1,
        stdout='{"event":"rank_setup"}\n',
        stderr="torchrun failure\n",
        commit=lambda: calls.append("commit"),
    )

    assert calls == ["commit"]
    assert outcome["status"] == "failed"
    assert outcome["torchrun_returncode"] == 1
    assert outcome["torchrun_stdout_tail"] == '{"event":"rank_setup"}\n'
    assert outcome["torchrun_stderr_tail"] == "torchrun failure\n"
    assert result_path.is_file()
    events = (root / "events.jsonl").read_text(encoding="utf-8")
    assert '"event": "torchrun_failed"' in events


def test_torchrun_start_failure_commits_the_failure_before_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = pii350_ddp_smoke.render_ddp_smoke_contract()
    spec = modal_smoke.ChildSpec(
        contract=contract,
        matched_constant3_contract=modal_smoke.matched_constant3_contract(),
        shard_paths=("/cache/hf/packed/data/train-00000.parquet",),
        expected_encoder_checkpoint_attestation={"body_tensor_count": 1},
        artifact_dir=str(tmp_path / "artifacts"),
        outer_modal_started_monotonic=100.0,
        absolute_training_deadline_monotonic=1_600.0,
        pre_torchrun_interconnect=_valid_pre_torchrun_evidence(),
    )
    commits: list[str] = []
    monkeypatch.setattr(
        modal_smoke.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("torchrun missing")),
    )
    monkeypatch.setattr(modal_smoke.artifacts, "commit", lambda: commits.append("commit"))

    with pytest.raises(RuntimeError, match="torchrun failed"):
        modal_smoke._run_isolated_torchrun(spec)

    assert commits == ["commit"]
    result = (tmp_path / "artifacts" / "result.json").read_text(encoding="utf-8")
    events = (tmp_path / "artifacts" / "events.jsonl").read_text(encoding="utf-8")
    assert "torchrun process could not start" in result
    assert '"event": "torchrun_failed"' in events


def test_expired_child_deadline_fails_before_child_contract_or_cuda_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = pii350_ddp_smoke.render_ddp_smoke_contract()
    spec = modal_smoke.ChildSpec(
        contract=contract,
        matched_constant3_contract=modal_smoke.matched_constant3_contract(),
        shard_paths=("/cache/hf/packed/data/train-00000.parquet",),
        expected_encoder_checkpoint_attestation={"body_tensor_count": 1},
        artifact_dir="/artifacts/pii350-ddp-smoke/test",
        outer_modal_started_monotonic=100.0,
        absolute_training_deadline_monotonic=1_600.0,
        pre_torchrun_interconnect=_valid_pre_torchrun_evidence(),
    )
    monkeypatch.setattr(modal_smoke.time, "monotonic", lambda: 1_600.0)
    monkeypatch.setattr(
        modal_smoke,
        "validate_child_spec",
        lambda _spec: pytest.fail("expired child must fail before child contract/CUDA setup"),
    )

    with pytest.raises(RuntimeError, match="before worker setup"):
        modal_smoke._run_worker(spec)


def test_torchrun_command_is_single_node_two_rank_and_no_hidden_background_process() -> None:
    command = modal_smoke.torchrun_command("/spec.json", "/result.json")
    assert command[:4] == ("torchrun", "--standalone", "--nproc_per_node=2", "-m")
    assert command[-4:] == (
        "--child-spec",
        "/spec.json",
        "--child-result",
        "/result.json",
    )


def test_a100_memory_attestation_accepts_allocatable_39_to_40_gib_not_impossible_exact_40() -> None:
    class Properties:
        name = "NVIDIA A100-SXM4-40GB"
        total_memory = 39 * 1024**3 + 512

    class Cuda:
        def is_available(self) -> bool:
            return True

        def device_count(self) -> int:
            return 2

        def set_device(self, index: int) -> None:
            assert index == 0

        def get_device_properties(self, index: int) -> Properties:
            assert index in {0, 1}
            return Properties()

    class Torch:
        cuda = Cuda()

    result = modal_smoke._require_exact_a100_topology(Torch(), 0)
    assert result["memory_attestation_bytes"]["minimum"] == 39 * 1024**3


def test_ddp_preflight_is_declared_on_this_app_not_a_cross_app_remote_call() -> None:
    assert modal_smoke.preflight_ddp_assets.app is modal_smoke.app


def test_ddp_prewarm_is_same_app_cpu_only_and_explicitly_online() -> None:
    assert modal_smoke.prewarm_ddp_assets.app is modal_smoke.app
    assert modal_smoke.CPU_PREWARM_OPTIONS == {
        "gpu": None,
        "cpu": 12.0,
        "memory": 64 * 1024,
        "timeout": 3_600,
        "max_containers": 1,
    }
    assert modal_smoke.ONLINE_CACHE_ENVIRONMENT["HF_HUB_OFFLINE"] == "0"
    assert modal_smoke.ONLINE_CACHE_ENVIRONMENT["HF_DATASETS_OFFLINE"] == "0"
    assert modal_smoke.ONLINE_CACHE_ENVIRONMENT["TRANSFORMERS_OFFLINE"] == "0"


@pytest.mark.parametrize(
    ("preflight_assets", "execute", "confirmation", "primary_action", "config_digest"),
    [
        (True, False, "", "", ""),
        (False, True, "", "", ""),
        (False, False, "unexpected", "", ""),
        (False, False, "", "unexpected", ""),
        (False, False, "", "", "digest"),
    ],
)
# reason: pytest binds this parameter from the parametrize argnames tuple, so the boolean is labelled at every call site.
def test_ddp_prewarm_rejects_every_receipt_or_paid_launch_flag(
    preflight_assets: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    execute: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    confirmation: str,
    primary_action: str,
    config_digest: str,
) -> None:
    with pytest.raises(RuntimeError, match="only --prewarm-assets"):
        modal_smoke.require_ddp_prewarm_only(
            preflight_assets=preflight_assets,
            execute=execute,
            confirmation=confirmation,
            primary_action=primary_action,
            config_digest=config_digest,
        )
