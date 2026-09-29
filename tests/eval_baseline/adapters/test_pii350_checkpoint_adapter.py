from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from meddies_pii.eval_baseline.adapters import pii350_checkpoint
from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_WEIGHTS_FILENAME,
    CLASSIFIER_FILENAME,
    LEGACY_STEP60_CHECKPOINT_DIGEST,
    LEGACY_STEP60_METADATA,
    LEGACY_STEP60_TRAJECTORY_DIGEST,
    METADATA_FILENAME,
    OPTIMIZER_FILENAME,
    RNG_FILENAME,
    STEP100_WAVE1_CHECKPOINT_DIGEST,
    STEP100_WAVE1_EXECUTION_CONTRACT_DIGEST,
    STEP100_WAVE1_PARENT_CHECKPOINT_DIGEST,
    STEP150_WAVE2_CHECKPOINT_DIGEST,
    STEP150_WAVE2_EXECUTION_CONTRACT_DIGEST,
    STEP150_WAVE2_PARENT_CHECKPOINT_DIGEST,
    CheckpointArtifact,
    _checkpoint_digest,
    verify_checkpoint_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path

STEP150_WAVE2_METADATA_FIXTURE = {
    "schema_version": 2,
    "trajectory_digest": LEGACY_STEP60_TRAJECTORY_DIGEST,
    "execution_contract_digest": STEP150_WAVE2_EXECUTION_CONTRACT_DIGEST,
    "parent_checkpoint_digest": STEP150_WAVE2_PARENT_CHECKPOINT_DIGEST,
    "optimizer_step": 150,
    "packed_cursor": 19_200,
    "lifecycle_state": "update_committed",
    "world_size": 4,
    "rank_mapping": [0, 1, 2, 3],
    "rng_transition": {
        "kind": "two_rank_to_four_rank_v1",
        "rank_mapping": [0, 1, 2, 3],
    },
    "rank_rng_states": [
        {
            "rank": rank,
            "cpu_rng": f"opaque-cpu-{rank}",
            "cuda_rng": f"opaque-cuda-{rank}",
        }
        for rank in range(4)
    ],
    "epoch_complete": False,
    "wave_profile": "huyhoang0411ha",
    "wave_cumulative_all_in_cost_usd": 1.680104012429088,
}
"""Immutable revision 310864af3f15645f70d1c016b8645e5988fbfbcd provenance contract.

Opaque RNG state bytes are only shape-validated; the allowlisted checkpoint digest and manifest bind their exact published
bytes at runtime.

The allowlisted artifact digest and verified manifest bind the opaque RNG bytes. This fixture verifies the semantic
transition contract.

"""


STEP100_WAVE1_METADATA_FIXTURE = {
    "schema_version": 2,
    "trajectory_digest": LEGACY_STEP60_TRAJECTORY_DIGEST,
    "execution_contract_digest": STEP100_WAVE1_EXECUTION_CONTRACT_DIGEST,
    "parent_checkpoint_digest": STEP100_WAVE1_PARENT_CHECKPOINT_DIGEST,
    "optimizer_step": 100,
    "packed_cursor": 12_800,
    "lifecycle_state": "update_committed",
    "world_size": 2,
    "rank_mapping": [0, 1],
    "rng_transition": {
        "kind": "single_rank_to_two_rank_v1",
        "rank_mapping": [0, 1],
    },
    "rank_rng_states": [
        {
            "rank": rank,
            "cpu_rng": f"immutable-cpu-rank-{rank}",
            "cuda_rng": f"immutable-cuda-rank-{rank}",
        }
        for rank in range(2)
    ],
    "epoch_complete": False,
    "wave_profile": "meddiesresearch",
    "wave_cumulative_all_in_cost_usd": 4.85716643914298,
}


def _legacy_step60_artifact() -> CheckpointArtifact:
    return CheckpointArtifact(
        repo_id="Meddies/private-pii350",
        revision="c" * 40,
        path="full-runs/pii350/checkpoints/step-00000060",
        checkpoint_digest=LEGACY_STEP60_CHECKPOINT_DIGEST,
    )


def _write_legacy_step60_artifact(root: Path) -> CheckpointArtifact:
    files = {
        ADAPTER_CONFIG_FILENAME: b'{"peft_type":"LORA","r":128,"lora_alpha":256}',
        ADAPTER_WEIGHTS_FILENAME: b"adapter-weights",
        CLASSIFIER_FILENAME: b"classifier-state",
        OPTIMIZER_FILENAME: b"optimizer-state",
        RNG_FILENAME: b"rng-state",
        METADATA_FILENAME: json.dumps(LEGACY_STEP60_METADATA, sort_keys=True).encode(),
    }
    manifest_body: dict[str, Any] = {
        "schema_version": 1,
        "committed": True,
        "files": [
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for relative, content in sorted(files.items())
        ],
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "manifest.json").write_text(
        json.dumps({**manifest_body, "checkpoint_digest": LEGACY_STEP60_CHECKPOINT_DIGEST}),
        encoding="utf-8",
    )
    return _legacy_step60_artifact()


def test_allowlisted_legacy_step60_metadata_is_verified_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _write_legacy_step60_artifact(tmp_path)
    monkeypatch.setattr(
        pii350_checkpoint,
        "_checkpoint_digest",
        lambda _metadata, _manifest: LEGACY_STEP60_CHECKPOINT_DIGEST,
    )

    verified = verify_checkpoint_artifact(tmp_path, artifact)

    assert verified.trajectory_digest == LEGACY_STEP60_TRAJECTORY_DIGEST
    assert verified.optimizer_step == 60


@pytest.mark.parametrize("field", sorted(LEGACY_STEP60_METADATA))
def test_allowlisted_legacy_step60_rejects_each_tampered_provenance_field(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    artifact = _legacy_step60_artifact()
    tampered = dict(LEGACY_STEP60_METADATA)
    tampered[field] = 61 if field in {"optimizer_step", "packed_cursor"} else "tampered"
    monkeypatch.setattr(
        pii350_checkpoint,
        "_checkpoint_digest",
        lambda _metadata, _manifest: LEGACY_STEP60_CHECKPOINT_DIGEST,
    )

    with pytest.raises(RuntimeError, match="legacy step-60 checkpoint provenance"):
        pii350_checkpoint._require_metadata(tampered, artifact, {"committed": True})


def test_allowlisted_legacy_step60_rejects_wrong_artifact_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong = CheckpointArtifact(
        repo_id="Meddies/private-pii350",
        revision="c" * 40,
        path="full-runs/pii350/checkpoints/step-00000060",
        checkpoint_digest="0" * 64,
    )
    monkeypatch.setattr(
        pii350_checkpoint,
        "_checkpoint_digest",
        lambda _metadata, _manifest: LEGACY_STEP60_CHECKPOINT_DIGEST,
    )

    with pytest.raises(RuntimeError, match="digest is not allowlisted"):
        pii350_checkpoint._require_metadata(dict(LEGACY_STEP60_METADATA), wrong, {"committed": True})


def test_allowlisted_step100_wave1_metadata_contract_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = CheckpointArtifact(
        repo_id="Meddies/pii350-trajectories-private",
        revision="310864af3f15645f70d1c016b8645e5988fbfbcd",
        path="checkpoints/step-00000100",
        checkpoint_digest=STEP100_WAVE1_CHECKPOINT_DIGEST,
    )
    monkeypatch.setattr(
        pii350_checkpoint,
        "_checkpoint_digest",
        lambda _metadata, _manifest: STEP100_WAVE1_CHECKPOINT_DIGEST,
    )
    assert pii350_checkpoint._require_metadata(dict(STEP100_WAVE1_METADATA_FIXTURE), artifact, {"committed": True})[
        1:
    ] == (100, 12_800, "update_committed", 2)


@pytest.mark.parametrize(
    ("artifact_digest", "metadata_override"),
    [
        ("0" * 64, {}),
        (STEP100_WAVE1_CHECKPOINT_DIGEST, {"world_size": 4}),
        (
            STEP100_WAVE1_CHECKPOINT_DIGEST,
            {
                "rng_transition": {
                    "kind": "four_rank_to_four_rank_v1",
                    "rank_mapping": [0, 1],
                },
            },
        ),
    ],
    ids=["wrong-digest", "wrong-world", "wrong-rng-transition"],
)
def test_allowlisted_step100_wave1_rejects_near_miss(
    monkeypatch: pytest.MonkeyPatch,
    artifact_digest: str,
    metadata_override: dict[str, object],
) -> None:
    artifact = CheckpointArtifact(
        repo_id="Meddies/pii350-trajectories-private",
        revision="310864af3f15645f70d1c016b8645e5988fbfbcd",
        path="checkpoints/step-00000100",
        checkpoint_digest=artifact_digest,
    )
    monkeypatch.setattr(
        pii350_checkpoint,
        "_checkpoint_digest",
        lambda _metadata, _manifest: artifact_digest,
    )
    with pytest.raises(RuntimeError, match=r"continuation schema|step-100 Wave1"):
        pii350_checkpoint._require_metadata(
            {**STEP100_WAVE1_METADATA_FIXTURE, **metadata_override},
            artifact,
            {"committed": True},
        )


def _write_artifact(root: Path, **metadata_overrides: object) -> CheckpointArtifact:
    files = {
        ADAPTER_CONFIG_FILENAME: b'{"peft_type":"LORA","r":128,"lora_alpha":256}',
        ADAPTER_WEIGHTS_FILENAME: b"adapter-weights",
        CLASSIFIER_FILENAME: b"classifier-state",
        OPTIMIZER_FILENAME: b"optimizer-state",
        RNG_FILENAME: b"rng-state",
    }
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "trajectory_digest": "a" * 64,
        "execution_contract_digest": "b" * 64,
        "parent_checkpoint_digest": "source-step60",
        "optimizer_step": 100,
        "packed_cursor": 12_800,
        "lifecycle_state": "update_committed",
        "world_size": 4,
        "rank_mapping": [0, 1, 2, 3],
        "rng_transition": {
            "kind": "four_rank_to_four_rank_v1",
            "rank_mapping": [0, 1, 2, 3],
        },
        "rank_rng_states": [{"rank": rank, "cpu_rng": f"cpu-{rank}", "cuda_rng": f"cuda-{rank}"} for rank in range(4)],
        "epoch_complete": False,
        "wave_profile": "meddiesresearch",
        "wave_cumulative_all_in_cost_usd": 14.0,
    }
    metadata.update(metadata_overrides)
    files[METADATA_FILENAME] = json.dumps(metadata, sort_keys=True).encode()
    manifest_body: dict[str, Any] = {
        "schema_version": 1,
        "committed": True,
        "files": [
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for relative, content in sorted(files.items())
        ],
    }
    digest = _checkpoint_digest(metadata, manifest_body)
    manifest = {**manifest_body, "checkpoint_digest": digest}
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return CheckpointArtifact(
        repo_id="Meddies/private-pii350",
        revision="b" * 40,
        path="step-00000100",
        checkpoint_digest=digest,
    )


def test_verified_checkpoint_rejects_corrupt_or_missing_manifest_before_loader(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(tmp_path)
    verified = verify_checkpoint_artifact(tmp_path, artifact)
    assert verified.optimizer_step == 100
    assert verified.trajectory_digest == "a" * 64

    (tmp_path / CLASSIFIER_FILENAME).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash verification"):
        verify_checkpoint_artifact(tmp_path, artifact)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"optimizer_step": 101}, "continuation schema"),
        ({"packed_cursor": 1}, "continuation schema"),
        ({"world_size": 1}, "continuation schema"),
        ({"lifecycle_state": "resume_source"}, "continuation schema"),
        ({"epoch_complete": True}, "continuation schema"),
        ({"wave_profile": "unknown"}, "continuation schema"),
        ({"wave_cumulative_all_in_cost_usd": -0.01}, "continuation schema"),
    ],
)
def test_verified_checkpoint_rejects_wrong_continuation_metadata(
    tmp_path: Path,
    overrides: dict[str, object],
    match: str,
) -> None:
    artifact = _write_artifact(tmp_path, **overrides)
    with pytest.raises(RuntimeError, match=match):
        verify_checkpoint_artifact(tmp_path, artifact)


def test_terminal_step_is_explicitly_rejected_from_cadence_full_benchmark(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path,
        optimizer_step=938,
        packed_cursor=120_064,
        lifecycle_state="terminal",
        epoch_complete=False,
    )
    with pytest.raises(RuntimeError, match="terminal checkpoint"):
        verify_checkpoint_artifact(tmp_path, artifact)


def test_allowlisted_terminal_step146_requires_exact_digest_cursor_and_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_metadata = {
        "optimizer_step": 146,
        "packed_cursor": 18_688,
        "lifecycle_state": "terminal",
        "world_size": 2,
        "rank_mapping": [0, 1],
        "rng_transition": {
            "kind": "four_rank_to_two_rank_v1",
            "rank_mapping": [0, 1],
        },
        "rank_rng_states": [{"rank": rank, "cpu_rng": f"cpu-{rank}", "cuda_rng": f"cuda-{rank}"} for rank in range(2)],
    }
    artifact = _write_artifact(tmp_path / "terminal", **terminal_metadata)
    monkeypatch.setattr(
        pii350_checkpoint,
        "TERMINAL_STEP146_CHECKPOINT_DIGEST",
        artifact.checkpoint_digest,
    )

    verified = verify_checkpoint_artifact(tmp_path / "terminal", artifact)

    assert (verified.optimizer_step, verified.packed_cursor) == (146, 18_688)
    assert (verified.lifecycle_state, verified.world_size) == ("terminal", 2)
    wrong_cursor = _write_artifact(tmp_path / "wrong-cursor", **{**terminal_metadata, "packed_cursor": 18_687})
    with pytest.raises(RuntimeError, match="step-146 contract"):
        verify_checkpoint_artifact(tmp_path / "wrong-cursor", wrong_cursor)
    wrong_digest = CheckpointArtifact(artifact.repo_id, artifact.revision, artifact.path, "0" * 64)
    with pytest.raises(RuntimeError, match="does not match requested checkpoint"):
        verify_checkpoint_artifact(tmp_path / "terminal", wrong_digest)
    manifest_path = tmp_path / "terminal" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["committed"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted"):
        verify_checkpoint_artifact(tmp_path / "terminal", artifact)


def test_manifest_digest_binds_producer_wave_metadata_parity(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path,
        wave_profile="meddies-ocr",
        wave_cumulative_all_in_cost_usd=0.0,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    manifest_body = {key: value for key, value in manifest.items() if key != "checkpoint_digest"}
    assert _checkpoint_digest(metadata, manifest_body) == artifact.checkpoint_digest
    assert verify_checkpoint_artifact(tmp_path, artifact).optimizer_step == 100


def test_checkpoint_digest_is_caller_bound_and_embedded_only_in_manifest(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(tmp_path)
    wrong = CheckpointArtifact(
        repo_id=artifact.repo_id,
        revision=artifact.revision,
        path=artifact.path,
        checkpoint_digest="0" * 64,
    )
    with pytest.raises(RuntimeError, match="does not match requested checkpoint"):
        verify_checkpoint_artifact(tmp_path, wrong)


def test_rejects_unmanifested_file_and_legacy_size_bytes_schema(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path)
    (tmp_path / "unverified.bin").write_bytes(b"must not deserialize")
    with pytest.raises(RuntimeError, match="unmanifested"):
        verify_checkpoint_artifact(tmp_path, artifact)

    artifact = _write_artifact(tmp_path / "legacy")
    manifest_path = tmp_path / "legacy" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["size_bytes"] = manifest["files"][0].pop("bytes")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="file entry is invalid"):
        verify_checkpoint_artifact(tmp_path / "legacy", artifact)


@pytest.mark.parametrize(
    ("optimizer_step", "packed_cursor", "wave_cost_usd"),
    [(200, 25_600, 7.688761635030914)],
    ids=["uploaded-step200"],
)
def test_wave2_four_to_four_checkpoint_profile_matches_producer_contract(
    tmp_path: Path,
    optimizer_step: int,
    packed_cursor: int,
    wave_cost_usd: float,
) -> None:
    artifact = _write_artifact(
        tmp_path,
        trajectory_digest="542478f63b1726aa863c9a0ecec1e84f6347855fa9b6776b451b40c37781dac4",
        optimizer_step=optimizer_step,
        packed_cursor=packed_cursor,
        wave_profile="huyhoang0411ha",
        wave_cumulative_all_in_cost_usd=wave_cost_usd,
    )

    verified = verify_checkpoint_artifact(tmp_path, artifact)

    assert (verified.optimizer_step, verified.packed_cursor) == (
        optimizer_step,
        packed_cursor,
    )


def test_wave2_profile_typo_is_not_a_valid_checkpoint_profile(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path,
        wave_profile="huyhoangha0411",
    )

    with pytest.raises(RuntimeError, match="continuation schema"):
        verify_checkpoint_artifact(tmp_path, artifact)


def test_allowlisted_step150_wave2_metadata_contract_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = CheckpointArtifact(
        repo_id="Meddies/pii350-trajectories-private",
        revision="74bfef38027a26a430cda60b9ee9e59b986c769e",
        path="trajectories/542478f63b1726aa863c9a0ecec1e84f6347855fa9b6776b451b40c37781dac4/checkpoints/step-00000150",
        checkpoint_digest=STEP150_WAVE2_CHECKPOINT_DIGEST,
    )
    monkeypatch.setattr(
        pii350_checkpoint,
        "_checkpoint_digest",
        lambda _metadata, _manifest: STEP150_WAVE2_CHECKPOINT_DIGEST,
    )

    assert pii350_checkpoint._require_metadata(dict(STEP150_WAVE2_METADATA_FIXTURE), artifact, {"committed": True})[
        1:
    ] == (150, 19_200, "update_committed", 4)


@pytest.mark.parametrize(
    ("artifact_digest", "metadata_override", "match"),
    [
        (
            STEP150_WAVE2_CHECKPOINT_DIGEST,
            {"parent_checkpoint_digest": "0" * 64},
            "step-150 Wave2",
        ),
        (
            STEP150_WAVE2_CHECKPOINT_DIGEST,
            {
                "rng_transition": {
                    "kind": "four_rank_to_four_rank_v1",
                    "rank_mapping": [0, 1, 2, 3],
                },
            },
            "step-150 Wave2",
        ),
        ("0" * 64, {}, "continuation schema"),
    ],
    ids=["wrong-parent", "wrong-transition", "unallowlisted-digest"],
)
def test_allowlisted_step150_wave2_rejects_near_miss(
    monkeypatch: pytest.MonkeyPatch,
    artifact_digest: str,
    metadata_override: dict[str, object],
    match: str,
) -> None:
    artifact = CheckpointArtifact(
        repo_id="Meddies/pii350-trajectories-private",
        revision="74bfef38027a26a430cda60b9ee9e59b986c769e",
        path=("trajectories/542478f63b1726aa863c9a0ecec1e84f6347855fa9b6776b451b40c37781dac4/checkpoints/step-00000150"),
        checkpoint_digest=artifact_digest,
    )
    monkeypatch.setattr(
        pii350_checkpoint,
        "_checkpoint_digest",
        lambda _metadata, _manifest: artifact_digest,
    )

    with pytest.raises(RuntimeError, match=match):
        pii350_checkpoint._require_metadata(
            {**STEP150_WAVE2_METADATA_FIXTURE, **metadata_override},
            artifact,
            {"committed": True},
        )
