from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
import ast
import inspect
import json
from hashlib import sha256
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.modal import (
    pii350_ddp_continuation as modal_continuation,
)
from meddies_pii.training.bioes.trainers import pii350_ddp_continuation as continuation


@runtime_checkable
class _Readable(Protocol):
    def read(self) -> str | bytes: ...


def _receipt_field(receipt: dict[str, object], key: str) -> dict[str, object]:
    """Return a nested receipt block, asserting the shape the upload path relies on.

    Returns:
        The mapping stored at `key`, proven to be string-keyed.

    """
    value = receipt[key]
    assert is_str_mapping(value)
    return dict(value)


def test_modal_contract_uses_exact_two_a100_for_wave_one_and_detached_gated_commands() -> None:
    contract = continuation.render_segment_contract("meddiesresearch")
    assert modal_continuation.TRAIN_OPTIONS == {
        "gpu": "A100-40GB:2",
        "cpu": 4.0,
        "memory": 64 * 1024,
        "timeout": 7_900,
        "max_containers": 1,
    }
    assert modal_continuation.train_options("huyhoang0411ha") == {
        "gpu": "A100-40GB:4",
        "cpu": 8.0,
        "memory": 128 * 1024,
        "timeout": 10_800,
        "max_containers": 1,
    }
    assert modal_continuation.train_options("meddies-ocr") == {
        "gpu": "A100-40GB:4",
        "cpu": 8.0,
        "memory": 128 * 1024,
        "timeout": 10_800,
        "max_containers": 1,
    }
    assert "--execute" not in continuation.render_cpu_receipt_command("meddiesresearch")
    train_command = continuation.render_train_command("meddiesresearch")
    assert "--detach" in train_command
    assert "--profile meddiesresearch" in train_command
    assert contract["execution_contract_digest"] in train_command
    assert "--confirmation LAUNCH_PII350_DDP2_CONTINUATION" in (continuation.render_train_command("meddiesresearch"))
    assert "--primary-action HA_AUTHORIZE_PII350_R128A256_B128_LR4E4_DDP2_CONTINUATION" in (
        continuation.render_train_command("meddiesresearch")
    )


def test_hub_transport_explicitly_enables_network_without_relaxing_training() -> None:
    assert modal_continuation.CACHE_ENVIRONMENT["HF_HUB_OFFLINE"] == "1"
    assert modal_continuation.CACHE_ENVIRONMENT["HF_DATASETS_OFFLINE"] == "1"
    assert modal_continuation.CACHE_ENVIRONMENT["TRANSFORMERS_OFFLINE"] == "1"
    assert modal_continuation.HUB_TRANSPORT_ENVIRONMENT["HF_HUB_OFFLINE"] == "0"
    assert modal_continuation.HUB_TRANSPORT_ENVIRONMENT["HF_DATASETS_OFFLINE"] == "0"
    assert modal_continuation.HUB_TRANSPORT_ENVIRONMENT["TRANSFORMERS_OFFLINE"] == "0"

    source = inspect.getsource(modal_continuation)
    assert source.count("@app.function(image=transport_image") == 2
    assert (
        "@app.function(image=transport_image, volumes=MOUNTS, secrets=[secret], **CPU_OPTIONS)\ndef upload_checkpoint("
    ) in source
    assert (
        "@app.function(image=transport_image, volumes=MOUNTS, secrets=[secret], **CPU_OPTIONS)\ndef download_checkpoint("
    ) in source
    assert ("@app.function(image=image, volumes=MOUNTS, secrets=[secret], **CPU_OPTIONS)\ndef cpu_preflight") in source
    assert "@app.function(\n    image=image," in source


def test_modal_image_branches_finish_with_local_source_after_every_build_step() -> None:
    source = inspect.getsource(modal_continuation)
    tree = ast.parse(source)
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"base_image", "image", "transport_image"}
    }

    def chained_methods(expression: ast.expr) -> tuple[str, ...]:
        methods: list[str] = []
        while isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute):
            methods.append(expression.func.attr)
            expression = expression.func.value
        return tuple(reversed(methods))

    assert chained_methods(assignments["base_image"]) == ()
    assert chained_methods(assignments["image"]) == ("env", "add_local_dir")
    assert chained_methods(assignments["transport_image"]) == (
        "env",
        "add_local_dir",
    )


def test_remote_authorization_fails_closed_on_cross_topology_tokens() -> None:
    wave_one = continuation.render_segment_contract("meddiesresearch")
    wave_two = continuation.render_segment_contract("huyhoang0411ha")
    ddp2 = wave_one["manual_launch"]
    ddp4 = wave_two["manual_launch"]

    with pytest.raises(RuntimeError, match="exact profile topology authorization"):
        modal_continuation._train_segment_impl(
            "meddiesresearch",
            wave_one["execution_contract_digest"],
            execute=True,
            confirmation=ddp4["confirmation"],
            primary_action=ddp4["primary_action"],
        )
    with pytest.raises(RuntimeError, match="source checkpoint receipt"):
        modal_continuation._train_segment_impl(
            "meddiesresearch",
            wave_one["execution_contract_digest"],
            execute=True,
            confirmation=ddp2["confirmation"],
            primary_action=ddp2["primary_action"],
        )
    with pytest.raises(RuntimeError, match="exact profile topology authorization"):
        modal_continuation._train_segment_impl(
            "huyhoang0411ha",
            wave_two["execution_contract_digest"],
            execute=True,
            confirmation=ddp2["confirmation"],
            primary_action=ddp2["primary_action"],
        )
    with pytest.raises(RuntimeError, match="source checkpoint receipt"):
        modal_continuation._train_segment_impl(
            "huyhoang0411ha",
            wave_two["execution_contract_digest"],
            execute=True,
            confirmation=ddp4["confirmation"],
            primary_action=ddp4["primary_action"],
        )


def test_destination_receipt_requires_digest_and_verifies_before_load(
    tmp_path: Path,
) -> None:
    files = {
        "adapter/adapter_config.json": b"adapter",
        "adapter/adapter_model.safetensors": b"weights",
        "classifier.pt": b"state",
        "optimizer.pt": b"optimizer",
        "rng.pt": b"rng",
        "metadata.json": b'{"optimizer_step":61,"packed_cursor":7808,"world_size":4}',
    }
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    body = continuation.build_manifest({name: tmp_path / name for name in files})
    manifest = {
        **body,
        "checkpoint_digest": continuation.checkpoint_digest(
            {"optimizer_step": 61, "packed_cursor": 7_808, "world_size": 4},
            body,
        ),
    }
    receipt = modal_continuation.verify_downloaded_checkpoint(tmp_path, manifest, purpose="evaluation")
    assert receipt["purpose"] == "evaluation"
    assert receipt["verified_files"] == sorted(files)
    with __import__("pytest").raises(RuntimeError, match="digest"):
        modal_continuation.verify_downloaded_checkpoint(tmp_path, body, purpose="evaluation")


def test_torchrun_executor_uses_contract_visible_local_ranks() -> None:
    # reason: `torchrun_command` composes an argv tuple and returns it. Both literals are arguments
    # reason: and the assertions read the composed strings, so nothing here opens a path — the same
    # reason: pure-composition class the open lane proved with an audit hook.
    command = modal_continuation.torchrun_command("/tmp/spec.json", "/tmp/result.json", world_size=2)  # ruff: ignore[hardcoded-temp-file]
    assert command[:3] == ("torchrun", "--standalone", "--nproc_per_node=2")
    assert "pii350_ddp_continuation" in command[4]
    assert modal_continuation.torchrun_command("/tmp/spec.json", "/tmp/result.json")[2] == "--nproc_per_node=4"  # ruff: ignore[hardcoded-temp-file]


def test_wave_one_modal_wrapper_and_batches_match_the_two_gpu_contract() -> None:
    source = inspect.getsource(modal_continuation)
    assert 'gpu="A100-40GB:2"' in source
    assert "def train_segment_7900_world2" in source
    assert "timeout=7_900" in source
    assert "(2, 7_900): train_segment_7900_world2" in source
    assert "train_segment_8300_world2" not in source
    assert "(2, 8_300)" not in source
    assert "def train_segment_7000" in source
    assert "timeout=7_000" in source
    assert "(4, 7_000): train_segment_7000" in source
    assert "def train_segment_6000" in source
    assert "timeout=6_000" in source
    assert "(4, 6_000): train_segment_6000" in source
    assert "(2, 8_000)" not in source
    assert "train_segment_8400_world2" not in source
    assert "(2, 8_400)" not in source
    rows = [{"row": index} for index in range(128)]
    batches = [modal_continuation._rank_batch(rows, cursor=7_680, rank=rank, world_size=2) for rank in range(2)]
    assert [len(batch) for batch in batches] == [64, 64]
    assert [row["row"] for batch in batches for row in batch] == list(range(128))


def test_manifest_digest_binds_metadata_without_cyclic_metadata_field(
    tmp_path: Path,
) -> None:
    files = {
        "adapter/adapter_config.json": b"adapter",
        "adapter/adapter_model.safetensors": b"weights",
        "classifier.pt": b"state",
        "optimizer.pt": b"optimizer",
        "rng.pt": b"rng",
        "metadata.json": b'{"optimizer_step":61,"packed_cursor":7808}',
    }
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest_body = continuation.build_manifest({name: tmp_path / name for name in files})
    manifest = {
        **manifest_body,
        "checkpoint_digest": continuation.checkpoint_digest({"optimizer_step": 61, "packed_cursor": 7_808}, manifest_body),
    }
    receipt = modal_continuation.verify_downloaded_checkpoint(tmp_path, manifest, purpose="training")
    assert receipt["checkpoint_digest"] == manifest["checkpoint_digest"]


def test_source_inventory_manifests_real_extra_files_and_keeps_canonical_payloads(
    tmp_path: Path,
) -> None:
    files = {
        "adapter/adapter_config.json": b"adapter",
        "adapter/adapter_model.safetensors": b"weights",
        "adapter/README.md": b"peft readme",
        "classifier.pt": b"state",
        "optimizer.pt": b"optimizer",
        "rng.pt": b"rng",
        "metadata.json": b'{"optimizer_step":60,"packed_cursor":7680}',
    }
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    receipt = modal_continuation.inventory_source_checkpoint(tmp_path)
    assert receipt["world_size"] == 1
    assert "adapter/README.md" in [item["path"] for item in receipt["manifest"]["files"]]


def test_legacy_step60_without_persisted_manifest_derives_only_the_verified_upload_manifest(
    tmp_path: Path,
) -> None:
    files = {
        "adapter/adapter_config.json": b"adapter",
        "adapter/adapter_model.safetensors": b"weights",
        "classifier.pt": b"state",
        "optimizer.pt": b"optimizer",
        "rng.pt": b"rng",
        "metadata.json": b'{"optimizer_step":60,"packed_cursor":7680}',
    }
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    source_receipt = modal_continuation.inventory_source_checkpoint(tmp_path)
    manifest, legacy = modal_continuation._validated_manifest_for_upload(tmp_path, source_receipt=source_receipt)
    assert legacy is True
    assert manifest["checkpoint_digest"] == source_receipt["checkpoint_digest"]
    (tmp_path / "classifier.pt").write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="prior hash-verified"):
        modal_continuation._validated_manifest_for_upload(tmp_path, source_receipt=source_receipt)


def _source_root_binding(
    launch: str,
    requested_root: str,
    resolved_artifact_root: str,
    resolved_root: str,
) -> str:
    return sha256(
        json.dumps(
            {
                "launch_digest": launch,
                "requested_source_root": requested_root,
                "resolved_artifact_root": resolved_artifact_root,
                "resolved_source_root": resolved_root,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
    ).hexdigest()


# reason: Checkpoint coordinates, digests, profile, world size, and roots are independent signed-receipt dimensions.
def _cpu_receipt_for_upload(  # ruff: ignore[too-many-arguments]
    *,
    checkpoint_root: str,
    step: int,
    cursor: int,
    checkpoint_digest: str = "a" * 64,
    profile: str = "meddiesresearch",
    world_size: int = 2,
    resolved_root: str | None = None,
    resolved_artifact_root: str = "/__modal/volumes/meddies-pii-bioes-artifacts",
) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "committed": True,
        "files": [
            {"path": name, "bytes": 1, "sha256": "b" * 64} for name in sorted(continuation.CHECKPOINT_REQUIRED_FILES)
        ],
        "checkpoint_digest": checkpoint_digest,
    }
    resolved = resolved_root or (resolved_artifact_root + checkpoint_root.removeprefix("/artifacts"))
    source = {
        "hash_verified": True,
        "checkpoint_digest": checkpoint_digest,
        "optimizer_step": step,
        "packed_cursor": cursor,
        "world_size": world_size,
        "epoch_complete": step == continuation.EPOCH_TERMINAL_STEP,
        "manifest": manifest,
        "source_root": resolved,
    }
    contract = continuation.render_segment_contract(profile)
    launch = continuation.launch_digest(contract["execution_contract_digest"], source)
    return {
        "profile": profile,
        "execution_contract_digest": contract["execution_contract_digest"],
        "source_checkpoint_receipt": source,
        "launch_digest": launch,
        "requested_source_root": checkpoint_root,
        "resolved_artifact_root": resolved_artifact_root,
        "source_root_binding": _source_root_binding(launch, checkpoint_root, resolved_artifact_root, resolved),
    }


def _write_checkpoint(tmp_path: Path, *, step: int = 100, cursor: int = 12_800) -> None:
    files = {
        "adapter/adapter_config.json": b"adapter",
        "adapter/adapter_model.safetensors": b"weights",
        "classifier.pt": b"state",
        "optimizer.pt": b"optimizer",
        "rng.pt": b"rng",
        "metadata.json": (f'{{"optimizer_step":{step},"packed_cursor":{cursor},"world_size":2}}').encode(),
    }
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_cpu_receipt_binds_requested_artifact_root_to_resolved_modal_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = "/artifacts/checkpoints/step-00000100"
    expected = _cpu_receipt_for_upload(checkpoint_root=checkpoint_root, step=100, cursor=12_800)
    source = expected["source_checkpoint_receipt"]
    assert isinstance(source, dict)
    monkeypatch.setattr(modal_continuation, "inventory_source_checkpoint", lambda _: source)
    monkeypatch.setattr(
        modal_continuation,
        "_resolved_artifact_root",
        lambda: "/__modal/volumes/meddies-pii-bioes-artifacts",
    )

    receipt = modal_continuation.cpu_receipt.local("meddiesresearch", checkpoint_root)

    assert receipt["requested_source_root"] == checkpoint_root
    assert receipt["source_checkpoint_receipt"]["source_root"].startswith("/__modal/volumes/")
    assert receipt["source_root_binding"] == _source_root_binding(
        receipt["launch_digest"],
        receipt["requested_source_root"],
        receipt["resolved_artifact_root"],
        receipt["source_checkpoint_receipt"]["source_root"],
    )


def test_persisted_step100_manifest_uses_strict_upload_branch(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    persisted = modal_continuation._manifest_with_digest(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(persisted), encoding="utf-8")

    manifest, legacy = modal_continuation._validated_manifest_for_upload(tmp_path)

    assert legacy is False
    assert manifest == persisted


def test_persisted_manifest_must_match_its_cpu_receipt(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    persisted = modal_continuation._manifest_with_digest(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(persisted), encoding="utf-8")
    source_receipt = modal_continuation.inventory_source_checkpoint(tmp_path)
    forged_receipt = {
        **source_receipt,
        "checkpoint_digest": "f" * 64,
    }

    with pytest.raises(RuntimeError, match="bytes do not match"):
        modal_continuation._validated_manifest_for_upload(tmp_path, source_receipt=forged_receipt)


def test_persisted_manifest_rejects_world_size_outside_profile_contract(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        '{"optimizer_step":100,"packed_cursor":12800,"world_size":1}',
        encoding="utf-8",
    )
    persisted = modal_continuation._manifest_with_digest(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(RuntimeError, match="world size does not match"):
        modal_continuation._validated_manifest_for_upload(tmp_path, profile="meddiesresearch")


@pytest.mark.parametrize("mutation_timing", ["before", "during"])
def test_upload_manifest_last_rejects_mutation_before_manifest_commit(tmp_path: Path, mutation_timing: str) -> None:
    _write_checkpoint(tmp_path)
    validated = modal_continuation._manifest_with_digest(tmp_path)
    uploaded_paths: list[str] = []

    class HubApi:
        mutated = False

        # reason: every method on the doubles in this module is called on an INSTANCE by the code under
        # reason: test — `api.create_repo(...)`, `api.upload_file(...)`, `volume.commit()`, `fn.remote(...)`.
        # reason: A staticmethod would not be reachable through the attribute lookup the production path
        # reason: performs, so `self` is contract even where the body ignores it. Same reasoning covers
        # reason: the unused `*args`/`**kwargs`: they absorb the real call shapes.
        def create_repo(self, **_: object) -> None:  # ruff: ignore[no-self-use]
            return None

        def upload_file(self, *, path_in_repo: str, **_: object) -> object:
            uploaded_paths.append(path_in_repo)
            if mutation_timing == "during" and not self.mutated:
                self.mutated = True
                (tmp_path / "classifier.pt").write_bytes(b"mutated")
            return type("Commit", (), {"oid": "commit-sha"})()

    if mutation_timing == "before":
        (tmp_path / "classifier.pt").write_bytes(b"mutated")

    with pytest.raises(RuntimeError, match="changed since validation"):
        modal_continuation.upload_manifest_last(
            HubApi(),
            root=tmp_path,
            repo_id="Meddies/private",
            remote_prefix="trajectories/x/checkpoints/step-00000100",
            manifest=validated,
        )

    assert not any(path.endswith("manifest.json") for path in uploaded_paths)


def test_upload_manifest_last_commits_the_validated_manifest_exactly(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)
    validated = modal_continuation._manifest_with_digest(tmp_path)
    uploaded_manifest: list[dict[str, object]] = []

    class HubApi:
        def create_repo(self, **_: object) -> None:  # ruff: ignore[no-self-use]
            return None

        def upload_file(self, *, path_in_repo: str, path_or_fileobj: object, **_: object) -> object:  # ruff: ignore[no-self-use]
            if path_in_repo.endswith("manifest.json"):
                assert isinstance(path_or_fileobj, _Readable)
                uploaded_manifest.append(json.loads(path_or_fileobj.read()))
            return type("Commit", (), {"oid": "commit-sha"})()

    uploaded = modal_continuation.upload_manifest_last(
        HubApi(),
        root=tmp_path,
        repo_id="Meddies/private",
        remote_prefix="trajectories/x/checkpoints/step-00000100",
        manifest=validated,
    )

    assert uploaded_manifest == [validated]
    assert uploaded["checkpoint_digest"] == validated["checkpoint_digest"]


def test_new_checkpoint_without_manifest_remains_rejected(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="only the verified legacy step-60"):
        modal_continuation._validated_manifest_for_upload(tmp_path)


# reason: Fixtures and checkpoint coordinates stay explicit because the test traces them across the CLI-to-remote seam.
@pytest.mark.parametrize(
    ("checkpoint_root", "step", "cursor", "checkpoint_digest"),
    [
        (
            "/artifacts/checkpoints/step-00000100",
            100,
            12_800,
            "a" * 64,
        ),
        (
            "/artifacts/checkpoints/terminal-step-00000146",
            146,
            18_688,
            "73a490a6a90da19d077951a673fa451e3350dc81f3e09ba594ee86e182985c76",
        ),
    ],
)
def test_upload_main_passes_valid_persisted_manifest_receipt_to_remote(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    checkpoint_root: str,
    step: int,
    cursor: int,
    checkpoint_digest: str,
) -> None:
    received: list[object] = []

    class Remote:
        def remote(self, *args: object) -> dict[str, str]:  # ruff: ignore[no-self-use]
            received.extend(args)
            return {"status": "queued"}

    receipt = _cpu_receipt_for_upload(
        checkpoint_root=checkpoint_root,
        step=step,
        cursor=cursor,
        checkpoint_digest=checkpoint_digest,
    )
    monkeypatch.setenv("MODAL_PROFILE", "meddiesresearch")
    monkeypatch.setattr(modal_continuation, "upload_checkpoint", Remote())
    modal_continuation.main(
        profile="meddiesresearch",
        upload=True,
        checkpoint_root=checkpoint_root,
        remote_prefix=f"trajectories/x/checkpoints/{Path(checkpoint_root).name}",
        source_receipt_json=json.dumps(receipt),
    )
    assert received[3] == receipt["source_checkpoint_receipt"]
    assert "queued" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("receipt_json", "error"),
    [
        ("not JSON", "JSON is invalid"),
        (json.dumps({}), "lacks source_checkpoint_receipt"),
        (
            json.dumps(_cpu_receipt_for_upload(checkpoint_root="/artifacts/wrong-root", step=100, cursor=12_800)),
            "requested root does not match checkpoint_root",
        ),
        (
            json.dumps(
                _cpu_receipt_for_upload(
                    checkpoint_root="/artifacts/checkpoints/step-00000100",
                    step=100,
                    cursor=12_800,
                    # reason: the path IS the forgery under test. The receipt verifier must refuse a
                    # reason: resolved root outside the Modal volume, and "resolved root is not a Modal
                    # reason: volume path" below is that refusal being asserted.
                    resolved_root="/tmp/forged-root",  # ruff: ignore[hardcoded-temp-file]
                ),
            ),
            "resolved root is not a Modal volume path",
        ),
        (
            json.dumps(
                _cpu_receipt_for_upload(
                    checkpoint_root="/artifacts/checkpoints/step-00000100",
                    step=100,
                    cursor=12_800,
                    resolved_root="/__modal/volumes/other/checkpoints/step-00000100",
                ),
            ),
            "resolved root is not mapped from checkpoint_root",
        ),
        (
            json.dumps({
                **_cpu_receipt_for_upload(
                    checkpoint_root="/artifacts/checkpoints/step-00000100",
                    step=100,
                    cursor=12_800,
                ),
                "source_root_binding": "0" * 64,
            }),
            "root binding is invalid",
        ),
        (
            json.dumps(
                _cpu_receipt_for_upload(
                    checkpoint_root="/artifacts/checkpoints/step-00000100",
                    step=100,
                    cursor=12_800,
                    profile="huyhoang0411ha",
                ),
            ),
            "does not match the requested profile contract",
        ),
        (
            json.dumps(
                _cpu_receipt_for_upload(
                    checkpoint_root="/artifacts/checkpoints/step-00000100",
                    step=100,
                    cursor=12_800,
                    world_size=1,
                ),
            ),
            "world size does not match",
        ),
        (
            json.dumps(
                _cpu_receipt_for_upload(
                    checkpoint_root="/artifacts/checkpoints/step-00000100",
                    step=100,
                    cursor=12_800,
                    world_size=4,
                ),
            ),
            "world size does not match",
        ),
        (
            json.dumps({
                **_cpu_receipt_for_upload(
                    checkpoint_root="/artifacts/checkpoints/step-00000100",
                    step=100,
                    cursor=12_800,
                ),
                "source_checkpoint_receipt": {
                    **_receipt_field(
                        _cpu_receipt_for_upload(
                            checkpoint_root="/artifacts/checkpoints/step-00000100",
                            step=100,
                            cursor=12_800,
                        ),
                        "source_checkpoint_receipt",
                    ),
                    "checkpoint_digest": "c" * 64,
                },
            }),
            "digest does not match manifest",
        ),
    ],
)
def test_upload_main_rejects_invalid_source_receipt_locally(
    monkeypatch: pytest.MonkeyPatch,
    receipt_json: str,
    error: str,
) -> None:
    class Remote:
        def remote(self, *args: object) -> dict[str, str]:  # ruff: ignore[no-self-use,unused-method-argument]
            msg = "invalid receipt must not dispatch to Modal"
            raise AssertionError(msg)

    monkeypatch.setenv("MODAL_PROFILE", "meddiesresearch")
    monkeypatch.setattr(modal_continuation, "upload_checkpoint", Remote())
    with pytest.raises(RuntimeError, match=error):
        modal_continuation.main(
            profile="meddiesresearch",
            upload=True,
            checkpoint_root="/artifacts/checkpoints/step-00000100",
            remote_prefix="trajectories/x/checkpoints/step-00000100",
            source_receipt_json=receipt_json,
        )


def test_upload_main_passes_exact_legacy_step60_receipt_to_remote(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: list[object] = []

    class Remote:
        def remote(self, *args: object) -> dict[str, str]:  # ruff: ignore[no-self-use]
            received.extend(args)
            return {"status": "queued"}

    checkpoint_root = "/artifacts/checkpoints/step-00000060"
    receipt = _cpu_receipt_for_upload(checkpoint_root=checkpoint_root, step=60, cursor=7_680, world_size=1)
    monkeypatch.setenv("MODAL_PROFILE", "meddiesresearch")
    monkeypatch.setattr(modal_continuation, "upload_checkpoint", Remote())
    modal_continuation.main(
        profile="meddiesresearch",
        upload=True,
        checkpoint_root=checkpoint_root,
        remote_prefix="trajectories/x/checkpoints/step-00000060",
        source_receipt_json=json.dumps(receipt),
    )
    assert received[3] == receipt["source_checkpoint_receipt"]
    assert "queued" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="prior CPU source receipt"):
        modal_continuation.main(
            profile="meddiesresearch",
            upload=True,
            checkpoint_root=checkpoint_root,
            remote_prefix="prefix",
            source_receipt_json="",
        )


def test_terminal_tail_rank_batch_has_exactly_25_rows_per_rank() -> None:
    rows = [{"row": index} for index in range(100)]
    batches = [modal_continuation._rank_batch(rows, cursor=119_936, rank=rank) for rank in range(4)]
    assert [len(batch) for batch in batches] == [25, 25, 25, 25]
    assert [row["row"] for batch in batches for row in batch] == list(range(100))


def test_cpu_receipt_cost_composer_has_reconciliation_fields() -> None:
    receipt = modal_continuation._with_cpu_cost({"kind": "preflight"}, started_monotonic=10.0)
    assert set(receipt["cpu_cost"]) == {
        "cpu_elapsed_seconds",
        "estimated_cpu_compute_cost_usd",
        "cpu_all_in_rate_usd_per_second",
    }
    deterministic = modal_continuation._cpu_cost_fields(10.0, now_monotonic=15.0)
    assert deterministic["cpu_elapsed_seconds"] == 5.0
    assert deterministic["estimated_cpu_compute_cost_usd"] > 0.0


def test_continuation_preflight_builds_missing_asset_receipt_in_its_own_cpu_call(
    tmp_path: Path,
) -> None:
    contract = continuation.render_segment_contract("meddiesresearch")
    baseline = contract["trajectory"]["baseline_full_run_contract"]
    receipt_path = tmp_path / "verification-receipt.json"
    calls: list[str] = []

    def build(candidate: str, **kwargs: object) -> dict[str, object]:
        assert candidate == "pii350"
        assert kwargs["contract"] == baseline
        calls.append("build")
        return {"receipt": {"receipt": "new"}}

    def persist(persisted_contract: object, receipt: object) -> Path:
        assert persisted_contract == baseline
        assert receipt == {"receipt": "new"}
        calls.append("persist")
        receipt_path.write_text("persisted", encoding="utf-8")
        return receipt_path

    def validate(validated_contract: object) -> tuple[tuple[str, ...], dict[str, str]]:
        assert validated_contract == baseline
        calls.append("validate")
        return ("/cache/packed.parquet",), {"body_tensor_values_sha256": "a" * 64}

    paths, attestation, built = modal_continuation._ensure_pii350_baseline_assets(
        baseline,
        receipt_path=receipt_path,
        preflight_candidate_assets=build,
        persist_preflight_receipt=persist,
        validated_receipt_inventory=validate,
        preflight_inputs={},
    )
    assert paths == ("/cache/packed.parquet",)
    assert attestation == {"body_tensor_values_sha256": "a" * 64}
    assert built is True
    assert calls == ["build", "persist", "validate"]
    assert "preflight_candidate_assets.remote" not in inspect.getsource(modal_continuation)


def test_continuation_preflight_reuses_existing_valid_asset_receipt(
    tmp_path: Path,
) -> None:
    contract = continuation.render_segment_contract("meddiesresearch")
    baseline = contract["trajectory"]["baseline_full_run_contract"]
    receipt_path = tmp_path / "verification-receipt.json"
    receipt_path.write_text("valid", encoding="utf-8")

    def unexpected(*args: object, **kwargs: object) -> object:  # ruff: ignore[unused-function-argument]
        msg = "existing receipt must not rebuild or rewrite"
        raise AssertionError(msg)

    paths, attestation, built = modal_continuation._ensure_pii350_baseline_assets(
        baseline,
        receipt_path=receipt_path,
        preflight_candidate_assets=unexpected,
        persist_preflight_receipt=unexpected,
        validated_receipt_inventory=lambda _: (
            ("/cache/packed.parquet",),
            {"body_tensor_count": 1},
        ),
        preflight_inputs={},
    )
    assert paths == ("/cache/packed.parquet",)
    assert attestation == {"body_tensor_count": 1}
    assert built is False


def test_continuation_preflight_refuses_corrupt_existing_asset_receipt(
    tmp_path: Path,
) -> None:
    contract = continuation.render_segment_contract("meddiesresearch")
    baseline = contract["trajectory"]["baseline_full_run_contract"]
    receipt_path = tmp_path / "verification-receipt.json"
    receipt_path.write_text("corrupt", encoding="utf-8")

    def unexpected(*args: object, **kwargs: object) -> object:  # ruff: ignore[unused-function-argument]
        msg = "corrupt receipt must fail instead of rebuilding"
        raise AssertionError(msg)

    with pytest.raises(RuntimeError, match="receipt pin mismatch"):
        modal_continuation._ensure_pii350_baseline_assets(
            baseline,
            receipt_path=receipt_path,
            preflight_candidate_assets=unexpected,
            persist_preflight_receipt=unexpected,
            validated_receipt_inventory=lambda _: (_ for _ in ()).throw(RuntimeError("receipt pin mismatch")),
            preflight_inputs={},
        )


def test_rendered_transport_commands_are_explicit_and_immutable() -> None:
    upload = continuation.render_upload_command(
        "meddiesresearch",
        "/artifacts/checkpoints/terminal-step-00000100",
        "stage1/step100",
        '{"source_checkpoint_receipt":{"hash_verified":true}}',
    )
    download = continuation.render_download_command(
        "huyhoang0411ha",
        "stage1/step100",
        "/artifacts/imported/step100",
        "abc123",
    )
    assert "--upload" in upload
    assert "--checkpoint-root" in upload
    assert "--remote-prefix" in upload
    assert "--source-receipt-json" in upload
    assert "--download" in download
    assert "--destination" in download
    assert "--revision 'abc123'" in download
    assert "MODAL_PROFILE=huyhoang0411ha" in download


def test_parameter_inventory_mismatch_fails_before_tensor_collectives() -> None:
    parameter_a = {
        "name": "backbone.layers.0.weight",
        "shape": [4, 4],
        "requires_grad": True,
        "tensor_present": True,
        "selected_tensor_dtype": "torch.bfloat16",
        "selected_tensor_device_type": "cuda",
        "selected_tensor_layout": "torch.strided",
        "selected_tensor_numel": 16,
    }
    parameter_b = {
        **parameter_a,
        "name": "classifier.weight",
        "shape": [37, 4],
        "selected_tensor_numel": 148,
    }
    records = [
        {"rank": 0, "parameters": [parameter_a, parameter_b]},
        {"rank": 1, "parameters": [parameter_b, parameter_a]},
        {"rank": 2, "parameters": [parameter_a, parameter_b]},
        {"rank": 3, "parameters": [parameter_a, parameter_b]},
    ]
    with pytest.raises(RuntimeError, match="name/order/shape/dtype inventory differs"):
        modal_continuation._require_matching_parameter_inventory(records, stage="post_restore_parameters")


def test_canonical_trainable_inventory_sorting_is_name_stable() -> None:
    class Parameter:
        requires_grad = True

    class Module:
        @staticmethod
        def named_parameters() -> list[tuple[str, Parameter]]:
            return [
                ("classifier.weight", Parameter()),
                ("backbone.layers.0.weight", Parameter()),
            ]

    assert [name for name, _parameter in modal_continuation._canonical_trainable_named_parameters(Module())] == [
        "backbone.layers.0.weight",
        "classifier.weight",
    ]


def test_launch_attempt_claim_rejects_modal_transparent_replay(tmp_path: Path) -> None:
    class Volume:
        def __init__(self) -> None:
            self.reloads = 0
            self.commits = 0

        def reload(self) -> None:
            self.reloads += 1

        def commit(self) -> None:
            self.commits += 1

    volume = Volume()
    contract = continuation.render_segment_contract("huyhoang0411ha")
    attempt_digest = continuation.launch_attempt_digest(contract["execution_contract_digest"], "a" * 64, "b" * 32)

    artifact_root = modal_continuation._claim_launch_attempt(
        tmp_path,
        attempt_digest=attempt_digest,
        launch_attempt_nonce="b" * 32,
        profile="huyhoang0411ha",
        source_launch_digest="a" * 64,
        volume=volume,
    )

    with pytest.raises(RuntimeError, match="already claimed"):
        modal_continuation._claim_launch_attempt(
            tmp_path,
            attempt_digest=attempt_digest,
            launch_attempt_nonce="b" * 32,
            profile="huyhoang0411ha",
            source_launch_digest="a" * 64,
            volume=volume,
        )

    assert artifact_root == tmp_path / "launch-attempts" / attempt_digest
    assert volume.reloads == 2
    assert volume.commits == 1
    assert json.loads((tmp_path / "launch-claims" / attempt_digest / "claim.json").read_text(encoding="utf-8")) == {
        "attempt_digest": attempt_digest,
        "launch_attempt_nonce": "b" * 32,
        "profile": "huyhoang0411ha",
        "source_launch_digest": "a" * 64,
    }


def test_train_segment_claim_blocks_a_transparent_modal_retry_before_torchrun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from meddies_pii.training.bioes.modal import full_run as modal_full_run

    class Volume:
        def reload(self) -> None:  # ruff: ignore[no-self-use]
            return None

        def commit(self) -> None:  # ruff: ignore[no-self-use]
            return None

    profile = "huyhoang0411ha"
    contract = continuation.render_segment_contract(profile)
    source = {
        "hash_verified": True,
        "checkpoint_digest": "a" * 64,
        "optimizer_step": 60,
        "packed_cursor": 7_680,
        "world_size": 1,
        "wave_profile": "meddiesresearch",
        "wave_cumulative_all_in_cost_usd": 0.0,
        "source_root": str(tmp_path / "step60"),
    }
    shards = ("/cache/packed-00000.parquet",)
    attestation = {"body_tensor_values_sha256": "b" * 64}
    packed_paths_sha256 = sha256("\n".join(shards).encode()).hexdigest()
    preflight = {
        "execution_contract_digest": contract["execution_contract_digest"],
        "packed_shards": {"count": 1, "paths_sha256": packed_paths_sha256},
        "encoder_checkpoint_attestation": attestation,
        "preflight_digest": continuation.preflight_digest(
            contract["execution_contract_digest"],
            packed_paths_sha256=packed_paths_sha256,
            shard_count=1,
            encoder_checkpoint_attestation=attestation,
        ),
    }
    source_launch_digest = continuation.launch_digest(contract["execution_contract_digest"], source)
    launched: list[object] = []
    monkeypatch.setattr(modal_continuation, "ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(modal_continuation, "artifacts", Volume())
    monkeypatch.setattr(modal_continuation, "inventory_source_checkpoint", lambda _: source)
    monkeypatch.setattr(
        modal_full_run,
        "_validated_receipt_inventory",
        lambda _: (shards, attestation),
    )
    monkeypatch.setattr(
        modal_continuation,
        "_run_isolated_torchrun",
        lambda spec: launched.append(spec) or {"status": "queued"},
    )

    first = modal_continuation._train_segment_impl(
        profile,
        contract["execution_contract_digest"],
        source,
        source_launch_digest,
        preflight,
        None,
        None,
        "c" * 32,
        10_800,
        execute=True,
        confirmation=contract["manual_launch"]["confirmation"],
        primary_action=contract["manual_launch"]["primary_action"],
    )
    assert first == {"status": "queued"}
    with pytest.raises(RuntimeError, match="already claimed"):
        modal_continuation._train_segment_impl(
            profile,
            contract["execution_contract_digest"],
            source,
            source_launch_digest,
            preflight,
            None,
            None,
            "c" * 32,
            10_800,
            execute=True,
            confirmation=contract["manual_launch"]["confirmation"],
            primary_action=contract["manual_launch"]["primary_action"],
        )
    assert len(launched) == 1
