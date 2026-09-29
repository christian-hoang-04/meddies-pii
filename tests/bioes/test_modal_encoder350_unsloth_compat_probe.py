from __future__ import annotations

# ruff: file-ignore[no-self-use]
# reason: stateless test doubles retain the bound method shape of the process interfaces they replace.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

import builtins
import inspect
import io
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from meddies_pii.training.bioes.modal import (
    encoder350_unsloth_compat_probe as modal_probe,
)
from meddies_pii.training.bioes.trainers import encoder350_unsloth_compat_probe as probe


def _spec() -> modal_probe.ChildSpec:
    return modal_probe.ChildSpec(
        contract=probe.render_probe(),
        shard_paths=("/cache/train-000.parquet",),
        expected_encoder_checkpoint_attestation={
            "body_tensor_count": 1,
            "body_tensor_names_sha256": "a" * 64,
            "body_tensor_values_sha256": "b" * 64,
        },
        artifact_dir="/artifacts/encoder350-unsloth-2026-7-4-compat/test",
    )


def test_one_step_endpoint_has_separately_versioned_image_and_contract() -> None:
    assert modal_probe.H100_OPTIONS["gpu"] == "H100!"
    assert modal_probe.H100_OPTIONS["timeout"] == probe.PROBE_HARD_TIMEOUT_SECONDS
    assert modal_probe.IMAGE_PACKAGES[-2:] == (
        "unsloth==2026.7.4",
        "unsloth_zoo==2026.7.4",
    )
    assert "transformers==5.2.0" in modal_probe.IMAGE_PACKAGES
    modal_probe._validate_child_spec(_spec())


@pytest.mark.parametrize("key", ["backend", "runtime_packages", "optimizer_steps"])
def test_child_refuses_wrong_backend_or_package_contract(key: str) -> None:
    bad = {**probe.render_probe(), key: "wrong"}
    with pytest.raises(ValueError, match=key):
        modal_probe._validate_child_spec(replace(_spec(), contract=bad))


def test_child_refuses_absent_cpu_hash_or_packed_shards() -> None:
    with pytest.raises(ValueError, match="body attestation"):
        modal_probe._validate_child_spec(replace(_spec(), expected_encoder_checkpoint_attestation={}))
    with pytest.raises(ValueError, match="packed shards"):
        modal_probe._validate_child_spec(replace(_spec(), shard_paths=()))


def test_h100_endpoint_requires_the_existing_cpu_receipt_before_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(modal_probe.artifacts, "commit", lambda: None)
    monkeypatch.setattr(
        modal_probe,
        "_validated_receipt_inventory",
        lambda _contract: (("/cache/train.parquet",), None),
    )
    with pytest.raises(RuntimeError, match="CPU receipt body attestation"):
        modal_probe.run_encoder350_unsloth_compat_probe.local(
            probe.render_probe(),
            execute=True,
            confirmation=probe.PROBE_CONFIRMATION,
        )


def test_child_main_reports_fail_closed_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec_path, result_path = tmp_path / "spec.json", tmp_path / "result.json"
    spec_path.write_text(
        json.dumps({
            "contract": _spec().contract,
            "shard_paths": list(_spec().shard_paths),
            "expected_encoder_checkpoint_attestation": _spec().expected_encoder_checkpoint_attestation,
            "artifact_dir": _spec().artifact_dir,
        }),
    )
    monkeypatch.setattr(modal_probe.artifacts, "commit", lambda: None)
    monkeypatch.setattr(
        modal_probe,
        "_run_child",
        lambda _spec: (_ for _ in ()).throw(RuntimeError("red")),
    )
    assert modal_probe._child_main(str(spec_path), str(result_path)) == 1
    assert json.loads(result_path.read_text())["status"] == "failed"


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


def test_child_stdout_is_forwarded_before_wait_returns() -> None:

    progress_seen = threading.Event()

    class _Process:
        # reason: `ChildProcess` declares these as mutable `IO[str] | None`, and a protocol attribute is
        # reason: invariant, so a bare `StringIO` inferred here does not satisfy it.
        stdout: IO[str] | None = io.StringIO('{"event":"runtime_attested"}\n')
        stderr: IO[str] | None = io.StringIO("")

        def wait(self) -> int:
            assert progress_seen.wait(timeout=1.0), "child progress was buffered until exit"
            return 0

    original_print = print

    # reason: builtins.print is a two-arm overload whose arms differ in both file and flush,
    # reason: so no single signature on this stand-in satisfies both the forwarding call and the assignment.
    def capture_print(
        *args,  # ruff: ignore[missing-type-args]
        **kwargs,  # ruff: ignore[missing-type-kwargs]
    ) -> None:
        if args and "runtime_attested" in str(args[0]):
            progress_seen.set()
        original_print(*args, **kwargs)

    old_print = builtins.print
    # reason: builtins.print is a two-arm overload whose arms differ in both `file` and `flush`, so no
    # reason: single signature on a stand-in satisfies the forwarding call and this assignment at once.
    builtins.print = capture_print  # ty: ignore[invalid-assignment]
    try:
        assert modal_probe._stream_child_output(_Process()) == (0, "")
    finally:
        builtins.print = old_print


def test_local_profile_is_encoder350_account_before_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    with pytest.raises(RuntimeError, match="huyhoang041100"):
        modal_probe._local_remote_call(probe.render_probe(), execute=True, confirmation=probe.PROBE_CONFIRMATION)


def test_compatibility_probe_uses_fail_closed_wrapper_to_body_provenance_transfer() -> None:

    source = inspect.getsource(modal_probe._run_child)
    assert "_propagate_unsloth_remote_code_provenance(wrapper, body)" in source
