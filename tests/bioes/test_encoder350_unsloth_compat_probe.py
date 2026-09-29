from __future__ import annotations

import json

# reason: these purity probes run this interpreter with fixed module/script argv and no shell.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

import pytest

from meddies_pii.training.bioes.trainers import encoder350_unsloth_compat_probe as probe


def test_one_step_encoder350_contract_is_exact_and_versioned() -> None:
    contract = probe.render_probe()
    assert contract["candidate_key"] == "encoder350"
    assert contract["batch_size"] == 192
    assert contract["optimizer_steps"] == 1
    assert contract["runtime_packages"] == {
        "torch": "2.10.0",
        "transformers": "5.2.0",
        "peft": "0.19.1",
        "unsloth": "2026.7.4",
        "unsloth_zoo": "2026.7.4",
    }
    assert contract["packed_dataset"]["segment_isolation"] is True
    assert contract["training"]["gradient_checkpointing"] == "unsloth"
    assert contract["training"]["task_type"] == "FEATURE_EXTRACTION"


@pytest.mark.parametrize("key", ["backend", "runtime_packages", "batch_size", "optimizer_steps", "loader"])
def test_contract_fails_closed_when_backend_or_version_contract_changes(
    key: str,
) -> None:
    mutated = {**probe.render_probe(), key: "wrong"}
    with pytest.raises(ValueError, match=key):
        probe.validate_probe_contract(mutated)


def test_runtime_package_attestation_is_exact_and_red_capable() -> None:
    assert probe.require_runtime_package_tuple(dict(probe.PROBE_RUNTIME_PACKAGES)) == probe.PROBE_RUNTIME_PACKAGES
    with pytest.raises(RuntimeError, match="package tuple mismatch"):
        probe.require_runtime_package_tuple({**probe.PROBE_RUNTIME_PACKAGES, "unsloth": "2026.5.2"})


def test_review_render_is_pure_and_paid_command_is_explicit() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "meddies_pii.training.bioes.trainers.encoder350_unsloth_compat_probe",
            "--render-config",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(completed.stdout)["optimizer_steps"] == 1
    modules = subprocess.run(
        [
            sys.executable,
            "-c",
            """import sys
import meddies_pii.training.bioes.trainers.encoder350_unsloth_compat_probe
assert "modal" not in sys.modules
assert "unsloth" not in sys.modules
""",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert modules.returncode == 0
    assert probe.PROBE_CONFIRMATION in probe.render_command()
    assert probe.render_command().startswith("MODAL_PROFILE=huyhoang041100 uv run modal run --timestamps -m")
