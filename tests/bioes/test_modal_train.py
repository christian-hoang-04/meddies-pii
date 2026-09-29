from __future__ import annotations

from meddies_pii.training.bioes.modal.train import (
    FULL_TRAINING_TIMEOUT_SECONDS,
    _artifact_payload,
)


def test_artifact_payload_records_durable_volume_metadata() -> None:
    payload = _artifact_payload(
        result={"model_artifact": "/artifacts/bioes/run/unsloth/backbone_adapter"},
        command_kwargs={"artifact_root": "/artifacts/bioes/run"},
        out="project-manager/preview/run.json",
        raw_log=None,
        modal_url=None,
        modal_profile="openmedical",
        artifact_root="/artifacts/bioes/run",
        artifact_volume="meddies-pii-bioes-artifacts",
        artifact_persisted=True,
    )

    assert payload["provenance"]["artifact_root"] == "/artifacts/bioes/run"
    assert payload["provenance"]["artifact_volume"] == "meddies-pii-bioes-artifacts"
    assert payload["provenance"]["artifact_persisted"] is True


def test_modal_timeout_is_long_enough_for_full_training() -> None:
    assert FULL_TRAINING_TIMEOUT_SECONDS >= 12 * 60 * 60
