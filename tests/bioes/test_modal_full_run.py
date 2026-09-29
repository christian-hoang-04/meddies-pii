from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
from typing import TYPE_CHECKING, NoReturn, cast

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.training.bioes.data.artifacts import PretrainedFactory

import pytest

from meddies_pii.training.bioes.modal import full_run
from meddies_pii.training.bioes.trainers import full_run as contract


def _artifact() -> dict[str, object]:
    return {
        "candidate": "encoder350",
        "batch_size": 192,
        **contract.MANUAL_TRANSFER_ACCEPTANCES["encoder350"],
    }


def _inventory() -> dict[str, object]:
    return {
        "id": "Meddies/meddies-pii-v2",
        "revision": "28aaef5dffd36aabead650c74658a6f814eb4db0",
        "config": "eval",
        "split": "train",
        "rows": 1700,
    }


def test_remote_call_refuses_objects_without_a_remote_method() -> None:
    with pytest.raises(TypeError, match="does not expose remote"):
        full_run._call_remote(object())


def test_remote_call_refuses_a_non_mapping_result() -> None:
    class _Endpoint:
        @staticmethod
        def remote() -> list[object]:
            return []

    with pytest.raises(TypeError, match="must return a mapping"):
        full_run._call_remote(_Endpoint())


def test_modal_uses_candidate_specific_unsloth_images_for_h100_endpoints() -> None:
    assert full_run.H100_OPTIONS["gpu"] == "H100!"
    assert full_run.H100_OPTIONS["timeout"] == 9_000
    assert full_run.UNSLOTH_CACHE_ENVIRONMENT["HF_HUB_CACHE"] == "/cache/hf"
    assert full_run.UNSLOTH_CACHE_ENVIRONMENT["UNSLOTH_DISABLE_STATISTICS"] == "1"
    import inspect

    source = inspect.getsource(full_run)
    assert "python_version=FULL_RUN_PYTHON_VERSION" in source
    assert ".pip_install(*FULL_RUN_IMAGE_PACKAGES)" in source
    assert full_run.FULL_RUN_IMAGE_PACKAGES == contract.FULL_RUN_IMAGE_PACKAGES
    assert full_run.ENCODER_FULL_RUN_IMAGE_PACKAGES == contract.ENCODER_FULL_RUN_IMAGE_PACKAGES
    assert source.count("image=unsloth_image, volumes=COMMON_MOUNTS, secrets=[secret], **H100_OPTIONS") == 2
    assert source.count("image=encoder_unsloth_image, volumes=COMMON_MOUNTS, secrets=[secret], **H100_OPTIONS") == 5


def test_candidate_specific_remote_guard_refuses_wrong_contract_and_unqualified() -> None:
    with pytest.raises(RuntimeError, match="different contract"):
        full_run._remote_contract(
            "encoder350",
            contract=contract.render_full_run("pii350"),
            qualification_artifact=_artifact(),
            eval_inventory=_inventory(),
        )
    with pytest.raises(RuntimeError, match="exact accepted"):
        full_run._remote_contract(
            "base230",
            contract=contract.render_full_run("base230"),
            qualification_artifact={"candidate": "base230"},
            eval_inventory=_inventory(),
        )
    with pytest.raises(RuntimeError, match="pinned eval inventory"):
        full_run._remote_contract(
            "encoder350",
            contract=contract.render_full_run("encoder350"),
            qualification_artifact=_artifact(),
            eval_inventory=None,
        )
    assert full_run._remote_contract(
        "encoder350",
        contract=contract.render_full_run("encoder350"),
        qualification_artifact=_artifact(),
        eval_inventory=_inventory(),
    ) == {"status": "validated_for_execution", "candidate": "encoder350"}


def test_base230_b384_remote_launch_requires_exact_persisted_b416_oom_and_action() -> None:
    fallback = contract.render_base230_oom_fallback_contract()
    with pytest.raises(RuntimeError, match="OOM evidence"):
        full_run._remote_base230_oom_fallback_contract(
            contract=fallback,
            primary_oom_evidence=None,
            primary_action=contract.BASE230_OOM_FALLBACK_PRIMARY_ACTION,
            eval_inventory=_inventory(),
        )
    with pytest.raises(RuntimeError, match="primary action"):
        full_run._remote_base230_oom_fallback_contract(
            contract=fallback,
            primary_oom_evidence=contract.BASE230_B416_OOM_EVIDENCE,
            primary_action="automatic",
            eval_inventory=_inventory(),
        )
    assert full_run._remote_base230_oom_fallback_contract(
        contract=fallback,
        primary_oom_evidence=contract.BASE230_B416_OOM_EVIDENCE,
        primary_action=contract.BASE230_OOM_FALLBACK_PRIMARY_ACTION,
        eval_inventory=_inventory(),
    ) == {"status": "validated_for_execution", "candidate": "base230"}


def test_pii350_utilization_remote_launch_requires_exact_source_run_and_action() -> None:
    utilization = contract.render_pii350_utilization_run_contract()
    with pytest.raises(RuntimeError, match="source run evidence"):
        full_run._remote_pii350_utilization_contract(
            contract=utilization,
            source_run_evidence=None,
            primary_action=contract.PII350_UTILIZATION_PRIMARY_ACTION,
            eval_inventory=_inventory(),
        )
    with pytest.raises(RuntimeError, match="primary action"):
        full_run._remote_pii350_utilization_contract(
            contract=utilization,
            source_run_evidence=contract.PII350_B192_COMPLETE_EVIDENCE,
            primary_action="automatic",
            eval_inventory=_inventory(),
        )
    assert full_run._remote_pii350_utilization_contract(
        contract=utilization,
        source_run_evidence=contract.PII350_B192_COMPLETE_EVIDENCE,
        primary_action=contract.PII350_UTILIZATION_PRIMARY_ACTION,
        eval_inventory=_inventory(),
    ) == {"status": "validated_for_execution", "candidate": "pii350"}


def test_pii350_capacity_remote_launch_requires_exact_source_run_and_action() -> None:
    capacity = contract.render_pii350_capacity_run_contract()
    with pytest.raises(RuntimeError, match="source run evidence"):
        full_run._remote_pii350_capacity_contract(
            contract=capacity,
            source_run_evidence=None,
            primary_action=contract.PII350_CAPACITY_PRIMARY_ACTION,
            eval_inventory=_inventory(),
        )
    with pytest.raises(RuntimeError, match="primary action"):
        full_run._remote_pii350_capacity_contract(
            contract=capacity,
            source_run_evidence=contract.PII350_B192_COMPLETE_EVIDENCE,
            primary_action="automatic",
            eval_inventory=_inventory(),
        )
    assert full_run._remote_pii350_capacity_contract(
        contract=capacity,
        source_run_evidence=contract.PII350_B192_COMPLETE_EVIDENCE,
        primary_action=contract.PII350_CAPACITY_PRIMARY_ACTION,
        eval_inventory=_inventory(),
    ) == {"status": "validated_for_execution", "candidate": "pii350"}


@pytest.mark.parametrize("candidate", ["base230", "encoder230"])
def test_m230_comparison_remote_launch_requires_exact_contract_action_and_eval(
    candidate: str,
) -> None:
    comparison = contract.render_m230_comparison_run_contract(candidate)
    action = contract.M230_COMPARISON_PRIMARY_ACTIONS[candidate]
    with pytest.raises(RuntimeError, match="primary action"):
        full_run._remote_m230_comparison_contract(
            candidate,
            contract=comparison,
            primary_action="automatic",
            eval_inventory=_inventory(),
        )
    with pytest.raises(RuntimeError, match="pinned eval inventory"):
        full_run._remote_m230_comparison_contract(
            candidate,
            contract=comparison,
            primary_action=action,
            eval_inventory=None,
        )
    assert full_run._remote_m230_comparison_contract(
        candidate,
        contract=comparison,
        primary_action=action,
        eval_inventory=_inventory(),
    ) == {"status": "validated_for_execution", "candidate": candidate}


@pytest.mark.parametrize("candidate", ["base230", "encoder230"])
def test_m230_comparison_uses_primary_asset_receipt_and_exact_profile(
    candidate: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = contract.render_m230_comparison_run_contract(candidate)
    assert full_run._asset_receipt_contract(comparison) == contract.render_full_run(candidate)

    class _Endpoint:
        @staticmethod
        def remote(*args: object) -> dict[str, tuple[object, ...]]:
            return {"args": args}

    expected_profile = {
        "base230": "meddies-pii",
        "encoder230": "hahuyhoang041100",
    }[candidate]
    monkeypatch.setenv("MODAL_PROFILE", expected_profile)
    result = full_run._local_m230_comparison_call(
        candidate,
        _Endpoint(),
        comparison,
        contract.M230_COMPARISON_PRIMARY_ACTIONS[candidate],
        _inventory(),
    )
    assert result["args"][0] == comparison
    monkeypatch.setenv("MODAL_PROFILE", "hahuyhoang411")
    with pytest.raises(RuntimeError, match=f"MODAL_PROFILE={expected_profile}"):
        full_run._local_m230_comparison_call(
            candidate,
            _Endpoint(),
            comparison,
            contract.M230_COMPARISON_PRIMARY_ACTIONS[candidate],
            _inventory(),
        )


def test_pii350_utilization_uses_primary_asset_receipt_and_exact_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utilization = contract.render_pii350_utilization_run_contract()
    assert full_run._asset_receipt_contract(utilization) == contract.render_full_run("pii350")

    class _Endpoint:
        @staticmethod
        def remote(*args: object) -> dict[str, tuple[object, ...]]:
            return {"args": args}

    monkeypatch.setenv("MODAL_PROFILE", "retraction")
    result = full_run._local_pii350_utilization_call(
        _Endpoint(),
        utilization,
        contract.PII350_B192_COMPLETE_EVIDENCE,
        contract.PII350_UTILIZATION_PRIMARY_ACTION,
        _inventory(),
    )
    assert result["args"][0] == utilization
    monkeypatch.setenv("MODAL_PROFILE", "huyhoang041100")
    with pytest.raises(RuntimeError, match="MODAL_PROFILE=retraction"):
        full_run._local_pii350_utilization_call(
            _Endpoint(),
            utilization,
            contract.PII350_B192_COMPLETE_EVIDENCE,
            contract.PII350_UTILIZATION_PRIMARY_ACTION,
            _inventory(),
        )


def test_full_run_profile_map_and_timeout_are_candidate_specific() -> None:
    assert full_run.FULL_RUN_PROFILES == {
        "base230": "hahuyhoang411",
        "encoder230": "huyhoangha0411",
        "encoder350": "huyhoang041100",
        "pii350": "huyhoang041100",
    }
    assert full_run.H100_OPTIONS["timeout"] == 9_000
    assert round(full_run.H100_OPTIONS["timeout"] * contract.FULL_RUN_RATE_USD_PER_SECOND, 6) == 9.873
    assert contract.PII350_UTILIZATION_PROFILE == "retraction"
    assert contract.PII350_CAPACITY_PROFILE == "retraction"


def test_encoder350_and_pii350_are_independently_launchable_on_shared_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Endpoint:
        def __init__(self, name: str) -> None:
            self.name = name

        def remote(self, *_args: object) -> dict[str, str]:
            calls.append(self.name)
            return {"candidate": self.name}

    monkeypatch.setenv("MODAL_PROFILE", "huyhoang041100")
    for candidate in ("encoder350", "pii350"):
        artifact = (
            _artifact()
            if candidate == "encoder350"
            else {
                "candidate": "pii350",
                "batch_size": 192,
                **contract.MANUAL_TRANSFER_ACCEPTANCES["pii350"],
            }
        )
        assert full_run._local_remote_call(
            candidate,
            _Endpoint(candidate),
            contract.render_full_run(candidate),
            artifact,
            _inventory(),
            None,
        ) == {"candidate": candidate}
    assert calls == ["encoder350", "pii350"]


def test_cpu_candidate_preflight_verifies_offline_pins_and_exact_eval_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    shard_one = tmp_path / "one.parquet"
    shard_two = tmp_path / "two.parquet"
    shard_one.write_bytes(b"one")
    shard_two.write_bytes(b"two")

    class _Packed:
        shard_paths = (str(shard_one), str(shard_two))

    monkeypatch.setattr(full_run, "verify_packed_artifact", lambda **_: _Packed())
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    def load_dataset(*args: object, **kwargs: object) -> list[None]:
        calls.append({"dataset": args, **kwargs})
        return [None] * 1700

    class _MaskedLM:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> NoReturn:
            msg = "covered by the wrapper-proof unit test"
            raise AssertionError(msg)

    monkeypatch.setattr(
        full_run,
        "_masked_lm_body_loading_evidence",
        lambda *_args, **_kwargs: {"checkpoint_body_attestation": {"body_tensor_count": 1}},
    )
    result = full_run._preflight_candidate_assets(
        "encoder230",
        snapshot_download=snapshot_download,
        load_dataset=load_dataset,
        auto_model_for_masked_lm=_MaskedLM,
    )
    assert {
        key: result[key]
        for key in (
            "candidate",
            "model_snapshot",
            "packed_shards",
            "eval_rows",
            "offline",
        )
    } == {
        "candidate": "encoder230",
        "model_snapshot": str(snapshot),
        "packed_shards": 2,
        "eval_rows": 1700,
        "offline": True,
    }
    assert result["receipt"]["packed_inventory"][0]["path"] == str(shard_one)
    assert calls[0]["local_files_only"] is True
    assert calls[0]["revision"] == "0b649ad0c684378b03d4d8304f7577a662ab89bc"
    assert calls[1]["revision"] == "28aaef5dffd36aabead650c74658a6f814eb4db0"


def test_cpu_prewarm_downloads_only_pinned_pii350_assets_with_twelve_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_snapshot = tmp_path / "packed"
    model_snapshot = tmp_path / "model"
    packed_snapshot.mkdir()
    model_snapshot.mkdir()
    shard_one = tmp_path / "one.parquet"
    shard_two = tmp_path / "two.parquet"
    shard_one.write_bytes(b"one")
    shard_two.write_bytes(b"two")

    class _Packed:
        shard_paths = (str(shard_one), str(shard_two))

    monkeypatch.setattr(full_run, "verify_packed_artifact", lambda **_: _Packed())
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(packed_snapshot if kwargs.get("repo_type") == "dataset" else model_snapshot)

    def load_dataset(*args: object, **kwargs: object) -> list[None]:
        calls.append({"dataset": args, **kwargs})
        return [None] * 1700

    result = full_run._prewarm_candidate_assets(
        "pii350",
        snapshot_download=snapshot_download,
        load_dataset=load_dataset,
    )
    assert result == {
        "candidate": "pii350",
        "model_snapshot": str(model_snapshot),
        "packed_snapshot": str(packed_snapshot),
        "packed_shards": 2,
        "eval_rows": 1700,
        "offline_verified": True,
    }
    assert calls[0]["repo_id"] == "Meddies/meddies-pii-mixed"
    assert calls[0]["revision"] == "11fd43ec9ebb187e1d0f94fe77bcf1090a2a18ee"
    assert calls[0]["allow_patterns"] == [
        "packed/manifest.json",
        "packed/data/*.parquet",
    ]
    assert calls[0]["max_workers"] == 12
    assert calls[1]["repo_id"] == "LiquidAI/LFM2.5-Encoder-350M-PII-Detector"
    assert calls[1]["max_workers"] == 12
    assert calls[2]["revision"] == "28aaef5dffd36aabead650c74658a6f814eb4db0"


def test_h100_requires_cpu_receipt_and_only_checks_local_shard_sizes(tmp_path: Path) -> None:
    import hashlib
    import json

    contract_payload = contract.render_full_run("encoder230")
    shard = tmp_path / "shard.parquet"
    shard.write_bytes(b"packed")
    inventory = [{"path": str(shard), "bytes": shard.stat().st_size}]
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    contract_payload["packed_dataset"]["shard_count"] = 1
    receipt = {
        "receipt_schema_version": full_run.PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "candidate": "encoder230",
        "model_id": contract_payload["model_id"],
        "model_revision": contract_payload["model_revision"],
        "packed_revision": contract.PACKED_DATASET_REVISION,
        "packed_manifest_sha256": contract.PACKED_MANIFEST_SHA256,
        "packed_shard_count": 1,
        "packed_inventory": inventory,
        "packed_inventory_sha256": digest,
        "eval": contract_payload["evaluation"],
        "config_digest": contract_payload["config_digest"],
        "masked_lm_body_loading": {
            "missing_body_keys": [],
            "mismatched_body_keys": [],
            "unexpected_wrapper_keys": [],
            "discarded_masked_lm_head_keys": ["lm_head.weight"],
            "checkpoint_body_attestation": {
                "body_tensor_count": 1,
                "body_tensor_names_sha256": "a" * 64,
                "body_tensor_values_sha256": "b" * 64,
            },
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert full_run._validated_shard_paths_from_receipt(contract_payload, receipt_path=receipt_path) == (str(shard),)
    shard.write_bytes(b"changed-size")
    with pytest.raises(RuntimeError, match="size"):
        full_run._validated_shard_paths_from_receipt(contract_payload, receipt_path=receipt_path)


def test_pii_h100_receipt_requires_complete_body_attestation(tmp_path: Path) -> None:
    import hashlib
    import json

    contract_payload = contract.render_full_run("pii350")
    shard = tmp_path / "shard.parquet"
    shard.write_bytes(b"packed")
    inventory = [{"path": str(shard), "bytes": shard.stat().st_size}]
    contract_payload["packed_dataset"]["shard_count"] = 1
    receipt = {
        "receipt_schema_version": full_run.PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "candidate": "pii350",
        "model_id": contract_payload["model_id"],
        "model_revision": contract_payload["model_revision"],
        "packed_revision": contract.PACKED_DATASET_REVISION,
        "packed_manifest_sha256": contract.PACKED_MANIFEST_SHA256,
        "packed_shard_count": 1,
        "packed_inventory": inventory,
        "packed_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest(),
        "eval": contract_payload["evaluation"],
        "config_digest": contract_payload["config_digest"],
        "pii_body_loading": {
            "missing_body_keys": [],
            "mismatched_body_keys": [],
        },
    }
    receipt_path = tmp_path / "pii-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="encoder body loading proof"):
        full_run._validated_shard_paths_from_receipt(contract_payload, receipt_path=receipt_path)

    receipt["pii_body_loading"] = {
        "missing_body_keys": [],
        "mismatched_body_keys": [],
        "unexpected_wrapper_keys": [],
        "discarded_pii_label_head_keys": [
            "class_weights",
            "classifier.bias",
            "classifier.weight",
        ],
        "discarded_pii_classifier_out_features": 161,
        "checkpoint_body_attestation": {
            "body_tensor_count": 1,
            "body_tensor_names_sha256": "a" * 64,
            "body_tensor_values_sha256": "b" * 64,
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert full_run._validated_shard_paths_from_receipt(contract_payload, receipt_path=receipt_path) == (str(shard),)


def test_pii_cpu_loading_evidence_requires_the_wrapper_and_all_body_tensors(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors.torch import save_file

    pii_contract = contract.render_full_run("pii350")
    snapshot = tmp_path / "pii-snapshot"
    snapshot.mkdir()
    save_file(
        {"lfm2.projection.weight": torch.ones((2, 2), dtype=torch.bfloat16)},
        str(snapshot / "model.safetensors"),
    )

    class _Body(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)

    class _Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lfm2 = _Body()
            self.classifier = torch.nn.Linear(2, 161, dtype=torch.bfloat16)
            self.register_buffer("class_weights", torch.ones(161, dtype=torch.bfloat16))

    class _TokenClassifier:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> tuple[_Wrapper, dict[str, list[str]]]:
            return _Wrapper(), {
                "missing_keys": [],
                "mismatched_keys": [],
                "unexpected_keys": [],
            }

    evidence = full_run._pii_body_loading_evidence(pii_contract, _TokenClassifier, snapshot=str(snapshot))
    assert {
        key: evidence[key]
        for key in (
            "missing_body_keys",
            "mismatched_body_keys",
            "unexpected_wrapper_keys",
            "discarded_pii_label_head_keys",
            "discarded_pii_classifier_out_features",
        )
    } == {
        "missing_body_keys": [],
        "mismatched_body_keys": [],
        "unexpected_wrapper_keys": [],
        "discarded_pii_label_head_keys": [
            "class_weights",
            "classifier.bias",
            "classifier.weight",
        ],
        "discarded_pii_classifier_out_features": 161,
    }
    assert evidence["body_state_attestation"]["body_tensor_count"] == 1
    assert evidence["checkpoint_body_attestation"]["body_tensor_count"] == 1

    class _MissingBodyTensor:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> tuple[_Wrapper, dict[str, list[str]]]:
            return _Wrapper(), {
                "missing_keys": ["lfm2.projection.weight"],
                "mismatched_keys": [],
                "unexpected_keys": [],
            }

    with pytest.raises(RuntimeError, match="missing, mismatched, or unexpected"):
        full_run._pii_body_loading_evidence(pii_contract, _MissingBodyTensor, snapshot=str(snapshot))

    class _DirectBody:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> tuple[_Body, dict[str, list[str]]]:
            return _Body(), {
                "missing_keys": [],
                "mismatched_keys": [],
                "unexpected_keys": [],
            }

    with pytest.raises(RuntimeError, match=r"no \.lfm2"):
        full_run._pii_body_loading_evidence(pii_contract, _DirectBody, snapshot=str(snapshot))


def test_masked_lm_cpu_loading_evidence_requires_complete_canonical_wrapper(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors.torch import save_file

    encoder_contract = contract.render_full_run("encoder230")
    snapshot = tmp_path / "encoder-snapshot"
    snapshot.mkdir()
    weight = torch.ones((2, 2), dtype=torch.bfloat16)
    save_file({"lfm2.projection.weight": weight}, str(snapshot / "model.safetensors"))

    class _Body(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)

    class _Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lfm2 = _Body()
            self.lm_head = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            self.lm_head.weight = self.lfm2.projection.weight

    class _MaskedLM:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> tuple[_Wrapper, dict[str, list[str]]]:
            return _Wrapper(), {
                "missing_keys": [],
                "mismatched_keys": [],
                "unexpected_keys": [],
            }

    evidence = full_run._masked_lm_body_loading_evidence(encoder_contract, _MaskedLM, snapshot=str(snapshot))
    assert evidence["discarded_masked_lm_head_keys"] == ["lm_head.weight"]
    assert evidence["checkpoint_body_attestation"]["body_tensor_count"] == 1

    class _MissingBodyTensor:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> tuple[_Wrapper, dict[str, list[str]]]:
            return _Wrapper(), {
                "missing_keys": ["lfm2.projection.weight"],
                "mismatched_keys": [],
                "unexpected_keys": [],
            }

    with pytest.raises(RuntimeError, match="missing, mismatched, or unexpected"):
        full_run._masked_lm_body_loading_evidence(encoder_contract, _MissingBodyTensor, snapshot=str(snapshot))


def test_pii350_milestone_remote_and_profile_gates_reject_swapped_profile_or_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = contract.render_pii350_milestone_run_contract("meddies-pii")
    with pytest.raises(RuntimeError, match="primary action"):
        full_run._remote_pii350_milestone_contract(
            profile="meddies-pii",
            contract=decision,
            source_run_evidence=contract.PII350_B192_COMPLETE_EVIDENCE,
            primary_action="HA_AUTHORIZE_PII350_R128A256_B160_MILESTONE7680_FULL_BUDGET",
            eval_inventory=_inventory(),
        )
    assert full_run._remote_pii350_milestone_contract(
        profile="meddies-pii",
        contract=decision,
        source_run_evidence=contract.PII350_B192_COMPLETE_EVIDENCE,
        primary_action=decision["manual_launch"]["requires_primary_action"],
        eval_inventory=_inventory(),
    ) == {"status": "validated_for_execution", "candidate": "pii350"}

    class _Endpoint:
        @staticmethod
        def remote(*args: object) -> dict[str, tuple[object, ...]]:
            return {"args": args}

    monkeypatch.setenv("MODAL_PROFILE", "anhthunguyenump")
    with pytest.raises(RuntimeError, match="MODAL_PROFILE=meddies-pii"):
        full_run._local_pii350_milestone_call(
            _Endpoint(),
            "meddies-pii",
            decision,
            contract.PII350_B192_COMPLETE_EVIDENCE,
            decision["manual_launch"]["requires_primary_action"],
            _inventory(),
        )
    assert full_run.PII350_MILESTONE_H100_OPTIONS["gpu"] == "H100!"
    assert full_run.PII350_MILESTONE_H100_OPTIONS["timeout"] == 7_474


@pytest.mark.parametrize(
    ("arm", "profile"),
    [
        ("lr1e-4", "hocdatacamp-hoangthu"),
        ("lr2e-4", "hocdatacamp-hoangthu"),
        ("lr3e-4", "hocdatacamp-hoangthu"),
        ("lr4e-4", "meddiesresearch"),
        ("wsd3e-4", "meddiesresearch"),
    ],
)
def test_pii350_scout_remote_and_profile_gates_bind_the_exact_arm(
    arm: str,
    profile: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scout = contract.render_pii350_scout_contract(arm)
    action = scout["manual_launch"]["requires_primary_action"]
    assert full_run._asset_receipt_contract(scout) == scout
    assert scout["config_digest"] in str(full_run._receipt_path(scout))

    with pytest.raises(RuntimeError, match="exact rendered contract"):
        full_run._remote_pii350_scout_contract(
            arm=arm,
            contract=contract.render_pii350_scout_contract("lr2e-4" if arm == "lr1e-4" else "lr1e-4"),
            source_run_evidence=contract.PII350_B192_COMPLETE_EVIDENCE,
            primary_action=action,
            eval_inventory=_inventory(),
        )
    with pytest.raises(RuntimeError, match="primary action"):
        full_run._remote_pii350_scout_contract(
            arm=arm,
            contract=scout,
            source_run_evidence=contract.PII350_B192_COMPLETE_EVIDENCE,
            primary_action="HA_AUTHORIZE_THE_OTHER_ARM",
            eval_inventory=_inventory(),
        )

    class _Endpoint:
        @staticmethod
        def remote(*args: object) -> dict[str, tuple[object, ...]]:
            return {"args": args}

    monkeypatch.setenv("MODAL_PROFILE", profile)
    preflight = full_run._local_pii350_scout_preflight_call(_Endpoint(), arm)
    assert preflight["args"] == ("pii350", scout)
    result = full_run._local_pii350_scout_call(
        _Endpoint(),
        arm,
        scout,
        contract.PII350_B192_COMPLETE_EVIDENCE,
        action,
        _inventory(),
    )
    assert result["args"] == (
        arm,
        scout,
        contract.PII350_B192_COMPLETE_EVIDENCE,
        action,
        _inventory(),
    )
    monkeypatch.setenv("MODAL_PROFILE", "hahuyhoang411")
    with pytest.raises(RuntimeError, match=f"MODAL_PROFILE={profile}"):
        full_run._local_pii350_scout_call(
            _Endpoint(),
            arm,
            scout,
            contract.PII350_B192_COMPLETE_EVIDENCE,
            action,
            _inventory(),
        )
    with pytest.raises(RuntimeError, match=f"MODAL_PROFILE={profile}"):
        full_run._local_pii350_scout_preflight_call(_Endpoint(), arm)

    monkeypatch.setenv("MODAL_PROFILE", profile)
    with pytest.raises(RuntimeError, match="exact rendered contract"):
        full_run._local_pii350_scout_call(
            _Endpoint(),
            arm,
            contract.render_pii350_scout_contract("lr3e-4" if arm == "lr4e-4" else "lr4e-4"),
            contract.PII350_B192_COMPLETE_EVIDENCE,
            action,
            _inventory(),
        )


@pytest.mark.parametrize("arm", ["lr1e-4", "lr2e-4", "lr3e-4", "lr4e-4", "wsd3e-4"])
def test_pii350_scout_cpu_preflight_receipt_uses_the_exact_arm_digest(
    arm: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scout = contract.render_pii350_scout_contract(arm)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    shard = tmp_path / "shard.parquet"
    shard.write_bytes(b"packed")

    class _Packed:
        shard_paths = (str(shard),)

    monkeypatch.setattr(full_run, "verify_packed_artifact", lambda **_: _Packed())
    monkeypatch.setattr(
        full_run,
        "_pii_body_loading_evidence",
        lambda *_args, **_kwargs: {"checkpoint_body_attestation": {"body_tensor_count": 1}},
    )
    result = full_run._preflight_candidate_assets(
        "pii350",
        snapshot_download=lambda **_: str(snapshot),
        load_dataset=lambda *_args, **_kwargs: [None] * 1700,
        # reason: the body-evidence function is replaced above, so this sentinel is never called.
        auto_model_for_token_classification=cast("PretrainedFactory", object),
        contract=scout,
    )
    receipt = result["receipt"]
    assert receipt["config_digest"] == scout["config_digest"]
    assert receipt["config_digest"] in str(full_run._receipt_path(scout))
    assert scout["training"]["batch_size"] == 128
    assert scout["training"]["lora"]["rank"] == 128
    assert scout["training"]["lora"]["alpha"] == 256
    assert scout["optimizer"]["lr"] in {1e-4, 2e-4, 3e-4, 4e-4}


def test_pii350_scout_rejects_a_generic_or_other_arm_receipt(tmp_path: Path) -> None:
    import json

    scout = contract.render_pii350_scout_contract("lr1e-4")
    other_arm = contract.render_pii350_scout_contract("lr2e-4")
    generic = contract.render_full_run("pii350")
    for name, receipt_contract in (("generic", generic), ("other-arm", other_arm)):
        receipt_path = tmp_path / f"{name}.json"
        receipt_path.write_text(
            json.dumps({
                "receipt_schema_version": full_run.PREFLIGHT_RECEIPT_SCHEMA_VERSION,
                "candidate": "pii350",
                "model_id": receipt_contract["model_id"],
                "model_revision": receipt_contract["model_revision"],
                "packed_revision": receipt_contract["packed_dataset"]["revision"],
                "packed_manifest_sha256": receipt_contract["packed_dataset"]["manifest_sha256"],
                "eval": receipt_contract["evaluation"],
                "config_digest": receipt_contract["config_digest"],
            }),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="receipt pin mismatch"):
            full_run._validated_receipt_inventory(scout, receipt_path=receipt_path)
