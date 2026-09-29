"""Guarded, candidate-specific Modal launch surface for full BIOES runs.

Use the pure renderer for review:
``python -m meddies_pii.training.bioes.trainers.full_run --render-config encoder350``.
This module intentionally has no default candidate and no multi-candidate lane.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack (torch, transformers, unsloth, datasets) is installed in the Modal image, so each
# reason: function body imports it inside the container.
# ruff: file-ignore[print]
# reason: this module runs inside a Modal container; its standard output is the operator's streamed run log.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
# ruff: file-ignore[implicit-namespace-package]
# reason: `modal/` is the only subpackage of `bioes/` without an `__init__.py` — assembly, data, eval,
# reason: reports and trainers all have one — so the asymmetry reads as an oversight, and adding the
# reason: file is likely inert under hatchling's src-layout discovery.
# reason: Deliberately deferred rather than fixed here: this directory holds every spend-authorization
# reason: gate, and its Modal-remote import paths have only fake-mediated local coverage, so adding
# reason: `__init__.py` is a post-merge change whose proof is a real GPU smoke run — owner ledger item.
import json
from collections.abc import Callable, Mapping, Sized
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast, runtime_checkable

import modal

from meddies_pii.training.bioes.data.artifacts import load_pretrained
from meddies_pii.training.bioes.modal.base_selection import verify_packed_artifact
from meddies_pii.training.bioes.modal.probe_scaffold import (
    VolumeMounts,
    artifact_volume,
    hf_cache_environment,
    hf_cache_volume,
    hf_secret,
    require_modal_profile,
    source_mounted_image,
    volume_mounts,
)
from meddies_pii.training.bioes.trainers.full_run import (
    ENCODER_FULL_RUN_IMAGE_PACKAGES as CONTRACT_ENCODER_FULL_RUN_IMAGE_PACKAGES,
)
from meddies_pii.training.bioes.trainers.full_run import (
    FULL_RUN_CONFIRMATION,
    FULL_RUN_HARD_TIMEOUT_SECONDS,
    FULL_RUN_MODAL_PROFILES,
    FULL_RUN_PYTHON_VERSION,
    M230_COMPARISON_CANDIDATES,
    M230_COMPARISON_PROFILES,
    PII350_CAPACITY_PROFILE,
    PII350_MILESTONE_HARD_TIMEOUT_SECONDS,
    PII350_UTILIZATION_PROFILE,
    render_base230_oom_fallback_contract,
    render_full_run,
    render_m230_comparison_run_contract,
    render_pii350_capacity_run_contract,
    render_pii350_milestone_run_contract,
    render_pii350_scout_contract,
    render_pii350_utilization_run_contract,
    require_base230_oom_fallback_execute,
    require_full_run_execute,
    require_m230_comparison_run_execute,
    require_pii350_capacity_run_execute,
    require_pii350_milestone_run_execute,
    require_pii350_scout_execute,
    require_pii350_utilization_run_execute,
    validate_pinned_eval_inventory,
)
from meddies_pii.training.bioes.trainers.full_run import (
    FULL_RUN_IMAGE_PACKAGES as CONTRACT_FULL_RUN_IMAGE_PACKAGES,
)
from meddies_pii.training.bioes.trainers.full_run_runtime import (
    lfm2_checkpoint_body_state_attestation,
    pii_body_state_attestation,
)


@runtime_checkable
class _RemoteEndpoint(Protocol):
    def remote(self, *args: object) -> object: ...


def _call_remote(endpoint: object, *args: object) -> dict[str, Any]:
    if not isinstance(endpoint, _RemoteEndpoint):
        msg = "Modal endpoint does not expose remote"
        raise TypeError(msg)
    result = endpoint.remote(*args)
    if not isinstance(result, Mapping) or not all(isinstance(key, str) for key in result):
        msg = "Modal endpoint must return a mapping with string keys"
        raise TypeError(msg)
    return dict(cast("Mapping[str, Any]", result))


class _H100Options(TypedDict):
    gpu: str
    cpu: float
    memory: int
    timeout: int
    max_containers: int


class _CpuPreflightOptions(TypedDict):
    gpu: None
    cpu: float
    memory: int
    timeout: int
    max_containers: int


class _CpuPrewarmOptions(TypedDict):
    gpu: None
    cpu: float
    memory: int
    timeout: int
    max_containers: int


PII350_LABEL_COUNT = 161

CACHE_MOUNT = "/cache"
ARTIFACT_MOUNT = "/artifacts"
FULL_RUN_IMAGE_PACKAGES: tuple[str, ...] = (
    "torch==2.10.0",
    "transformers==5.2.0",
    "peft==0.19.1",
    "pyarrow==23.0.0",
    "datasets==4.3.0",
    "huggingface_hub==1.11.0",
    "nvidia-ml-py==13.590.44",
    "unsloth==2026.5.2",
    "unsloth_zoo==2026.5.1",
)
"""The dependency inventory gate audits pins by reading this file's literals.

So the pins are spelled out here and checked against the renderer's contract below. A starred import would leave the
shipped image unaudited.

"""
ENCODER_FULL_RUN_IMAGE_PACKAGES: tuple[str, ...] = (
    "torch==2.10.0",
    "transformers==5.2.0",
    "peft==0.19.1",
    "pyarrow==23.0.0",
    "datasets==4.3.0",
    "huggingface_hub==1.11.0",
    "nvidia-ml-py==13.590.44",
    "unsloth==2026.7.4",
    "unsloth_zoo==2026.7.4",
)
if (
    FULL_RUN_IMAGE_PACKAGES != CONTRACT_FULL_RUN_IMAGE_PACKAGES
    or ENCODER_FULL_RUN_IMAGE_PACKAGES != CONTRACT_ENCODER_FULL_RUN_IMAGE_PACKAGES
):
    msg = "Modal full-run image pins drifted from the rendered run contract"
    raise RuntimeError(msg)
if FULL_RUN_PYTHON_VERSION != "3.11":
    msg = "Modal full-run Python drifted from its pinned registry image"
    raise RuntimeError(msg)
FULL_CACHE_ENVIRONMENT = hf_cache_environment()
"""CPU prewarm stores Hugging Face snapshots directly in /cache/hf."""
UNSLOTH_CACHE_ENVIRONMENT = FULL_CACHE_ENVIRONMENT
ONLINE_CACHE_ENVIRONMENT = hf_cache_environment(
    offline=None,
    disable_hub_telemetry=True,
)
FULL_RUN_PROFILES = FULL_RUN_MODAL_PROFILES
H100_OPTIONS: _H100Options = {
    "gpu": "H100!",
    "cpu": 4.0,
    "memory": 64 * 1024,
    "timeout": FULL_RUN_HARD_TIMEOUT_SECONDS,
    "max_containers": 1,
}
PII350_MILESTONE_H100_OPTIONS: _H100Options = {
    **H100_OPTIONS,
    "timeout": PII350_MILESTONE_HARD_TIMEOUT_SECONDS,
}
CPU_PREFLIGHT_OPTIONS: _CpuPreflightOptions = {
    "gpu": None,
    "cpu": 2.0,
    "memory": 8 * 1024,
    "timeout": 600,
    "max_containers": 1,
}
CPU_PREWARM_OPTIONS: _CpuPrewarmOptions = {
    "gpu": None,
    "cpu": 12.0,
    "memory": 64 * 1024,
    "timeout": 3_600,
    "max_containers": 1,
}
cache = hf_cache_volume()
artifacts = artifact_volume()
secret = hf_secret()
unsloth_image = source_mounted_image(
    modal.Image
    .from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941")
    .pip_install(*FULL_RUN_IMAGE_PACKAGES)
    .env(UNSLOTH_CACHE_ENVIRONMENT),
)
"""The pinned registry image replaces the floating ``python_version=FULL_RUN_PYTHON_VERSION`` constructor.

while keeping 3.11.

"""
encoder_unsloth_image = source_mounted_image(
    modal.Image
    .from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941")
    .pip_install(*ENCODER_FULL_RUN_IMAGE_PACKAGES)
    .env(UNSLOTH_CACHE_ENVIRONMENT),
)
prewarm_image = source_mounted_image(
    modal.Image
    .from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941")
    .pip_install(*ENCODER_FULL_RUN_IMAGE_PACKAGES)
    .env(ONLINE_CACHE_ENVIRONMENT),
)
app = modal.App("meddies-lfm25-bioes-full-runs")
COMMON_MOUNTS: VolumeMounts = volume_mounts(cache, artifacts)
PREFLIGHT_RECEIPT_SCHEMA_VERSION = 2


def _canonical_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _receipt_path(contract: Mapping[str, Any]) -> Path:
    return (
        Path("/artifacts/full-runs/preflight")
        / str(contract["candidate_key"])
        / str(contract["config_digest"])
        / f"verification-receipt-v{PREFLIGHT_RECEIPT_SCHEMA_VERSION}.json"
    )


def _asset_receipt_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the batch-independent primary contract used to prove cached assets.

    Returns:
        The contract whose receipt path the cached assets are proven against. A launch
        variant normally re-renders to its primary contract, so batch-size arms share one
        receipt instead of each demanding its own CPU preflight. The two named anchor
        purposes are the exception and keep their own contract, because their assets differ.

    """
    manual = contract.get("manual_launch")
    if isinstance(manual, Mapping) and manual.get("purpose") in {
        "anchor_pii350_r128a256_batch128_learning_rate",
        "anchor_pii350_r128a256_batch128_wsd_learning_rate",
    }:
        return contract
    if contract.get("fallback_launch") is not None or contract.get("manual_launch") is not None:
        return render_full_run(str(contract["candidate_key"]))
    return contract


def _pii_body_loading_evidence(
    contract: Mapping[str, Any],
    auto_model_for_token_classification: object,
    *,
    snapshot: str,
) -> dict[str, Any]:
    """Load the PII wrapper so ``lfm2.*`` checkpoint tensors retain their namespace.

    Returns:
        The body-loading evidence: the missing, mismatched and unexpected key lists (all
        proven empty to get here), the classifier keys deliberately discarded, the 161-label
        head width, and the attestations over both the loaded body state and the checkpoint
        on disk. Two attestations rather than one is the point — they let a later reader
        prove the body it holds is the body the pinned checkpoint contained.

    Raises:
        RuntimeError: If the loader does not return loading information, or returns it in an
            unusable shape; if any tensor is missing, mismatched or unexpected; if the
            wrapper exposes no ``.lfm2`` body; if the body's tensors are not all retained
            under the ``lfm2.`` prefix; if anything outside the classifier head sits beyond
            the body; or if the classifier is not the expected 161-label head. Loading
            through the PII wrapper rather than the bare body is what preserves the
            namespace, and these checks are what prove the preservation actually happened.

    """
    import torch

    loaded = load_pretrained(
        auto_model_for_token_classification,
        str(contract["model_id"]),
        {
            "revision": str(contract["model_revision"]),
            "trust_remote_code": True,
            "output_loading_info": True,
            "torch_dtype": torch.bfloat16,
            "cache_dir": "/cache/hf",
            "local_files_only": True,
        },
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "PII token-classification wrapper did not expose loading information"
        raise RuntimeError(msg)
    wrapper, loading_info = loaded
    if not isinstance(loading_info, Mapping):
        msg = "PII token-classification loading information is invalid"
        raise RuntimeError(msg)
    missing = list(loading_info.get("missing_keys", ()))
    mismatched = list(loading_info.get("mismatched_keys", ()))
    unexpected = list(loading_info.get("unexpected_keys", ()))
    if missing or mismatched or unexpected:
        msg = "PII token-classification wrapper has missing, mismatched, or unexpected tensors"
        raise RuntimeError(msg)
    body = getattr(wrapper, "lfm2", None)
    if body is None:
        msg = "PII token-classification wrapper has no .lfm2 encoder body"
        raise RuntimeError(msg)
    wrapper_state = wrapper.state_dict()
    body_state = body.state_dict()
    body_wrapper_keys = {f"lfm2.{name}" for name in body_state}
    if not body_wrapper_keys or not body_wrapper_keys <= set(wrapper_state):
        msg = "PII wrapper does not retain every lfm2 body tensor"
        raise RuntimeError(msg)
    discarded_head_keys = sorted(set(wrapper_state) - body_wrapper_keys)
    expected_head_keys = {"classifier.weight", "classifier.bias", "class_weights"}
    if set(discarded_head_keys) != expected_head_keys:
        msg = "PII wrapper has non-classifier tensors outside the encoder body"
        raise RuntimeError(msg)
    classifier = getattr(wrapper, "classifier", None)
    if getattr(classifier, "out_features", None) != PII350_LABEL_COUNT:
        msg = "PII wrapper classifier is not the expected 161-label head"
        raise RuntimeError(msg)
    return {
        "missing_body_keys": missing,
        "mismatched_body_keys": mismatched,
        "unexpected_wrapper_keys": unexpected,
        "discarded_pii_label_head_keys": discarded_head_keys,
        "discarded_pii_classifier_out_features": 161,
        "body_state_attestation": pii_body_state_attestation(body),
        "checkpoint_body_attestation": lfm2_checkpoint_body_state_attestation(snapshot),
    }


def _masked_lm_body_loading_evidence(
    contract: Mapping[str, Any],
    auto_model_for_masked_lm: object,
    *,
    snapshot: str,
) -> dict[str, Any]:
    """Load the canonical masked-LM wrapper and prove its complete LFM2 body.

    Returns:
        The same evidence shape as the PII path, with the tied ``lm_head.weight`` recorded as
        the one discarded key. Keeping the two evidence builders separate rather than
        parameterized is deliberate: the expected head differs, and a shared builder would
        make the head expectation a runtime argument instead of a fixed fact per candidate.

    Raises:
        RuntimeError: If the loader does not return loading information, or returns it in an
            unusable shape; if any tensor is missing, mismatched or unexpected; if the
            wrapper exposes no ``.lfm2`` body; if the body's tensors are not all retained
            under the ``lfm2.`` prefix; if anything other than exactly ``lm_head.weight``
            sits outside the body; or if that weight is absent.

    """
    import torch

    loaded = load_pretrained(
        auto_model_for_masked_lm,
        str(contract["model_id"]),
        {
            "revision": str(contract["model_revision"]),
            "trust_remote_code": True,
            "output_loading_info": True,
            "torch_dtype": torch.bfloat16,
            "cache_dir": "/cache/hf",
            "local_files_only": True,
        },
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "masked-LM wrapper did not expose loading information"
        raise RuntimeError(msg)
    wrapper, loading_info = loaded
    if not isinstance(loading_info, Mapping):
        msg = "masked-LM loading information is invalid"
        raise RuntimeError(msg)
    missing = list(loading_info.get("missing_keys", ()))
    mismatched = list(loading_info.get("mismatched_keys", ()))
    unexpected = list(loading_info.get("unexpected_keys", ()))
    if missing or mismatched or unexpected:
        msg = "masked-LM wrapper has missing, mismatched, or unexpected tensors"
        raise RuntimeError(msg)
    body = getattr(wrapper, "lfm2", None)
    if body is None:
        msg = "masked-LM wrapper has no .lfm2 encoder body"
        raise RuntimeError(msg)
    wrapper_state = wrapper.state_dict()
    body_wrapper_keys = {f"lfm2.{name}" for name in body.state_dict()}
    if not body_wrapper_keys or not body_wrapper_keys <= set(wrapper_state):
        msg = "masked-LM wrapper does not retain every lfm2 body tensor"
        raise RuntimeError(msg)
    discarded_head_keys = sorted(set(wrapper_state) - body_wrapper_keys)
    if discarded_head_keys != ["lm_head.weight"]:
        msg = "masked-LM wrapper has tensors outside the tied lm_head"
        raise RuntimeError(msg)
    if getattr(getattr(wrapper, "lm_head", None), "weight", None) is None:
        msg = "masked-LM wrapper has no lm_head weight"
        raise RuntimeError(msg)
    return {
        "missing_body_keys": missing,
        "mismatched_body_keys": mismatched,
        "unexpected_wrapper_keys": unexpected,
        "discarded_masked_lm_head_keys": discarded_head_keys,
        "body_state_attestation": pii_body_state_attestation(body),
        "checkpoint_body_attestation": lfm2_checkpoint_body_state_attestation(snapshot),
    }


# reason: preflight exposes candidate/contract as its Modal schema; bundling would break callers.
def _preflight_candidate_assets(  # ruff: ignore[too-many-arguments]
    candidate_key: str,
    *,
    snapshot_download: Callable[..., object],
    load_dataset: Callable[..., Sized],
    auto_model_for_token_classification: object | None = None,
    auto_model_for_masked_lm: object | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """CPU-only complete asset proof that the H100 can validate cheaply.

    Returns:
        The preflight result — candidate, resolved model snapshot, shard and eval row counts,
        an ``offline`` marker — wrapped around the durable ``receipt`` the caller persists.
        The receipt carries the inventory and its digest, so the H100 can revalidate by
        comparing digests and stat sizes rather than re-downloading or re-loading anything.

    Raises:
        RuntimeError: If the pinned model snapshot is absent from the offline cache; if the
            pinned evaluation split is not exactly its expected row count; if the PII
            candidate is preflighted without a token-classification loader, or an encoder
            candidate without a masked-LM loader; or, from the body-loading evidence, if the
            checkpoint tensors do not load cleanly. Doing all of this on CPU is the whole
            design: every one of these refusals costs cache time rather than H100 time.

    """
    contract = contract or render_full_run(candidate_key)
    packed = verify_packed_artifact(revision=str(contract["packed_dataset"]["revision"]))
    snapshot = snapshot_download(
        repo_id=str(contract["model_id"]),
        revision=str(contract["model_revision"]),
        cache_dir="/cache/hf",
        local_files_only=True,
    )
    if not isinstance(snapshot, str) or not Path(snapshot).is_dir():
        msg = "pinned model snapshot is absent from the offline cache"
        raise RuntimeError(msg)
    evaluation = contract["evaluation"]
    dataset = load_dataset(
        str(evaluation["id"]),
        str(evaluation["config"]),
        split=str(evaluation["split"]),
        revision=str(evaluation["revision"]),
        cache_dir="/cache/hf",
    )
    if len(dataset) != int(evaluation["expected_rows"]):
        msg = "pinned eval/train inventory is not exactly 1,700 rows"
        raise RuntimeError(msg)
    inventory = [{"path": path, "bytes": Path(path).stat().st_size} for path in packed.shard_paths]
    receipt: dict[str, Any] = {
        "receipt_schema_version": PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "candidate": candidate_key,
        "model_id": contract["model_id"],
        "model_revision": contract["model_revision"],
        "packed_revision": contract["packed_dataset"]["revision"],
        "packed_manifest_sha256": contract["packed_dataset"]["manifest_sha256"],
        "packed_shard_count": len(inventory),
        "packed_inventory": inventory,
        "packed_inventory_sha256": _canonical_sha256(inventory),
        "eval": evaluation,
        "config_digest": contract["config_digest"],
    }
    if candidate_key == "pii350":
        if auto_model_for_token_classification is None:
            msg = "PII CPU preflight requires token-classification loading evidence"
            raise RuntimeError(msg)
        receipt["pii_body_loading"] = _pii_body_loading_evidence(
            contract,
            auto_model_for_token_classification,
            snapshot=snapshot,
        )
    elif candidate_key in {"encoder230", "encoder350"}:
        if auto_model_for_masked_lm is None:
            msg = "encoder CPU preflight requires masked-LM loading evidence"
            raise RuntimeError(msg)
        receipt["masked_lm_body_loading"] = _masked_lm_body_loading_evidence(
            contract,
            auto_model_for_masked_lm,
            snapshot=snapshot,
        )
    return {
        "candidate": candidate_key,
        "model_snapshot": snapshot,
        "packed_shards": len(inventory),
        "eval_rows": len(dataset),
        "offline": True,
        "receipt": receipt,
    }


def _prewarm_candidate_assets(
    candidate_key: str,
    *,
    snapshot_download: Callable[..., object],
    load_dataset: Callable[..., Sized],
) -> dict[str, Any]:
    """Download the pinned assets, then verify the packed shards offline.

    Returns:
        The hydration result: the candidate, both resolved snapshot paths, the verified shard
        count, the eval row count, and an ``offline_verified`` marker. This is the only
        online step in the pipeline; everything downstream runs with
        ``local_files_only=True``, so its job is to leave the cache in a state the offline
        preflight can prove.

    Raises:
        RuntimeError: If the pinned evaluation split is not exactly its expected row count,
            or if either snapshot download did not resolve to a local directory. Verifying
            the packed shards here, after download and still on CPU, is what stops a
            truncated or partial cache from being discovered inside a paid container.

    """
    contract = render_full_run(candidate_key)
    packed = contract["packed_dataset"]
    evaluation = contract["evaluation"]
    print(
        json.dumps(
            {
                "event": "packed_cache_download_started",
                "candidate": candidate_key,
                "expected_shards": packed["shard_count"],
                "workers": 12,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    packed_snapshot = snapshot_download(
        repo_id=str(packed["id"]),
        repo_type="dataset",
        revision=str(packed["revision"]),
        allow_patterns=["packed/manifest.json", "packed/data/*.parquet"],
        cache_dir="/cache/hf",
        local_files_only=False,
        max_workers=12,
    )
    print(
        json.dumps(
            {
                "event": "packed_cache_download_complete",
                "candidate": candidate_key,
                "snapshot": packed_snapshot,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        json.dumps(
            {"event": "model_cache_download_started", "candidate": candidate_key},
            sort_keys=True,
        ),
        flush=True,
    )
    model_snapshot = snapshot_download(
        repo_id=str(contract["model_id"]),
        revision=str(contract["model_revision"]),
        cache_dir="/cache/hf",
        local_files_only=False,
        max_workers=12,
    )
    print(
        json.dumps(
            {
                "event": "model_cache_download_complete",
                "candidate": candidate_key,
                "snapshot": model_snapshot,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    verified = verify_packed_artifact(
        revision=str(packed["revision"]),
        cache_dir="/cache/hf",
    )
    print(
        json.dumps(
            {
                "event": "packed_cache_verification_complete",
                "candidate": candidate_key,
                "verified_shards": len(verified.shard_paths),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dataset = load_dataset(
        str(evaluation["id"]),
        str(evaluation["config"]),
        split=str(evaluation["split"]),
        revision=str(evaluation["revision"]),
        cache_dir="/cache/hf",
    )
    if len(dataset) != int(evaluation["expected_rows"]):
        msg = "pinned eval/train inventory is not exactly 1,700 rows"
        raise RuntimeError(msg)
    if not isinstance(packed_snapshot, str) or not Path(packed_snapshot).is_dir():
        msg = "packed snapshot did not resolve to a local directory"
        raise RuntimeError(msg)
    if not isinstance(model_snapshot, str) or not Path(model_snapshot).is_dir():
        msg = "model snapshot did not resolve to a local directory"
        raise RuntimeError(msg)
    return {
        "candidate": candidate_key,
        "model_snapshot": model_snapshot,
        "packed_snapshot": packed_snapshot,
        "packed_shards": len(verified.shard_paths),
        "eval_rows": len(dataset),
        "offline_verified": True,
    }


# reason: validated receipt owns read text and asset together; splitting would fragment diagnostics.
def _validated_receipt_inventory(  # ruff: ignore[complex-structure,too-many-branches,too-many-locals,too-many-statements]
    contract: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
) -> tuple[tuple[str, ...], Mapping[str, Any] | None]:
    """Fail closed on receipt pins, then return local shards and PII body proof.

    Returns:
        The verified shard paths and the encoder checkpoint attestation, or ``None`` for that
        attestation on ``base230``, which has no encoder body to prove. Every returned path
        was confirmed present on this container's cache at the recorded byte size, so the
        caller receives paths it can open rather than paths a receipt once claimed.

    Raises:
        RuntimeError: If the CPU verification receipt is absent or not an object; if any
            pinned field — schema version, candidate, model id and revision, packed revision
            and manifest digest, eval block, config digest — differs from the contract; if
            the shard inventory is missing, empty, disagrees with the recorded or contracted
            shard count, or does not match its own canonical digest; if the encoder body
            loading proof or its attestation fields are missing or malformed; or if any
            listed shard is not a file on this cache or has changed size. Fail-closed is the
            design: the receipt is required BEFORE launch, so a missing receipt refuses the
            run rather than triggering a fresh in-container verification.

    """
    receipt_contract = _asset_receipt_contract(contract)
    path = receipt_path or _receipt_path(receipt_contract)
    if not path.is_file():
        msg = "CPU verification receipt is required before H100 launch"
        raise RuntimeError(msg)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        msg = "CPU verification receipt is invalid"
        raise RuntimeError(msg)
    expected = {
        "receipt_schema_version": PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "candidate": receipt_contract["candidate_key"],
        "model_id": receipt_contract["model_id"],
        "model_revision": receipt_contract["model_revision"],
        "packed_revision": receipt_contract["packed_dataset"]["revision"],
        "packed_manifest_sha256": receipt_contract["packed_dataset"]["manifest_sha256"],
        "eval": receipt_contract["evaluation"],
        "config_digest": receipt_contract["config_digest"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        msg = "CPU verification receipt pin mismatch"
        raise RuntimeError(msg)
    inventory = receipt.get("packed_inventory")
    if not isinstance(inventory, list) or not inventory:
        msg = "CPU verification receipt has no shard inventory"
        raise RuntimeError(msg)
    if (
        receipt.get("packed_shard_count") != len(inventory)
        or receipt.get("packed_shard_count") != contract["packed_dataset"]["shard_count"]
    ):
        msg = "CPU verification receipt shard count mismatch"
        raise RuntimeError(msg)
    if receipt.get("packed_inventory_sha256") != _canonical_sha256(inventory):
        msg = "CPU verification receipt inventory digest mismatch"
        raise RuntimeError(msg)
    encoder_checkpoint_attestation: Mapping[str, Any] | None = None
    candidate_key = str(contract["candidate_key"])
    proof_key = "pii_body_loading" if candidate_key == "pii350" else "masked_lm_body_loading"
    if candidate_key != "base230":
        evidence = receipt.get(proof_key)
        checkpoint_attestation = evidence.get("checkpoint_body_attestation") if isinstance(evidence, Mapping) else None
        expected_discarded = (
            ["class_weights", "classifier.bias", "classifier.weight"] if candidate_key == "pii350" else ["lm_head.weight"]
        )
        discarded_key = "discarded_pii_label_head_keys" if candidate_key == "pii350" else "discarded_masked_lm_head_keys"
        # reason: validated receipt keeps missing/mismatched in one gate; helper predicates would scatter the rule.
        if (
            not isinstance(evidence, Mapping)  # ruff: ignore[too-many-boolean-expressions]
            or evidence.get("missing_body_keys") != []
            or evidence.get("mismatched_body_keys") != []
            or evidence.get("unexpected_wrapper_keys") != []
            or evidence.get(discarded_key) != expected_discarded
            or (candidate_key == "pii350" and evidence.get("discarded_pii_classifier_out_features") != PII350_LABEL_COUNT)
            or not isinstance(checkpoint_attestation, Mapping)
            or not isinstance(checkpoint_attestation.get("body_tensor_count"), int)
            or checkpoint_attestation.get("body_tensor_count", 0) <= 0
            or not isinstance(checkpoint_attestation.get("body_tensor_names_sha256"), str)
            or not isinstance(checkpoint_attestation.get("body_tensor_values_sha256"), str)
        ):
            msg = "CPU verification receipt lacks encoder body loading proof"
            raise RuntimeError(msg)
        encoder_checkpoint_attestation = dict(checkpoint_attestation)
    paths: list[str] = []
    for item in inventory:
        if not isinstance(item, Mapping):
            msg = "CPU verification receipt shard inventory is invalid"
            raise RuntimeError(msg)
        shard_path, byte_count = item.get("path"), item.get("bytes")
        if not isinstance(shard_path, str) or not isinstance(byte_count, int):
            msg = "CPU verification receipt shard entry is invalid"
            raise RuntimeError(msg)
        if not Path(shard_path).is_file():
            msg = "CPU-verified shard is absent on the H100 cache"
            raise RuntimeError(msg)
        if Path(shard_path).stat().st_size != byte_count:
            msg = "CPU-verified shard size changed before H100 launch"
            raise RuntimeError(msg)
        paths.append(shard_path)
    return tuple(paths), encoder_checkpoint_attestation


def _persist_preflight_receipt(contract: Mapping[str, Any], receipt: Mapping[str, Any]) -> Path:
    """Persist one canonical receipt; a conflicting existing receipt is fatal.

    Returns:
        The receipt path, whether this call wrote it or found it already written with
        byte-identical content. Re-running the preflight is therefore safe and idempotent.

    Raises:
        RuntimeError: If a receipt already exists at that path with different bytes. Treating
            that as fatal rather than overwriting is what makes the receipt immutable: the
            path is keyed by candidate and config digest, so differing content means two
            preflights disagreed about the same configuration, and silently taking the newer
            one would destroy the evidence that they did.

    """
    root = _receipt_path(contract).parent
    root.mkdir(parents=True, exist_ok=True)
    path = _receipt_path(contract)
    encoded = json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        msg = "immutable CPU verification receipt already differs"
        raise RuntimeError(msg)
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    return path


def _validated_shard_paths_from_receipt(
    contract: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
) -> tuple[str, ...]:
    """Compatibility wrapper for callers that need only validated shard paths.

    Returns:
        Just the verified shard paths. Discarding the encoder attestation here narrows what
        the caller receives, never what was checked — the full validation still runs, so a
        caller taking this wrapper cannot skip the body proof by asking for less.

    """
    paths, _ = _validated_receipt_inventory(contract, receipt_path=receipt_path)
    return paths


@app.function(
    image=unsloth_image,
    volumes=COMMON_MOUNTS,
    secrets=[secret],
    **CPU_PREFLIGHT_OPTIONS,
)
def preflight_candidate_assets(candidate_key: str, contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """CPU-only cache preflight, emitting durable JSON evidence to the volume.

    Returns:
        The preflight result with ``receipt`` replaced by ``receipt_path``. The receipt is
        popped and persisted rather than returned, so what crosses the Modal boundary is a
        pointer to committed volume evidence instead of a payload a caller could edit and
        hand back. The volume is committed before this returns.

    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification

    print(
        json.dumps({"event": "asset_preflight_started", "candidate": candidate_key}),
        flush=True,
    )
    result = _preflight_candidate_assets(
        candidate_key,
        snapshot_download=snapshot_download,
        load_dataset=load_dataset,
        auto_model_for_token_classification=AutoModelForTokenClassification,
        auto_model_for_masked_lm=AutoModelForMaskedLM,
        contract=contract,
    )
    receipt = result.pop("receipt")
    contract = contract or render_full_run(candidate_key)
    result["receipt_path"] = str(_persist_preflight_receipt(contract, receipt))
    artifacts.commit()
    print(
        json.dumps({"event": "asset_preflight_complete", **result}, sort_keys=True),
        flush=True,
    )
    return result


@app.function(
    image=prewarm_image,
    volumes={CACHE_MOUNT: cache},
    secrets=[secret],
    **CPU_PREWARM_OPTIONS,
)
def prewarm_candidate_assets(candidate_key: str) -> dict[str, Any]:
    """CPU-only online cache hydration with visible Hugging Face progress.

    Returns:
        The hydration result unchanged from the inner call. The cache volume is committed in
        a ``finally`` block, so a partially hydrated cache is still committed and the next
        run resumes from it rather than re-downloading from zero.

    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    print(
        json.dumps({"event": "asset_prewarm_started", "candidate": candidate_key}),
        flush=True,
    )
    try:
        result = _prewarm_candidate_assets(
            candidate_key,
            snapshot_download=snapshot_download,
            load_dataset=load_dataset,
        )
    finally:
        cache.commit()
    print(
        json.dumps({"event": "asset_prewarm_complete", **result}, sort_keys=True),
        flush=True,
    )
    return result


def _remote_contract(
    candidate_key: str,
    *,
    contract: Mapping[str, Any],
    qualification_artifact: Mapping[str, Any] | None,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, str]:
    if contract.get("candidate_key") != candidate_key:
        msg = "candidate-specific Modal endpoint received a different contract"
        raise RuntimeError(msg)
    require_full_run_execute(
        contract,
        execute=True,
        confirmation=FULL_RUN_CONFIRMATION,
        qualification_artifact=qualification_artifact,
    )
    if eval_inventory is None:
        msg = "full H100 run requires a pinned eval inventory before encoding"
        raise RuntimeError(msg)
    validate_pinned_eval_inventory(eval_inventory)
    return {"status": "validated_for_execution", "candidate": candidate_key}


def _remote_base230_oom_fallback_contract(
    *,
    contract: Mapping[str, Any],
    primary_oom_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, str]:
    require_base230_oom_fallback_execute(
        contract,
        primary_oom_evidence=primary_oom_evidence,
        primary_action=primary_action,
    )
    if eval_inventory is None:
        msg = "full H100 run requires a pinned eval inventory before encoding"
        raise RuntimeError(msg)
    validate_pinned_eval_inventory(eval_inventory)
    return {"status": "validated_for_execution", "candidate": "base230"}


def _remote_m230_comparison_contract(
    candidate_key: str,
    *,
    contract: Mapping[str, Any],
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, str]:
    require_m230_comparison_run_execute(
        contract,
        candidate_key=candidate_key,
        primary_action=primary_action,
    )
    if eval_inventory is None:
        msg = "full H100 run requires a pinned eval inventory before encoding"
        raise RuntimeError(msg)
    validate_pinned_eval_inventory(eval_inventory)
    return {"status": "validated_for_execution", "candidate": candidate_key}


def _remote_pii350_utilization_contract(
    *,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, str]:
    require_pii350_utilization_run_execute(
        contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
    )
    if eval_inventory is None:
        msg = "full H100 run requires a pinned eval inventory before encoding"
        raise RuntimeError(msg)
    validate_pinned_eval_inventory(eval_inventory)
    return {"status": "validated_for_execution", "candidate": "pii350"}


def _remote_pii350_capacity_contract(
    *,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, str]:
    require_pii350_capacity_run_execute(
        contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
    )
    if eval_inventory is None:
        msg = "full H100 run requires a pinned eval inventory before encoding"
        raise RuntimeError(msg)
    validate_pinned_eval_inventory(eval_inventory)
    return {"status": "validated_for_execution", "candidate": "pii350"}


def _remote_pii350_milestone_contract(
    *,
    profile: str,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, str]:
    require_pii350_milestone_run_execute(
        contract,
        profile=profile,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
    )
    if eval_inventory is None:
        msg = "full H100 run requires a pinned eval inventory before encoding"
        raise RuntimeError(msg)
    validate_pinned_eval_inventory(eval_inventory)
    return {"status": "validated_for_execution", "candidate": "pii350"}


def _remote_pii350_scout_contract(
    *,
    arm: str,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, str]:
    require_pii350_scout_execute(
        contract,
        arm=arm,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
    )
    if eval_inventory is None:
        msg = "full H100 run requires a pinned eval inventory before encoding"
        raise RuntimeError(msg)
    validate_pinned_eval_inventory(eval_inventory)
    return {"status": "validated_for_execution", "candidate": "pii350"}


def _execute_remote(
    candidate_key: str,
    *,
    contract: Mapping[str, Any],
    qualification_artifact: Mapping[str, Any] | None,
    eval_inventory: Mapping[str, Any] | None,
    resume_checkpoint: str | None,
) -> dict[str, Any]:
    """Verify data before running the single-candidate CUDA runtime.

    An unqualified contract becomes eligible only after an actual runtime artifact is present on the mounted volume.  A
    caller-supplied JSON object alone is not qualification evidence.

    Returns:
        The completed run result from the CUDA runtime. Reaching the runtime call is itself
        the statement that the launch was authorized: the contract gate runs first, and the
        artifacts volume is committed in a ``finally`` block so a crashed run still leaves
        its partial evidence behind.

    Raises:
        RuntimeError: This ENFORCES a spend-authorization gate. It first calls the contract
            check, which passes ``execute=True`` with ``FULL_RUN_CONFIRMATION`` into
            ``require_full_run_execute`` and refuses a contract that is not byte-equal to the
            freshly rendered one, then refuses a run with no pinned eval inventory. For an
            unqualified contract it additionally refuses if no qualification artifact was
            supplied, if the artifact does not name an ``/artifacts/`` volume root, if that
            root holds no ``result.json``, or if any of the candidate, runtime backend, batch
            size, optimizer steps, status or execution id differs from the saved artifact.
            That last comparison is the load-bearing one: it is what makes qualification mean
            a run that actually happened on the volume rather than a JSON object a caller
            typed, and it is why the artifact is re-read from disk instead of trusted.

    """
    _remote_contract(
        candidate_key,
        contract=contract,
        qualification_artifact=qualification_artifact,
        eval_inventory=eval_inventory,
    )
    if contract["qualification"]["status"] == "unqualified":
        if qualification_artifact is None:
            msg = "missing qualification artifact"
            raise RuntimeError(msg)
        root = qualification_artifact.get("artifact_root")
        if not isinstance(root, str) or not root.startswith("/artifacts/"):
            msg = "runtime qualification must name an artifact-volume root"
            raise RuntimeError(msg)
        result_path = Path(root) / "result.json"
        if not result_path.is_file():
            msg = "runtime qualification result.json is absent"
            raise RuntimeError(msg)
        recorded = json.loads(result_path.read_text(encoding="utf-8"))
        for key in (
            "candidate",
            "runtime_backend",
            "batch_size",
            "optimizer_steps",
            "status",
            "execution_id",
        ):
            if recorded.get(key) != qualification_artifact.get(key):
                msg = "runtime qualification does not match its saved artifact"
                raise RuntimeError(msg)
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        execute_full_candidate,
    )

    shard_paths, encoder_checkpoint_attestation = _validated_receipt_inventory(contract)
    try:
        return execute_full_candidate(
            contract,
            shard_paths=shard_paths,
            artifact_parent="/artifacts/full-runs",
            commit=artifacts.commit,
            resume_checkpoint=resume_checkpoint,
            encoder_checkpoint_attestation=encoder_checkpoint_attestation,
        )
    finally:
        artifacts.commit()


def _execute_base230_oom_fallback_remote(
    *,
    contract: Mapping[str, Any],
    primary_oom_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Execute the separately authorized fresh Base230 batch-384 fallback.

    Returns:
        The completed run result. "Separately authorized" is the fact worth carrying: this
        endpoint enforces its OWN gate through the fallback contract check, which demands the
        recorded out-of-memory evidence and this run's own primary action, so approval to
        launch the primary Base230 run does not reach this one. It never resumes a
        checkpoint — the fallback is a fresh run by definition, and passing
        ``resume_checkpoint=None`` is what keeps it that way.

    """
    _remote_base230_oom_fallback_contract(
        contract=contract,
        primary_oom_evidence=primary_oom_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        execute_full_candidate,
    )

    shard_paths, encoder_checkpoint_attestation = _validated_receipt_inventory(contract)
    try:
        return execute_full_candidate(
            contract,
            shard_paths=shard_paths,
            artifact_parent="/artifacts/full-runs",
            commit=artifacts.commit,
            resume_checkpoint=None,
            encoder_checkpoint_attestation=encoder_checkpoint_attestation,
        )
    finally:
        artifacts.commit()


def _execute_m230_comparison_remote(
    candidate_key: str,
    *,
    contract: Mapping[str, Any],
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Execute one fresh Base230-320 or Encoder230-288 comparison.

    Returns:
        The completed run result for the named comparison arm. The candidate key is passed
        through to the contract check as well as the runtime, so an arm cannot be launched
        under another arm's contract. Always a fresh run: a comparison that resumed from a
        checkpoint would not be comparable to the arm it is measured against.

    """
    _remote_m230_comparison_contract(
        candidate_key,
        contract=contract,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        execute_full_candidate,
    )

    shard_paths, encoder_checkpoint_attestation = _validated_receipt_inventory(contract)
    try:
        return execute_full_candidate(
            contract,
            shard_paths=shard_paths,
            artifact_parent="/artifacts/full-runs",
            commit=artifacts.commit,
            resume_checkpoint=None,
            encoder_checkpoint_attestation=encoder_checkpoint_attestation,
        )
    finally:
        artifacts.commit()


def _execute_pii350_utilization_remote(
    *,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Execute the separately approved fresh PII350 utilization run.

    Returns:
        The completed run result. This and the capacity endpoint below take the SAME
        batch-192 source evidence but demand DIFFERENT primary actions, so completing batch
        192 unlocks each only through its own action and one approval cannot be replayed to
        launch the other. Always a fresh run.

    """
    _remote_pii350_utilization_contract(
        contract=contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        execute_full_candidate,
    )

    shard_paths, encoder_checkpoint_attestation = _validated_receipt_inventory(contract)
    try:
        return execute_full_candidate(
            contract,
            shard_paths=shard_paths,
            artifact_parent="/artifacts/full-runs",
            commit=artifacts.commit,
            resume_checkpoint=None,
            encoder_checkpoint_attestation=encoder_checkpoint_attestation,
        )
    finally:
        artifacts.commit()


def _execute_pii350_capacity_remote(
    *,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Execute the separately approved fresh PII350 r128/alpha256 run.

    Returns:
        The completed run result. This is the capacity half of the pair described on the
        utilization endpoint above: same batch-192 source evidence, its own required primary
        action, its own approval. Always a fresh run.

    """
    _remote_pii350_capacity_contract(
        contract=contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        execute_full_candidate,
    )

    shard_paths, encoder_checkpoint_attestation = _validated_receipt_inventory(contract)
    try:
        return execute_full_candidate(
            contract,
            shard_paths=shard_paths,
            artifact_parent="/artifacts/full-runs",
            commit=artifacts.commit,
            resume_checkpoint=None,
            encoder_checkpoint_attestation=encoder_checkpoint_attestation,
        )
    finally:
        artifacts.commit()


def _execute_pii350_milestone_remote(
    *,
    profile: str,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Execute one fresh PII350 equal-cursor milestone decision run.

    Returns:
        The completed run result for one milestone arm. The profile is passed to the contract
        check, which reads the action it requires back off the rendered contract rather than
        a constant — so an action approved for one profile cannot launch another arm. Every
        arm must reach the same packed cursor to be comparable, which is why this is a fresh
        run and why the contract fixes the cursor rather than the step count.

    """
    _remote_pii350_milestone_contract(
        profile=profile,
        contract=contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        execute_full_candidate,
    )

    shard_paths, encoder_checkpoint_attestation = _validated_receipt_inventory(contract)
    try:
        return execute_full_candidate(
            contract,
            shard_paths=shard_paths,
            artifact_parent="/artifacts/full-runs",
            commit=artifacts.commit,
            resume_checkpoint=None,
            encoder_checkpoint_attestation=encoder_checkpoint_attestation,
        )
    finally:
        artifacts.commit()


def _execute_pii350_scout_remote(
    *,
    arm: str,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _remote_pii350_scout_contract(
        arm=arm,
        contract=contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        execute_full_candidate,
    )

    shard_paths, encoder_checkpoint_attestation = _validated_receipt_inventory(contract)
    try:
        return execute_full_candidate(
            contract,
            shard_paths=shard_paths,
            artifact_parent="/artifacts/full-runs",
            commit=artifacts.commit,
            resume_checkpoint=None,
            encoder_checkpoint_attestation=encoder_checkpoint_attestation,
        )
    finally:
        artifacts.commit()


def _require_candidate_profile(candidate_key: str) -> None:
    _require_profile(FULL_RUN_PROFILES[candidate_key], candidate_key)


def _require_profile(expected: str, label: str) -> None:
    require_modal_profile(expected, label)


# reason: local remote call exposes candidate/resume as its Modal schema; bundling would break callers.
def _local_remote_call(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    candidate_key: str,
    endpoint: object,
    contract: Mapping[str, Any],
    qualification_artifact: Mapping[str, Any] | None,
    eval_inventory: Mapping[str, Any] | None,
    resume_checkpoint: str | None,
) -> dict[str, Any]:
    _require_candidate_profile(candidate_key)
    return _call_remote(
        endpoint,
        contract,
        qualification_artifact,
        eval_inventory,
        resume_checkpoint,
    )


def _local_preflight_call(candidate_key: str, *, expected_profile: str | None = None) -> dict[str, Any]:
    _require_profile(
        expected_profile or FULL_RUN_PROFILES[candidate_key],
        f"{candidate_key} asset preflight",
    )
    return preflight_candidate_assets.remote(candidate_key)


def _render_pii350_scout_contract_and_profile(arm: str) -> tuple[dict[str, Any], str]:
    """Bind a scout arm to its rendered contract and Modal profile.

    Returns:
        The arm's freshly rendered contract and the Modal profile it must run under. Both
        come from one render, so the contract and the workspace it is billed to cannot be
        chosen independently — which is what makes the caller's profile gate meaningful.

    Raises:
        RuntimeError: If the scout contract carries no manual launch metadata, or no Modal
            profile string within it. An arm with no declared profile has no workspace this
            can be pinned to, so dispatching it would bill an unverified account.

    """
    scout_contract = render_pii350_scout_contract(arm)
    manual_launch = scout_contract.get("manual_launch")
    if not isinstance(manual_launch, Mapping):
        msg = "PII350 scout contract requires manual launch metadata"
        raise RuntimeError(msg)
    profile = manual_launch.get("modal_profile")
    if not isinstance(profile, str):
        msg = "PII350 scout contract requires a Modal profile"
        raise RuntimeError(msg)
    return scout_contract, profile


def _local_pii350_scout_preflight_call(endpoint: object, arm: str) -> dict[str, Any]:
    """Dispatch a CPU receipt only after pinning the scout profile and arm.

    Returns:
        The remote preflight result for this arm. The profile gate runs BEFORE the dispatch,
        so a mismatched local Modal profile refuses here rather than after a container has
        started somewhere else. The rendered contract is passed rather than re-derived
        remotely, which keeps the arm the caller pinned and the arm preflighted identical.

    """
    scout_contract, profile = _render_pii350_scout_contract_and_profile(arm)
    _require_profile(profile, "PII350 scout asset preflight")
    return _call_remote(endpoint, "pii350", scout_contract)


def _local_prewarm_call(candidate_key: str, *, expected_profile: str) -> dict[str, Any]:
    _require_profile(expected_profile, f"{candidate_key} asset prewarm")
    return prewarm_candidate_assets.remote(candidate_key)


def _local_base230_oom_fallback_call(
    contract: Mapping[str, Any],
    primary_oom_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require_candidate_profile("base230")
    return run_base230_oom_fallback.remote(
        contract,
        primary_oom_evidence,
        primary_action,
        eval_inventory,
    )


def _local_m230_comparison_call(
    candidate_key: str,
    endpoint: object,
    contract: Mapping[str, Any],
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require_profile(
        M230_COMPARISON_PROFILES[candidate_key],
        f"{candidate_key} 230M comparison",
    )
    return _call_remote(
        endpoint,
        contract,
        primary_action,
        eval_inventory,
    )


def _local_pii350_utilization_call(
    endpoint: object,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require_profile(PII350_UTILIZATION_PROFILE, "PII350 utilization run")
    return _call_remote(
        endpoint,
        contract,
        source_run_evidence,
        primary_action,
        eval_inventory,
    )


def _local_pii350_capacity_call(
    endpoint: object,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require_profile(PII350_CAPACITY_PROFILE, "PII350 capacity run")
    return _call_remote(
        endpoint,
        contract,
        source_run_evidence,
        primary_action,
        eval_inventory,
    )


# reason: PII350 training endpoint, candidate, contract, and eval controls are one Modal call schema.
def _local_pii350_milestone_call(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    endpoint: object,
    profile: str,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require_profile(profile, "PII350 milestone run")
    return _call_remote(
        endpoint,
        profile,
        contract,
        source_run_evidence,
        primary_action,
        eval_inventory,
    )


# reason: PII350 scout endpoint, candidate, contract, and eval controls are one Modal call schema.
def _local_pii350_scout_call(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    endpoint: object,
    arm: str,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Dispatch one H100 scout arm after the local profile gate.

    Returns:
        The remote scout run result. Both checks run locally, before any paid container
        starts: the caller's contract must equal the freshly rendered one, and the active
        Modal profile must be the arm's own.

    Raises:
        RuntimeError: If the supplied contract is not byte-equal to the contract this arm
            renders. Comparing the whole dict rather than a digest or a few fields is the
            point — a contract that merely looks similar is refused, so a hand-edited launch
            payload cannot reach an H100.

    """
    expected_contract, profile = _render_pii350_scout_contract_and_profile(arm)
    if dict(contract) != expected_contract:
        msg = "PII350 scout requires the exact rendered contract"
        raise RuntimeError(msg)
    _require_profile(profile, "PII350 scout run")
    return _call_remote(
        endpoint,
        arm,
        contract,
        source_run_evidence,
        primary_action,
        eval_inventory,
    )


@app.function(image=unsloth_image, volumes=COMMON_MOUNTS, secrets=[secret], **H100_OPTIONS)
def run_base230(
    contract: Mapping[str, Any],
    qualification_artifact: Mapping[str, Any] | None,
    eval_inventory: Mapping[str, Any] | None,
    resume_checkpoint: str | None = None,
) -> dict[str, Any]:
    return _execute_remote(
        "base230",
        contract=contract,
        qualification_artifact=qualification_artifact,
        eval_inventory=eval_inventory,
        resume_checkpoint=resume_checkpoint,
    )


@app.function(image=unsloth_image, volumes=COMMON_MOUNTS, secrets=[secret], **H100_OPTIONS)
def run_base230_oom_fallback(
    contract: Mapping[str, Any],
    primary_oom_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _execute_base230_oom_fallback_remote(
        contract=contract,
        primary_oom_evidence=primary_oom_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )


@app.function(
    image=encoder_unsloth_image,
    volumes=COMMON_MOUNTS,
    secrets=[secret],
    **H100_OPTIONS,
)
def run_base230_comparison(
    contract: Mapping[str, Any],
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _execute_m230_comparison_remote(
        "base230",
        contract=contract,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )


@app.function(
    image=encoder_unsloth_image,
    volumes=COMMON_MOUNTS,
    secrets=[secret],
    **H100_OPTIONS,
)
def run_encoder230_comparison(
    contract: Mapping[str, Any],
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _execute_m230_comparison_remote(
        "encoder230",
        contract=contract,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )


@app.function(image=encoder_unsloth_image, volumes=COMMON_MOUNTS, secrets=[secret], **H100_OPTIONS)
def run_encoder230(
    contract: Mapping[str, Any],
    qualification_artifact: Mapping[str, Any] | None,
    eval_inventory: Mapping[str, Any] | None,
    resume_checkpoint: str | None = None,
) -> dict[str, Any]:
    return _execute_remote(
        "encoder230",
        contract=contract,
        qualification_artifact=qualification_artifact,
        eval_inventory=eval_inventory,
        resume_checkpoint=resume_checkpoint,
    )


@app.function(image=encoder_unsloth_image, volumes=COMMON_MOUNTS, secrets=[secret], **H100_OPTIONS)
def run_encoder350(
    contract: Mapping[str, Any],
    qualification_artifact: Mapping[str, Any] | None,
    eval_inventory: Mapping[str, Any] | None,
    resume_checkpoint: str | None = None,
) -> dict[str, Any]:
    return _execute_remote(
        "encoder350",
        contract=contract,
        qualification_artifact=qualification_artifact,
        eval_inventory=eval_inventory,
        resume_checkpoint=resume_checkpoint,
    )


@app.function(image=encoder_unsloth_image, volumes=COMMON_MOUNTS, secrets=[secret], **H100_OPTIONS)
def run_pii350(
    contract: Mapping[str, Any],
    qualification_artifact: Mapping[str, Any] | None,
    eval_inventory: Mapping[str, Any] | None,
    resume_checkpoint: str | None = None,
) -> dict[str, Any]:
    return _execute_remote(
        "pii350",
        contract=contract,
        qualification_artifact=qualification_artifact,
        eval_inventory=eval_inventory,
        resume_checkpoint=resume_checkpoint,
    )


@app.function(image=encoder_unsloth_image, volumes=COMMON_MOUNTS, secrets=[secret], **H100_OPTIONS)
def run_pii350_utilization(
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _execute_pii350_utilization_remote(
        contract=contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )


@app.function(image=encoder_unsloth_image, volumes=COMMON_MOUNTS, secrets=[secret], **H100_OPTIONS)
def run_pii350_capacity(
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _execute_pii350_capacity_remote(
        contract=contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )


@app.function(
    image=encoder_unsloth_image,
    volumes=COMMON_MOUNTS,
    secrets=[secret],
    **PII350_MILESTONE_H100_OPTIONS,
)
def run_pii350_milestone(
    profile: str,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _execute_pii350_milestone_remote(
        profile=profile,
        contract=contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )


@app.function(
    image=encoder_unsloth_image,
    volumes=COMMON_MOUNTS,
    secrets=[secret],
    **PII350_MILESTONE_H100_OPTIONS,
)
def run_pii350_scout(
    arm: str,
    contract: Mapping[str, Any],
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
    eval_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _execute_pii350_scout_remote(
        arm=arm,
        contract=contract,
        source_run_evidence=source_run_evidence,
        primary_action=primary_action,
        eval_inventory=eval_inventory,
    )


# reason: full run main owns and profile and full run together; splitting would let sample setup drift.
@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(  # ruff: ignore[complex-structure,too-many-return-statements,too-many-branches,too-many-arguments,too-many-locals,too-many-statements,too-many-positional-arguments]
    candidate: str = "",
    execute: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    confirmation: str = "",
    qualification_artifact_json: str = "",
    eval_inventory_json: str = "",
    resume_checkpoint: str = "",
    preflight_assets: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    prewarm_assets: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    base230_oom_fallback: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    m230_comparison_run: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    pii350_utilization_run: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    pii350_capacity_run: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    pii350_milestone_run: str = "",
    pii350_scout_run: str = "",
    primary_oom_evidence_json: str = "",
    source_run_evidence_json: str = "",
    primary_action: str = "",
) -> None:
    if candidate not in {"base230", "encoder230", "encoder350", "pii350"}:
        msg = "choose exactly one explicit --candidate"
        raise RuntimeError(msg)
    if pii350_milestone_run:
        if candidate != "pii350":
            msg = "PII350 milestone run requires --candidate pii350"
            raise RuntimeError(msg)
        render_pii350_milestone_run_contract(pii350_milestone_run)
    if pii350_scout_run:
        if candidate != "pii350":
            msg = "PII350 scout run requires --candidate pii350"
            raise RuntimeError(msg)
        render_pii350_scout_contract(pii350_scout_run)
    if m230_comparison_run:
        expected_asset_profile = M230_COMPARISON_PROFILES[candidate]
    elif pii350_utilization_run or pii350_capacity_run:
        expected_asset_profile = PII350_UTILIZATION_PROFILE
    elif pii350_milestone_run:
        expected_asset_profile = pii350_milestone_run
    elif pii350_scout_run:
        _, expected_asset_profile = _render_pii350_scout_contract_and_profile(pii350_scout_run)
    else:
        expected_asset_profile = FULL_RUN_PROFILES[candidate]
    if prewarm_assets:
        print(
            json.dumps(
                _local_prewarm_call(candidate, expected_profile=expected_asset_profile),
                indent=2,
                sort_keys=True,
            ),
        )
        return
    if preflight_assets:
        if pii350_scout_run:
            print(
                json.dumps(
                    _local_pii350_scout_preflight_call(preflight_candidate_assets, pii350_scout_run),
                    indent=2,
                    sort_keys=True,
                ),
            )
            return
        print(
            json.dumps(
                _local_preflight_call(candidate, expected_profile=expected_asset_profile),
                indent=2,
                sort_keys=True,
            ),
        )
        return
    if base230_oom_fallback:
        if candidate != "base230":
            msg = "Base230 OOM fallback requires --candidate base230"
            raise RuntimeError(msg)
        if not execute or confirmation != FULL_RUN_CONFIRMATION:
            msg = "Base230 OOM fallback requires --execute and confirmation"
            raise RuntimeError(msg)
        if resume_checkpoint:
            msg = "Base230 OOM fallback must start fresh without resume"
            raise RuntimeError(msg)
        fallback_contract = render_base230_oom_fallback_contract()
        parsed_evidence = json.loads(primary_oom_evidence_json) if primary_oom_evidence_json else None
        evidence = parsed_evidence if isinstance(parsed_evidence, Mapping) else None
        inventory = json.loads(eval_inventory_json) if eval_inventory_json else None
        require_base230_oom_fallback_execute(
            fallback_contract,
            primary_oom_evidence=evidence,
            primary_action=primary_action,
        )
        print(
            json.dumps(
                _local_base230_oom_fallback_call(fallback_contract, evidence, primary_action, inventory),
                indent=2,
                sort_keys=True,
            ),
        )
        return
    if m230_comparison_run:
        if candidate not in M230_COMPARISON_CANDIDATES:
            msg = "230M comparison run requires --candidate base230 or encoder230"
            raise RuntimeError(msg)
        if not execute or confirmation != FULL_RUN_CONFIRMATION:
            msg = "230M comparison run requires --execute and confirmation"
            raise RuntimeError(msg)
        if resume_checkpoint:
            msg = "230M comparison run must start fresh without resume"
            raise RuntimeError(msg)
        comparison_contract = render_m230_comparison_run_contract(candidate)
        inventory = json.loads(eval_inventory_json) if eval_inventory_json else None
        require_m230_comparison_run_execute(
            comparison_contract,
            candidate_key=candidate,
            primary_action=primary_action,
        )
        comparison_endpoints = {
            "base230": run_base230_comparison,
            "encoder230": run_encoder230_comparison,
        }
        print(
            json.dumps(
                _local_m230_comparison_call(
                    candidate,
                    comparison_endpoints[candidate],
                    comparison_contract,
                    primary_action,
                    inventory,
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        return
    if pii350_utilization_run:
        if candidate != "pii350":
            msg = "PII350 utilization run requires --candidate pii350"
            raise RuntimeError(msg)
        if not execute or confirmation != FULL_RUN_CONFIRMATION:
            msg = "PII350 utilization run requires --execute and confirmation"
            raise RuntimeError(msg)
        if resume_checkpoint:
            msg = "PII350 utilization run must start fresh without resume"
            raise RuntimeError(msg)
        utilization_contract = render_pii350_utilization_run_contract()
        parsed_evidence = json.loads(source_run_evidence_json) if source_run_evidence_json else None
        evidence = parsed_evidence if isinstance(parsed_evidence, Mapping) else None
        inventory = json.loads(eval_inventory_json) if eval_inventory_json else None
        require_pii350_utilization_run_execute(
            utilization_contract,
            source_run_evidence=evidence,
            primary_action=primary_action,
        )
        print(
            json.dumps(
                _local_pii350_utilization_call(
                    run_pii350_utilization,
                    utilization_contract,
                    evidence,
                    primary_action,
                    inventory,
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        return
    if pii350_capacity_run:
        if candidate != "pii350":
            msg = "PII350 capacity run requires --candidate pii350"
            raise RuntimeError(msg)
        if not execute or confirmation != FULL_RUN_CONFIRMATION:
            msg = "PII350 capacity run requires --execute and confirmation"
            raise RuntimeError(msg)
        if resume_checkpoint:
            msg = "PII350 capacity run must start fresh without resume"
            raise RuntimeError(msg)
        capacity_contract = render_pii350_capacity_run_contract()
        parsed_evidence = json.loads(source_run_evidence_json) if source_run_evidence_json else None
        evidence = parsed_evidence if isinstance(parsed_evidence, Mapping) else None
        inventory = json.loads(eval_inventory_json) if eval_inventory_json else None
        require_pii350_capacity_run_execute(
            capacity_contract,
            source_run_evidence=evidence,
            primary_action=primary_action,
        )
        print(
            json.dumps(
                _local_pii350_capacity_call(
                    run_pii350_capacity,
                    capacity_contract,
                    evidence,
                    primary_action,
                    inventory,
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        return
    if pii350_milestone_run:
        if candidate != "pii350":
            msg = "PII350 milestone run requires --candidate pii350"
            raise RuntimeError(msg)
        if not execute or confirmation != FULL_RUN_CONFIRMATION:
            msg = "PII350 milestone run requires --execute and confirmation"
            raise RuntimeError(msg)
        if resume_checkpoint:
            msg = "PII350 milestone run must start fresh without resume"
            raise RuntimeError(msg)
        milestone_contract = render_pii350_milestone_run_contract(pii350_milestone_run)
        parsed_evidence = json.loads(source_run_evidence_json) if source_run_evidence_json else None
        evidence = parsed_evidence if isinstance(parsed_evidence, Mapping) else None
        inventory = json.loads(eval_inventory_json) if eval_inventory_json else None
        require_pii350_milestone_run_execute(
            milestone_contract,
            profile=pii350_milestone_run,
            source_run_evidence=evidence,
            primary_action=primary_action,
        )
        print(
            json.dumps(
                _local_pii350_milestone_call(
                    run_pii350_milestone,
                    pii350_milestone_run,
                    milestone_contract,
                    evidence,
                    primary_action,
                    inventory,
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        return
    if pii350_scout_run:
        if not execute or confirmation != FULL_RUN_CONFIRMATION:
            msg = "PII350 scout run requires --execute and confirmation"
            raise RuntimeError(msg)
        if resume_checkpoint:
            msg = "PII350 scout run must start fresh without resume"
            raise RuntimeError(msg)
        scout_contract = render_pii350_scout_contract(pii350_scout_run)
        parsed = json.loads(source_run_evidence_json) if source_run_evidence_json else None
        evidence = parsed if isinstance(parsed, Mapping) else None
        inventory = json.loads(eval_inventory_json) if eval_inventory_json else None
        require_pii350_scout_execute(
            scout_contract,
            arm=pii350_scout_run,
            source_run_evidence=evidence,
            primary_action=primary_action,
        )
        print(
            json.dumps(
                _local_pii350_scout_call(
                    run_pii350_scout,
                    pii350_scout_run,
                    scout_contract,
                    evidence,
                    primary_action,
                    inventory,
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        return
    contract = render_full_run(candidate)
    artifact = json.loads(qualification_artifact_json) if qualification_artifact_json else None
    inventory = json.loads(eval_inventory_json) if eval_inventory_json else None
    require_full_run_execute(
        contract,
        execute=execute,
        confirmation=confirmation,
        qualification_artifact=artifact,
    )
    endpoints = {
        "base230": run_base230,
        "encoder230": run_encoder230,
        "encoder350": run_encoder350,
        "pii350": run_pii350,
    }
    print(
        json.dumps(
            _local_remote_call(
                candidate,
                endpoints[candidate],
                contract,
                artifact,
                inventory,
                resume_checkpoint or None,
            ),
            indent=2,
            sort_keys=True,
        ),
    )
