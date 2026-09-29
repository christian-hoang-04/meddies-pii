"""Hash-verified PII350 continuation-checkpoint adapter for the full benchmark.

The continuation producer publishes a complete checkpoint through a manifest-last
protocol.  Every checkpoint byte is verified before any model deserializer runs;
evaluation then loads only the LoRA adapter and classifier head.
"""

from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: the two cumulative-cost checks compare against a pinned allowlist constant the
# reason: continuation producer writes verbatim, so an approximate match would admit a
# reason: checkpoint that is not the allowlisted one. Both sit at 81 characters inside a long
# reason: `or` chain, where the 43-character per-site directive would open line-too-long.
# ruff: file-ignore[import-outside-top-level]
# reason: the adapter loads its model stack inside load(), not when the module is imported.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard, cast, override

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    decode_bioes_from_offsets,
    viterbi_decode_logits,
)
from meddies_pii.eval_baseline.baseline.adapter import PiiAdapter
from meddies_pii.evaluation.identity import canonical_sha256, file_sha256, is_sha256
from meddies_pii.taxonomy import PII_LABEL_SET, PiiLabel

if TYPE_CHECKING:
    from meddies_pii.spans import CharSpan

GIT_SHA_HEX_LENGTH = 40

BASE_MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M-PII-Detector"
BASE_MODEL_REVISION = "b8c9cf3d2d6ae52501b35a27ba46f271449c9ce2"
MANIFEST_SCHEMA_VERSION = 1
METADATA_SCHEMA_VERSION = 2
LEGACY_STEP60_CHECKPOINT_DIGEST = "dacc5dfdfc8a2f0f38bd2631c5db16f9af5760ef0c968bb0269703d18f7fc939"
LEGACY_STEP60_TRAJECTORY_DIGEST = "542478f63b1726aa863c9a0ecec1e84f6347855fa9b6776b451b40c37781dac4"
STEP100_WAVE1_CHECKPOINT_DIGEST = "20b5bcd29d45353f344f0058c93dec0fa4a588dfe3285e3bd9c1506fa7be2dc0"
STEP100_WAVE1_EXECUTION_CONTRACT_DIGEST = "bad3a3000c01411411071b0d86ac115f680e9c159e148c18e915afdd5aa5fc81"
STEP100_WAVE1_PARENT_CHECKPOINT_DIGEST = LEGACY_STEP60_CHECKPOINT_DIGEST
STEP100_WAVE1_OPTIMIZER_STEP = 100
STEP100_WAVE1_PACKED_CURSOR = 12_800
STEP100_WAVE1_WORLD_SIZE = 2
STEP100_WAVE1_CUMULATIVE_ALL_IN_COST_USD = 4.85716643914298
TERMINAL_STEP146_CHECKPOINT_DIGEST = "73a490a6a90da19d077951a673fa451e3350dc81f3e09ba594ee86e182985c76"
TERMINAL_STEP146_OPTIMIZER_STEP = 146
TERMINAL_STEP146_PACKED_CURSOR = 18_688
TERMINAL_STEP146_WORLD_SIZE = 2
STEP150_WAVE2_CHECKPOINT_DIGEST = "904a279177f47b9c9976c2fc16af25c0a9c4521910898fdd1d24fc65008ee1b5"
STEP150_WAVE2_EXECUTION_CONTRACT_DIGEST = "ed3eb5c002584941d94682899431c5b4edd8e9117632c92fe1fa07f7198215fb"
STEP150_WAVE2_PARENT_CHECKPOINT_DIGEST = TERMINAL_STEP146_CHECKPOINT_DIGEST
STEP150_WAVE2_OPTIMIZER_STEP = 150
STEP150_WAVE2_PACKED_CURSOR = 19_200
STEP150_WAVE2_WORLD_SIZE = 4
STEP150_WAVE2_CUMULATIVE_ALL_IN_COST_USD = 1.680104012429088
LEGACY_STEP60_METADATA = {
    "candidate_key": "pii350",
    "config_digest": "2e5bdd06d0e2268deb5192d7a68a3d0cd370bd43c4a21fc2139b3d5a041cd2c8",
    "eval_revision": "28aaef5dffd36aabead650c74658a6f814eb4db0",
    "packed_revision": "11fd43ec9ebb187e1d0f94fe77bcf1090a2a18ee",
    "packed_manifest_sha256": "7cadda8e81ef4b2a1111f37a8b508492158a983e90f54fecafdc24764f792969",
    "lifecycle_state": "optimizer_step_cap_reached",
    "optimizer_step": 60,
    "packed_cursor": 7_680,
}
LORA_RANK = 128
LORA_ALPHA = 256
HEAD_HIDDEN_SIZE = 1024
HEAD_LABEL_COUNT = 37
GLOBAL_BATCH_SIZE = 128
WORLD_SIZE = 4
WAVE_PROFILES = frozenset({"meddiesresearch", "huyhoang0411ha", "meddies-ocr"})
MAX_SEQUENCE_LENGTH = 8192
MAX_BATCH_SIZE = 8
MAX_BATCH_TOKENS = 32_768
MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata.json"
CLASSIFIER_FILENAME = "classifier.pt"
ADAPTER_CONFIG_FILENAME = "adapter/adapter_config.json"
ADAPTER_WEIGHTS_FILENAME = "adapter/adapter_model.safetensors"
OPTIMIZER_FILENAME = "optimizer.pt"
RNG_FILENAME = "rng.pt"
EVALUATION_REQUIRED_FILES = frozenset({
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_WEIGHTS_FILENAME,
    CLASSIFIER_FILENAME,
    OPTIMIZER_FILENAME,
    RNG_FILENAME,
    METADATA_FILENAME,
})
CONTINUATION_METADATA_KEYS = frozenset({
    "schema_version",
    "trajectory_digest",
    "execution_contract_digest",
    "parent_checkpoint_digest",
    "optimizer_step",
    "packed_cursor",
    "lifecycle_state",
    "world_size",
    "rank_mapping",
    "rng_transition",
    "rank_rng_states",
    "epoch_complete",
    "wave_profile",
    "wave_cumulative_all_in_cost_usd",
})


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    repo_id: str
    revision: str
    path: str
    checkpoint_digest: str

    def __post_init__(self) -> None:
        if not self.repo_id or not self.path or not self.revision:
            msg = "checkpoint artifact requires repo, immutable revision, and path"
            raise ValueError(msg)
        if not is_sha256(self.checkpoint_digest):
            msg = "checkpoint artifact digest must be a lowercase SHA-256"
            raise ValueError(msg)
        if len(self.revision) != GIT_SHA_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            msg = "checkpoint artifact revision must be an immutable Git SHA"
            raise ValueError(msg)
        artifact_path = Path(self.path)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            msg = "checkpoint artifact path must remain below its root"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class VerifiedCheckpoint:
    root: Path
    artifact: CheckpointArtifact
    trajectory_digest: str
    optimizer_step: int
    packed_cursor: int
    lifecycle_state: str
    world_size: int
    label_vocabulary: tuple[str, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ContinuationMetadata:
    trajectory: str
    execution: str
    step: int
    cursor: int
    lifecycle: str
    wave_profile: str
    cumulative_cost: int | float
    rng_transition: Mapping[str, Any]
    rank_rng_states: list[Mapping[str, Any]]


def _is_schema_int(value: object) -> TypeGuard[int]:
    return type(value) is int


def _is_finite_cost(value: object) -> TypeGuard[int | float]:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(float(value))


def _is_rank_rng_states(value: object) -> TypeGuard[list[Mapping[str, Any]]]:
    return isinstance(value, list) and all(
        isinstance(record, Mapping) and isinstance(record.get("cpu_rng"), str) and isinstance(record.get("cuda_rng"), str)
        for record in value
    )


def _validated_continuation_metadata(metadata: Mapping[str, Any]) -> _ContinuationMetadata:
    if set(metadata) != CONTINUATION_METADATA_KEYS or metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        msg = "checkpoint metadata does not match the continuation schema"
        raise RuntimeError(msg)
    parent_digest = metadata.get("parent_checkpoint_digest")
    if not isinstance(parent_digest, str) or not parent_digest:
        msg = "checkpoint metadata does not match the continuation schema"
        raise RuntimeError(msg)
    trajectory = metadata.get("trajectory_digest")
    execution = metadata.get("execution_contract_digest")
    if not is_sha256(trajectory) or not is_sha256(execution):
        msg = "checkpoint metadata does not match the continuation schema"
        raise RuntimeError(msg)
    step = metadata.get("optimizer_step")
    cursor = metadata.get("packed_cursor")
    if not _is_schema_int(step) or not _is_schema_int(cursor):
        msg = "checkpoint metadata does not match the continuation schema"
        raise RuntimeError(msg)
    lifecycle = metadata.get("lifecycle_state")
    wave_profile = metadata.get("wave_profile")
    if not isinstance(lifecycle, str) or not isinstance(wave_profile, str) or wave_profile not in WAVE_PROFILES:
        msg = "checkpoint metadata does not match the continuation schema"
        raise RuntimeError(msg)
    cumulative_cost = metadata.get("wave_cumulative_all_in_cost_usd")
    if not _is_finite_cost(cumulative_cost) or cumulative_cost < 0:
        msg = "checkpoint metadata does not match the continuation schema"
        raise RuntimeError(msg)
    rng_transition = metadata.get("rng_transition")
    rank_rng_states = metadata.get("rank_rng_states")
    if not isinstance(rng_transition, Mapping) or not _is_rank_rng_states(rank_rng_states):
        msg = "checkpoint metadata does not match the continuation schema"
        raise RuntimeError(msg)
    if metadata.get("epoch_complete") is not False:
        msg = "checkpoint metadata does not match the continuation schema"
        raise RuntimeError(msg)
    return _ContinuationMetadata(
        trajectory,
        execution,
        step,
        cursor,
        lifecycle,
        wave_profile,
        cumulative_cost,
        rng_transition,
        rank_rng_states,
    )


def _read_json(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"checkpoint {role} is missing or invalid"
        raise RuntimeError(msg) from error
    if not isinstance(payload, Mapping):
        msg = f"checkpoint {role} must be a JSON object"
        raise RuntimeError(msg)
    return payload


def _checkpoint_digest(metadata: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """Match the continuation producer's manifest-last checkpoint identity.

    Returns:
        The canonical SHA-256 over the metadata and manifest pair, computed the same way the
        producer computes it so the two identities compare byte for byte.

    """
    return canonical_sha256({"metadata": dict(metadata), "manifest": dict(manifest)})


def _relative_file(root: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        msg = "checkpoint manifest contains an invalid file path"
        raise RuntimeError(msg)
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        msg = "checkpoint manifest file escapes artifact root"
        raise RuntimeError(msg) from error
    return value, candidate


def _require_complete_file_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    """Verify every supplied byte and reject partial or unmanifested checkpoint trees.

    Raises:
        RuntimeError: on any of these, each of which fails the tree closed — a `files` list that
            is absent or empty; an entry that is not a mapping, carries a non-integer or negative
            byte count, a malformed digest, or a duplicate path; a path that is absolute, contains
            `..`, or resolves outside the artifact root; a file whose size or SHA-256 does not
            match its entry; a directory holding ANY file the manifest does not list, or missing
            one it does; and a manifest lacking the evaluation-required files.

    """
    files = manifest.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or not files:
        msg = "checkpoint manifest files must be a non-empty list"
        raise RuntimeError(msg)
    observed: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            msg = "checkpoint manifest file entry is invalid"
            raise RuntimeError(msg)
        relative, path = _relative_file(root, entry.get("path"))
        byte_count = entry.get("bytes")
        digest = entry.get("sha256")
        if type(byte_count) is not int or byte_count < 0 or not is_sha256(digest) or relative in observed:
            msg = "checkpoint manifest file entry is invalid"
            raise RuntimeError(msg)
        if not path.is_file() or path.stat().st_size != byte_count or file_sha256(str(path)) != digest:
            msg = f"checkpoint artifact file failed hash verification: {relative}"
            raise RuntimeError(msg)
        observed.add(relative)
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != MANIFEST_FILENAME
    }
    if actual != observed:
        msg = "checkpoint artifact has missing or unmanifested files"
        raise RuntimeError(msg)
    if not observed >= EVALUATION_REQUIRED_FILES:
        missing = sorted(EVALUATION_REQUIRED_FILES - observed)
        msg = f"checkpoint receipt lacks evaluation-required files: {missing}"
        raise RuntimeError(msg)


# reason: Checkpoint pins, rank receipts, and updates form one loader verdict; splitting forks diagnostics.
def _require_metadata(  # ruff: ignore[complex-structure]
    metadata: Mapping[str, Any],
    artifact: CheckpointArtifact,
    manifest: Mapping[str, Any],
) -> tuple[str, int, int, str, int]:
    if set(metadata) == set(LEGACY_STEP60_METADATA):
        if artifact.checkpoint_digest != LEGACY_STEP60_CHECKPOINT_DIGEST:
            msg = "legacy step-60 checkpoint digest is not allowlisted"
            raise RuntimeError(msg)
        if dict(metadata) != LEGACY_STEP60_METADATA:
            msg = "legacy step-60 checkpoint provenance is invalid"
            raise RuntimeError(msg)
        if _checkpoint_digest(metadata, manifest) != artifact.checkpoint_digest:
            msg = "checkpoint manifest digest is invalid"
            raise RuntimeError(msg)
        return (
            LEGACY_STEP60_TRAJECTORY_DIGEST,
            60,
            # reason: the allowlisted constant is the authority for both; reading it keeps one source.
            cast("int", LEGACY_STEP60_METADATA["packed_cursor"]),
            cast("str", LEGACY_STEP60_METADATA["lifecycle_state"]),
            WORLD_SIZE,
        )

    validated = _validated_continuation_metadata(metadata)
    trajectory = validated.trajectory
    execution = validated.execution
    step = validated.step
    cursor = validated.cursor
    lifecycle = validated.lifecycle
    wave_profile = validated.wave_profile
    cumulative_cost = validated.cumulative_cost
    rng_transition = validated.rng_transition
    rank_rng_states = validated.rank_rng_states

    if artifact.checkpoint_digest == STEP100_WAVE1_CHECKPOINT_DIGEST:
        # reason: require metadata keeps update/single in one gate; helper predicates would scatter the rule.
        if (
            lifecycle != "update_committed"  # ruff: ignore[too-many-boolean-expressions]
            or trajectory != LEGACY_STEP60_TRAJECTORY_DIGEST
            or execution != STEP100_WAVE1_EXECUTION_CONTRACT_DIGEST
            or metadata.get("parent_checkpoint_digest") != STEP100_WAVE1_PARENT_CHECKPOINT_DIGEST
            or step != STEP100_WAVE1_OPTIMIZER_STEP
            or cursor != STEP100_WAVE1_PACKED_CURSOR
            or metadata.get("world_size") != STEP100_WAVE1_WORLD_SIZE
            or metadata.get("rank_mapping") != [0, 1]
            or rng_transition.get("kind") != "single_rank_to_two_rank_v1"
            or rng_transition.get("rank_mapping") != [0, 1]
            or [record.get("rank") for record in rank_rng_states] != [0, 1]
            or wave_profile != "meddiesresearch"
            or float(cumulative_cost) != STEP100_WAVE1_CUMULATIVE_ALL_IN_COST_USD
        ):
            msg = "step-100 Wave1 checkpoint does not match its allowlisted contract"
            raise RuntimeError(msg)
    elif artifact.checkpoint_digest == STEP150_WAVE2_CHECKPOINT_DIGEST:
        # reason: require metadata keeps update/two rank in one gate; helper predicates would scatter the rule.
        if (
            lifecycle != "update_committed"  # ruff: ignore[too-many-boolean-expressions]
            or trajectory != LEGACY_STEP60_TRAJECTORY_DIGEST
            or execution != STEP150_WAVE2_EXECUTION_CONTRACT_DIGEST
            or metadata.get("parent_checkpoint_digest") != STEP150_WAVE2_PARENT_CHECKPOINT_DIGEST
            or step != STEP150_WAVE2_OPTIMIZER_STEP
            or cursor != STEP150_WAVE2_PACKED_CURSOR
            or metadata.get("world_size") != STEP150_WAVE2_WORLD_SIZE
            or metadata.get("rank_mapping") != [0, 1, 2, 3]
            or rng_transition.get("kind") != "two_rank_to_four_rank_v1"
            or rng_transition.get("rank_mapping") != [0, 1, 2, 3]
            or [record.get("rank") for record in rank_rng_states] != [0, 1, 2, 3]
            or wave_profile != "huyhoang0411ha"
            or float(cumulative_cost) != STEP150_WAVE2_CUMULATIVE_ALL_IN_COST_USD
        ):
            msg = "step-150 Wave2 checkpoint does not match its allowlisted contract"
            raise RuntimeError(msg)
    elif lifecycle == "terminal":
        # reason: require metadata keeps checkpoint/world size in one gate; helper predicates would scatter the rule.
        if (
            artifact.checkpoint_digest != TERMINAL_STEP146_CHECKPOINT_DIGEST  # ruff: ignore[too-many-boolean-expressions]
            or step != TERMINAL_STEP146_OPTIMIZER_STEP
            or cursor != TERMINAL_STEP146_PACKED_CURSOR
            or metadata.get("world_size") != TERMINAL_STEP146_WORLD_SIZE
            or metadata.get("rank_mapping") != [0, 1]
            or [record.get("rank") for record in rank_rng_states] != [0, 1]
            or rng_transition.get("rank_mapping") != [0, 1]
        ):
            msg = "terminal checkpoint does not match the step-146 contract"
            raise RuntimeError(msg)
    # reason: require metadata keeps update/four rank in one gate; helper predicates would scatter the rule.
    elif (
        lifecycle != "update_committed"  # ruff: ignore[too-many-boolean-expressions]
        or step < 0
        or step % 50 != 0
        or cursor != step * GLOBAL_BATCH_SIZE
        or metadata.get("world_size") != WORLD_SIZE
        or metadata.get("rank_mapping") != [0, 1, 2, 3]
        or rng_transition.get("kind") != "four_rank_to_four_rank_v1"
        or rng_transition.get("rank_mapping") != [0, 1, 2, 3]
        or [record.get("rank") for record in rank_rng_states] != [0, 1, 2, 3]
    ):
        msg = "checkpoint metadata does not match the continuation schema"
        raise RuntimeError(msg)
    if _checkpoint_digest(metadata, manifest) != artifact.checkpoint_digest:
        msg = "checkpoint manifest digest is invalid"
        raise RuntimeError(msg)
    return trajectory, step, cursor, str(lifecycle), int(metadata["world_size"])


def verify_checkpoint_artifact(root: str | Path, artifact: CheckpointArtifact) -> VerifiedCheckpoint:
    """Verify the complete producer checkpoint before any model deserialization.

    Returns:
        The `VerifiedCheckpoint` describing the tree, its trajectory, step, cursor, lifecycle,
        world size, label space and manifest.

    Raises:
        RuntimeError: if the manifest key set is not exactly the expected schema, if it is
            uncommitted or reports the wrong schema version, or if its embedded digest is not the
            requested checkpoint. Refusals from the file-manifest and metadata checks it delegates
            to propagate unchanged.

    """
    checkpoint_root = Path(root)
    persisted_manifest = _read_json(checkpoint_root / MANIFEST_FILENAME, role="manifest")
    expected_manifest_keys = {
        "schema_version",
        "committed",
        "files",
        "checkpoint_digest",
    }
    if set(persisted_manifest) != expected_manifest_keys:
        msg = "checkpoint manifest schema keys are invalid"
        raise RuntimeError(msg)
    embedded_digest = persisted_manifest.get("checkpoint_digest")
    manifest = {key: value for key, value in persisted_manifest.items() if key != "checkpoint_digest"}
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get("committed") is not True:
        msg = "checkpoint manifest is absent or uncommitted"
        raise RuntimeError(msg)
    if embedded_digest != artifact.checkpoint_digest:
        msg = "checkpoint manifest digest does not match requested checkpoint"
        raise RuntimeError(msg)
    _require_complete_file_manifest(checkpoint_root, manifest)
    metadata = _read_json(checkpoint_root / METADATA_FILENAME, role="metadata")
    trajectory, step, cursor, lifecycle, world_size = _require_metadata(metadata, artifact, manifest)
    return VerifiedCheckpoint(
        checkpoint_root,
        artifact,
        trajectory,
        step,
        cursor,
        lifecycle,
        world_size,
        tuple(build_bioes_label_space(ENTITY_LABELS)),
        manifest,
    )


def _validate_adapter_config(verified: VerifiedCheckpoint) -> None:
    config = _read_json(verified.root / ADAPTER_CONFIG_FILENAME, role="adapter config")
    if config.get("peft_type") != "LORA" or config.get("r") != LORA_RANK or config.get("lora_alpha") != LORA_ALPHA:
        msg = "verified checkpoint adapter config disagrees with PII350 LoRA contract"
        raise RuntimeError(msg)


def build_checkpoint_tagger(verified: VerifiedCheckpoint) -> tuple[Any, Any]:
    """Load the evaluation subset only after the complete tree passed verification.

    Returns:
        The assembled tagger and its tokenizer.

    Raises:
        RuntimeError: if the base wrapper exposes no `lfm2` encoder body, or if the persisted
            classifier state is not a state-dict mapping.

    """
    # reason: transformers declares its public names only under `TYPE_CHECKING` and serves them at
    # reason: runtime through `_LazyModule`, so a static reader cannot prove the symbol is present.
    # reason: Verified against the pinned 5.14.1: `hasattr(transformers, "AutoTokenizer")` is True.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForTokenClassification, AutoTokenizer  # ty: ignore[possibly-missing-import]

    from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger

    _validate_adapter_config(verified)
    wrapper = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    body = getattr(wrapper, "lfm2", None)
    if body is None:
        msg = "PII350 base wrapper does not expose its lfm2 encoder body"
        raise RuntimeError(msg)
    backbone = PeftModel.from_pretrained(body, verified.root / "adapter", is_trainable=False)
    tagger = HiddenStateTokenTagger(
        backbone,
        HEAD_HIDDEN_SIZE,
        HEAD_LABEL_COUNT,
        request_hidden_states=False,
        classifier_dtype=torch.bfloat16,
        dropout=0.1,
        packed_segment_isolation=False,
    )
    classifier = torch.load(verified.root / CLASSIFIER_FILENAME, map_location="cpu", weights_only=True)
    if not isinstance(classifier, Mapping):
        msg = "checkpoint classifier state is not a state-dict mapping"
        raise RuntimeError(msg)
    tagger.classifier.load_state_dict(classifier, strict=True)
    tagger = tagger.to(device="cuda", dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    return tagger, tokenizer


class Pii350CheckpointAdapter(PiiAdapter):
    """Pinned PII350 BIOES inference with length-aware A10G batches."""

    name = "pii350-checkpoint"
    supported_labels: frozenset[PiiLabel] = PII_LABEL_SET

    def __init__(
        self,
        verified: VerifiedCheckpoint,
        *,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        max_batch_size: int = MAX_BATCH_SIZE,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if max_sequence_length != MAX_SEQUENCE_LENGTH or max_batch_size <= 0 or max_batch_tokens < max_sequence_length:
            msg = "checkpoint adapter batch contract is invalid"
            raise ValueError(msg)
        self.verified = verified
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.on_progress = on_progress
        self._tagger: Any | None = None
        self._tokenizer: Any | None = None
        self.truncated_documents = 0

    @override
    def load(self) -> None:
        if self._tagger is None:
            self._tagger, self._tokenizer = build_checkpoint_tagger(self.verified)

    def _batches(self, texts: Sequence[str]) -> list[list[int]]:
        tokenizer = self._tokenizer
        if tokenizer is None:
            msg = "Pii350CheckpointAdapter.load() must be called before predict()"
            raise RuntimeError(msg)
        lengths = [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts]
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for index, length in enumerate(lengths):
            bounded = min(length, self.max_sequence_length)
            proposed_max = max(current_max, bounded)
            if current and (
                len(current) >= self.max_batch_size or proposed_max * (len(current) + 1) > self.max_batch_tokens
            ):
                batches.append(current)
                current, current_max = [], 0
            current.append(index)
            current_max = max(current_max, bounded)
        if current:
            batches.append(current)
        return batches

    # reason: Pii350CheckpointAdapter coordinates load with batches; extra seams would misattribute row errors.
    @override
    def predict(self, texts: list[str]) -> list[list[CharSpan]]:  # ruff: ignore[too-many-locals]
        self.load()
        import torch

        tagger, tokenizer = self._tagger, self._tokenizer
        if tagger is None or tokenizer is None:
            msg = "checkpoint adapter did not load"
            raise RuntimeError(msg)
        results: list[list[CharSpan] | None] = [None] * len(texts)
        id_to_label = dict(enumerate(self.verified.label_vocabulary))
        completed = 0
        with torch.inference_mode():
            for indices in self._batches(texts):
                batch_texts = [texts[index] for index in indices]
                encoded = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_sequence_length,
                    padding=True,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                offsets = encoded.pop("offset_mapping")
                attention_mask = encoded["attention_mask"]
                device = next(tagger.parameters()).device
                inputs = {key: value.to(device) for key, value in encoded.items()}
                logits = tagger(**inputs)["logits"]
                for row_index, text, offset_row, mask_row, logits_row in zip(
                    indices,
                    batch_texts,
                    offsets,
                    attention_mask,
                    logits,
                    strict=True,
                ):
                    active = int(mask_row.sum().item())
                    offsets_row = [(int(pair[0]), int(pair[1])) for pair in offset_row[:active].tolist()]
                    if offsets_row and max(end for _start, end in offsets_row) < len(text):
                        self.truncated_documents += 1
                    tags = viterbi_decode_logits(logits_row[:active], id_to_label, offsets_row)
                    results[row_index] = list(decode_bioes_from_offsets(text, offsets_row, tags, id_to_label))
                    completed += 1
                    if self.on_progress is not None:
                        self.on_progress(completed, len(texts))
        if any(item is None for item in results):
            msg = "checkpoint adapter lost a prediction during batched decode"
            raise RuntimeError(msg)
        return [item for item in results if item is not None]
