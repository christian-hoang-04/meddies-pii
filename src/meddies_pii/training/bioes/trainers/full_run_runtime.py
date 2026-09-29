"""Actual, single-candidate full-run runtime; imported only inside Modal containers.

This module deliberately has no Modal decoration and no local rendering path.  It
receives an already-validated contract plus verified local Parquet paths, then
performs one resumable packed epoch and one final, length-aware evaluation.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack is an optional extra with no macOS wheels, so it loads inside the call that needs it.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
# ruff: file-ignore[any-type]
# reason: every Any here is a torch, unsloth or peft object; the training stack is an optional extra with no macOS
# reason: wheels and unsloth publishes none, so a real annotation would need the module-scope import this module
# reason: exists to avoid.
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from itertools import islice
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, TypedDict
from uuid import uuid4

from .full_run import (
    FULL_RUN_LABEL_VOCABULARY,
    FULL_RUN_TRAINING_DEADLINE_SECONDS,
)
from .full_run_engine import (
    RuntimeResumeState,
    RunWriter,
    StopReason,
    checkpoint_metadata,
    cost_fields,
    next_step_reason,
    validate_resume_state,
)
from .pins import EVAL_ROWS, PACKED_UNIT_COUNT

if TYPE_CHECKING:
    from meddies_pii.eval_baseline.baseline.models import EvalRow


class _FinalEvalEntry(TypedDict):
    """One pinned eval row paired with the tokenization the final evaluation reads it through."""

    row: EvalRow
    tokens: dict[str, Any]


WSD_SCOUT_OPTIMIZER_STEPS = 60

RATE_USD_PER_SECOND = 0.001097
EVAL_BATCH_SIZE = 16
EVAL_PROGRESS_INTERVAL_ROWS = 160


@dataclass(frozen=True, slots=True)
class AdamWParameters:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    fused: bool


@dataclass(frozen=True, slots=True)
class WSDParameters:
    peak_lr: float
    total_steps: int
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    warmup_type: str
    decay_type: str
    min_lr_ratio: float


def _canonical_pii_body_tensor_name(name: str) -> str:
    """Translate equivalent wrapper namespaces to the encoder-body namespace.

    Returns:
        The name with every ``base_model.model.``, ``model.`` and ``lfm2.`` prefix removed,
        stripped repeatedly rather than once, so a doubly-wrapped tensor lands in the same
        namespace as the same tensor loaded bare. Two spellings of one tensor must produce one
        canonical name, or the attestation digests would differ across equivalent wrappers.

    Raises:
        RuntimeError: If stripping consumes the whole name. An empty canonical name would make
            distinct tensors collide under one key, so it refuses rather than fingerprinting it.

    """
    canonical = name
    stripped = True
    while stripped:
        stripped = False
        for prefix in ("base_model.model.", "model.", "lfm2."):
            if canonical.startswith(prefix):
                canonical = canonical.removeprefix(prefix)
                stripped = True
                break
    if not canonical:
        msg = "PII encoder body has an empty canonical tensor name"
        raise RuntimeError(msg)
    return canonical


def pii_tensor_state_attestation(state: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint every PII encoder tensor after namespace normalization.

    Returns:
        Three fields that together pin the body: ``body_tensor_count``, a digest over the
        sorted canonical names, and a digest over name, dtype, shape and raw bytes of every
        tensor in sorted-name order. The values digest reads each tensor as ``uint8`` on CPU,
        so it compares the exact stored bits rather than a float rendering.

    Raises:
        RuntimeError: If the state mapping is empty; if any entry has a non-string name or a
            non-tensor value; or if two source names normalize to the same canonical name,
            which would silently drop one tensor from the fingerprint. The empty-canonical-name
            refusal of ``_canonical_pii_body_tensor_name`` reaches a caller through here too.

    """
    import torch

    if not state:
        msg = "PII encoder body state_dict is empty or invalid"
        raise RuntimeError(msg)
    digest = sha256()
    tensors: dict[str, Any] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            msg = "PII encoder body state_dict contains an invalid tensor"
            raise RuntimeError(msg)
        canonical_name = _canonical_pii_body_tensor_name(name)
        if canonical_name in tensors:
            msg = "PII encoder body has duplicate canonical tensor names"
            raise RuntimeError(msg)
        tensors[canonical_name] = tensor
    names: list[str] = []
    for name, tensor in sorted(tensors.items()):
        names.append(name)
        value = tensor.detach().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    encoded_names = json.dumps(names, separators=(",", ":")).encode()
    return {
        "body_tensor_count": len(names),
        "body_tensor_names_sha256": sha256(encoded_names).hexdigest(),
        "body_tensor_values_sha256": digest.hexdigest(),
    }


def pii_body_state_attestation(body: Any) -> dict[str, Any]:
    """Fingerprint a loaded PII body at its current module boundary.

    Returns:
        The same three attestation fields ``pii_tensor_state_attestation`` produces, computed
        over whatever ``state_dict()`` reports at the boundary this is called on. Calling it on
        the wrapper and on the unwrapped body gives the same digests, because the names are
        canonicalized first.

    Raises:
        RuntimeError: If the object exposes no ``state_dict`` attribute, or if that call returns
            something other than a mapping. Every refusal of ``pii_tensor_state_attestation`` --
            empty state, an invalid tensor entry, duplicate canonical names -- reaches a caller
            through here as well.

    """
    if not hasattr(body, "state_dict"):
        msg = "PII encoder body does not expose a state_dict"
        raise RuntimeError(msg)
    state = body.state_dict()
    if not isinstance(state, Mapping):
        msg = "PII encoder body state_dict is invalid"
        raise RuntimeError(msg)
    return pii_tensor_state_attestation(state)


def lfm2_checkpoint_body_state_attestation(snapshot: str) -> dict[str, Any]:
    """Attest source checkpoint body tensors before Unsloth may transform them.

    Returns:
        The three attestation fields over the ``lfm2.``-prefixed tensors of the snapshot's
        ``model.safetensors``, read on CPU. This runs before ``FastModel`` sees the checkpoint,
        so the digests describe the bytes on disk rather than anything a loader patched.

    Raises:
        RuntimeError: If the snapshot directory holds no ``model.safetensors``, or if the file
            contains no ``lfm2.``-prefixed tensor at all. An empty body would fingerprint to a
            stable digest over nothing, which would then match any other empty body.

    """
    from safetensors.torch import load_file

    weights_path = Path(snapshot) / "model.safetensors"
    if not weights_path.is_file():
        msg = "PII checkpoint snapshot lacks model.safetensors"
        raise RuntimeError(msg)
    state = load_file(str(weights_path), device="cpu")
    body = {name.removeprefix("lfm2."): tensor for name, tensor in state.items() if name.startswith("lfm2.")}
    if not body:
        msg = "PII checkpoint has no lfm2 body tensors"
        raise RuntimeError(msg)
    return pii_tensor_state_attestation(body)


def require_pii_body_state_attestation(expected: Mapping[str, Any] | None, body: Any) -> dict[str, Any]:
    """Reject a PII body unless it exactly matches the CPU-verified body.

    Returns:
        The freshly recomputed attestation of the live body, returned only once all three
        fields equal the CPU receipt. Callers persist the returned value rather than the
        expected one, so what lands in the run record is what was actually measured on device.

    Raises:
        RuntimeError: If ``expected`` is not a mapping, which is how an absent CPU receipt
            arrives, or if the count, the names digest or the values digest differs. The
            comparison reads ``expected`` with ``.get``, so a receipt missing one of the three
            keys compares ``None`` against a real digest and is refused rather than skipped --
            an incomplete receipt cannot pass by omitting the field it would fail.

    """
    if not isinstance(expected, Mapping):
        msg = "PII runtime requires a CPU-verified body attestation"
        raise RuntimeError(msg)
    actual = pii_body_state_attestation(body)
    keys = (
        "body_tensor_count",
        "body_tensor_names_sha256",
        "body_tensor_values_sha256",
    )
    if any(expected.get(key) != actual[key] for key in keys):
        msg = "PII runtime body tensors do not match the CPU receipt"
        raise RuntimeError(msg)
    return actual


def require_lfm2_checkpoint_body_state_attestation(expected: Mapping[str, Any] | None, snapshot: str) -> dict[str, Any]:
    """Reject a changed or incomplete cached checkpoint before FastModel loads it.

    Returns:
        The attestation recomputed from the snapshot on disk, returned only once all three
        fields equal the CPU receipt. This runs before ``FastModel.from_pretrained``, so a
        cache that was swapped or truncated between verification and load is caught here rather
        than trained on.

    Raises:
        RuntimeError: If ``expected`` is not a mapping, which is how an absent CPU receipt
            arrives, or if the count, the names digest or the values digest differs. As with
            the body gate, the comparison uses ``.get`` on ``expected``, so a receipt that
            omits one of the three keys is refused rather than passing on the two it carries.

    """
    if not isinstance(expected, Mapping):
        msg = "PII runtime requires a CPU-verified checkpoint attestation"
        raise RuntimeError(msg)
    actual = lfm2_checkpoint_body_state_attestation(snapshot)
    keys = (
        "body_tensor_count",
        "body_tensor_names_sha256",
        "body_tensor_values_sha256",
    )
    if any(expected.get(key) != actual[key] for key in keys):
        msg = "PII checkpoint body tensors do not match the CPU receipt"
        raise RuntimeError(msg)
    return actual


def _runtime_label_vocabulary() -> list[str]:
    from meddies_pii.annotations.bioes.vocabulary import (
        ENTITY_LABELS,
        build_bioes_label_space,
    )

    return list(build_bioes_label_space(ENTITY_LABELS))


def _validate_runtime_label_vocabulary(contract: Mapping[str, Any]) -> list[str]:
    expected = contract.get("label_vocabulary")
    actual = _runtime_label_vocabulary()
    if expected != actual or actual != list(FULL_RUN_LABEL_VOCABULARY):
        msg = "runtime label vocabulary mismatch with launch contract"
        raise RuntimeError(msg)
    return actual


def _persist_label_vocabulary(writer: RunWriter, labels: Sequence[str]) -> None:
    writer.atomic_json("label_vocabulary.json", {"labels": list(labels)})


def _iter_packed_rows(paths: Sequence[str]) -> Iterator[Mapping[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            yield from batch.to_pylist()


def _classifier_digest(classifier: Any) -> str:
    """Hash the seeded fp16/bf16 head bytes without serializing a checkpoint.

    Returns:
        A hex sha256 over the raw bytes of the classifier weight and then the bias, each read
        as ``uint8`` on CPU. Reading the bits rather than a float rendering makes the digest
        exact at bf16, where two distinct values can print the same.

    """
    import torch

    digest = sha256()
    for parameter in (classifier.weight, classifier.bias):
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _packed_order_attestation(rows: Sequence[Mapping[str, Any]], *, packed_cursor: int) -> dict[str, int | str]:
    """Hash packed-unit source identities, never the large CUDA token tensors.

    Returns:
        The cursor span this batch covers, the number of source identities inside it, and a
        digest over those identities in order. A row contributes every element of its
        ``row_uids`` when it packs several sources, and its own ``uid`` otherwise, so the
        identity count exceeds the row count on packed batches.

    Raises:
        ValueError: If the cursor is negative or the batch is empty. Either would attest an
            ordering that no step actually consumed.

    """
    if packed_cursor < 0 or not rows:
        msg = "packed order attestation requires a non-empty valid batch"
        raise ValueError(msg)
    identities: list[str] = []
    for row in rows:
        row_uids = row.get("row_uids")
        if isinstance(row_uids, Sequence) and not isinstance(row_uids, str):
            identities.extend(str(uid) for uid in row_uids)
        else:
            identities.append(str(row.get("uid", "")))
    encoded = json.dumps(identities, ensure_ascii=False, separators=(",", ":"))
    return {
        "first_packed_unit_cursor": packed_cursor,
        "last_packed_unit_cursor": packed_cursor + len(rows) - 1,
        "source_identity_count": len(identities),
        "ordered_source_identity_sha256": sha256(encoded.encode()).hexdigest(),
    }


def _physical_batches(rows: Iterator[Mapping[str, Any]], batch_size: int) -> Iterator[list[Mapping[str, Any]]]:
    """Yield fixed full batches and the one final non-empty physical batch.

    Yields:
        Successive lists of ``batch_size`` rows drawn from the iterator, and then one shorter
        final list when the source does not divide evenly. It never yields an empty list, so a
        drained iterator ends the loop rather than producing a zero-row step.

    Raises:
        ValueError: If the batch size is not positive. A zero size would make ``islice`` return
            an empty list forever, so the loop would never terminate.

    """
    if batch_size <= 0:
        msg = "physical batch size must be positive"
        raise ValueError(msg)
    while batch := list(islice(rows, batch_size)):
        yield batch


def _packed_batch(rows: Sequence[Mapping[str, Any]], device: str) -> tuple[dict[str, Any], int]:
    import torch

    ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"] for row in rows], dtype=torch.long, device=device)
    positions = torch.tensor([row["position_ids"] for row in rows], dtype=torch.long, device=device)
    lengths = torch.tensor(
        [length for row in rows for length in row["seq_lengths"] if length > 0],
        dtype=torch.int32,
        device=device,
    )
    if int(lengths.sum().item()) != int(ids.numel()):
        msg = "packed segment lengths do not cover the physical batch"
        raise RuntimeError(msg)
    return {
        "input_ids": ids,
        "labels": labels,
        "position_ids": positions,
        "packed_seq_lengths": lengths,
    }, sum(int(row["real_token_count"]) for row in rows)


class _NvmlSampler:
    """Capture NVML while CUDA work is in flight, rather than after it idles."""

    # reason: read only through dict(self._empty_sample) in __init__, never mutated in place.
    _empty_sample: dict[str, float | int | None] = {  # ruff: ignore[mutable-class-default]
        "gpu_sample_monotonic": None,
        "gpu_sm_utilization": None,
        "gpu_memory_controller_utilization": None,
        "device_vram_bytes": None,
        "device_vram_used_bytes": None,
    }

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._latest = dict(self._empty_sample)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            # reason: pynvml ships with the NVIDIA driver container and is absent here; every use sits inside try/except.
            pynvml = import_module("pynvml")

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # reason: NVML is telemetry; any pynvml failure must degrade the sample, never abort a paid H100 run.
        except Exception:  # ruff: ignore[blind-except]
            return

        def sample_once() -> None:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                sample: dict[str, float | int | None] = {
                    "gpu_sample_monotonic": time.monotonic(),
                    "gpu_sm_utilization": float(utilization.gpu),
                    "gpu_memory_controller_utilization": float(utilization.memory),
                    "device_vram_bytes": int(memory.total),
                    "device_vram_used_bytes": int(memory.used),
                }
            # reason: NVML is telemetry; any pynvml failure must degrade the sample, never abort a paid H100 run.
            except Exception:  # ruff: ignore[blind-except]
                return
            with self._lock:
                self._latest = sample

        def loop() -> None:
            while not self._stop.wait(self.interval_seconds):
                sample_once()

        sample_once()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def latest(self) -> dict[str, float | int | None]:
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 0.1)


def _optimizer_progress(
    *,
    cursor: int,
    batch_size: int,
    durations: Sequence[float],
    elapsed_seconds: float,
    training_deadline_seconds: float = FULL_RUN_TRAINING_DEADLINE_SECONDS,
) -> dict[str, float | int]:
    if not durations:
        msg = "optimizer progress requires a completed step"
        raise ValueError(msg)
    rolling_median = float(median(durations[-5:]))
    remaining = PACKED_UNIT_COUNT - cursor
    remaining_steps = (remaining + batch_size - 1) // batch_size
    deadline_remaining = max(0.0, training_deadline_seconds - elapsed_seconds)
    return {
        "packed_epoch_progress_fraction": cursor / PACKED_UNIT_COUNT,
        "packed_units_remaining": remaining,
        "training_deadline_seconds_remaining": deadline_remaining,
        "rolling_median_step_seconds": rolling_median,
        "projected_epoch_eta_seconds": remaining_steps * rolling_median,
        "projected_steps_before_deadline": int(deadline_remaining // rolling_median),
    }


def _contract_runtime_packages(contract: Mapping[str, object]) -> tuple[str, ...]:
    """Read the immutable launch image pins instead of candidate defaults.

    Returns:
        The contract's ``runtime.packages`` list as a tuple, each entry still in its
        ``name==version`` form. These are the pins the launch image was built from, so the
        attestation compares against them rather than against whatever a candidate default says.

    Raises:
        RuntimeError: If the ``runtime`` block is missing or is not a mapping; if ``packages``
            is not a list or is empty; or if any entry is not a string containing ``==``. An
            unpinned entry would let the attestation pass for an image built from a different
            version, so a loose spelling is refused rather than normalized.

    """
    runtime = contract.get("runtime")
    packages = runtime.get("packages") if isinstance(runtime, Mapping) else None
    if (
        not isinstance(packages, list)
        or not packages
        or any(not isinstance(package, str) or "==" not in package for package in packages)
    ):
        msg = "launch contract has invalid runtime package pins"
        raise RuntimeError(msg)
    return tuple(str(package) for package in packages)


def _require_exact_runtime_package_versions(
    runtime_packages: Sequence[str],
    observed: Mapping[str, str],
) -> dict[str, str]:
    """Fail rather than write a backend attestation for the wrong Modal image.

    Returns:
        The pinned name-to-version mapping parsed out of the contract, returned only when the
        observed versions equal it exactly.

    Raises:
        RuntimeError: If any pinned package reports a different installed version, or reports
            none at all. The comparison builds its actual side with ``observed.get``, so a
            package absent from the running image compares ``None`` against a version and is
            refused -- a missing dependency fails as loudly as a wrong one.

    """
    expected = {package.split("==", maxsplit=1)[0]: package.split("==", maxsplit=1)[1] for package in runtime_packages}
    actual = {name: observed.get(name) for name in expected}
    if actual != expected:
        msg = f"runtime package attestation mismatch: expected={expected}, observed={actual}"
        raise RuntimeError(msg)
    return expected


# reason: runtime keeps contract/encoder at its adapter seam; bundling would hide required inputs.
def _runtime_attestation(  # ruff: ignore[too-many-arguments]
    *,
    contract: Mapping[str, Any],
    fast_model: Any,
    model: Any,
    tagger: Any,
    candidate_key: str,
    encoder_checkpoint_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist only observable Unsloth-loader and adapter facts.

    Returns:
        The attestation written into the run record: the loader class and module, the returned
        model class, the installed versions of every contract-pinned package, the Torch CUDA
        version and device name, the observed gradient-checkpointing and cache flags, the
        resolved LoRA target modules and adapter dtypes, the classifier's dtype, output width,
        trainability and initialization digest, and the candidate key. For a non-``base230``
        candidate it also carries the encoder checkpoint attestation. Every field is read back
        off the live objects rather than copied from the contract, so it records what loaded.

    Raises:
        RuntimeError: If the candidate is not ``base230`` and no encoder checkpoint attestation
            was supplied, because the record would then claim a verified encoder it cannot
            name. Three delegated gates refuse before this one is reached and their failures
            reach a caller here: the adapter check refuses a model with no trainable LoRA
            tensors, trainable non-LoRA tensors, missing target modules or non-fp32 adapters;
            the training-state check refuses a model that did not enable gradient checkpointing
            or disable ``use_cache``; and the package check refuses a mismatched image.

    """
    from importlib.metadata import version

    import torch

    from .config import LORA_TARGET_MODULES

    lora_proof = _validate_trainable_lora_parameters(model, LORA_TARGET_MODULES)
    observed_training_state = _validate_observed_training_state(model)
    runtime_packages = _contract_runtime_packages(contract)
    installed_package_versions = {
        package.split("==", maxsplit=1)[0]: version(package.split("==", maxsplit=1)[0]) for package in runtime_packages
    }
    _require_exact_runtime_package_versions(runtime_packages, installed_package_versions)
    result = {
        "runtime_backend": "unsloth",
        "unsloth_version": installed_package_versions["unsloth"],
        "installed_package_versions": installed_package_versions,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "loader": fast_model.__name__,
        "loader_class": f"{fast_model.__module__}.{fast_model.__name__}",
        "returned_model_class": type(model).__name__,
        "returned_model_module": type(model).__module__,
        "gradient_checkpointing_mode": "unsloth",
        **observed_training_state,
        "observed_attention_implementation": _observed_attention_implementation(model),
        "packed_segment_isolation": bool(tagger.packed_segment_isolation),
        "request_hidden_states": bool(tagger.request_hidden_states),
        "classifier_dtype": str(tagger.classifier.weight.dtype),
        "classifier_trainable": all(parameter.requires_grad for parameter in tagger.classifier.parameters()),
        "classifier_output_labels": int(tagger.classifier.out_features),
        "classifier_initialization_sha256": _classifier_digest(tagger.classifier),
        **lora_proof,
        "candidate": candidate_key,
    }
    if candidate_key != "base230":
        if encoder_checkpoint_attestation is None:
            msg = "encoder runtime checkpoint attestation is absent"
            raise RuntimeError(msg)
        result["encoder_checkpoint_attestation"] = dict(encoder_checkpoint_attestation)
    return result


def _observed_attention_implementation(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "_attn_implementation", getattr(config, "attn_implementation", None))


def _validate_observed_training_state(model: Any) -> dict[str, bool]:
    observed_gradient_checkpointing = getattr(model, "is_gradient_checkpointing", None)
    if callable(observed_gradient_checkpointing):
        observed_gradient_checkpointing = observed_gradient_checkpointing()
    if observed_gradient_checkpointing is not True:
        msg = "Unsloth model did not enable gradient checkpointing"
        raise RuntimeError(msg)
    observed_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if observed_use_cache is not False:
        msg = "Unsloth model did not disable use_cache during training"
        raise RuntimeError(msg)
    return {
        "observed_gradient_checkpointing": True,
        "observed_use_cache": False,
    }


def _validate_trainable_lora_parameters(model: Any, target_modules: Sequence[str]) -> dict[str, list[str]]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    adapters = [(name, parameter) for name, parameter in trainable if ".lora_" in name]
    if not adapters:
        msg = "Unsloth model has no trainable LoRA adapter tensors"
        raise RuntimeError(msg)
    non_lora = [name for name, _ in trainable if ".lora_" not in name]
    if non_lora:
        msg = f"Unsloth model has trainable non-LoRA backbone tensors: {non_lora}"
        raise RuntimeError(msg)
    resolved_targets = sorted({target for name, _ in adapters for target in target_modules if f".{target}." in name})
    missing = sorted(set(target_modules) - set(resolved_targets))
    if missing:
        msg = f"Unsloth LoRA target modules lack adapters: {missing}"
        raise RuntimeError(msg)
    dtypes = sorted({str(parameter.dtype) for _, parameter in adapters})
    if dtypes != ["torch.float32"]:
        msg = f"Unsloth LoRA adapter tensors must be fp32, got {dtypes}"
        raise RuntimeError(msg)
    return {
        "lora_trainable_parameter_dtypes": dtypes,
        "resolved_lora_modules": resolved_targets,
    }


def _contract_adamw_parameters(contract: Mapping[str, Any]) -> AdamWParameters:
    """Bind AdamW to the immutable contract and reject drift before training.

    Returns:
        The learning rate, betas, epsilon, weight decay, ``amsgrad`` and ``fused`` flags, taken
        from the contract's public ``optimizer`` block only after every one has been proved
        equal to the Unsloth ``mechanics.adamw`` copy. The two representations must agree, so
        the optimizer the run builds is the one the contract was approved with.

    Raises:
        RuntimeError: If the ``optimizer`` or ``training`` block is absent or not a mapping; if
            ``training.mechanics`` or its ``adamw`` sub-block is absent; if the optimizer name
            is not ``adamw``; if the schedule is neither ``constant`` nor ``wsd``; if a constant
            schedule carries a non-null scheduler; if any parameter is the wrong type or out of
            range -- non-positive learning rate or epsilon, negative weight decay, a betas list
            that is not exactly two numbers, or a non-``bool`` for either flag; or if any of the
            six values disagrees with the Unsloth mechanics copy of it.

    """
    optimizer = contract.get("optimizer")
    training = contract.get("training")
    if not isinstance(optimizer, Mapping) or not isinstance(training, Mapping):
        msg = "launch contract has no optimizer or training configuration"
        raise RuntimeError(msg)
    mechanics = training.get("mechanics")
    if not isinstance(mechanics, Mapping):
        msg = "launch contract has no Unsloth mechanics"
        raise RuntimeError(msg)
    mechanics_adamw = mechanics.get("adamw")
    if not isinstance(mechanics_adamw, Mapping):
        msg = "launch contract has no AdamW mechanics"
        raise RuntimeError(msg)
    if optimizer.get("name") != "adamw":
        msg = "launch contract optimizer must be AdamW"
        raise RuntimeError(msg)
    schedule = optimizer.get("schedule")
    if schedule not in {"constant", "wsd"}:
        msg = "launch contract optimizer schedule must be constant or wsd"
        raise RuntimeError(msg)
    if schedule == "constant" and mechanics.get("scheduler") is not None:
        msg = "Unsloth mechanics scheduler must be null"
        raise RuntimeError(msg)
    lr = optimizer.get("lr")
    betas = optimizer.get("betas")
    eps = optimizer.get("eps")
    weight_decay = optimizer.get("weight_decay")
    amsgrad = optimizer.get("amsgrad")
    fused = optimizer.get("fused")
    # reason: contract adamw keeps lr/eps in one gate; helper predicates would scatter the rule.
    if (
        not isinstance(lr, (int, float))  # ruff: ignore[too-many-boolean-expressions]
        or lr <= 0
        or not isinstance(betas, list)
        or len(betas) != 2  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        or any(not isinstance(beta, (int, float)) for beta in betas)
        or not isinstance(eps, (int, float))
        or eps <= 0
        or not isinstance(weight_decay, (int, float))
        or weight_decay < 0
        or type(amsgrad) is not bool
        or type(fused) is not bool
    ):
        msg = "launch contract has invalid AdamW parameters"
        raise RuntimeError(msg)
    for key, value in (
        ("lr", lr),
        ("betas", betas),
        ("eps", eps),
        ("weight_decay", weight_decay),
        ("amsgrad", amsgrad),
        ("fused", fused),
    ):
        if mechanics_adamw.get(key) != value:
            msg = "optimizer fields disagree with Unsloth AdamW mechanics"
            raise RuntimeError(msg)
    return AdamWParameters(
        lr=float(lr),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(eps),
        weight_decay=float(weight_decay),
        amsgrad=amsgrad,
        fused=fused,
    )


# reason: contract combines wsdparameters and optimizer; splitting would split parity state.
def _contract_scheduler_parameters(contract: Mapping[str, Any]) -> WSDParameters | None:  # ruff: ignore[complex-structure,too-many-branches,too-many-statements]
    """Validate the scheduler in both public and Unsloth runtime representations.

    Returns:
        ``None`` for a ``constant`` schedule, which is the ordinary full run and builds no
        scheduler at all; otherwise the WSD peak learning rate, total, warmup, stable and decay
        step counts, the two phase-shape names and the minimum learning-rate ratio. As with
        AdamW, every field must already agree across the public and Unsloth representations.

    Raises:
        RuntimeError: If the ``optimizer`` or ``training`` block is absent or not a mapping; if
            ``training.mechanics`` is absent; if a constant schedule carries a scheduler; if the
            schedule is neither ``constant`` nor ``wsd``; if a WSD schedule has no matching
            ``wsd`` runtime scheduler; if any of the eight fields disagrees across the two
            representations; if a parameter is the wrong type or out of range -- non-positive
            peak learning rate, a negative or non-``int`` phase length, a ratio outside 0 to 1;
            if a phase type is not ``linear``, ``cosine`` or ``1-sqrt``; if the three phase
            lengths do not sum to the total; if the total is not the fixed 60-step scout cap or
            ``training.optimizer_step_cap`` does not equal it; if the run also sets a milestone,
            which the scout does not use; if it is not a fresh run or names a resume checkpoint;
            or if the optimizer learning rate is not the peak learning rate.

    """
    optimizer = contract.get("optimizer")
    training = contract.get("training")
    if not isinstance(optimizer, Mapping) or not isinstance(training, Mapping):
        msg = "launch contract has no optimizer or training configuration"
        raise RuntimeError(msg)
    mechanics = training.get("mechanics")
    if not isinstance(mechanics, Mapping):
        msg = "launch contract has no Unsloth mechanics"
        raise RuntimeError(msg)
    schedule = optimizer.get("schedule")
    scheduler = mechanics.get("scheduler")
    if schedule == "constant":
        if scheduler is not None:
            msg = "constant optimizer must not have a scheduler"
            raise RuntimeError(msg)
        return None
    if schedule != "wsd":
        msg = "launch contract has an unsupported optimizer schedule"
        raise RuntimeError(msg)
    if not isinstance(scheduler, Mapping) or scheduler.get("name") != "wsd":
        msg = "WSD optimizer requires WSD runtime scheduler mechanics"
        raise RuntimeError(msg)

    field_names = (
        "peak_lr",
        "total_steps",
        "warmup_steps",
        "stable_steps",
        "decay_steps",
        "warmup_type",
        "decay_type",
        "min_lr_ratio",
    )
    if any(optimizer.get(field) != scheduler.get(field) for field in field_names):
        msg = "optimizer fields disagree with Unsloth scheduler mechanics"
        raise RuntimeError(msg)
    peak_lr = optimizer.get("peak_lr")
    total_steps = optimizer.get("total_steps")
    warmup_steps = optimizer.get("warmup_steps")
    stable_steps = optimizer.get("stable_steps")
    decay_steps = optimizer.get("decay_steps")
    warmup_type = optimizer.get("warmup_type")
    decay_type = optimizer.get("decay_type")
    min_lr_ratio = optimizer.get("min_lr_ratio")
    # reason: contract keeps peak lr/warmup in one gate; helper predicates would scatter the rule.
    if (
        not isinstance(peak_lr, (int, float))  # ruff: ignore[too-many-boolean-expressions]
        or peak_lr <= 0
        or any(type(value) is not int or value < 0 for value in (total_steps, warmup_steps, stable_steps, decay_steps))
        or not isinstance(warmup_type, str)
        or not isinstance(decay_type, str)
        or not isinstance(min_lr_ratio, (int, float))
        or not 0 <= min_lr_ratio <= 1
    ):
        msg = "launch contract has invalid WSD scheduler parameters"
        raise RuntimeError(msg)
    # reason: inert narrowing - the guard above already raised RuntimeError on any non-int, so stripping these
    # reason: under -O changes no reachable behavior; they exist so the type checker sees the int.
    assert isinstance(total_steps, int)  # ruff: ignore[assert]
    assert isinstance(warmup_steps, int)  # ruff: ignore[assert]
    assert isinstance(stable_steps, int)  # ruff: ignore[assert]
    assert isinstance(decay_steps, int)  # ruff: ignore[assert]
    if warmup_type not in {"linear", "cosine", "1-sqrt"} or decay_type not in {
        "linear",
        "cosine",
        "1-sqrt",
    }:
        msg = "launch contract has unsupported WSD phase types"
        raise RuntimeError(msg)
    if warmup_steps + stable_steps + decay_steps != total_steps:
        msg = "WSD phase lengths must sum to the total steps"
        raise RuntimeError(msg)
    if total_steps != WSD_SCOUT_OPTIMIZER_STEPS or training.get("optimizer_step_cap") != total_steps:
        msg = "WSD scout requires the fixed 60-step optimizer cap"
        raise RuntimeError(msg)
    if training.get("milestone") is not None:
        msg = "WSD scout must use only final evaluation"
        raise RuntimeError(msg)
    if training.get("fresh_run") is not True or training.get("resume_from_checkpoint") is not None:
        msg = "WSD scout must start fresh without a resume checkpoint"
        raise RuntimeError(msg)
    if optimizer.get("lr") != peak_lr:
        msg = "WSD optimizer learning rate must equal the peak learning rate"
        raise RuntimeError(msg)
    return WSDParameters(
        peak_lr=float(peak_lr),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        warmup_type=warmup_type,
        decay_type=decay_type,
        min_lr_ratio=float(min_lr_ratio),
    )


def _build_scheduler(optimizer: Any, parameters: WSDParameters | None) -> Any | None:
    """Create the pinned Transformers WSD scheduler only for the approved scout.

    Returns:
        ``None`` when there are no WSD parameters, which is the ordinary full run and steps the
        optimizer at a constant rate; otherwise the Transformers ``get_wsd_schedule`` built from
        exactly the contract's phase lengths, shapes and minimum ratio. The import stays inside
        the branch that needs it, so a constant run does not load ``transformers`` here.

    """
    if parameters is None:
        return None
    from transformers import get_wsd_schedule

    return get_wsd_schedule(
        optimizer,
        num_warmup_steps=parameters.warmup_steps,
        num_stable_steps=parameters.stable_steps,
        num_decay_steps=parameters.decay_steps,
        warmup_type=parameters.warmup_type,
        decay_type=parameters.decay_type,
        min_lr_ratio=parameters.min_lr_ratio,
    )


def _apply_optimizer_step(optimizer: Any, scheduler: Any | None) -> dict[str, Any]:
    """Record the learning rate used by an update before advancing its schedule.

    Returns:
        ``lr``, the rate this step actually applied, read before ``optimizer.step()`` so it
        describes the update rather than the next one. With a scheduler it also carries
        ``next_lr``, read after the scheduler advanced, and the ``wsd`` scheduler name; a
        constant run returns ``lr`` alone.

    """
    applied_lr = float(optimizer.param_groups[0]["lr"])
    optimizer.step()
    if scheduler is None:
        return {"lr": applied_lr}
    scheduler.step()
    return {
        "lr": applied_lr,
        "next_lr": float(optimizer.param_groups[0]["lr"]),
        "scheduler": "wsd",
    }


def _require_fresh_scheduler_run(parameters: WSDParameters | None, *, resume_checkpoint: str | None) -> None:
    """Reject WSD resume because its exact scheduler state is intentionally absent.

    Raises:
        RuntimeError: If a WSD schedule is paired with a resume checkpoint. Checkpoints save
            adapter, head, optimizer and RNG but not scheduler state, so a resumed WSD run
            would restart its phase clock and apply a warmup rate mid-decay. Refusing keeps the
            scout's learning-rate trajectory exactly the one that was approved.

    """
    if parameters is not None and resume_checkpoint is not None:
        msg = "WSD scout cannot resume without an exact scheduler state"
        raise RuntimeError(msg)


def _feature_extraction_task_type_name(task_type: object) -> str:
    value = getattr(task_type, "value", task_type)
    return str(value)


def _require_feature_extraction_peft_task(model: Any, task_type: object) -> None:
    configs = getattr(model, "peft_config", None)
    values = list(configs.values()) if isinstance(configs, Mapping) else [configs]
    if not values or any(
        _feature_extraction_task_type_name(getattr(config, "task_type", None))
        != _feature_extraction_task_type_name(task_type)
        for config in values
    ):
        msg = "Unsloth LoRA adapter is not FEATURE_EXTRACTION"
        raise RuntimeError(msg)


def _contract_lora_mechanics(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Read every adapter parameter from the immutable launch contract.

    Returns:
        The adapter shape ``get_peft_model`` is called with: rank, alpha, dropout, bias mode,
        target modules, random state, gradient-checkpointing mode, and the ``use_rslora`` and
        ``use_dora`` flags. Rank, alpha, dropout and target modules must already match the
        public ``training.lora`` block, so the two contract views cannot describe two adapters.

    Raises:
        RuntimeError: If the ``training`` block, its ``mechanics``, or either LoRA block is
            absent or not a mapping; if rank or alpha is not a positive integer, dropout not a
            number, bias not a string, target modules not a non-empty list of strings, random
            state not an integer, or either flag not a ``bool``; if gradient checkpointing is
            not ``unsloth``, since the attestation later asserts that exact mode; or if rank,
            alpha, dropout or target modules disagree between the two contract blocks.

    """
    training = contract.get("training")
    if not isinstance(training, Mapping):
        msg = "launch contract has no training configuration"
        raise RuntimeError(msg)
    mechanics = training.get("mechanics")
    if not isinstance(mechanics, Mapping):
        msg = "launch contract has no Unsloth mechanics"
        raise RuntimeError(msg)
    lora = mechanics.get("lora")
    training_lora = training.get("lora")
    if not isinstance(lora, Mapping) or not isinstance(training_lora, Mapping):
        msg = "launch contract has no LoRA configuration"
        raise RuntimeError(msg)
    rank = lora.get("rank")
    alpha = lora.get("alpha")
    dropout = lora.get("dropout")
    bias = lora.get("bias")
    target_modules = lora.get("target_modules")
    random_state = lora.get("random_state")
    use_rslora = lora.get("use_rslora")
    use_dora = lora.get("use_dora")
    gradient_checkpointing = mechanics.get("gradient_checkpointing")
    # reason: contract lora keeps unsloth/rank in one gate; helper predicates would scatter the rule.
    if (
        not isinstance(rank, int)  # ruff: ignore[too-many-boolean-expressions]
        or rank <= 0
        or not isinstance(alpha, int)
        or alpha <= 0
        or not isinstance(dropout, (int, float))
        or not isinstance(bias, str)
        or not isinstance(target_modules, list)
        or not target_modules
        or any(not isinstance(target, str) for target in target_modules)
        or not isinstance(random_state, int)
        or type(use_rslora) is not bool
        or type(use_dora) is not bool
        or gradient_checkpointing != "unsloth"
    ):
        msg = "launch contract has invalid Unsloth LoRA mechanics"
        raise RuntimeError(msg)
    for key, value in (
        ("rank", rank),
        ("alpha", alpha),
        ("dropout", dropout),
        ("target_modules", target_modules),
    ):
        if training_lora.get(key) != value:
            msg = "training LoRA fields disagree with Unsloth mechanics"
            raise RuntimeError(msg)
    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": float(dropout),
        "bias": bias,
        "target_modules": list(target_modules),
        "random_state": random_state,
        "gradient_checkpointing": gradient_checkpointing,
        "use_rslora": use_rslora,
        "use_dora": use_dora,
    }


def _attach_unsloth_feature_extraction_lora(
    fast_model: Any,
    model: Any,
    *,
    task_type: object,
    contract: Mapping[str, Any],
) -> Any:
    """Use Unsloth's adapter path while rejecting causal-generation PEFT wrappers.

    Returns:
        The adapted model from Unsloth's own ``get_peft_model``, returned only after every
        attached PEFT config is confirmed to carry the requested task type. Going through
        Unsloth's path rather than PEFT's directly keeps its gradient-checkpointing patches
        applied; the task-type check is what stops a causal-generation wrapper from being
        trained as a token tagger.

    """
    lora = _contract_lora_mechanics(contract)
    adapted = fast_model.get_peft_model(
        model,
        r=lora["rank"],
        target_modules=lora["target_modules"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        use_gradient_checkpointing=lora["gradient_checkpointing"],
        use_rslora=lora["use_rslora"],
        use_dora=lora["use_dora"],
        random_state=lora["random_state"],
        task_type=task_type,
    )
    _require_feature_extraction_peft_task(adapted, task_type)
    return adapted


def _propagate_unsloth_remote_code_provenance(wrapper: Any, body: Any) -> None:
    """Preserve FastModel's loader provenance when the BIOES head unwraps ``.lfm2``.

    Unsloth records this boolean on the outer remote-code wrapper, then reads it
    later from the model passed to ``get_peft_model``. The tagger must adapt the
    encoder body, so fail closed unless the exact true provenance transfers.

    Raises:
        RuntimeError: If the wrapper's flag is not the ``bool`` ``True`` -- a truthy value of
            another type is refused, not coerced -- or if the body does not read back ``True``
            after the assignment. Either way Unsloth would later see an unwrapped body with no
            recorded remote-code provenance, and this fails the run instead of letting the
            adapter attach under a provenance nobody granted.

    """
    provenance = getattr(wrapper, "_unsloth_trust_remote_code", None)
    if type(provenance) is not bool or provenance is not True:
        msg = "Unsloth wrapper remote-code provenance must be exactly True"
        raise RuntimeError(msg)
    # reason: third-party internal - unsloth records provenance on the outer wrapper and reads it back off the model
    # reason: passed to get_peft_model; the tagger adapts the unwrapped body, so the flag must transfer. Guarded on
    # reason: both sides: the wrapper flag must be exactly True and the body must read it back.
    body._unsloth_trust_remote_code = provenance  # ruff: ignore[private-member-access]
    if getattr(body, "_unsloth_trust_remote_code", None) is not True:
        msg = "Unsloth encoder body did not retain remote-code provenance"
        raise RuntimeError(msg)


def _unsloth_encoder_tagger(
    candidate_key: str,
    *,
    contract: Mapping[str, Any],
    expected_encoder_checkpoint_attestation: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load one encoder through Unsloth ``FastModel`` with no PEFT fallback.

    Unsloth installs its Transformers and PEFT patches at import time.

    Returns:
        The BIOES tagger already moved to CUDA and in train mode, the tokenizer with a pad
        token guaranteed, and the runtime attestation for the loaded stack. The tagger wraps
        the unwrapped ``.lfm2`` body with the adapter attached, its head seeded under the fixed
        gate seed so the classifier digest is reproducible across runs.

    Raises:
        ValueError: If the candidate's loader is neither ``masked_lm_body`` nor
            ``token_classifier_body``, which means the key names a non-encoder candidate.
        RuntimeError: If the ``FastModel`` wrapper exposes no ``.lfm2`` encoder body, or if the
            adapted model comes back without a config -- ``use_cache`` is set on that config a
            moment later, so an absent one would silently leave caching enabled during
            training. The checkpoint attestation gate runs before the loader and its refusal of
            a changed or unverified snapshot reaches a caller here as well.

    """
    import torch

    # reason: unsloth publishes no macOS wheel and installs in the training container, so this import is env-only.
    # reason: unsloth patches transformers and peft at import time, so it must load first.
    fast_model = import_module("unsloth").FastModel

    from huggingface_hub import snapshot_download
    from peft import TaskType
    from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification

    # reason: deferred import inside a block whose order is pinned by two tests; unsloth must import before
    # reason: transformers and peft.
    from ..data.tagger import HiddenStateTokenTagger  # ruff: ignore[relative-imports]
    from .base230_unsloth_probe import resolve_candidate_snapshot
    from .base_selection import CANDIDATES, GATE_SEED

    candidate = CANDIDATES[candidate_key]
    if candidate.loader not in {"masked_lm_body", "token_classifier_body"}:
        msg = f"{candidate_key} is not an encoder candidate"
        raise ValueError(msg)
    snapshot = resolve_candidate_snapshot(candidate_key, snapshot_download)
    auto_model = AutoModelForTokenClassification if candidate_key == "pii350" else AutoModelForMaskedLM
    encoder_checkpoint_attestation = require_lfm2_checkpoint_body_state_attestation(
        expected_encoder_checkpoint_attestation,
        snapshot,
    )
    model, tokenizer = fast_model.from_pretrained(
        model_name=candidate.model_id,
        revision=candidate.revision,
        tokenizer_name=snapshot,
        max_seq_length=8192,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        dtype=torch.bfloat16,
        full_finetuning=False,
        fast_inference=False,
        trust_remote_code=candidate.trust_remote_code,
        auto_model=auto_model,
    )
    body = getattr(model, "lfm2", None)
    if body is None:
        msg = "encoder FastModel wrapper has no .lfm2 encoder body"
        raise RuntimeError(msg)
    _propagate_unsloth_remote_code_provenance(model, body)
    model = body
    model = _attach_unsloth_feature_extraction_lora(
        fast_model,
        model,
        task_type=TaskType.FEATURE_EXTRACTION,
        contract=contract,
    )
    if getattr(model, "config", None) is None:
        msg = "Unsloth encoder returned a model without config"
        raise RuntimeError(msg)
    model.config.use_cache = False
    torch.manual_seed(GATE_SEED)
    tagger = (
        HiddenStateTokenTagger(
            model,
            int(model.config.hidden_size),
            37,
            request_hidden_states=False,
            classifier_dtype=torch.bfloat16,
            packed_segment_isolation=True,
        )
        .cuda()
        .train()
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return (
        tagger,
        tokenizer,
        _runtime_attestation(
            contract=contract,
            fast_model=fast_model,
            model=model,
            tagger=tagger,
            candidate_key=candidate_key,
            encoder_checkpoint_attestation=encoder_checkpoint_attestation,
        ),
    )


def _unsloth_tagger(*, contract: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from huggingface_hub import snapshot_download

    # reason: unsloth publishes no macOS wheel, so a static reader cannot resolve this import.
    fast_language_model = import_module("unsloth").FastLanguageModel

    # reason: deferred import inside a block whose order is pinned by two tests; unsloth must import before
    # reason: transformers and peft.
    from ..data.tagger import HiddenStateTokenTagger  # ruff: ignore[relative-imports]
    from .base230_unsloth_probe import (
        build_unsloth_base230,
        resolve_base230_snapshot,
    )
    from .base_selection import GATE_SEED

    model, tokenizer = build_unsloth_base230(
        fast_language_model,
        dtype=torch.bfloat16,
        tokenizer_name=resolve_base230_snapshot(snapshot_download),
    )
    if getattr(model, "config", None) is None:
        msg = "Unsloth Base230 returned a model without config"
        raise RuntimeError(msg)
    model.config.use_cache = False
    torch.manual_seed(GATE_SEED)
    tagger = (
        HiddenStateTokenTagger(
            model,
            int(model.config.hidden_size),
            37,
            request_hidden_states=True,
            classifier_dtype=torch.bfloat16,
            packed_segment_isolation=True,
        )
        .cuda()
        .train()
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return (
        tagger,
        tokenizer,
        _runtime_attestation(
            contract=contract,
            fast_model=fast_language_model,
            model=model,
            tagger=tagger,
            candidate_key="base230",
        ),
    )


def _build_tagger(
    candidate_key: str,
    *,
    contract: Mapping[str, Any],
    expected_encoder_checkpoint_attestation: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    if candidate_key == "base230":
        return _unsloth_tagger(contract=contract)
    return _unsloth_encoder_tagger(
        candidate_key,
        contract=contract,
        expected_encoder_checkpoint_attestation=expected_encoder_checkpoint_attestation,
    )


def _save_checkpoint(
    writer: RunWriter,
    *,
    tagger: Any,
    optimizer: Any,
    state: RuntimeResumeState,
    reason: StopReason,
) -> None:
    """Save adapter, head, optimizer, and RNG, then atomically publish metadata."""
    import torch

    root = writer.root / "checkpoints" / f"step-{state.optimizer_step:08d}"
    relative_root = str(root.relative_to(writer.root))
    writer.append(
        {
            "event": "checkpoint_started",
            "checkpoint_path": relative_root,
            "step": state.optimizer_step,
            "packed_cursor": state.packed_cursor,
            "reason": reason,
        },
    )
    root.mkdir(parents=True, exist_ok=True)
    tagger.backbone.save_pretrained(root / "adapter")
    torch.save(tagger.classifier.state_dict(), root / "classifier.pt")
    torch.save(optimizer.state_dict(), root / "optimizer.pt")
    torch.save(
        {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
        root / "rng.pt",
    )
    writer.atomic_json(
        str(root.relative_to(writer.root) / "metadata.json"),
        checkpoint_metadata(state, reason=reason),
    )
    writer.append(
        {
            "event": "checkpoint_complete",
            "checkpoint_path": relative_root,
            "step": state.optimizer_step,
            "packed_cursor": state.packed_cursor,
            "reason": reason,
        },
    )


def _active_trainable_adapter_parameters(backbone: Any) -> dict[str, Any]:
    parameters = {
        name: parameter for name, parameter in backbone.named_parameters() if parameter.requires_grad and "lora_" in name
    }
    if not parameters:
        msg = "adapter resume found no active trainable LoRA parameters"
        raise RuntimeError(msg)
    return parameters


def _canonical_adapter_state_dict(backbone: Any, adapter_state: Mapping[str, Any], *, adapter_name: str) -> dict[str, Any]:
    """Apply PEFT's own saved-key to model-key normalization before verification.

    Returns:
        The checkpoint's adapter state re-keyed the way PEFT itself names those tensors on a
        loaded model, by inserting the active adapter name into each key. Verification compares
        checkpoint keys against live parameter names, and the two spellings differ before this
        normalization, so comparing raw saved keys would report every tensor as missing.

    Raises:
        RuntimeError: If the backbone carries no ``peft_config`` mapping or that mapping has no
            entry for the named adapter; or if the configured PEFT type does not map to the
            ``lora_`` prefix. A different prefix means a different adapter family, and applying
            LoRA's key mapping to it would produce names that match nothing.

    """
    # reason: THIRD-PARTY INTERNAL, not a first-party privacy question. peft 0.20.0 exposes no public
    # reason: equivalent — `save_and_load` publishes only `get_peft_model_state_dict` and friends, which
    # reason: operate on a live model rather than a saved state dict, so none performs this saved-key to
    # reason: model-key rename. Any peft release that moves or renames this symbol breaks adapter resume
    # reason: at call time, inside the verification path. Owner ledger item; see the run report.
    from peft import PeftType
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
    from peft.utils.save_and_load import _insert_adapter_name_into_state_dict  # ruff: ignore[import-private-name]

    configs = getattr(backbone, "peft_config", None)
    if not isinstance(configs, Mapping) or adapter_name not in configs:
        msg = "adapter resume lacks the active PEFT adapter configuration"
        raise RuntimeError(msg)
    peft_type = getattr(configs[adapter_name], "peft_type", None)
    if not isinstance(peft_type, PeftType):
        msg = "adapter resume lacks a recognized PEFT adapter type"
        raise RuntimeError(msg)
    parameter_prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(peft_type)
    if parameter_prefix != "lora_":
        msg = "adapter resume requires the configured LoRA key mapping"
        raise RuntimeError(msg)
    return dict(
        _insert_adapter_name_into_state_dict(
            dict(adapter_state),
            adapter_name=adapter_name,
            parameter_prefix=parameter_prefix,
        ),
    )


def _require_exact_adapter_load(
    outcome: Any,
    *,
    checkpoint_model_keys: Sequence[str] | None = None,
    active_trainable_adapter_keys: Sequence[str] | None = None,
) -> None:
    missing = list(getattr(outcome, "missing_keys", []) or [])
    unexpected = list(getattr(outcome, "unexpected_keys", []) or [])
    if checkpoint_model_keys is None or active_trainable_adapter_keys is None:
        relevant_missing = missing
    else:
        required_keys = set(checkpoint_model_keys) | set(active_trainable_adapter_keys)
        relevant_missing = [name for name in missing if name in required_keys]
    if relevant_missing or unexpected:
        msg = f"adapter resume state mismatch: missing={sorted(relevant_missing)} unexpected={sorted(unexpected)}"
        raise RuntimeError(
            msg,
        )


def _require_exact_adapter_tensor_values(backbone: Any, checkpoint_model_state: Mapping[str, Any]) -> None:
    """Prove every saved LoRA tensor now equals its active runtime parameter.

    Raises:
        RuntimeError: If the checkpoint's key set is not exactly the active trainable adapter
            key set -- compared as sets in both directions, so an extra saved tensor is refused
            as loudly as a missing one -- or if any tensor differs in kind, shape, dtype or
            value. The value check is a bitwise ``torch.equal`` on CPU rather than a tolerance,
            because a resume must continue from the saved weights, not from something close to
            them. ``set_peft_model_state_dict`` reports which keys it consumed; this reads the
            live parameters back afterwards, so it also catches a load that reported success
            while writing nothing. The no-active-adapter refusal of the parameter lookup reaches
            a caller through here.

    """
    import torch

    active = _active_trainable_adapter_parameters(backbone)
    checkpoint_keys = set(checkpoint_model_state)
    active_keys = set(active)
    if checkpoint_keys != active_keys:
        msg = f"adapter resume key mismatch: checkpoint={sorted(checkpoint_keys)} active={sorted(active_keys)}"
        raise RuntimeError(
            msg,
        )
    for name, checkpoint_tensor in checkpoint_model_state.items():
        parameter = active[name]
        if not isinstance(checkpoint_tensor, torch.Tensor):
            msg = f"adapter resume checkpoint tensor is invalid: {name}"
            raise RuntimeError(msg)
        if parameter.shape != checkpoint_tensor.shape:
            msg = f"adapter resume tensor shape mismatch: {name}"
            raise RuntimeError(msg)
        if parameter.dtype != checkpoint_tensor.dtype:
            msg = f"adapter resume tensor dtype mismatch: {name}"
            raise RuntimeError(msg)
        if not torch.equal(parameter.detach().cpu(), checkpoint_tensor.detach().cpu()):
            msg = f"adapter resume tensor value mismatch: {name}"
            raise RuntimeError(msg)


def _move_optimizer_state_to_parameter_devices(optimizer: Any) -> None:
    """Keep restored Adam moments colocated with the parameter each moment updates."""
    import torch

    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.device != parameter.device:
                state[key] = value.to(parameter.device)


def _assert_optimizer_owns_active_adapter(tagger: Any, optimizer: Any) -> None:
    optimizer_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    missing = [
        name
        for name, parameter in tagger.backbone.named_parameters()
        if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids
    ]
    if missing:
        msg = f"optimizer does not own active adapter tensors: {sorted(missing)}"
        raise RuntimeError(msg)


def _restore_pii_training_payloads(root: Path, *, tagger: Any, optimizer: Any) -> Mapping[str, Any]:
    """Restore mutable PII training payloads after a caller verified checkpoint identity.

    Returns:
        The RNG payload read off the checkpoint, carrying the saved CPU and CUDA generator
        states. It is returned rather than applied, because rank-local policy decides whether a
        distributed worker adopts them; the adapter, classifier and optimizer are restored in
        place before it returns.

    Raises:
        RuntimeError: If the RNG payload is not a mapping. Four verification gates run before
            that point and their refusals reach a caller through here: PEFT's key normalization
            refuses a missing or non-LoRA adapter config; the exact-load check refuses missing
            or unexpected keys; the tensor-value check refuses any adapter tensor that does not
            match bitwise; and the optimizer-ownership check refuses an optimizer whose
            parameter groups do not cover every active adapter tensor, which would otherwise
            train an adapter that receives no updates. Classifier and RNG tensors load with
            ``weights_only=True``; the optimizer state does not, since its payload holds
            non-tensor entries.

    """
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_state = load_file(str(root / "adapter" / "adapter_model.safetensors"))
    checkpoint_model_state = _canonical_adapter_state_dict(tagger.backbone, adapter_state, adapter_name="default")
    active_adapter_parameters = _active_trainable_adapter_parameters(tagger.backbone)
    outcome = set_peft_model_state_dict(tagger.backbone, adapter_state, adapter_name="default")
    _require_exact_adapter_load(
        outcome,
        checkpoint_model_keys=tuple(checkpoint_model_state),
        active_trainable_adapter_keys=tuple(active_adapter_parameters),
    )
    tagger.backbone.set_adapter("default")
    _require_exact_adapter_tensor_values(tagger.backbone, checkpoint_model_state)
    tagger.classifier.load_state_dict(torch.load(root / "classifier.pt", map_location="cuda", weights_only=True))
    optimizer.load_state_dict(torch.load(root / "optimizer.pt", map_location="cuda", weights_only=False))
    _move_optimizer_state_to_parameter_devices(optimizer)
    _assert_optimizer_owns_active_adapter(tagger, optimizer)
    rng = torch.load(root / "rng.pt", map_location="cpu", weights_only=True)
    if not isinstance(rng, Mapping):
        msg = "checkpoint RNG payload is invalid"
        raise RuntimeError(msg)
    return rng


def load_verified_pii_resume_state(
    resume_dir: str,
    *,
    tagger: Any,
    optimizer: Any,
    verified_metadata: Mapping[str, Any],
    rank_local_rng_policy: Callable[[Mapping[str, Any]], None],
) -> dict[str, int]:
    """Restore a hash-verified continuation checkpoint without legacy config coupling.

    The caller owns manifest/schema verification. This seam rejects any unverified
    state before deserializing adapter, classifier, optimizer, or RNG bytes.

    Returns:
        The ``optimizer_step`` and ``packed_cursor`` the run resumes from, taken from the
        verified receipt rather than recomputed, and returned only after the payloads have been
        restored and the rank-local RNG policy has been applied.

    Raises:
        RuntimeError: If the receipt does not carry ``hash_verified`` exactly ``True``, which
            is the check that keeps unverified bytes from reaching ``torch.load`` at all; if
            the step or cursor is not an ``int``, the step is negative, or the cursor is not
            exactly ``step * 128``, so a receipt cannot claim a position its step count does
            not imply; if the world size is not 1, 2 or 4; or if the receipt carries no string
            ``checkpoint_digest``. Every refusal raised while restoring the payloads --
            adapter config, key set, bitwise tensor equality, optimizer ownership and RNG shape
            -- reaches a caller through here as well.

    """
    if verified_metadata.get("hash_verified") is not True:
        msg = "continuation resume requires a hash-verified checkpoint receipt"
        raise RuntimeError(msg)
    step = verified_metadata.get("optimizer_step")
    cursor = verified_metadata.get("packed_cursor")
    world_size = verified_metadata.get("world_size")
    if type(step) is not int or type(cursor) is not int or step < 0 or cursor != step * 128:
        msg = "continuation checkpoint step/cursor is invalid"
        raise RuntimeError(msg)
    if world_size not in {1, 2, 4}:
        msg = "continuation checkpoint world size is invalid"
        raise RuntimeError(msg)
    if not isinstance(verified_metadata.get("checkpoint_digest"), str):
        msg = "continuation checkpoint receipt lacks its digest"
        raise RuntimeError(msg)
    rng = _restore_pii_training_payloads(Path(resume_dir), tagger=tagger, optimizer=optimizer)
    rank_local_rng_policy(rng)
    return {"optimizer_step": step, "packed_cursor": cursor}


def _resume_if_requested(
    resume_dir: str | None,
    *,
    tagger: Any,
    optimizer: Any,
    candidate_key: str,
    config_digest: str,
) -> RuntimeResumeState | None:
    if not resume_dir:
        return None
    import torch

    root = Path(resume_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    state = RuntimeResumeState(**{key: metadata[key] for key in RuntimeResumeState.__dataclass_fields__})
    validate_resume_state(state, candidate_key=candidate_key, config_digest=config_digest)
    rng = _restore_pii_training_payloads(root, tagger=tagger, optimizer=optimizer)
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state_all(rng["cuda"])
    return state


def _load_final_eval_rows() -> list[EvalRow]:
    """Load immutable eval rows without training's label round-trip filter.

    Returns:
        Every pinned evaluation row, converted from the dataset revision named in the contract
        module and tagged with the ``v2-eval`` dataset and ``eval`` shard. No row is dropped:
        unlike the training path there is no label round-trip filter here, so a row whose gold
        spans do not survive re-tokenization still counts against recall rather than vanishing.

    Raises:
        RuntimeError: If the pinned dataset does not hold exactly the expected row count, which
            means the revision moved underneath the pin; or if conversion returns fewer rows
            than it read, which would quietly shrink the denominator every metric divides by.

    """
    from datasets import load_dataset

    from meddies_pii.eval_baseline.baseline.datasets import row_from_record

    from .pins import (
        EVAL_CONFIG,
        EVAL_DATASET_ID,
        EVAL_DATASET_REVISION,
        EVAL_SPLIT,
    )

    dataset = load_dataset(
        EVAL_DATASET_ID,
        EVAL_CONFIG,
        split=EVAL_SPLIT,
        revision=EVAL_DATASET_REVISION,
        cache_dir="/cache/hf",
    )
    if len(dataset) != EVAL_ROWS:
        msg = f"pinned eval inventory changed: expected {EVAL_ROWS}, got {len(dataset)}"
        raise RuntimeError(msg)
    rows = [row_from_record(dict(row), dataset="v2-eval", shard="eval", index=index) for index, row in enumerate(dataset)]
    if len(rows) != EVAL_ROWS:
        msg = "final eval did not retain all pinned source rows"
        raise RuntimeError(msg)
    return rows


def _inference_tokenization(tokenizer: Any, text: str) -> dict[str, Any]:
    """Tokenize for prediction only, preserving visible offsets and truncation.

    Returns:
        The first window's ``input_ids``, ``attention_mask`` and ``offset_mapping``, plus a
        ``truncated`` flag. Overflow windows are requested so that truncation is detectable but
        only the first is scored, and the offsets are kept as character spans into the original
        text so a predicted span maps back to what a reader sees. The evaluation counts gold
        spans past the visible offsets against recall rather than discarding them.

    Raises:
        RuntimeError: If the three per-token fields come back at different lengths. Positions
            are read by index across all three, so a length disagreement would silently pair a
            label with another token's offsets.

    """
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=8192,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]
    nested = bool(input_ids) and isinstance(input_ids[0], Sequence)
    if nested:
        first_ids = list(input_ids[0])
        first_mask = list(attention_mask[0])
        first_offsets = [tuple(offset) for offset in offsets[0]]
        truncated = len(input_ids) > 1
    else:
        first_ids = list(input_ids)
        first_mask = list(attention_mask)
        first_offsets = [tuple(offset) for offset in offsets]
        truncated = bool(encoded.get("num_truncated_tokens", 0))
    if not (len(first_ids) == len(first_mask) == len(first_offsets)):
        msg = "inference tokenizer returned inconsistent field lengths"
        raise RuntimeError(msg)
    return {
        "input_ids": first_ids,
        "attention_mask": first_mask,
        "offset_mapping": first_offsets,
        "truncated": truncated,
    }


def _inference_batch(entries: Sequence[Mapping[str, Any]], tokenizer: Any, device: str) -> dict[str, Any]:
    import torch

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    max_length = max(len(entry["tokens"]["input_ids"]) for entry in entries)
    input_ids = torch.full((len(entries), max_length), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(entries), max_length), dtype=torch.long, device=device)
    for index, entry in enumerate(entries):
        token_ids = entry["tokens"]["input_ids"]
        mask = entry["tokens"]["attention_mask"]
        input_ids[index, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=device)
        attention_mask[index, : len(mask)] = torch.tensor(mask, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# reason: final evaluate orders eval rows before to id; helper seams would duplicate totals or escaping.
def _final_evaluate(  # ruff: ignore[too-many-locals,too-many-statements]
    tagger: Any,
    tokenizer: Any,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    progress_event: str = "final_evaluation_progress",
) -> dict[str, Any]:
    import torch

    from meddies_pii.annotations.bioes.spans import decode_bioes_from_offsets
    from meddies_pii.annotations.bioes.viterbi import viterbi_decode_logits
    from meddies_pii.annotations.bioes.vocabulary import (
        ENTITY_LABELS,
        build_bioes_label_space,
        build_label_to_id,
    )
    from meddies_pii.evaluation.span_metrics import (
        containment_span_prf_by_label,
        containment_span_report_by_doc,
        exact_span_prf_by_label,
        exact_span_report_by_doc,
    )

    rows = _load_final_eval_rows()
    entries: list[_FinalEvalEntry] = [{"row": row, "tokens": _inference_tokenization(tokenizer, row.text)} for row in rows]
    label_to_id = build_label_to_id(build_bioes_label_space(ENTITY_LABELS))
    id_to_label = {index: label for label, index in label_to_id.items()}
    ordered = sorted(entries, key=lambda entry: len(entry["tokens"]["input_ids"]))
    predicted: dict[str, Any] = {}
    gold: dict[str, Any] = {}
    languages: dict[str, str] = {}
    truncation = {
        "truncated_rows": 0,
        "gold_spans_beyond_visible_offsets": 0,
        "gold_spans_without_visible_token": 0,
    }
    tagger.eval()
    device = str(next(tagger.parameters()).device)
    started = time.monotonic()
    processed_rows = 0
    last_reported_rows = 0
    with torch.no_grad():
        for start in range(0, len(ordered), EVAL_BATCH_SIZE):
            batch_entries = ordered[start : start + EVAL_BATCH_SIZE]
            batch = _inference_batch(batch_entries, tokenizer, device)
            logits = tagger(**batch)["logits"]
            for local_index, entry in enumerate(batch_entries):
                row = entry["row"]
                tokens = entry["tokens"]
                offsets = tokens["offset_mapping"]
                visible_end = max((end for _begin, end in offsets), default=0)
                if tokens["truncated"]:
                    truncation["truncated_rows"] += 1
                for span in row.gold_spans:
                    if span.end > visible_end:
                        truncation["gold_spans_beyond_visible_offsets"] += 1
                    if not any(begin < span.end and end > span.start for begin, end in offsets):
                        truncation["gold_spans_without_visible_token"] += 1
                ids = viterbi_decode_logits(logits[local_index, : len(offsets)], id_to_label, offsets)
                key = row.stable_id
                predicted[key] = decode_bioes_from_offsets(row.text, offsets, ids, id_to_label)
                gold[key] = row.gold_spans
                languages[key] = row.language
            processed_rows += len(batch_entries)
            if progress is not None and (
                processed_rows - last_reported_rows >= EVAL_PROGRESS_INTERVAL_ROWS or processed_rows == len(ordered)
            ):
                elapsed = time.monotonic() - started
                throughput = processed_rows / elapsed if elapsed else 0.0
                progress(
                    {
                        "event": progress_event,
                        "processed_rows": processed_rows,
                        "total_rows": EVAL_ROWS,
                        "elapsed_seconds": elapsed,
                        "rows_per_second": throughput,
                        "eta_seconds": ((len(ordered) - processed_rows) / throughput if throughput else None),
                    },
                )
                last_reported_rows = processed_rows
    exact = exact_span_report_by_doc(predicted, gold)
    containment = containment_span_report_by_doc(predicted, gold)
    exact_by_label = exact_span_prf_by_label(predicted, gold)
    containment_by_label = containment_span_prf_by_label(predicted, gold)
    language_recall: dict[str, float] = {}
    for language in sorted(set(languages.values())):
        keys = [key for key in gold if languages[key] == language]
        recall = exact_span_report_by_doc({key: predicted[key] for key in keys}, {key: gold[key] for key in keys})[
            "recall"
        ]
        if not isinstance(recall, (int, float)):
            msg = "exact-span report has a non-numeric recall"
            raise RuntimeError(msg)
        language_recall[language] = float(recall)
    return {
        "rows": len(rows),
        "batching": "length_aware",
        "batch_size": EVAL_BATCH_SIZE,
        "truncation": truncation,
        "exact_span": exact,
        "containment_span": containment,
        "per_label_exact": exact_by_label,
        "per_label_containment": containment_by_label,
        "per_language_recall": language_recall,
    }


# reason: require milestone owns training and milestone together; splitting would fragment diagnostics.
def _require_milestone_state(contract: Mapping[str, Any], state: RuntimeResumeState) -> Mapping[str, Any]:  # ruff: ignore[complex-structure]
    """Attest that the selected batch reached the one equal-cursor milestone.

    Returns:
        The contract's ``training.milestone`` block, returned only once the live state is
        proved to sit exactly on it. Callers read ``evaluation_rows`` off the returned block
        and check the evaluation scored that many.

    Raises:
        RuntimeError: If the ``training`` block or its ``milestone`` is absent or not a
            mapping; if the batch size, expected step or expected cursor is not an ``int``; if
            batch size times step does not equal the cursor, which is the arithmetic that makes
            one milestone reachable by exactly one batch size; if the live step or cursor is
            not the expected one; if the milestone's row count is not the pinned evaluation
            size; if either ``checkpoint`` or ``evaluate`` is not ``True``, since a milestone
            must both save and score; if ``continues_training`` is not ``True``, because a
            milestone marks a point in the run rather than its end; or if the run also sets an
            optimizer-step stop cap, which would make the milestone and the cap disagree about
            when training ends.

    """
    training = contract.get("training")
    if not isinstance(training, Mapping):
        msg = "milestone contract has no training configuration"
        raise RuntimeError(msg)
    milestone = training.get("milestone")
    if not isinstance(milestone, Mapping):
        msg = "milestone evaluation is absent from the launch contract"
        raise RuntimeError(msg)
    batch_size = training.get("batch_size")
    expected_step = milestone.get("optimizer_step")
    expected_cursor = milestone.get("packed_cursor")
    if not isinstance(batch_size, int) or not isinstance(expected_step, int) or not isinstance(expected_cursor, int):
        msg = "milestone contract has invalid batch, step, or cursor"
        raise RuntimeError(msg)
    if batch_size * expected_step != expected_cursor:
        msg = "milestone batch and step do not equal the packed cursor"
        raise RuntimeError(msg)
    if state.optimizer_step != expected_step:
        msg = "milestone optimizer step does not match the launch contract"
        raise RuntimeError(msg)
    if state.packed_cursor != expected_cursor:
        msg = "milestone packed cursor does not match the launch contract"
        raise RuntimeError(msg)
    if milestone.get("evaluation_rows") != EVAL_ROWS:
        msg = "milestone evaluation row count does not match the pinned eval"
        raise RuntimeError(msg)
    if milestone.get("checkpoint") is not True or milestone.get("evaluate") is not True:
        msg = "milestone checkpoint and evaluation must both be enabled"
        raise RuntimeError(msg)
    if milestone.get("continues_training") is not True:
        msg = "milestone cannot stop training"
        raise RuntimeError(msg)
    if training.get("optimizer_step_cap") is not None:
        msg = "milestone run cannot set an optimizer-step stop cap"
        raise RuntimeError(msg)
    return milestone


def _is_milestone_step(contract: Mapping[str, Any], state: RuntimeResumeState) -> bool:
    """Keep milestone persistence independent from periodic checkpoint cadence.

    Returns:
        ``True`` only when the contract carries a milestone mapping and the live optimizer step
        equals its step. This is a cheap predicate on the hot loop, deliberately separate from
        the full attestation: it answers whether to open the milestone path, and the
        attestation then re-checks every field before anything is written.

    """
    training = contract.get("training")
    if not isinstance(training, Mapping):
        return False
    milestone = training.get("milestone")
    return isinstance(milestone, Mapping) and state.optimizer_step == milestone.get("optimizer_step")


def _evaluate_milestone(
    contract: Mapping[str, Any],
    state: RuntimeResumeState,
    tagger: Any,
    tokenizer: Any,
    writer: RunWriter,
) -> dict[str, Any]:
    """Evaluate the saved milestone, then restore train mode for later steps.

    Returns:
        The milestone metrics, already persisted under ``milestones/step-NNNNNNNN``. The
        ``finally`` restores whatever training mode the tagger was in beforehand, so the loop
        continues stepping after a milestone rather than staying in eval mode.

    Raises:
        RuntimeError: If the evaluation did not score every pinned row, which would let a
            milestone record metrics over a partial set. The milestone attestation runs first
            and its refusals arrive before any write. The ``except`` here appends a
            ``milestone_evaluation_error`` event and re-raises, so this and any failure inside
            the evaluation itself reach the caller -- the handler records, it does not absorb.

    """
    milestone = _require_milestone_state(contract, state)
    event_fields = {
        "optimizer_step": state.optimizer_step,
        "packed_cursor": state.packed_cursor,
        "rows": milestone["evaluation_rows"],
    }
    writer.append({"event": "milestone_evaluation_started", **event_fields})
    was_training = bool(getattr(tagger, "training", False))
    try:
        metrics = _final_evaluate(
            tagger,
            tokenizer,
            progress=writer.append,
            progress_event="milestone_evaluation_progress",
        )
        if metrics.get("rows") != milestone["evaluation_rows"]:
            msg = "milestone evaluation did not score all pinned rows"
            # reason: the raise is the point - the except records milestone_evaluation_error and re-raises, so the
            # reason: handler observes this failure rather than absorbing it.
            raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
        writer.atomic_json(f"milestones/step-{state.optimizer_step:08d}/evaluation.json", metrics)
    except Exception as error:
        writer.append(
            {
                "event": "milestone_evaluation_error",
                **event_fields,
                "error": repr(error),
            },
        )
        raise
    finally:
        tagger.train(was_training)
    writer.append(
        {
            "event": "milestone_evaluation_completed",
            **event_fields,
            "metrics": metrics,
        },
    )
    return metrics


# reason: execute full coordinates validate with build tagger; extra seams would split cleanup from writes.
def _execute_full_candidate(  # ruff: ignore[complex-structure,too-many-branches,too-many-locals,too-many-statements]
    contract: Mapping[str, Any],
    *,
    shard_paths: Sequence[str],
    writer: RunWriter,
    resume_checkpoint: str | None = None,
    encoder_checkpoint_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one candidate. Callers must have checked launch gates and inventory.

    Returns:
        The run result already written to ``result.json``: status ``ok``, the candidate, the
        execution id and artifact root, the lifecycle reason training stopped for, the final
        optimizer step and packed cursor, the full evaluation report, the runtime attestation
        and the accrued cost fields. Reaching this point means a terminal checkpoint exists,
        because the loop saves one on exit unless the cap already saved it.

    Raises:
        RuntimeError: If CUDA is unavailable; if ``optimizer_step_cap`` is neither ``None`` nor
            an ``int``; if the loop reaches a stop reason that is not the deadline or the step
            cap, which would mean the lifecycle gained a state this branch does not handle; if
            the classifier returns no loss or a non-finite loss, each checked before
            ``backward`` so a diverged step is never applied; or if a contract that declares a
            milestone reaches final evaluation without having completed it. The step body sits
            inside a ``try`` that catches only out-of-memory, so these propagate through it.
        OutOfMemoryError: Re-raised after the loop writes an ``oom`` event and a
            ``failed_oom`` ``result.json``. The record carries the failing step, the physical
            batch size and whether it was the first step, which is what distinguishes a
            configuration too large to ever run from one that survived and later grew.

    """
    import torch

    candidate = str(contract["candidate_key"])
    batch_size = int(contract["training"]["batch_size"])
    training_deadline_seconds = float(contract["budget"]["training_deadline_seconds"])
    milestone_config = contract["training"].get("milestone")
    milestone_completed = milestone_config is None
    if not torch.cuda.is_available():
        msg = "full H100 run requires CUDA"
        raise RuntimeError(msg)
    started = time.monotonic()
    writer.set_runtime_context(
        phase="runtime_initialization",
        started_monotonic=started,
        candidate=candidate,
        config_digest=str(contract["config_digest"]),
        optimizer_step=0,
        packed_cursor=0,
    )
    writer.append(
        {
            "event": "runtime_initialization_started",
            "candidate": candidate,
            "config_digest": contract["config_digest"],
        },
    )
    label_vocabulary = _validate_runtime_label_vocabulary(contract)
    _persist_label_vocabulary(writer, label_vocabulary)
    tagger, tokenizer, runtime_attestation = _build_tagger(
        candidate,
        contract=contract,
        expected_encoder_checkpoint_attestation=encoder_checkpoint_attestation,
    )
    adamw = _contract_adamw_parameters(contract)
    optimizer = torch.optim.AdamW(
        [p for p in tagger.parameters() if p.requires_grad],
        lr=adamw.lr,
        betas=adamw.betas,
        eps=adamw.eps,
        weight_decay=adamw.weight_decay,
        amsgrad=adamw.amsgrad,
        fused=adamw.fused,
    )
    scheduler_parameters = _contract_scheduler_parameters(contract)
    _require_fresh_scheduler_run(scheduler_parameters, resume_checkpoint=resume_checkpoint)
    scheduler = _build_scheduler(optimizer, scheduler_parameters)
    resumed = _resume_if_requested(
        resume_checkpoint,
        tagger=tagger,
        optimizer=optimizer,
        candidate_key=candidate,
        config_digest=str(contract["config_digest"]),
    )
    cursor, step = (resumed.packed_cursor, resumed.optimizer_step) if resumed else (0, 0)
    initialization_elapsed = time.monotonic() - started
    writer.set_runtime_context(phase="training", optimizer_step=step, packed_cursor=cursor)
    writer.append(
        {
            "event": "runtime_initialization_complete",
            "candidate": candidate,
            "config_digest": contract["config_digest"],
            "runtime_attestation": runtime_attestation,
            **cost_fields(initialization_elapsed, RATE_USD_PER_SECOND),
        },
    )
    writer.append(
        {
            "event": "run_started",
            "candidate": candidate,
            "config_digest": contract["config_digest"],
            "batch_size": batch_size,
            "resume": resumed is not None,
            "packed_cursor": cursor,
            "runtime_attestation": runtime_attestation,
        },
    )
    iterator = _iter_packed_rows(shard_paths)
    for _ in range(cursor):
        next(iterator)
    durations: list[float] = []
    reason: StopReason = "epoch_complete"
    terminal_checkpoint_saved = False
    optimizer_step_cap = contract["training"].get("optimizer_step_cap")
    if optimizer_step_cap is not None and type(optimizer_step_cap) is not int:
        msg = "optimizer step cap must be an integer or null"
        raise RuntimeError(msg)
    sampler = _NvmlSampler()
    sampler.start()
    for rows in _physical_batches(iterator, batch_size):
        predicted = max(durations[-5:] or [5.0])
        elapsed = time.monotonic() - started
        next_reason = next_step_reason(
            elapsed_seconds=elapsed,
            predicted_next_step_seconds=predicted,
            training_deadline_seconds=training_deadline_seconds,
            epoch_complete=False,
            optimizer_step=step,
            optimizer_step_cap=optimizer_step_cap,
        )
        if next_reason != "training":
            reason = next_reason
            if reason == "deadline_reached":
                event_name = "training_deadline_reached"
            elif reason == "optimizer_step_cap_reached":
                event_name = "optimizer_step_cap_reached"
            else:
                msg = f"unexpected pre-step lifecycle reason: {reason}"
                raise RuntimeError(msg)
            writer.append(
                {
                    "event": event_name,
                    "step": step,
                    "packed_cursor": cursor,
                    "training_deadline_seconds": training_deadline_seconds,
                    "optimizer_step_cap": optimizer_step_cap,
                    **cost_fields(elapsed, RATE_USD_PER_SECOND),
                },
            )
            break
        # reason: execute full's try keeps packed batch with apply step; splitting would split cleanup from writes.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            batch, real_tokens = _packed_batch(rows, "cuda")
            torch.cuda.synchronize()
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            output = tagger(**batch)
            loss = output["loss"]
            if loss is None:
                msg = "BIOES classifier returned no loss"
                raise RuntimeError(msg)
            if not torch.isfinite(loss):
                msg = "non-finite BIOES loss"
                raise RuntimeError(msg)
            loss.backward()
            learning_rate_event = _apply_optimizer_step(optimizer, scheduler)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as error:
            elapsed = time.monotonic() - started
            failure = {
                "status": "failed_oom",
                "candidate": candidate,
                "execution_id": writer.root.name,
                "artifact_root": str(writer.root),
                "failed_optimizer_step": step + 1,
                "first_optimizer_step_stability_check": step == 0,
                "batch_size": batch_size,
                "physical_batch_size": len(rows),
                "packed_cursor": cursor,
                "error": repr(error),
                **cost_fields(elapsed, RATE_USD_PER_SECOND),
            }
            writer.append({"event": "oom", **failure})
            writer.atomic_json("result.json", failure)
            sampler.stop()
            raise
        duration = time.monotonic() - before
        durations.append(duration)
        step += 1
        cursor += len(rows)
        elapsed = time.monotonic() - started
        writer.set_runtime_context(optimizer_step=step, packed_cursor=cursor)
        writer.append(
            {
                "event": "optimizer_step",
                "step": step,
                "first_optimizer_step_stability_check": step == 1,
                "physical_batch_size": len(rows),
                "packed_cursor": cursor,
                **(
                    {"packed_order_attestation": _packed_order_attestation(rows, packed_cursor=cursor - len(rows))}
                    if step == 1
                    else {}
                ),
                "real_tokens": real_tokens,
                "real_tokens_per_second": real_tokens / duration,
                "loss": float(loss.detach().float().cpu()),
                **learning_rate_event,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                **_optimizer_progress(
                    cursor=cursor,
                    batch_size=batch_size,
                    durations=durations,
                    elapsed_seconds=elapsed,
                    training_deadline_seconds=training_deadline_seconds,
                ),
                **sampler.latest(),
                **cost_fields(elapsed, RATE_USD_PER_SECOND),
            },
        )
        state = RuntimeResumeState(
            candidate,
            str(contract["config_digest"]),
            str(contract["packed_dataset"]["revision"]),
            str(contract["packed_dataset"]["manifest_sha256"]),
            str(contract["evaluation"]["revision"]),
            cursor,
            step,
        )
        reached_optimizer_step_cap = optimizer_step_cap is not None and step >= optimizer_step_cap
        if step % int(contract["training"]["checkpoint_every_optimizer_steps"]) == 0:
            _save_checkpoint(
                writer,
                tagger=tagger,
                optimizer=optimizer,
                state=state,
                reason=("optimizer_step_cap_reached" if reached_optimizer_step_cap else "training"),
            )
            terminal_checkpoint_saved = reached_optimizer_step_cap
        if reached_optimizer_step_cap:
            reason = "optimizer_step_cap_reached"
            writer.append(
                {
                    "event": "optimizer_step_cap_reached",
                    "step": step,
                    "packed_cursor": cursor,
                    "training_deadline_seconds": training_deadline_seconds,
                    "optimizer_step_cap": optimizer_step_cap,
                    **cost_fields(elapsed, RATE_USD_PER_SECOND),
                },
            )
            break
        if milestone_config is not None and _is_milestone_step(contract, state):
            _require_milestone_state(contract, state)
            writer.append(
                {
                    "event": "milestone_checkpoint_started",
                    "optimizer_step": step,
                    "packed_cursor": cursor,
                    "milestone_attested": True,
                },
            )
            try:
                _save_checkpoint(
                    writer,
                    tagger=tagger,
                    optimizer=optimizer,
                    state=state,
                    reason="training",
                )
            except Exception as error:
                writer.append(
                    {
                        "event": "milestone_checkpoint_error",
                        "optimizer_step": step,
                        "packed_cursor": cursor,
                        "error": repr(error),
                    },
                )
                sampler.stop()
                raise
            writer.append(
                {
                    "event": "milestone_checkpoint_completed",
                    "optimizer_step": step,
                    "packed_cursor": cursor,
                },
            )
            try:
                _evaluate_milestone(contract, state, tagger, tokenizer, writer)
            except Exception:
                sampler.stop()
                raise
            milestone_completed = True
    sampler.stop()
    if not milestone_completed:
        msg = "milestone checkpoint and evaluation did not complete before final evaluation"
        raise RuntimeError(msg)
    writer.set_runtime_context(phase="final_evaluation")
    final_state = RuntimeResumeState(
        candidate,
        str(contract["config_digest"]),
        str(contract["packed_dataset"]["revision"]),
        str(contract["packed_dataset"]["manifest_sha256"]),
        str(contract["evaluation"]["revision"]),
        cursor,
        step,
    )
    if not terminal_checkpoint_saved:
        _save_checkpoint(
            writer,
            tagger=tagger,
            optimizer=optimizer,
            state=final_state,
            reason=reason,
        )
    writer.append(
        {
            "event": "final_evaluation_started",
            "reason": reason,
            "optimizer_step": step,
            "packed_cursor": cursor,
            **cost_fields(time.monotonic() - started, RATE_USD_PER_SECOND),
        },
    )
    evaluation = _final_evaluate(tagger, tokenizer, progress=writer.append)
    writer.atomic_json("final_evaluation.json", evaluation)
    elapsed = time.monotonic() - started
    result = {
        "status": "ok",
        "candidate": candidate,
        "execution_id": writer.root.name,
        "artifact_root": str(writer.root),
        "lifecycle_state": reason,
        "optimizer_steps": step,
        "packed_cursor": cursor,
        "evaluation": evaluation,
        "runtime_attestation": runtime_attestation,
        **cost_fields(elapsed, RATE_USD_PER_SECOND),
    }
    writer.atomic_json("result.json", result)
    writer.append(
        {
            "event": "run_completed",
            **{key: value for key, value in result.items() if key != "evaluation"},
        },
    )
    return result


# reason: execute full exposes contract/encoder as its public contract; bundling would break callers.
def execute_full_candidate(  # ruff: ignore[too-many-arguments]
    contract: Mapping[str, Any],
    *,
    shard_paths: Sequence[str],
    artifact_parent: str,
    commit: Callable[[], None] | None = None,
    resume_checkpoint: str | None = None,
    encoder_checkpoint_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist every terminal failure; retain the OOM result written in-loop.

    Returns:
        The inner run's result dict unchanged. This wrapper adds the artifact root, writes the
        launch contract next to it, and guarantees a ``result.json`` on every terminal path.

    On failure it writes a ``failed`` record carrying the phase, config digest, optimizer step
    and packed cursor from the runtime context, then re-raises rather than swallowing. The
    write is guarded on ``result.json`` not already existing, which is what keeps the in-loop
    ``failed_oom`` record -- with its failing step and physical batch size -- from being
    overwritten by the coarser outer one. Every exception from the inner run reaches the caller.

    """
    candidate = str(contract["candidate_key"])
    writer = RunWriter(Path(artifact_parent) / candidate / uuid4().hex, commit=commit)
    writer.set_runtime_context(
        phase="launch",
        candidate=candidate,
        config_digest=str(contract["config_digest"]),
        optimizer_step=None,
        packed_cursor=None,
    )
    writer.atomic_json("launch_contract.json", contract)
    try:
        return _execute_full_candidate(
            contract,
            shard_paths=shard_paths,
            writer=writer,
            resume_checkpoint=resume_checkpoint,
            encoder_checkpoint_attestation=encoder_checkpoint_attestation,
        )
    except Exception as error:
        result_path = writer.root / "result.json"
        if not result_path.exists():
            context = writer.runtime_context
            started = context.get("started_monotonic")
            elapsed = max(0.0, time.monotonic() - started) if isinstance(started, (int, float)) else 0.0
            failure = {
                "status": "failed",
                "candidate": candidate,
                "execution_id": writer.root.name,
                "artifact_root": str(writer.root),
                "phase": context["phase"],
                "config_digest": context["config_digest"],
                "optimizer_step": context["optimizer_step"],
                "packed_cursor": context["packed_cursor"],
                "error": repr(error),
                **cost_fields(elapsed, RATE_USD_PER_SECOND),
            }
            writer.append({"event": "run_failed", **failure})
            writer.atomic_json("result.json", failure)
        raise
