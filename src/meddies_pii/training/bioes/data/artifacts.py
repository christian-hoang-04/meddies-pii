"""BIOES probe and artifact builders.

Builds ``ProbeArtifacts`` (tagger + tokenizer + label vocab + index map)
from either a vanilla HuggingFace backbone or an Unsloth-wrapped LFM backbone.
It can also reassemble a historical PEFT checkpoint for evaluation.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack is an optional extra; importing it at module load would make the package unimportable without
# reason: it.
# ruff: file-ignore[print]
# reason: results and progress travel back through the streamed run log, because Modal's large-result blob path is
# reason: unimplemented in this workspace.
import importlib
import json
import os
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast, runtime_checkable

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    IGNORE_INDEX,
    build_bioes_label_space,
    build_label_to_id,
)
from meddies_pii.training.bioes.trainers.config import LORA_TARGET_MODULES

from .tagger import Backbone, HiddenStateTokenTagger

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from transformers import PreTrainedTokenizerBase

DEFAULT_TEXTS: tuple[str, ...] = (
    "John Smith visited Mercy General Hospital on 03/15/1985.",
    "Email john.smith@email.com or call (555) 123-4567.",
)


def _disable_unsloth_statistics_timeout() -> None:
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")


@dataclass(slots=True)
class ProbeOutcome:
    name: str
    success: bool
    model_class: str | None = None
    backbone_class: str | None = None
    hidden_size: int | None = None
    num_labels: int | None = None
    trainable_params: int | None = None
    gradients_seen: int | None = None
    artifact_path: str | None = None
    error: str | None = None


@dataclass(slots=True)
class ProbeArtifacts:
    tagger: HiddenStateTokenTagger
    tokenizer: PreTrainedTokenizerBase
    label_vocab: tuple[str, ...]
    label_to_id: dict[str, int]


class ParameterModule(Protocol):
    def parameters(self) -> Iterator[torch.nn.Parameter]: ...


@runtime_checkable
class ConfiguredBackbone(Backbone, ParameterModule, Protocol):
    config: object

    def to(self, device: str) -> Self: ...


@runtime_checkable
class HiddenSizeConfig(Protocol):
    hidden_size: int


@runtime_checkable
class ConvolutionCacheConfig(Protocol):
    # reason: the name is fixed by the object this Protocol matches, not chosen here. It is the attribute the
    # reason: model config carries, read at :496 through a runtime_checkable isinstance and again by
    # reason: packing_runtime's getattr, so renaming it would make the structural check never match.
    conv_L_cache: object  # ruff: ignore[mixed-case-variable-in-class-scope]


@runtime_checkable
class TokenizerRuntime(Protocol):
    pad_token: str | None
    eos_token: str | None
    unk_token: str | None

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> Mapping[str, Tensor]: ...


@runtime_checkable
class PretrainedFactory(Protocol):
    def from_pretrained(self, model_id: str, **kwargs: object) -> object: ...


@runtime_checkable
class PeftModelFactory(Protocol):
    def from_pretrained(self, model: object, adapter_dir: str) -> object: ...


@runtime_checkable
class UnslothModelApi(Protocol):
    def from_pretrained(self, **kwargs: object) -> tuple[object, object]: ...

    def get_peft_model(self, model: object, **kwargs: object) -> object: ...


@runtime_checkable
class UnslothApi(Protocol):
    FastLanguageModel: UnslothModelApi


@runtime_checkable
class AdapterBackbone(Protocol):
    peft_config: object

    def save_pretrained(self, destination: Path) -> object: ...


@runtime_checkable
class Optimizer(Protocol):
    def zero_grad(self, *, set_to_none: bool) -> None: ...

    def step(self) -> object: ...


def _require_torch() -> None:
    try:
        __import__("torch")
    except ImportError as exc:
        msg = "torch is required for the runtime probe"
        raise ImportError(msg) from exc


def _require_unsloth() -> UnslothModelApi:
    module = importlib.import_module("unsloth")
    if not isinstance(module, UnslothApi):
        msg = "unsloth does not expose FastLanguageModel"
        # reason: the guard reports a missing optional dependency, not a caller passing the wrong type, so TypeError
        # reason: would misdescribe it; the same function raises this type from non-isinstance guards too.
        raise ImportError(msg)  # ruff: ignore[type-check-without-type-error]
    return _as_unsloth_model_api(module.FastLanguageModel)


def load_pretrained(factory: object, model_id: str, kwargs: Mapping[str, object]) -> object:
    if not isinstance(factory, PretrainedFactory):
        msg = "Transformers factory does not expose from_pretrained"
        # reason: the guard reports a missing optional dependency, not a caller passing the wrong type, so TypeError
        # reason: would misdescribe it; the same function raises this type from non-isinstance guards too.
        raise ImportError(msg)  # ruff: ignore[type-check-without-type-error]
    return factory.from_pretrained(model_id, **kwargs)


def _load_peft_model(factory: object, model: object, adapter_dir: str) -> object:
    if not isinstance(factory, PeftModelFactory):
        msg = "PEFT model does not expose from_pretrained"
        # reason: the guard reports a missing optional dependency, not a caller passing the wrong type, so TypeError
        # reason: would misdescribe it; the same function raises this type from non-isinstance guards too.
        raise ImportError(msg)  # ruff: ignore[type-check-without-type-error]
    return factory.from_pretrained(model, adapter_dir)


def _as_unsloth_model_api(candidate: object) -> UnslothModelApi:
    if not isinstance(candidate, UnslothModelApi):
        msg = "unsloth FastLanguageModel does not expose its loaders"
        # reason: the guard reports a missing optional dependency, not a caller passing the wrong type, so TypeError
        # reason: would misdescribe it; the same function raises this type from non-isinstance guards too.
        raise ImportError(msg)  # ruff: ignore[type-check-without-type-error]
    return candidate


def _as_tokenizer(candidate: object) -> PreTrainedTokenizerBase:
    _as_tokenizer_runtime(candidate)
    return cast("PreTrainedTokenizerBase", candidate)


def _as_tokenizer_runtime(candidate: object) -> TokenizerRuntime:
    if not isinstance(candidate, TokenizerRuntime):
        msg = "Loaded tokenizer does not satisfy the tokenizer interface"
        raise TypeError(msg)
    return candidate


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    runtime_tokenizer = _as_tokenizer_runtime(tokenizer)
    if runtime_tokenizer.pad_token is None:
        runtime_tokenizer.pad_token = runtime_tokenizer.eos_token or runtime_tokenizer.unk_token


def _as_backbone(candidate: object) -> ConfiguredBackbone:
    if not isinstance(candidate, ConfiguredBackbone):
        msg = "Loaded model does not satisfy the backbone interface"
        raise TypeError(msg)
    return candidate


def _as_optimizer(candidate: object) -> Optimizer:
    if not isinstance(candidate, Optimizer):
        msg = "AdamW does not satisfy the optimizer interface"
        raise TypeError(msg)
    return candidate


def _hidden_size(backbone: ConfiguredBackbone) -> int:
    if not isinstance(backbone.config, HiddenSizeConfig):
        msg = "Loaded model config does not expose an integer hidden_size"
        raise TypeError(msg)
    return backbone.config.hidden_size


def _device_for(module: ParameterModule) -> str:
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return "cpu"


def infer_floating_dtype(module: ParameterModule) -> torch.dtype:
    _require_torch()
    import torch

    for parameter in module.parameters():
        if torch.is_floating_point(parameter):
            return parameter.dtype
    return torch.float32


def _make_synthetic_batch(
    tokenizer: PreTrainedTokenizerBase,
    label_to_id: Mapping[str, int],
    max_length: int = 64,
    device: str = "cpu",
) -> dict[str, Tensor]:
    _require_torch()
    import torch

    runtime_tokenizer = _as_tokenizer_runtime(tokenizer)
    encoded = runtime_tokenizer(
        list(DEFAULT_TEXTS),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = torch.full(
        encoded["input_ids"].shape,
        label_to_id["O"],
        dtype=torch.long,
        device=device,
    )
    non_o = [label for label in label_to_id if label != "O"]
    for row_index in range(labels.size(0)):
        if labels.size(1) > 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            labels[row_index, 1] = label_to_id[non_o[row_index % len(non_o)]]
        if labels.size(1) > 3:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            labels[row_index, 2] = label_to_id[non_o[(row_index + 1) % len(non_o)]]
    labels = labels.masked_fill(encoded["attention_mask"] == 0, IGNORE_INDEX)
    encoded["labels"] = labels
    return encoded


def _count_trainable_params(module: ParameterModule) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _count_seen_gradients(module: ParameterModule) -> int:
    return sum(1 for parameter in module.parameters() if parameter.requires_grad and parameter.grad is not None)


def _save_classifier_state(tagger: HiddenStateTokenTagger, destination: Path) -> str:
    _require_torch()
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "classifier": tagger.classifier.state_dict(),
        "num_labels": tagger.classifier.out_features,
    }
    target = destination / "classifier.pt"
    torch.save(payload, target)
    reloaded = torch.load(target, map_location="cpu")
    if not isinstance(reloaded, Mapping):
        msg = "Classifier reload did not return a mapping"
        # reason: every guard here reports an environment or contract failure - a missing asset, an unverified
        # reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
        # reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
        # reason: make one failure class signal two exception types.
        raise RuntimeError(msg)  # ruff: ignore[type-check-without-type-error]
    reloaded_mapping = cast("Mapping[object, object]", reloaded)
    if reloaded_mapping.get("num_labels") != tagger.classifier.out_features:
        msg = "Classifier reload sanity check failed"
        raise RuntimeError(msg)
    return str(target)


def _save_backbone_adapter(backbone: object, destination: Path) -> str | None:
    if not isinstance(backbone, AdapterBackbone):
        return None
    adapter_dir = destination / "backbone_adapter"
    backbone.save_pretrained(adapter_dir)
    return str(adapter_dir)


def _run_single_step(tagger: HiddenStateTokenTagger, batch: Mapping[str, Tensor]) -> tuple[float, tuple[int, ...]]:
    _require_torch()
    import torch

    tagger.train()
    optimizer = _as_optimizer(
        torch.optim.AdamW(
            [parameter for parameter in tagger.parameters() if parameter.requires_grad],
            lr=1e-4,
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    result = tagger(**batch)
    loss = result["loss"]
    # reason: narrowing a model output. The tagger was called with labels, so the forward path that omits `loss`
    # reason: is unreachable here, but the output mapping is typed loosely enough to admit None. A raise would
    # reason: add a runtime branch for a case this call cannot produce.
    assert loss is not None  # ruff: ignore[assert]
    loss.backward()
    optimizer.step()
    logits = result["logits"]
    return float(loss.detach().cpu().item()), tuple(int(dimension) for dimension in logits.shape)


# reason: build hf exposes model id/trust code as its public contract; bundling would break callers.
def build_hf_artifacts(  # ruff: ignore[too-many-arguments]
    model_id: str,
    entity_labels: Sequence[str] = ENTITY_LABELS,
    *,
    model_revision: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    trust_remote_code: bool = False,
) -> ProbeArtifacts:
    _require_torch()
    import torch
    import transformers

    label_vocab = build_bioes_label_space(entity_labels)
    label_to_id = build_label_to_id(label_vocab)
    resolved_tokenizer_id = tokenizer_id or model_id
    tokenizer_kwargs: dict[str, object] = {}
    if tokenizer_revision or model_revision:
        tokenizer_kwargs["revision"] = tokenizer_revision or model_revision
    if trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = trust_remote_code
    tokenizer = _as_tokenizer(
        load_pretrained(
            # reason: transformers serves its public names through `_LazyModule` at runtime and
            # reason: declares them only under `TYPE_CHECKING`, so attribute presence is not
            # reason: statically provable. Verified against the pinned 5.14.1 in this
            # reason: environment: `hasattr(transformers, "AutoTokenizer")` is True.
            cast("object", transformers.AutoTokenizer),  # ty: ignore[possibly-missing-attribute]
            resolved_tokenizer_id,
            tokenizer_kwargs,
        ),
    )
    _ensure_pad_token(tokenizer)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, object] = {"dtype": dtype}
    if model_revision:
        model_kwargs["revision"] = model_revision
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = trust_remote_code
    backbone = _as_backbone(
        # reason: transformers serves its public names through `_LazyModule` at runtime and
        # reason: declares them only under `TYPE_CHECKING`, so attribute presence is not
        # reason: statically provable. Verified against the pinned 5.14.1:
        # reason: `hasattr(transformers, "AutoModel")` is True.
        load_pretrained(
            cast("object", transformers.AutoModel),  # ty: ignore[possibly-missing-attribute]
            model_id,
            model_kwargs,
        ),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone.to(device)
    backbone_dtype = infer_floating_dtype(backbone)
    tagger = HiddenStateTokenTagger(
        backbone=backbone,
        hidden_size=_hidden_size(backbone),
        num_labels=len(label_vocab),
        request_hidden_states=False,
        classifier_dtype=backbone_dtype,
    ).to(device)
    return ProbeArtifacts(
        tagger=tagger,
        tokenizer=tokenizer,
        label_vocab=label_vocab,
        label_to_id=label_to_id,
    )


# reason: build unsloth exposes model id/trust code as its public contract; bundling would break callers.
def build_unsloth_artifacts(  # ruff: ignore[too-many-arguments]
    model_id: str,
    entity_labels: Sequence[str] = ENTITY_LABELS,
    *,
    max_seq_length: int = 256,
    lora_rank: int = 4,
    lora_alpha: int | None = None,
    model_revision: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    trust_remote_code: bool = False,
) -> ProbeArtifacts:
    """Build the accepted Unsloth LoRA training surface for BIOES.

    Returns:
        The probe artifacts: the adapted model, its tokenizer, and the BIOES label vocabulary
        with its id mapping, all built under the accepted Unsloth path.

    Raises:
        ValueError: If a tokenizer id is given that differs from the model id. The two must
            name the same checkpoint, because a tokenizer from elsewhere would produce offsets
            the model's own vocabulary does not match, and the mismatch would show up as
            degraded accuracy rather than as an error.

    """
    _require_torch()
    import torch

    _disable_unsloth_statistics_timeout()
    fast_language_model = _require_unsloth()

    label_vocab = build_bioes_label_space(entity_labels)
    label_to_id = build_label_to_id(label_vocab)
    resolved_tokenizer_id = tokenizer_id or model_id
    if resolved_tokenizer_id != model_id:
        msg = "Unsloth artifact builder requires tokenizer_id to match model_id"
        raise ValueError(msg)
    if tokenizer_revision is not None and tokenizer_revision != model_revision:
        msg = "Unsloth artifact builder requires tokenizer_revision to match model_revision"
        raise ValueError(msg)
    unsloth_kwargs: dict[str, object] = {
        "model_name": model_id,
        "max_seq_length": max_seq_length,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "load_in_16bit": True,
        "full_finetuning": False,
        "fast_inference": False,
    }
    if model_revision:
        unsloth_kwargs["revision"] = model_revision
    if trust_remote_code:
        unsloth_kwargs["trust_remote_code"] = trust_remote_code
    loaded_model, loaded_tokenizer = fast_language_model.from_pretrained(**unsloth_kwargs)
    model = _as_backbone(
        fast_language_model.get_peft_model(
            loaded_model,
            r=lora_rank,
            target_modules=list(LORA_TARGET_MODULES),
            lora_alpha=(lora_alpha if lora_alpha is not None else max(8, lora_rank * 2)),
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        ),
    )
    tokenizer = _as_tokenizer(loaded_tokenizer)
    _ensure_pad_token(tokenizer)
    device = "cuda" if torch.cuda.is_available() else _device_for(model)
    backbone_dtype = infer_floating_dtype(model)
    tagger = HiddenStateTokenTagger(
        backbone=model,
        hidden_size=_hidden_size(model),
        num_labels=len(label_vocab),
        request_hidden_states=True,
        classifier_dtype=backbone_dtype,
        packed_segment_isolation=(
            bool(model.config.conv_L_cache) if isinstance(model.config, ConvolutionCacheConfig) else False
        ),
    ).to(device)
    return ProbeArtifacts(
        tagger=tagger,
        tokenizer=tokenizer,
        label_vocab=label_vocab,
        label_to_id=label_to_id,
    )


def build_native_checkpoint_artifacts(model_id: str, adapter_dir: Path, *, model_revision: str) -> ProbeArtifacts:
    """Load a historical CausalLM LoRA checkpoint without Unsloth.

    The saved PEFT adapter configuration is authoritative for its LoRA shape and
    target modules. This is evaluation-only and does not replace training.

    Returns:
        The probe artifacts loaded through native Transformers and PEFT. The LoRA shape and
        target modules come from the checkpoint's own saved adapter configuration rather than
        from current defaults, so a historical checkpoint loads as it was trained even after
        those defaults have moved.

    """
    _require_torch()
    import torch
    import transformers
    from peft import PeftModel

    tokenizer = _as_tokenizer(
        load_pretrained(
            # reason: transformers serves its public names through `_LazyModule` at runtime and
            # reason: declares them only under `TYPE_CHECKING`, so attribute presence is not
            # reason: statically provable. Verified against the pinned 5.14.1 in this
            # reason: environment: `hasattr(transformers, "AutoTokenizer")` is True.
            cast("object", transformers.AutoTokenizer),  # ty: ignore[possibly-missing-attribute]
            model_id,
            {"revision": model_revision},
        ),
    )
    _ensure_pad_token(tokenizer)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = load_pretrained(
        # reason: transformers serves its public names through `_LazyModule` at runtime and
        # reason: declares them only under `TYPE_CHECKING`, so attribute presence is not
        # reason: statically provable. Verified against the pinned 5.14.1:
        # reason: `hasattr(transformers, "AutoModelForCausalLM")` is True.
        cast("object", transformers.AutoModelForCausalLM),  # ty: ignore[possibly-missing-attribute]
        model_id,
        {"revision": model_revision, "dtype": dtype},
    )
    backbone = _as_backbone(_load_peft_model(cast("object", PeftModel), base_model, str(adapter_dir)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone.to(device)
    label_vocab = build_bioes_label_space(ENTITY_LABELS)
    tagger = HiddenStateTokenTagger(
        backbone=backbone,
        hidden_size=_hidden_size(backbone),
        num_labels=len(label_vocab),
        request_hidden_states=True,
        classifier_dtype=infer_floating_dtype(backbone),
    ).to(device)
    return ProbeArtifacts(
        tagger=tagger,
        tokenizer=tokenizer,
        label_vocab=label_vocab,
        label_to_id=build_label_to_id(label_vocab),
    )


def _probe(name: str, builder: Callable[[], ProbeArtifacts], artifact_root: Path) -> ProbeOutcome:
    # reason: probe's try keeps single step with builder; splitting would misattribute row errors.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        artifacts = builder()
        batch = _make_synthetic_batch(
            artifacts.tokenizer,
            artifacts.label_to_id,
            device=_device_for(artifacts.tagger),
        )
        loss, logits_shape = _run_single_step(artifacts.tagger, batch)
        artifact_dir = artifact_root / name
        classifier_path = _save_classifier_state(artifacts.tagger, artifact_dir)
        adapter_path = _save_backbone_adapter(artifacts.tagger.backbone, artifact_dir)
        artifact_summary = f"classifier={classifier_path}"
        if adapter_path is not None:
            artifact_summary += f" | adapter={adapter_path}"
        artifact_summary += f" | loss={loss:.4f} | logits_shape={logits_shape}"
        return ProbeOutcome(
            name=name,
            success=True,
            model_class=artifacts.tagger.__class__.__name__,
            backbone_class=artifacts.tagger.backbone.__class__.__name__,
            hidden_size=artifacts.tagger.classifier.in_features,
            num_labels=artifacts.tagger.classifier.out_features,
            trainable_params=_count_trainable_params(artifacts.tagger),
            gradients_seen=_count_seen_gradients(artifacts.tagger),
            artifact_path=artifact_summary,
        )
    # reason: a probe harness reports the failure it saw; the traceback is captured into the outcome, so
    # reason: breadth IS the contract and narrowing would let an unanticipated probe failure kill the run.
    except Exception as exc:  # ruff: ignore[blind-except]
        return ProbeOutcome(
            name=name,
            success=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def recommendation_for(hf: ProbeOutcome, unsloth: ProbeOutcome) -> str:
    if unsloth.success:
        return (
            "The Unsloth technical wrapper path is viable for one-step training: keep the BIOES head "
            "outside the backbone, "
            "request hidden states from the CausalLM path, and treat this as infrastructure evidence "
            "only — not a task-ready model."
        )
    if hf.success:
        return (
            "The plain Transformers wrapper path works, but the Unsloth path failed. Use Transformers "
            "+ PEFT for BIOES unless we invest in a custom Unsloth integration."
        )
    return "Both probes failed. Fix the baseline HF path before making any stack decision."


def run_all_probes(
    model_id: str = "LiquidAI/LFM2.5-350M-Base",
    # reason: a named scratch root inside the Modal container, not a shared-tmp security hazard. It is a
    # reason: default the caller overrides, and it must be stable rather than randomized so a later probe in
    # reason: the same container finds what an earlier one wrote; mkdtemp would break that handoff.
    artifact_root: str = "/tmp/meddies-pii-bioes-artifacts",  # ruff: ignore[hardcoded-temp-file]
) -> dict[str, object]:
    artifact_path = Path(artifact_root)
    artifact_path.mkdir(parents=True, exist_ok=True)

    unsloth = _probe("unsloth_backbone", lambda: build_unsloth_artifacts(model_id), artifact_path)
    hf = _probe("hf_backbone", lambda: build_hf_artifacts(model_id), artifact_path)

    return {
        "model_id": model_id,
        "entity_labels": list(ENTITY_LABELS),
        "label_vocab_size": len(build_bioes_label_space(ENTITY_LABELS)),
        "hf_backbone": asdict(hf),
        "unsloth_backbone": asdict(unsloth),
        "recommendation": recommendation_for(hf, unsloth),
    }


if __name__ == "__main__":
    print(json.dumps(run_all_probes(), indent=2))
