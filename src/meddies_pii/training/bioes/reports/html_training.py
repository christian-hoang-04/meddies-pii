"""Render attached training configuration and the reproducible Modal command."""

from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: the multiplication sign is typography meaning "by" in an operator-facing report label
# reason: (source x language x bucket, 2xA100); the ASCII letter x would misrender the heading.
import json
from html import escape
from typing import TYPE_CHECKING, Any

from meddies_pii.training.bioes.trainers.config import LORA_TARGET_MODULES

from .json_narrowing import JsonMap, int_like, map_at
from .models import DEFAULT_MODAL_PROFILE, _planned_training_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


PLANNED_BATCH_SIZE = 128
PLANNED_LORA_RANK = 32
PLANNED_LORA_ALPHA = 64


# reason: Planned stack, installed versions, and module map produce one reproducibility table.
def _training_stack_html(modal: Mapping[str, Any], modal_summary: Mapping[str, object]) -> str:  # ruff: ignore[too-many-locals]
    result = map_at(modal, "result")
    attached_config = map_at(result, "config")
    config = dict(attached_config) if attached_config else _planned_training_config()
    versions = map_at(result, "package_versions")
    model_id = str(modal_summary.get("model_id") or config.get("model_id"))
    dataset_id = str(modal_summary.get("dataset_id") or config.get("dataset_id"))
    backend = str(modal_summary.get("backend") or config.get("backend"))
    torch_version = str(versions.get("torch") or "n/a")
    transformers_version = str(versions.get("transformers") or "n/a")
    datasets_version = str(versions.get("datasets") or "n/a")
    unsloth_version = str(versions.get("unsloth") or "n/a")
    unsloth_zoo_version = str(versions.get("unsloth_zoo") or "n/a")
    batch_size = int_like(config.get("batch_size") or modal_summary.get("batch_size") or 128)
    grad_accum = int_like(config.get("gradient_accumulation_steps") or 1)
    packing = bool(config.get("packing"))
    probe = map_at(result, "packing_attention_probe")
    if packing and probe and probe.get("passed") is True:
        packing_text = (
            "enabled with BIOES packed metadata; contamination probe passed "
            f"(<code>max_abs_diff={escape(str(probe.get('max_abs_diff')))}</code>). "
            "LFM2 conv models use segment-isolated packed fallback, not native full-forward packing."
        )
    elif packing and not attached_config:
        packing_text = (
            "planned, but no all-language Modal contamination probe is attached yet; "
            "smoke the current remote config before the real run."
        )
    elif packing:
        packing_text = (
            "requested, but no passing contamination probe is attached; do not launch a full run from this report alone."
        )
    else:
        packing_text = "disabled; rows are padded normally."
    return f"""<div class="stack">
      <div><b>Primary model</b><br><code>{escape(model_id)}</code></div>
      <div><b>Dataset</b><br><code>{escape(dataset_id)}</code> / \
<code>{escape(str(modal_summary.get("train_config") or config.get("train_config") or "pii-bioes"))}</code></div>
      <div><b>Train/eval \
split</b><br><code>{
        escape(str(modal_summary.get("dataset_split") or config.get("dataset_split") or "train"))
    }</code> → <code>{
        escape(str(modal_summary.get("eval_dataset_split") or config.get("eval_dataset_split") or "validation"))
    }</code></div>
      <div><b>Backend</b><br><code>{escape(backend)}</code> token-classification smoke path</div>
      <div><b>Batch decision</b><br><code>{escape(str(batch_size))} physical</code>; \
<code>grad_accum={escape(str(grad_accum))}</code>; {packing_text}</div>
      <div><b>Head</b><br>37-class BIOES token classifier: <code>O + 9 labels × B/I/E/S</code></div>
      <div><b>Decoder</b><br>constrained BIOES/Viterbi-compatible span decoding path</div>
      <div><b>Runtime</b><br><code>torch {escape(torch_version)}</code>, <code>transformers \
{escape(transformers_version)}</code>, <code>datasets {escape(datasets_version)}</code>, \
<code>unsloth {escape(unsloth_version)}</code>, <code>unsloth_zoo \
{escape(unsloth_zoo_version)}</code></div>
      <div><b>Primary eval metric</b><br>typed exact-span precision/recall/F1 plus containment span report</div>
      <div><b>Challenger baseline</b><br><code>fastino/gliner2-multi-v1</code> against the same validation split</div>
    </div>
"""


def _result_and_config(modal: Mapping[str, Any]) -> tuple[JsonMap, JsonMap]:
    result = map_at(modal, "result")
    config = map_at(result, "config")
    return result, config


def _config_table(rows: Sequence[tuple[str, object]]) -> str:
    html_rows = ["<tr><th>config</th><th>value</th></tr>"]
    for key, value in rows:
        if isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = "n/a" if value is None else str(value)
        html_rows.append(f"<tr><td><code>{escape(key)}</code></td><td><code>{escape(rendered)}</code></td></tr>")
    return '<table class="dense">' + "".join(html_rows) + "</table>"


# reason: launch decision owns and config and config table together; splitting would duplicate totals or escaping.
def _launch_decision_config_html(modal: Mapping[str, Any], *, dataset_summary: Mapping[str, Any] | None = None) -> str:  # ruff: ignore[too-many-locals]
    result, attached_config = _result_and_config(modal)
    dataset_summary = dataset_summary or {}
    all_rows_summary = map_at(dataset_summary, "all_rows_summary")
    max_length_filter = map_at(all_rows_summary, "max_length_filter")
    max_length = attached_config.get("max_length") or max_length_filter.get("max_length") or 4096
    config = dict(attached_config) if attached_config else _planned_training_config(max_length=max_length)
    batch_size = int_like(config.get("batch_size") or 128)
    grad_accum = int_like(config.get("gradient_accumulation_steps") or 1)
    packing = bool(config.get("packing"))
    length_bucketing = bool(config.get("length_bucketing"))
    train_packed_examples = result.get("train_packed_examples")
    train_packing_utilization = result.get("train_packing_utilization")
    packing_boundary_tokens = result.get("train_packing_boundary_token_count")
    packing_probe = map_at(result, "packing_attention_probe")
    packing_probe_passed = packing_probe.get("passed")
    if packing and packing_probe_passed is True:
        packing_status = "enabled_contamination_probe_passed"
        native_unsloth_packing = "not_for_lfm2_full_forward; using segment_isolated_fallback"
        recommended_batching = (
            "Correctness-gated packing is safe for smoke/full launch; benchmark "
            "throughput before assuming it is faster, because LFM2 uses the "
            "segment-isolated fallback."
        )
    elif packing and not attached_config:
        packing_status = "planned_needs_all_language_modal_probe"
        native_unsloth_packing = "not_for_lfm2_full_forward; planned segment_isolated_fallback"
        recommended_batching = (
            "Planned full-run shape is batch 128 with packing, but do not launch "
            "until the current remote all-language config passes the Modal "
            "contamination probe."
        )
    elif packing:
        packing_status = "requested_without_passing_probe_do_not_launch"
        native_unsloth_packing = "unsafe_or_unverified"
        recommended_batching = "Do not launch full packed training until the contamination probe passes."
    else:
        packing_status = "disabled"
        native_unsloth_packing = "not_used"
        recommended_batching = "Use padded rows; increase effective batch with grad accumulation if needed."
    train_examples = result.get("train_examples")
    avg_source_rows_per_pack: float | None = None
    if isinstance(train_examples, int) and isinstance(train_packed_examples, int) and train_packed_examples > 0:
        avg_source_rows_per_pack = train_examples / train_packed_examples
    batch_128_status = (
        "passed"
        if (
            result
            and batch_size == PLANNED_BATCH_SIZE
            and int_like(config.get("lora_rank") or 0) == PLANNED_LORA_RANK
            and int_like(config.get("lora_alpha") or 0) == PLANNED_LORA_ALPHA
            and packing
            and packing_probe_passed is True
        )
        else f"not_attached_for_config_batch_{batch_size}"
        if result
        else "planned_needs_all_language_smoke"
    )
    rows = [
        ("chosen_max_length", max_length),
        ("drop_rows_over_max_length", True),
        (
            "rows_over_4096_dropped_before_split",
            max_length_filter.get("dropped_rows", 1184),
        ),
        (
            "rows_over_4096_kept_after_filter",
            0 if max_length_filter else "not_attached",
        ),
        ("physical_batch_size_packed_or_rows", batch_size),
        ("gradient_accumulation_steps", grad_accum),
        ("effective_batch_size_packed_or_rows", batch_size * grad_accum),
        ("packing_status", packing_status),
        (
            "length_bucketing_status",
            "disabled_until_upstream_sampler_or_measured_benefit" if length_bucketing else "disabled",
        ),
        ("native_unsloth_packing_for_bioes", native_unsloth_packing),
        ("packing_attention_probe_passed", packing_probe_passed),
        (
            "packing_attention_probe_max_abs_diff",
            packing_probe.get("max_abs_diff"),
        ),
        (
            "packing_attention_probe_compared_tokens",
            packing_probe.get("compared_tokens"),
        ),
        ("train_packing_boundary_token_count", packing_boundary_tokens),
        ("train_packed_examples", train_packed_examples),
        ("train_packing_utilization", train_packing_utilization),
        ("avg_source_rows_per_pack_in_smoke", avg_source_rows_per_pack),
        ("batch_128_r32_alpha64_packed_status", batch_128_status),
        ("batch_256_unpacked_accum1_status", "failed_h100_oom"),
        (
            "batch_256_unpacked_accum1_failure",
            "tried to allocate 17.25 GiB with 16.21 GiB free; failure in LFM2 FFN/LoRA forward",
        ),
        ("recommended_next_batching", recommended_batching),
        ("train_examples_in_smoke", result.get("train_examples")),
        ("eval_examples_in_smoke", result.get("eval_examples")),
        ("first_train_loss", result.get("first_train_loss")),
        ("final_train_loss", result.get("final_train_loss")),
        ("train_loss_delta", result.get("train_loss_delta")),
    ]
    return _config_table(rows)


def _adapter_config_html(modal: Mapping[str, Any]) -> str:
    _result, attached_config = _result_and_config(modal)
    config = dict(attached_config) if attached_config else _planned_training_config()
    backend = str(config.get("backend") or "unsloth")
    if backend != "unsloth":
        return '<p class="empty">Attached smoke is not an Unsloth LoRA run; no LoRA adapter config applies.</p>'
    lora_rank = int_like(config.get("lora_rank") or 4)
    lora_alpha = int_like(config.get("lora_alpha") or max(8, lora_rank * 2))
    rows = [
        ("lora_rank_r", lora_rank),
        ("lora_alpha", lora_alpha),
        ("lora_dropout", 0),
        ("lora_bias", "none"),
        ("lora_target_modules", list(LORA_TARGET_MODULES)),
        ("use_gradient_checkpointing", "unsloth"),
        ("random_state", 3407),
        ("load_in_4bit", False),
        ("load_in_8bit", False),
        ("load_in_16bit", True),
        ("full_finetuning", False),
        ("fast_inference", False),
        ("request_hidden_states", True),
        ("causal_lm_logits_to_keep", 1),
        ("classifier_head", "linear hidden_size -> 37 BIOES labels"),
        ("classifier_dropout", 0.1),
    ]
    return _config_table(rows)


def _hyperparameter_table(modal: Mapping[str, Any]) -> str:
    _result, attached_config = _result_and_config(modal)
    config = dict(attached_config) if attached_config else _planned_training_config()
    priority_keys = [
        "config_source",
        "model_id",
        "backend",
        "dataset_id",
        "dataset_split",
        "train_config",
        "eval_config",
        "eval_dataset_split",
        "max_length",
        "batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "packing",
        "length_bucketing",
        "lora_rank",
        "lora_alpha",
        "steps",
        "learning_rate",
        "train_limit",
        "eval_limit",
        "target_eval_limit",
        "target_eval_scan_multiplier",
        "fused_adamw",
        "trust_remote_code",
        "allow_label_repairs",
        "require_label_json",
        "viterbi_transition_biases",
        "viterbi_calibration_path",
    ]
    keys = priority_keys + sorted(set(config) - set(priority_keys))
    rows: list[tuple[str, object]] = [(key, config.get(key)) for key in keys]
    return _config_table(rows)


def _complete_config_json_html(modal: Mapping[str, Any]) -> str:
    _result, attached_config = _result_and_config(modal)
    config = dict(attached_config) if attached_config else {}
    provenance = map_at(modal, "provenance")
    command_kwargs = map_at(provenance, "command_kwargs")
    if not config and not command_kwargs:
        planned_payload = {
            "planned_config": _planned_training_config(),
            "note": (
                "No Modal smoke JSON is attached for this rebuilt all-language split; "
                "this is the intended launch shape pending Modal smoke."
            ),
        }
        return (
            "<pre><code>"
            + escape(json.dumps(planned_payload, ensure_ascii=False, indent=2, sort_keys=True))
            + "</code></pre>"
        )
    payload = {
        "smoke_config": dict(config),
        "command_kwargs": dict(command_kwargs),
    }
    return f"<pre>{escape(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))}</pre>"


# reason: modal command orders map at before replace; helper seams would duplicate totals or escaping.
def _modal_command_html(modal: Mapping[str, Any]) -> str:  # ruff: ignore[complex-structure,too-many-branches]
    provenance = map_at(modal, "provenance")
    result = map_at(modal, "result")
    config = map_at(result, "config")
    command_kwargs = map_at(provenance, "command_kwargs")
    command: dict[str, object] = dict(command_kwargs)
    config_fallback_keys = [
        "backend",
        "steps",
        "epochs",
        "logging_steps",
        "max_length",
        "train_limit",
        "eval_limit",
        "batch_size",
        "gradient_accumulation_steps",
        "lora_rank",
        "lora_alpha",
        "packing",
        "length_bucketing",
        "viterbi_calibration_path",
        "target_eval_slice",
        "target_eval_limit",
        "target_eval_scan_multiplier",
        "dataset_id",
        "dataset_split",
        "eval_dataset_split",
        "dataset_revision",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "train_config",
        "eval_config",
        "trust_remote_code",
        "artifact_root",
    ]
    for key in config_fallback_keys:
        if key not in command and config.get(key) is not None:
            command[key] = config[key]
    if "out" not in command and provenance.get("out"):
        command["out"] = provenance["out"]
    if "raw_log" not in command and provenance.get("raw_log"):
        command["raw_log"] = provenance["raw_log"]
    if not command:
        return '<p class="empty">No command kwargs attached.</p>'
    ordered = [
        "gpu",
        "steps",
        "epochs",
        "logging_steps",
        "max_length",
        "train_limit",
        "eval_limit",
        "batch_size",
        "gradient_accumulation_steps",
        "lora_rank",
        "lora_alpha",
        "packing",
        "length_bucketing",
        "train_config",
        "eval_config",
        "eval_dataset_split",
        "dataset_split",
        "dataset_id",
        "model_id",
        "backend",
        "trust_remote_code",
        "artifact_root",
        "raw_log",
        "out",
    ]
    modal_profile = str(provenance.get("modal_profile") or DEFAULT_MODAL_PROFILE)
    parts = [f"MODAL_PROFILE={modal_profile} uv run modal run src/meddies_pii/training/bioes/modal/train.py"]
    for key in ordered:
        if key not in command:
            continue
        value = command[key]
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        else:
            parts.append(f"{flag} {value}")
    for key, value in command.items():
        if key in ordered or value is None:
            continue
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        else:
            parts.append(f"{flag} {value}")
    return f"<pre>{escape(' \\\\\n  '.join(parts))}</pre>"
