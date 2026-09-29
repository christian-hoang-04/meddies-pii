from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .contamination import run_packed_attention_contamination_probe
from .packing import PackedTrainingUnit, pack_prepared_rows, packing_utilization

if TYPE_CHECKING:
    import torch

    from meddies_pii.training.bioes.data.preparation import PreparedRow

    from .config import SmokeTrainingConfig


MIN_PACKING_PROBE_ROWS = 2


@dataclass(frozen=True, slots=True)
class PreparedTrainingUnits:
    units: Sequence[PreparedRow | PackedTrainingUnit]
    packed_rows: list[PackedTrainingUnit] | None
    utilization: float | None
    boundary_token_count: int | None
    attention_probe: dict[str, Any] | None


# reason: the tokenizer arrives from `AutoTokenizer.from_pretrained`, which the pinned transformers declares as
# reason: `Unknown | TokenizersBackend | None | SentencePieceBackend`. There is no single honest static type to write,
# reason: and spelling the union out would add an `Unknown` arm and a `None` the pinned version never returns.
def _resolve_pad_token_id(tokenizer: Any) -> int:  # ruff: ignore[any-type]
    token_id = getattr(tokenizer, "pad_token_id", None)
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is None:
        return 0
    return int(token_id)


def _packing_boundary_token_count_for_model(model: torch.nn.Module) -> int:
    if getattr(model, "packed_segment_isolation", False):
        return 0
    config = getattr(getattr(model, "backbone", model), "config", None)
    conv_cache = getattr(config, "conv_L_cache", None)
    if isinstance(conv_cache, int) and conv_cache > 1:
        layer_types = getattr(config, "layer_types", None)
        if isinstance(layer_types, Sequence):
            conv_layers = sum(1 for layer_type in layer_types if layer_type != "full_attention")
            return (conv_cache - 1) * max(1, conv_layers)
        return conv_cache - 1
    return 0


def _mutate_row_input_ids_same_length(row: PreparedRow, *, replacement_token_id: int) -> PreparedRow:
    mutated_ids = [replacement_token_id for _token_id in row.tokenized.input_ids]
    tokenized = replace(row.tokenized, input_ids=mutated_ids)
    return replace(row, tokenized=tokenized)


# reason: prepare training exposes config/log as its public contract; bundling would break callers.
def prepare_training_units(  # ruff: ignore[too-many-arguments]
    config: SmokeTrainingConfig,
    *,
    model: torch.nn.Module,
    # reason: the tokenizer arrives from `AutoTokenizer.from_pretrained`, which the pinned transformers declares as
    # reason: `Unknown | TokenizersBackend | None | SentencePieceBackend`. There is no single honest static type to
    # reason: write, and spelling the union out would add an `Unknown` arm and a `None` the pin never returns.
    tokenizer: Any,  # ruff: ignore[any-type]
    train_rows: Sequence[PreparedRow],
    device: str,
    log: Callable[[str], None],
) -> PreparedTrainingUnits:
    if not config.packing:
        return PreparedTrainingUnits(
            units=train_rows,
            packed_rows=None,
            utilization=None,
            boundary_token_count=None,
            attention_probe=None,
        )

    pad_token_id = _resolve_pad_token_id(tokenizer)
    boundary_token_count = _packing_boundary_token_count_for_model(model)
    packed_rows = pack_prepared_rows(
        train_rows,
        max_length=config.max_length,
        pad_token_id=pad_token_id,
        boundary_token_count=boundary_token_count,
    )
    if not packed_rows:
        msg = "packing=True produced 0 packed training units"
        raise RuntimeError(msg)
    utilization = packing_utilization(packed_rows)
    log(
        "packing prepared "
        f"packed_units={len(packed_rows)} "
        f"source_examples={len(train_rows)} "
        f"boundary_tokens={boundary_token_count} "
        f"utilization={utilization:.6f}",
    )
    if len(train_rows) < MIN_PACKING_PROBE_ROWS:
        msg = "packing=True requires at least 2 prepared train rows for the attention contamination probe"
        raise RuntimeError(msg)
    probe_prefix, probe_target = train_rows[:2]
    clean_probe_unit = pack_prepared_rows(
        [probe_prefix, probe_target],
        max_length=config.max_length,
        pad_token_id=pad_token_id,
        boundary_token_count=boundary_token_count,
    )[0]
    mutated_probe_unit = pack_prepared_rows(
        [
            _mutate_row_input_ids_same_length(probe_prefix, replacement_token_id=pad_token_id),
            probe_target,
        ],
        max_length=config.max_length,
        pad_token_id=pad_token_id,
        boundary_token_count=boundary_token_count,
    )[0]
    log(f"packing_attention_probe start prefix_uid={probe_prefix.uid} target_uid={probe_target.uid}")
    attention_probe = run_packed_attention_contamination_probe(
        model,
        clean_unit=clean_probe_unit,
        mutated_unit=mutated_probe_unit,
        target_row_uid=probe_target.uid,
        device=device,
        atol=0.0,
        rtol=0.0,
    ).to_dict()
    log(
        "packing_attention_probe done "
        f"passed={json.dumps(attention_probe['passed'])} "
        f"max_abs_diff={attention_probe['max_abs_diff']}",
    )
    if not attention_probe["passed"]:
        msg = f"Packed attention contamination probe failed: max_abs_diff={attention_probe['max_abs_diff']}"
        raise RuntimeError(msg)
    return PreparedTrainingUnits(
        units=packed_rows,
        packed_rows=packed_rows,
        utilization=utilization,
        boundary_token_count=boundary_token_count,
        attention_probe=attention_probe,
    )
