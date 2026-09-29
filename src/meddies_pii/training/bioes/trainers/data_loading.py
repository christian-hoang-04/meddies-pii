from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack is an optional extra; importing it at module load would make the package unimportable without
# reason: it.
# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import load_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.training.bioes.eval.selection import TargetedSliceSelection

    from .config import SmokeTrainingConfig


# reason: load rows keeps config name/revision at its adapter seam; bundling would hide required inputs.
def _load_rows(  # ruff: ignore[too-many-arguments]
    config_name: str,
    limit: int,
    scan_multiplier: int = 8,
    *,
    dataset_id: str = "Meddies/meddies-pii",
    split: str = "train",
    revision: str | None = None,
) -> list[dict[str, Any]]:
    if scan_multiplier < 1:
        msg = "scan_multiplier must be >= 1"
        raise ValueError(msg)
    dataset_path = Path(dataset_id)
    if dataset_path.is_file():
        local_rows: list[dict[str, Any]] = []
        max_rows = limit * scan_multiplier
        with dataset_path.open(encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if idx >= max_rows:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    msg = f"local JSONL row {idx} in {dataset_path} is not an object"
                    raise ValueError(msg)
                row.setdefault("_dataset_index", idx)
                row.setdefault("uid", f"{dataset_path}:{idx}")
                local_rows.append(row)
        return local_rows
    ds = load_dataset(dataset_id, config_name, split=split, revision=revision)
    hf_rows: list[dict[str, Any]] = []
    for idx in range(min(limit * scan_multiplier, len(ds))):
        row = dict(ds[idx])
        row.setdefault("_dataset_index", idx)
        row.setdefault("uid", f"{config_name}:{split}:{idx}")
        hf_rows.append(row)
    return hf_rows


def _load_targeted_eval_selection(
    config: SmokeTrainingConfig,
    *,
    # reason: the tokenizer arrives from `AutoTokenizer.from_pretrained`, which the pinned transformers declares as
    # reason: `Unknown | TokenizersBackend | None | SentencePieceBackend`. There is no single honest static type to
    # reason: write, and spelling the union out would add an `Unknown` arm and a `None` the pin never returns.
    tokenizer: Any,  # ruff: ignore[any-type]
    id_to_label: dict[int, str],
    entity_labels: Sequence[str],
) -> TargetedSliceSelection | None:
    if config.target_eval_slice is None:
        return None
    if config.target_eval_limit <= 0:
        msg = "target_eval_limit must be positive when target_eval_slice is set"
        raise ValueError(msg)
    if config.target_eval_scan_multiplier < 1:
        msg = "target_eval_scan_multiplier must be >= 1"
        raise ValueError(msg)

    from meddies_pii.training.bioes.eval.selection import (
        at_dot_obfuscate_email_rows,
        select_prepared_rows_for_adversarial_slice,
    )

    raw_rows = _load_rows(
        config.eval_config,
        config.target_eval_limit,
        config.target_eval_scan_multiplier,
        dataset_id=config.resolved_eval_dataset_id(),
        split=config.eval_dataset_split or config.dataset_split,
        revision=config.resolved_eval_revision(),
    )
    candidate_rows = raw_rows
    if config.target_eval_slice == "at_dot_obfuscation":
        candidate_rows = at_dot_obfuscate_email_rows(raw_rows, limit=config.target_eval_limit)

    selection = select_prepared_rows_for_adversarial_slice(
        candidate_rows,
        tokenizer,
        slice_name=config.target_eval_slice,
        max_length=config.max_length,
        limit=config.target_eval_limit,
        id_to_label=id_to_label,
        entity_labels=entity_labels,
        allow_label_repairs=config.allow_label_repairs,
        require_label_json=config.require_label_json,
    )
    if config.target_eval_require_support and not selection.prepared_rows:
        msg = (
            "target_eval_slice="
            f"{config.target_eval_slice!r} produced 0 prepared rows; "
            "increase target_eval_limit/target_eval_scan_multiplier or disable "
            "target_eval_require_support"
        )
        raise RuntimeError(
            msg,
        )
    return selection
