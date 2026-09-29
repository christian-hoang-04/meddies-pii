#!/usr/bin/env python
"""Build OpenPII general-domain Meddies Labels augmentation candidates."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

# reason: transformers declares its public names only under `TYPE_CHECKING` and serves them at
# reason: runtime through `_LazyModule`, so a static reader cannot prove the symbol is present.
# reason: Verified against the pinned 5.14.1: `hasattr(transformers, "AutoTokenizer")` is True.
from datasets import load_dataset
from transformers import AutoTokenizer  # ty: ignore[possibly-missing-import]

from meddies_pii.training.bioes.data.augmentation import (
    info_value,
    load_current_splits,
    text_hash,
    write_json,
    write_jsonl,
)
from meddies_pii.training.bioes.data.openpii_candidates import (
    DEFAULT_OPENPII_DATASET_ID,
    parse_language_quotas,
    select_openpii_candidates_from_rows,
)
from meddies_pii.training.bioes.data.splits import normalize_text
from meddies_pii.training.bioes.data.tokenizer_security import (
    validate_remote_code_tokenizer_policy,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class _LengthTokenizer(Protocol):
    """The single call shape this script needs to measure token length."""

    def __call__(self, text: str, *, add_special_tokens: bool, truncation: bool) -> Mapping[str, Sequence[int]]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_OPENPII_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--quota",
        action="append",
        help="Language quota in LANG=COUNT form. Defaults to en=4000,de=1500,es=1500,fr=1500,pt=1500.",
    )
    parser.add_argument("--current-dataset-id", default="Meddies/meddies-pii")
    parser.add_argument("--current-config-name", default="pii-bioes")
    parser.add_argument("--tokenizer-model-id", default="LiquidAI/LFM2.5-350M-Base")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-scan", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bioes-v2/candidates/openpii_candidates.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/bioes-v2/candidates/openpii_candidates.summary.json"),
    )
    return parser.parse_args()


def existing_text_hashes_from_rows(rows: Sequence[Mapping[str, object]]) -> set[str]:
    return {text_hash(normalize_text(str(row.get("text") or ""))) for row in rows}


def main() -> None:
    args = parse_args()
    quotas = parse_language_quotas(args.quota)
    current_train, current_validation = load_current_splits(args.current_dataset_id, args.current_config_name)
    existing_ids = {info_value(row, "id", "") for row in [*current_train, *current_validation]}
    existing_text_hashes = existing_text_hashes_from_rows([*current_train, *current_validation])
    validate_remote_code_tokenizer_policy(
        tokenizer_model_id=args.tokenizer_model_id,
        tokenizer_revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    # reason: from_pretrained is declared wide enough to include None, which is not callable; the
    # reason: remote-code policy above is what actually admits this tokenizer.
    tokenizer: _LengthTokenizer = cast(
        "_LengthTokenizer",
        AutoTokenizer.from_pretrained(
            args.tokenizer_model_id,
            revision=args.tokenizer_revision,
            trust_remote_code=args.trust_remote_code,
        ),
    )

    def token_length(record: Mapping[str, Any]) -> int:
        encoded = tokenizer(
            str(record.get("text") or ""),
            add_special_tokens=False,
            truncation=False,
        )
        return len(encoded["input_ids"])

    rows = load_dataset(args.dataset_id, split=args.split, streaming=True)
    selected, summary = select_openpii_candidates_from_rows(
        rows,
        quotas=quotas,
        existing_ids=existing_ids,
        existing_text_hashes=existing_text_hashes,
        dataset_id=args.dataset_id,
        token_length_fn=token_length,
        max_length=args.max_length,
        max_scan=args.max_scan,
    )
    write_jsonl(args.output, selected)
    summary = {
        **summary,
        "output": str(args.output),
        "current_dataset_id": args.current_dataset_id,
        "current_config_name": args.current_config_name,
        "tokenizer_model_id": args.tokenizer_model_id,
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
