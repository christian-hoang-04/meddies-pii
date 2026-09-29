#!/usr/bin/env python
"""Launch the LFM2.5-230M BIOES fine-tune on ``meddies-pii-mixed``.

Thin call site over the existing ``run_smoke_training`` — all training logic lives in
``src/meddies_pii/training/bioes/``; this file only supplies the settled configuration
for the settled 230M recipe documented in issue #78:
https://github.com/meddies-ai/meddies-pii/issues/78

  * model     LiquidAI/LFM2.5-230M-Base      (base tokenizer loads without remote code)
  * train     Meddies/meddies-pii-mixed / default / train   (1,000,000 rows)
  * eval      Meddies/meddies-pii-v2 / eval / train         (1,700 rows, proven disjoint)
  * max_len   8192   (drops ~0.044% of the corpus; protects high-fertility languages)
  * backend   unsloth, fused_adamw=False, no custom kernel  (issue #78 settled defaults)

Cross-repo eval (train on mixed, eval on v2) uses the ``eval_dataset_id`` field.

Compute note: ``--backend unsloth`` needs a CUDA GPU (real runs go to Modal H100). For a
LOCAL wiring smoke on CPU use ``--backend hf`` with tiny limits — it exercises the full
config -> tokenize -> BIOES head -> loss -> eval path without GPU kernels, just slowly.
Prints the result JSON; writes it to ``--out`` if given. Never pushes anything.

Packing needs the unsloth backend (validate_smoke_training_config enforces this); a local --backend hf smoke can't pack, so
disable it rather than crash.

"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.training.bioes.trainers.trainer import (
    SmokeTrainingConfig,
    run_smoke_training,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ISSUE_78_URL = "https://github.com/meddies-ai/meddies-pii/issues/78"
MODEL_ID = "LiquidAI/LFM2.5-230M-Base"
TRAIN_REPO = "Meddies/meddies-pii-mixed"
EVAL_REPO = "Meddies/meddies-pii-v2"


def build_config(args: argparse.Namespace) -> SmokeTrainingConfig:
    """v2 'eval' config stores its rows under a 'train' split."""
    packing = args.packing
    if packing and args.backend != "unsloth":
        print("note: packing requires the unsloth backend; disabling for backend=hf")
        packing = False
    return SmokeTrainingConfig(
        backend=args.backend,
        model_id=MODEL_ID,
        model_revision=args.model_revision,
        dataset_id=TRAIN_REPO,
        train_config="default",
        dataset_split="train",
        dataset_revision=args.train_revision,
        eval_dataset_id=EVAL_REPO,
        eval_config="eval",
        eval_dataset_split="train",
        eval_dataset_revision=args.eval_revision,
        max_length=args.max_length,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        steps=args.steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=1e-4,
        packing=packing,
        checkpoint_every_steps=args.checkpoint_every,
        fused_adamw=False,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Defaults mirror coworker's proven production recipe.

    Bioes/20260604_h100_8192_r128a256_pack_bs128_150step_ckpt10), swapping only model 350M->230M and data
    pii-bioes->meddies-pii-mixed. Override for a local smoke: --backend hf --no-packing --batch-size 2 --steps 2
    --train-limit 8.

    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        default="unsloth",
        choices=("unsloth", "hf"),
        help="unsloth (GPU, real recipe) or hf (CPU-capable local wiring smoke).",
    )
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument(
        "--train-limit",
        type=int,
        default=173652,
        help="Row pool loaded before quality filtering (coworker's value; "
        "mixed has 1,000,000 rows, so raise to use more).",
    )
    p.add_argument("--eval-limit", type=int, default=1700, help="Full v2 eval is 1,700 rows.")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument(
        "--lora-rank",
        type=int,
        default=128,
        help="Issue #78 recipe binds LoRA rank 128; override only for a deliberate different run.",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=256,
        help="Issue #78 recipe binds alpha 256; override only for a deliberate different run.",
    )
    p.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Greedy 8192-token packing (unsloth only). --no-packing to disable.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save a checkpoint every N steps (0 = only final).",
    )
    p.add_argument("--model-revision", default=None)
    p.add_argument("--train-revision", default=None)
    p.add_argument("--eval-revision", default=None)
    p.add_argument("--out", default=None, help="Write result JSON to this path.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    print(
        f"train {TRAIN_REPO}/default  eval {EVAL_REPO}/eval  model {MODEL_ID}\n"
        f"backend={config.backend} max_length={config.max_length} "
        f"lora_rank={config.lora_rank} lora_alpha={config.lora_alpha or 'auto(256)'} "
        f"packing={config.packing} batch_size={config.batch_size} "
        f"steps={config.steps} ckpt_every={config.checkpoint_every_steps} "
        f"train_limit={config.train_limit} eval_limit={config.eval_limit}",
        flush=True,
    )
    result = run_smoke_training(config)
    rendered = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(f"result written to {args.out}")
    print(
        "eval_exact_span_f1=",
        result.get("eval_exact_span_f1"),
        " train_loss_delta=",
        result.get("train_loss_delta"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
