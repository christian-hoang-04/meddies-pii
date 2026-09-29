# Meddies PII training surfaces

This package has one active training path. Historical SFT/GRPO code remains in Git history.

## Current default: Meddies Labels BIOES

`meddies_pii.training.bioes` is the current token-classification path for the next PII detector.

- **Taxonomy:** Meddies Labels (`address`, `company_name`, `date`, `email_address`, `human_name`, `id_number`, `phone_number`, `private_url`, `secret`).
- **Dataset config:** `Meddies/meddies-pii` / `pii-bioes`.
- **Record shape:** raw `text` plus `label` character spans (`category`, `start`, `end`, `text`).
- **Model target:** BIOES token labels over the nine span classes plus `O`.
- **Launch policy:** LFM2.5 tokenizer max length 4096; rows over max length are dropped before training.
- **Backend:** Unsloth LoRA is the accepted training backend. A native Transformers
  plus PEFT loader evaluates historical checkpoints only after span parity passes.

Key modules:

| Module | Role |
| --- | --- |
| `bioes/schema.py` | Meddies Labels/BIOES output schema and span record parsing. |
| `bioes/data/legacy_to_pii_labels.py` | Legacy Meddies row migration into high-confidence Meddies Labels spans. |
| `bioes/data/build_legacy_pii_label_corpus.py` | Package-backed builder for Meddies-language source pulls. |
| `bioes/data/mixed.py` | General-domain external dataset adapters into the Meddies Labels span contract. |
| `bioes/data/mixed_build.py` | Package-backed builder for medical/general mixture policy. |
| `bioes/data/splits.py` | No-leakage train/validation split policy. |
| `bioes/data/augmentation.py` | Merge targeted generated rows into the published `pii-bioes` split and emit HF-ready parquet/report artifacts. |
| `bioes/data/preparation.py` | Character-span to token-label alignment for model input. |
| `bioes/data/tagger.py` | LFM2.5 hidden-state token tagger. |
| `bioes/trainers/config.py` | Training config/result dataclasses and validation. |
| `bioes/trainers/packing.py` | BIOES packing utilities for the accepted Unsloth training path. |
| `bioes/trainers/trainer.py` | Unsloth BIOES smoke/full-run orchestration. |
| `bioes/report.py` | Public report facade; scanning/highlighting/rendering live under `bioes/reports/`. |

Common commands:

```bash
# Build/update the static training readiness report
uv run python scripts/reports/build_training_report.py \
  --train-jsonl <train.jsonl> \
  --validation-jsonl <validation.jsonl> \
  --output <training_readiness_report.html>

# Merge targeted generated rows into a pii-bioes artifact bundle
uv run python scripts/migrations/merge_pii_bioes_augmentation.py \
  --augmentation-file <accepted.vi.jsonl> \
  --augmentation-file <accepted.en.jsonl> \
  --augmentation-file <repaired_candidates.clean.vi.jsonl> \
  --augmentation-file <repaired_candidates.clean.en.jsonl>

# Modal smoke against the Hub config
MODAL_PROFILE=openmedical uv run modal run \
  src/meddies_pii/training/bioes/modal/train.py \
  --backend unsloth --gpu A10G --steps 5 --max-length 4096 \
  --train-config pii-bioes --eval-config pii-bioes \
  --eval-dataset-split validation --trust-remote-code
```

## Archived SFT/GRPO JSON extraction

The seven-label JSON SFT/GRPO tree was retired because the active nine-label BIOES path superseded it. The full implementation and its former data helpers remain in Git history; there are no operational callers outside the retired tree.

## Historical baseline migration status

`scripts/ops/run_lfm_bioes_baseline.py::run_native_parity` passed 8/8 persisted
reference rows on A10G with zero missing or unexpected spans. The separate native
Meddies-v1 candidate failed 2/8 rows, so its adapter and runner were removed rather
than presenting an unproven replacement for the historical vLLM baseline.

## Architecture decisions

The decisions that shape this package are enforced in code and tests rather than by private in-repo planning paths:

- Meddies Labels taxonomy is single-sourced in `meddies_pii.taxonomy`.
- BIOES data policy is enforced by corpus-gate, split, and tokenizer-policy tests.
- BIOES packing stays behind the trainer/packing modules and is separate from legacy SFT assumptions.
