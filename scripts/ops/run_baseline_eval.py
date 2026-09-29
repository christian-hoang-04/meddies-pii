"""Run-1 baseline measurement: score an EXISTING trained BIOES checkpoint.

Loads the trained adapter + classifier head through native Transformers + PEFT, then scores \
per-label and per-language F1 on the
Run-1 eval JSONL sets. No retrain, no HF push, no generation — load + eval only.

Two entry points (run smoke first, it's the risky adapter+classifier load):

    MODAL_PROFILE=openmedical uv run modal run scripts/ops/run_baseline_eval.py::smoke
    MODAL_PROFILE=openmedical uv run modal run scripts/ops/run_baseline_eval.py::full

The checkpoint lives on Modal Volume ``meddies-pii-bioes-artifacts`` (profile
openmedical) under ``CHECKPOINT_DIR``; eval sets under ``run1/``. Results are
PRINTED (``SMOKE_OK::`` / ``BASELINE_RESULT::``), never returned — the Modal
large-result blob path is broken and a single log line truncates ~64KB.
"""

from __future__ import annotations

# ruff: file-ignore[import-private-name]
# reason: this script reuses the training path's OWN select, prepare, collate and evaluate helpers so the number it
# reason: reports is the number training produces; re-implementing them here would let the baseline and the trainer
# reason: drift apart silently. The fix that would satisfy the rule is a public re-export inside `src/`, which this
# reason: lane does not own, so it is declared here and flagged rather than worked around.
# ruff: file-ignore[implicit-namespace-package]
# reason: this module is launched as `uv run modal run <this path>` and is never imported, so it is a script
# reason: rather than a package member. An `__init__.py` would declare this directory a package it is not, and
# reason: the sibling scripts here that run under `python` say so with a shebang instead.
# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import json
import random
from typing import TYPE_CHECKING, Any

import modal

from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)

if TYPE_CHECKING:
    from meddies_pii.training.bioes.data.artifacts import ProbeArtifacts

ARTIFACT_VOLUME_NAME = "meddies-pii-bioes-artifacts"
ARTIFACT_VOLUME_MOUNT = "/artifacts"
CHECKPOINT_DIR = "bioes/20260604_h100_8192_r128a256_pack_bs128_150step_ckpt10/unsloth"
EVAL_FILES = {
    "heldout": "run1/heldout.jsonl",
    "component_b": "run1/component_b.jsonl",
}

MODEL_ID = "LiquidAI/LFM2.5-350M-Base"
LORA_RANK = 128
LORA_ALPHA = 256
MAX_SEQ_LENGTH = 8192
SEED = 3407

H100_GPU = "H100"
TIMEOUT_SECONDS = 60 * 60
PINNED_PACKAGES = (
    "torch==2.13.0",
    "transformers==5.11.0",
    "datasets==4.5.0",
    "peft==0.19.1",
    "safetensors==0.8.0",
    "python-dotenv==1.2.2",
)
"""Native Transformers + PEFT reconstruction replaces the historical Unsloth runtime."""

image = add_source_pythonpath(
    modal.Image
    .from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941")
    .pip_install(*PINNED_PACKAGES)
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT),
)

artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
app = modal.App("meddies-bioes-run1-baseline-eval", image=image)


def _seed_everything(seed: int) -> None:

    import numpy as np
    import torch

    random.seed(seed)
    # reason: seeding the GLOBAL numpy RNG is the whole point — the datasets and transformers code
    # reason: downstream calls `np.random.*` directly, and a `Generator` instance would leave every one
    # reason: of those calls unseeded. The rule's replacement cannot reach them.
    np.random.seed(seed)  # ruff: ignore[numpy-legacy-random]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loaded_artifacts() -> tuple[ProbeArtifacts, dict[str, Any]]:
    """Build artifacts and load the trained adapter + classifier into them.

    Returns ``(artifacts, load_report)``. Mirrors the SAVE functions in
    ``data/artifacts.py`` and the resume-load in ``trainers/trainer.py``
    (``load_state_dict(..., strict=False)``).
    """
    from pathlib import Path

    import torch

    from meddies_pii.training.bioes.data.artifacts import (
        build_native_checkpoint_artifacts,
    )

    checkpoint = Path(ARTIFACT_VOLUME_MOUNT) / CHECKPOINT_DIR
    adapter_dir = checkpoint / "backbone_adapter"
    classifier_path = checkpoint / "classifier.pt"
    if not adapter_dir.is_dir():
        msg = f"Missing adapter dir: {adapter_dir}"
        raise FileNotFoundError(msg)
    if not classifier_path.is_file():
        msg = f"Missing classifier: {classifier_path}"
        raise FileNotFoundError(msg)

    artifacts = build_native_checkpoint_artifacts(
        MODEL_ID,
        adapter_dir,
        model_revision="9960764e30892e01f29a6dc23df2533fcd8bd5ae",
    )

    classifier_payload = torch.load(classifier_path, map_location="cpu", weights_only=True)
    saved_num_labels = int(classifier_payload["num_labels"])
    head_num_labels = int(artifacts.tagger.classifier.out_features)
    vocab_size = len(artifacts.label_vocab)
    if saved_num_labels != head_num_labels or saved_num_labels != vocab_size:
        msg = f"num_labels mismatch: classifier.pt={saved_num_labels} head={head_num_labels} vocab={vocab_size}"
        raise RuntimeError(
            msg,
        )
    classifier_state = {
        key: value.to(
            dtype=artifacts.tagger.classifier.weight.dtype,
            device=artifacts.tagger.classifier.weight.device,
        )
        for key, value in classifier_payload["classifier"].items()
    }
    artifacts.tagger.classifier.load_state_dict(classifier_state, strict=True)
    artifacts.tagger.eval()

    report = {
        "num_labels": saved_num_labels,
        "vocab_size": vocab_size,
        "hidden_size": int(artifacts.tagger.classifier.in_features),
        "adapter_runtime": "native_transformers_peft",
        "device": str(next(artifacts.tagger.parameters()).device),
    }
    return artifacts, report


def _synthetic_records() -> list[dict[str, Any]]:
    """Hand-made {text, spans, info} records covering several labels + langs.

    Spans are exact substrings (start/end/text validated by the prep auditor).
    """

    def span(text: str, value: str, label: str) -> dict[str, Any]:
        start = text.index(value)
        return {
            "label": label,
            "start": start,
            "end": start + len(value),
            "text": value,
        }

    rows: list[tuple[str, list[tuple[str, str]], str]] = [
        (
            "Patient John Smith was seen at Mercy General Hospital on 03/15/1985.",
            [
                ("John Smith", "human_name"),
                ("Mercy General Hospital", "company_name"),
                ("03/15/1985", "date"),
            ],
            "English",
        ),
        (
            "Contact: john.smith@email.com or call 0225 786 3719.",
            [
                ("john.smith@email.com", "email_address"),
                ("0225 786 3719", "phone_number"),
            ],
            "English",
        ),
        (
            "Bệnh nhân Nguyễn Hoa, mã số M-24-000748, khám tại Bệnh viện Bạch Mai.",
            [
                ("Nguyễn Hoa", "human_name"),
                ("M-24-000748", "id_number"),
                ("Bệnh viện Bạch Mai", "company_name"),
            ],
            "Vietnamese",
        ),
        (
            "Adresse: 12 Rue de la Paix, 75002 Paris. Tel: 01 42 86 00 00.",
            [
                ("12 Rue de la Paix, 75002 Paris", "address"),
                ("01 42 86 00 00", "phone_number"),
            ],
            "French",
        ),
        (
            "Dr. Maria Garcia signed the report on 2023-11-15 at Clinica Norte.",
            [
                ("Maria Garcia", "human_name"),
                ("2023-11-15", "date"),
                ("Clinica Norte", "company_name"),
            ],
            "Spanish",
        ),
    ]
    records: list[dict[str, Any]] = []
    for idx, (text, entities, language) in enumerate(rows):
        spans = [span(text, value, label) for value, label in entities]
        records.append({
            "text": text,
            "spans": spans,
            "info": {
                "language": language,
                "source": "smoke",
                "uid": f"smoke-{idx}",
            },
        })
    return records


def _records_to_audit_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map {text, spans:[{label,...}], info} into the prep auditor row shape.

    ``_audit_label_list_row`` reads ``row['label']`` as a list of
    ``{category, start, end, text}`` — so spans' ``label`` becomes ``category``.
    This reuses the existing prep path instead of hand-rolling tokenization.

    Canonical records (mixed.py `_record`) use `label:[{category,...}]`; legacy internal records used
    `spans:[{label,...}]`. Read whichever is present and per-span accept `category` or `label`, so every source prepares
    (not just internal). This was the Component-B / external-row drop bug.

    """
    audit_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        info = record.get("info") or {}
        raw_spans = record.get("label") or record.get("spans") or []
        label_list = []
        for span in raw_spans:
            start, end = int(span["start"]), int(span["end"])
            label_list.append({
                "category": span.get("category") or span.get("label"),
                "start": start,
                "end": end,
                "text": span.get("text", record["text"][start:end]),
            })
        audit_rows.append({
            "text": record["text"],
            "label": label_list,
            "uid": str(info.get("uid") or info.get("id") or f"row-{idx}"),
            "language": info.get("language", "unknown"),
        })
    return audit_rows


def _prepare_eval_rows(records: list[dict[str, Any]], artifacts: ProbeArtifacts) -> tuple[list[Any], dict[str, str]]:
    """Prepare rows via the existing select+prepare path; return rows + uid→lang."""
    from meddies_pii.training.bioes.data.preparation import (
        _prepare_rows,
        _select_source_rows,
    )

    audit_rows = _records_to_audit_rows(records)
    uid_to_language = {row["uid"]: row["language"] for row in audit_rows}
    id_to_label = {idx: label for label, idx in artifacts.label_to_id.items()}

    selected, _ = _select_source_rows(
        audit_rows,
        limit=None,
        allow_label_repairs=False,
        require_label_json=False,
        sort_by_length=False,
    )
    prepared, _ = _prepare_rows(
        selected,
        artifacts.tokenizer,
        max_length=MAX_SEQ_LENGTH,
        limit=None,
        id_to_label=id_to_label,
    )
    return prepared, uid_to_language


def _read_jsonl_from_volume(volume_path: str) -> list[dict[str, Any]] | None:
    from pathlib import Path

    path = Path(ARTIFACT_VOLUME_MOUNT) / volume_path
    if not path.is_file():
        return None
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


# reason: smoke combines build loaded and eval rows; splitting would misattribute row errors.
@app.function(
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def smoke() -> None:  # ruff: ignore[too-many-locals]
    _seed_everything(SEED)
    import torch

    from meddies_pii.annotations.bioes import (
        decode_bioes_from_offsets,
        viterbi_decode_logits,
    )
    from meddies_pii.training.bioes.trainers.batching import _collate

    artifacts, load_report = _build_loaded_artifacts()
    print(f"LOAD_REPORT::{json.dumps(load_report, ensure_ascii=False)}", flush=True)

    records = _synthetic_records()
    prepared, _ = _prepare_eval_rows(records, artifacts)
    if not prepared:
        msg = "Smoke prepared 0 rows from synthetic records"
        raise RuntimeError(msg)

    model = artifacts.tagger
    model.eval()
    id_to_label = {idx: label for label, idx in artifacts.label_to_id.items()}
    device = str(next(model.parameters()).device)
    logits_shape: tuple[int, ...] | None = None
    with torch.no_grad():
        for row in prepared[:2]:
            batch = _collate([row], artifacts.tokenizer, device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
            logits_shape = tuple(logits.shape)
            pred_ids = viterbi_decode_logits(logits[0], id_to_label, row.tokenized.offset_mapping)
            predicted = decode_bioes_from_offsets(row.raw, row.tokenized.offset_mapping, pred_ids, id_to_label)
            gold = [(s.label, s.text) for s in row.parsed.spans]
            pred = [(s.label, s.text) for s in predicted]
            print(f"ROW uid={row.uid}", flush=True)
            print(f"  GOLD={gold}", flush=True)
            print(f"  PRED={pred}", flush=True)

    smoke_ok = {
        "rows_prepared": len(prepared),
        "logits_shape": list(logits_shape) if logits_shape else None,
        "num_labels": load_report["num_labels"],
        "vocab_size": load_report["vocab_size"],
        "labels_match": load_report["num_labels"] == load_report["vocab_size"],
        "adapter_tensors_loaded": load_report["adapter_tensors_loaded"],
        "device": device,
    }
    print(f"SMOKE_OK::{json.dumps(smoke_ok, ensure_ascii=False)}", flush=True)


def _evaluate_group(artifacts: ProbeArtifacts, rows: list[Any]) -> dict[str, Any]:
    """Run _evaluate + per-label F1 for a row group; return a compact summary."""
    from meddies_pii.evaluation.span_metrics import (
        SpanMetricBlock,
        containment_span_prf_by_label,
        exact_span_prf_by_label,
    )
    from meddies_pii.training.bioes.trainers.evaluation import _evaluate

    metrics = _evaluate(artifacts, rows)

    predicted_by_doc: dict[str, Any] = {}
    gold_by_doc: dict[str, Any] = {}
    for index, row in enumerate(rows):
        import torch

        from meddies_pii.annotations.bioes import (
            decode_bioes_from_offsets,
            viterbi_decode_logits,
        )
        from meddies_pii.training.bioes.trainers.batching import _collate

        device = str(next(artifacts.tagger.parameters()).device)
        with torch.no_grad():
            batch = _collate([row], artifacts.tokenizer, device)
            logits = artifacts.tagger(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
        id_to_label = {idx: label for label, idx in artifacts.label_to_id.items()}
        pred_ids = viterbi_decode_logits(logits[0], id_to_label, row.tokenized.offset_mapping)
        decoded = decode_bioes_from_offsets(row.raw, row.tokenized.offset_mapping, pred_ids, id_to_label)
        doc_key = f"{row.uid}:{index}"
        predicted_by_doc[doc_key] = decoded
        gold_by_doc[doc_key] = row.parsed.spans

    exact_by_label = exact_span_prf_by_label(predicted_by_doc, gold_by_doc)
    containment_by_label = containment_span_prf_by_label(predicted_by_doc, gold_by_doc)

    def slim(block: SpanMetricBlock) -> dict[str, float]:
        return {
            "precision": round(float(block["precision"]), 4),
            "recall": round(float(block["recall"]), 4),
            "f1": round(float(block["f1"]), 4),
            "gold_total": int(block["gold_total"]),
            "pred_total": int(block["pred_total"]),
        }

    return {
        "support": len(rows),
        "exact_typed": slim(metrics["typed"]),
        "containment_typed": slim(metrics["containment_span"]["typed"]),
        "exact_per_label": {label: slim(block) for label, block in exact_by_label.items()},
        "containment_per_label": {label: slim(block) for label, block in containment_by_label.items()},
    }


@app.function(
    gpu=H100_GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def full() -> None:
    """Per-dataset line stays under Modal's ~64KB single-log-line truncation.

    The combined result with per-label + per-language for both datasets overflowed it). Parse these lines, not the slim
    summary below.

    """
    _seed_everything(SEED)

    artifacts, load_report = _build_loaded_artifacts()
    print(f"LOAD_REPORT::{json.dumps(load_report, ensure_ascii=False)}", flush=True)

    result: dict[str, Any] = {
        "checkpoint": CHECKPOINT_DIR,
        "model_id": MODEL_ID,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": MAX_SEQ_LENGTH,
        "seed": SEED,
        "num_labels": load_report["num_labels"],
        "datasets": {},
    }

    for name, volume_path in EVAL_FILES.items():
        records = _read_jsonl_from_volume(volume_path)
        if records is None:
            print(f"BASELINE_SKIP::{name} path={volume_path} (missing)", flush=True)
            continue
        prepared, uid_to_language = _prepare_eval_rows(records, artifacts)
        if not prepared:
            print(
                f"BASELINE_SKIP::{name} path={volume_path} (0 prepared rows)",
                flush=True,
            )
            continue

        overall = _evaluate_group(artifacts, prepared)

        by_language: dict[str, list[Any]] = {}
        for row in prepared:
            language = uid_to_language.get(row.uid, "unknown")
            by_language.setdefault(language, []).append(row)
        per_language = {
            language: _evaluate_group(artifacts, lang_rows) for language, lang_rows in sorted(by_language.items())
        }

        result["datasets"][name] = {
            "source_records": len(records),
            "prepared_rows": len(prepared),
            "overall": overall,
            "per_language": per_language,
        }
        print(
            f"BASELINE_DATASET::{name}::{json.dumps(result['datasets'][name], ensure_ascii=False)}",
            flush=True,
        )

    summary = {
        "checkpoint": result.get("checkpoint"),
        "datasets": {
            n: {
                "prepared_rows": ds["prepared_rows"],
                "exact_f1": ds["overall"].get("exact_typed", {}).get("f1"),
            }
            for n, ds in result["datasets"].items()
        },
    }
    print(f"BASELINE_RESULT::{json.dumps(summary, ensure_ascii=False)}", flush=True)
