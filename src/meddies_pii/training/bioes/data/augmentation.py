from __future__ import annotations

# ruff: file-ignore[print]
# reason: results and progress travel back through the streamed run log, because Modal's large-result blob path is
# reason: unimplemented in this workspace.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from datasets import load_dataset

# reason: transformers declares its public names only under `TYPE_CHECKING` and serves them at
# reason: runtime through `_LazyModule`, so a static reader cannot prove the symbol is present.
# reason: Verified against the pinned 5.14.1: `hasattr(transformers, "AutoTokenizer")` is True.
from transformers import AutoTokenizer  # ty: ignore[possibly-missing-import]

from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.taxonomy import PII_LABELS
from meddies_pii.training.bioes.data.augmentation_artifacts import (
    read_jsonl,
    text_hash,
    write_json,
    write_jsonl,
    write_parquet,
)
from meddies_pii.training.bioes.data.augmentation_tokens import token_lengths
from meddies_pii.training.bioes.data.record_schema import (
    BatchLengthTokenizer,
    NormalizedRecord,
    Record,
    record_from_object,
    record_sequence_from_object,
)
from meddies_pii.training.bioes.data.tokenizer_security import (
    validate_remote_code_tokenizer_policy,
)

INVALID_EXAMPLE_LIMIT = 100


def _info_mapping(record: Mapping[str, object]) -> Record:
    info = record.get("info")
    if isinstance(info, Mapping):
        return record_from_object(info, source="record.info")
    return {}


# reason: normalize record owns parse record and info mapping together; splitting would misattribute row errors.
def normalize_record(record: Mapping[str, object], *, default_id: str) -> NormalizedRecord:  # ruff: ignore[too-many-locals]
    example_id, text, spans = parse_labeled_record(record, default_id=default_id)
    raw_info = _info_mapping(record)
    source_dataset = str(raw_info.get("source_dataset") or "UNKNOWN")
    source = str(raw_info.get("source") or source_dataset)
    language = str(raw_info.get("language") or "UNKNOWN")
    domain_bucket = str(raw_info.get("domain_bucket") or "UNKNOWN")
    domain_profile = str(raw_info.get("domain_profile") or domain_bucket)
    document_type = str(raw_info.get("document_type") or "")
    generation_model = str(raw_info.get("generation_model") or "")
    language_bucket = str(raw_info.get("language_bucket") or "other")
    label_policy = str(raw_info.get("label_policy") or "")
    original_id = str(raw_info.get("original_id") or raw_info.get("id") or example_id)
    normalized_id = str(raw_info.get("id") or example_id)
    scenario = str(raw_info.get("scenario") or "")
    split_purpose = str(raw_info.get("split_purpose") or "")
    text_format = str(raw_info.get("text_format") or "")
    return {
        "text": text,
        "label": [
            {
                "category": span.label,
                "start": int(span.start),
                "end": int(span.end),
                "text": span.text,
            }
            for span in sorted(spans, key=lambda item: (item.start, item.end, item.label))
        ],
        "info": {
            "domain_bucket": domain_bucket,
            "domain_profile": domain_profile,
            "document_type": document_type,
            "generation_model": generation_model,
            "id": normalized_id,
            "language": language,
            "language_bucket": language_bucket,
            "label_policy": label_policy,
            "original_id": original_id,
            "scenario": scenario,
            "source": source,
            "source_dataset": source_dataset,
            "split_purpose": split_purpose,
            "text_format": text_format,
        },
    }


def validate_record(record: Mapping[str, object], *, default_id: str) -> list[str]:
    issues: list[str] = []
    try:
        example_id, text, spans = parse_labeled_record(record, default_id=default_id)
    except ValueError as exc:
        return [f"parse_error:{exc}"]
    if not spans:
        issues.append("empty_labels")
    cursor = -1
    for span in sorted(spans, key=lambda item: (item.start, item.end, item.label)):
        if span.start < cursor:
            issues.append(f"overlap:{example_id}:{span.label}:{span.start}-{span.end}")
        cursor = max(cursor, span.end)
        if span.label not in PII_LABELS:
            issues.append(f"bad_label:{span.label}")
        if text[span.start : span.end] != span.text:
            issues.append(f"bad_offset:{example_id}:{span.label}:{span.text!r}")
    return issues


def load_current_splits(dataset_id: str, config_name: str) -> tuple[list[NormalizedRecord], list[NormalizedRecord]]:
    dataset = load_dataset(dataset_id, config_name)
    if not isinstance(dataset, Mapping):
        msg = "Loaded dataset must expose named splits"
        raise TypeError(msg)
    train_rows = record_sequence_from_object(dataset.get("train"), source="train")
    validation_rows = record_sequence_from_object(dataset.get("validation"), source="validation")
    train = [
        normalize_record(
            record_from_object(row, source=f"train[{idx}]"),
            default_id=f"current-train-{idx}",
        )
        for idx, row in enumerate(train_rows)
    ]
    validation = [
        normalize_record(
            record_from_object(row, source=f"validation[{idx}]"),
            default_id=f"current-validation-{idx}",
        )
        for idx, row in enumerate(validation_rows)
    ]
    return train, validation


# reason: load augmentation keeps read jsonl beside check record; splitting would fragment diagnostics.
def load_augmentation_rows(  # ruff: ignore[too-many-arguments,too-many-locals]
    paths: Sequence[Path],
    *,
    tokenizer: BatchLengthTokenizer,
    max_length: int,
    token_batch_size: int,
    existing_ids: set[str],
    existing_text_hashes: set[str],
) -> tuple[list[NormalizedRecord], Record]:
    raw_rows_by_path: dict[str, list[Record]] = {str(path): read_jsonl(path) for path in paths}
    normalized: list[tuple[str, int, NormalizedRecord]] = []
    source_file_counts: Counter[str] = Counter()
    invalid_rows: list[Record] = []
    for path, rows in raw_rows_by_path.items():
        source_file_counts[path] = len(rows)
        for idx, row in enumerate(rows):
            default_id = f"{Path(path).stem}:{idx}"
            issues = validate_record(row, default_id=default_id)
            if issues:
                invalid_rows.append({"path": path, "row_index": idx, "issues": issues})
                continue
            normalized.append((path, idx, normalize_record(row, default_id=default_id)))

    lengths = token_lengths(
        [record["text"] for _, _, record in normalized],
        tokenizer=tokenizer,
        batch_size=token_batch_size,
    )
    kept: list[NormalizedRecord] = []
    dropped_over_length: list[Record] = []
    duplicate_ids: list[str] = []
    duplicate_texts: list[str] = []
    seen_ids = set(existing_ids)
    seen_text_hashes = set(existing_text_hashes)
    kept_by_file: Counter[str] = Counter()
    dropped_over_length_by_file: Counter[str] = Counter()
    for (path, idx, record), length in zip(normalized, lengths, strict=True):
        info = record["info"]
        record_id = str(info["id"])
        digest = text_hash(str(record["text"]))
        if length > max_length:
            dropped_over_length.append({
                "path": path,
                "row_index": idx,
                "id": record_id,
                "tokens": length,
            })
            dropped_over_length_by_file[path] += 1
            continue
        if record_id in seen_ids:
            duplicate_ids.append(record_id)
            continue
        if digest in seen_text_hashes:
            duplicate_texts.append(record_id)
            continue
        seen_ids.add(record_id)
        seen_text_hashes.add(digest)
        kept_by_file[path] += 1
        kept.append(record)

    summary = {
        "source_file_counts": dict(source_file_counts),
        "candidate_rows": sum(source_file_counts.values()),
        "valid_candidate_rows": len(normalized),
        "kept_rows": len(kept),
        "kept_by_file": dict(kept_by_file),
        "invalid_rows": invalid_rows[:100],
        "invalid_row_count": len(invalid_rows),
        "dropped_over_max_length": len(dropped_over_length),
        "dropped_over_max_length_by_file": dict(dropped_over_length_by_file),
        "dropped_over_max_length_examples": sorted(
            dropped_over_length,
            key=_dropped_over_length_tokens,
            reverse=True,
        )[:20],
        "duplicate_id_dropped": len(duplicate_ids),
        "duplicate_id_examples": duplicate_ids[:20],
        "duplicate_text_dropped": len(duplicate_texts),
        "duplicate_text_examples": duplicate_texts[:20],
    }
    return kept, summary


def _dropped_over_length_tokens(row: Mapping[str, object]) -> int:
    tokens = row.get("tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        msg = "dropped-over-length record must include integer tokens"
        raise TypeError(msg)
    return tokens


def labels(row: Mapping[str, object]) -> set[str]:
    out: set[str] = set()
    raw_labels = row.get("label")
    if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes)):
        for label in raw_labels:
            if not isinstance(label, Mapping):
                continue
            category = label.get("category")
            if isinstance(category, str):
                out.add(category)
    return out


def info_value(row: Mapping[str, object], key: str, default: str = "UNKNOWN") -> str:
    info = _info_mapping(row)
    value = info.get(key)
    return str(value) if value is not None else default


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> Record:
    source_counts: Counter[str] = Counter()
    domain_bucket_counts: Counter[str] = Counter()
    language_bucket_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    source_language_counts: Counter[str] = Counter()
    label_doc_counts: Counter[str] = Counter()
    label_span_counts: Counter[str] = Counter()
    for row in rows:
        source_dataset = info_value(row, "source_dataset")
        language = info_value(row, "language")
        language_bucket = info_value(row, "language_bucket", "other")
        source_counts[source_dataset] += 1
        domain_bucket_counts[info_value(row, "domain_bucket")] += 1
        language_bucket_counts[language_bucket] += 1
        language_counts[language] += 1
        source_language_counts[f"{source_dataset} | {language} | {language_bucket}"] += 1
        label_doc_counts.update(labels(row))
        raw_labels = row.get("label")
        if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes)):
            for label in raw_labels:
                if not isinstance(label, Mapping):
                    continue
                category = label.get("category")
                if isinstance(category, str):
                    label_span_counts[category] += 1
    return {
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "domain_bucket_counts": dict(sorted(domain_bucket_counts.items())),
        "language_bucket_counts": dict(sorted(language_bucket_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "source_language_counts": dict(sorted(source_language_counts.items())),
        "label_doc_counts": {label: label_doc_counts.get(label, 0) for label in PII_LABELS},
        "label_span_counts": {label: label_span_counts.get(label, 0) for label in PII_LABELS},
    }


def validate_splits(
    *,
    train_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    tokenizer: BatchLengthTokenizer,
    max_length: int,
    token_batch_size: int,
) -> Record:
    train_ids = [info_value(row, "id", "") for row in train_rows]
    validation_ids = [info_value(row, "id", "") for row in validation_rows]
    train_hashes = [text_hash(str(row.get("text") or "")) for row in train_rows]
    validation_hashes = [text_hash(str(row.get("text") or "")) for row in validation_rows]
    invalid_examples: list[Record] = []
    empty_label_rows = 0
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for idx, row in enumerate(rows):
            issues = validate_record(row, default_id=f"{split}:{idx}")
            if "empty_labels" in issues:
                empty_label_rows += 1
            if issues and len(invalid_examples) < INVALID_EXAMPLE_LIMIT:
                invalid_examples.append({"split": split, "row_index": idx, "issues": issues})

    train_lengths = token_lengths(
        [str(row.get("text") or "") for row in train_rows],
        tokenizer=tokenizer,
        batch_size=token_batch_size,
    )
    validation_lengths = token_lengths(
        [str(row.get("text") or "") for row in validation_rows],
        tokenizer=tokenizer,
        batch_size=token_batch_size,
    )
    all_rows_summary = summarize_rows([*train_rows, *validation_rows])
    label_doc_counts = all_rows_summary.get("label_doc_counts")
    if not isinstance(label_doc_counts, Mapping):
        msg = "row summary did not contain label document counts"
        raise RuntimeError(msg)
    return {
        "train_validation_id_overlap": len(set(train_ids) & set(validation_ids)),
        "train_validation_text_overlap": len(set(train_hashes) & set(validation_hashes)),
        "duplicate_train_ids": len(train_ids) - len(set(train_ids)),
        "duplicate_validation_ids": len(validation_ids) - len(set(validation_ids)),
        "duplicate_train_texts": len(train_hashes) - len(set(train_hashes)),
        "duplicate_validation_texts": len(validation_hashes) - len(set(validation_hashes)),
        "invalid_row_count": len(invalid_examples),
        "invalid_examples": invalid_examples,
        "empty_label_rows": empty_label_rows,
        "missing_labels": [label for label in PII_LABELS if label_doc_counts.get(label, 0) == 0],
        "max_length": max_length,
        "train_rows_over_max_length": sum(length > max_length for length in train_lengths),
        "validation_rows_over_max_length": sum(length > max_length for length in validation_lengths),
        "train_max_tokens": max(train_lengths, default=0),
        "validation_max_tokens": max(validation_lengths, default=0),
    }


# reason: augmentation main coordinates parse args with load splits; extra seams would fragment diagnostics.
def main() -> None:  # ruff: ignore[too-many-locals]
    parser = argparse.ArgumentParser(
        description="Merge targeted Meddies Labels augmentation rows into the current Hugging Face pii-bioes split.",
    )
    parser.add_argument("--dataset-id", default="Meddies/meddies-pii")
    parser.add_argument("--config-name", default="pii-bioes")
    parser.add_argument(
        "--augmentation-file",
        type=Path,
        action="append",
        required=True,
        help="Validated augmentation JSONL. Pass multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bioes-v2/base/augmented"),
    )
    parser.add_argument("--tokenizer-model-id", default="LiquidAI/LFM2.5-350M-Base")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--token-batch-size", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_train, current_validation = load_current_splits(args.dataset_id, args.config_name)
    current_ids = {info_value(row, "id", "") for row in [*current_train, *current_validation]}
    current_text_hashes = {text_hash(str(row.get("text") or "")) for row in [*current_train, *current_validation]}
    validate_remote_code_tokenizer_policy(
        tokenizer_model_id=args.tokenizer_model_id,
        tokenizer_revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_id,
        revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer is None:
        msg = f"tokenizer {args.tokenizer_model_id!r} could not be loaded"
        raise ValueError(msg)
    augmentation_rows, augmentation_summary = load_augmentation_rows(
        args.augmentation_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        token_batch_size=args.token_batch_size,
        existing_ids=current_ids,
        existing_text_hashes=current_text_hashes,
    )
    train_rows = [*current_train, *augmentation_rows]
    validation_rows = current_validation
    split_validation = validate_splits(
        train_rows=train_rows,
        validation_rows=validation_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        token_batch_size=args.token_batch_size,
    )

    train_jsonl = args.output_dir / f"train.{LEGACY_ARTIFACT_TOKEN}.all_lang_unique_augmented.jsonl"
    validation_jsonl = args.output_dir / f"validation.{LEGACY_ARTIFACT_TOKEN}.all_lang_unique_augmented.jsonl"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(validation_jsonl, validation_rows)
    write_parquet(args.output_dir / "hf_upload/pii-bioes/train-00000-of-00001.parquet", train_rows)
    write_parquet(
        args.output_dir / "hf_upload/pii-bioes/validation-00000-of-00001.parquet",
        validation_rows,
    )

    split_summary = {
        "input_rows": len(train_rows) + len(validation_rows),
        "train": summarize_rows(train_rows),
        "validation": summarize_rows(validation_rows),
        "split_validation": split_validation,
        "validation_policy": {
            "source": "unchanged from prior pii-bioes split",
            "validation_rows": len(validation_rows),
        },
    }
    all_rows_summary = summarize_rows([*train_rows, *validation_rows])
    dataset_summary = {
        "artifact_type": "pii_bioes_dataset_summary",
        "dataset_id": args.dataset_id,
        "config_name": args.config_name,
        "total_rows": len(train_rows) + len(validation_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "base_remote_train_rows": len(current_train),
        "base_remote_validation_rows": len(current_validation),
        "augmentation": augmentation_summary,
        "max_length_filter": {
            "tokenizer_model_id": args.tokenizer_model_id,
            "max_length": args.max_length,
            "policy": "Rows over max_length are dropped before merge.",
        },
        "all_rows_summary": all_rows_summary,
        "split_summary": split_summary,
        "unique_no_duplicate_policy": (
            "No exact duplicate ids/texts within split or across train/validation; "
            "augmentation rows over 4096 LFM2.5 tokens are dropped before merge."
        ),
    }
    train_summary = summarize_rows(train_rows)
    train_summary["augmentation"] = augmentation_summary
    write_json(
        args.output_dir / f"train.{LEGACY_ARTIFACT_TOKEN}.all_lang_unique_augmented.summary.json",
        train_summary,
    )
    write_json(args.output_dir / "pii-bioes.split.summary.json", split_summary)
    write_json(args.output_dir / "pii-bioes.summary.json", dataset_summary)
    write_json(args.output_dir / "merge_validation.summary.json", split_validation)
    write_json(
        args.output_dir / "hf_upload/reports/pii-bioes.split.summary.json",
        split_summary,
    )
    write_json(args.output_dir / "hf_upload/reports/pii-bioes.summary.json", dataset_summary)

    print(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "total_rows": len(train_rows) + len(validation_rows),
                "augmentation": augmentation_summary,
                "split_validation": split_validation,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
