from __future__ import annotations

# ruff: file-ignore[print]
# reason: results and progress travel back through the streamed run log, because Modal's large-result blob path is
# reason: unimplemented in this workspace.
import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import load_dataset
from huggingface_hub import HfApi

# reason: transformers declares its public names only under `TYPE_CHECKING` and serves them at
# reason: runtime through `_LazyModule`, so a static reader cannot prove the symbol is present.
# reason: Verified against the pinned 5.14.1: `hasattr(transformers, "AutoTokenizer")` is True.
from transformers import AutoTokenizer  # ty: ignore[possibly-missing-import]

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    build_label_to_id,
    decode_bioes_from_offsets,
    tokenize_and_align,
)
from meddies_pii.training.bioes.eval.harness import ADVERSARIAL_SLICE_NAMES

from .manifest import (
    WorkloadCandidate,
    build_workload_manifest,
    row_hash_from_parts,
)
from .preparation import (
    AuditedSourceRow,
    PreparationStats,
    _select_source_rows,
    _span_signature,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_DATASET_ID = "Meddies/meddies-pii"
DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-350M-Base"
DEFAULT_MANIFEST_NAME = "lfm25-bioes-4096-longest-eval-pinned"
DEFAULT_SELECTION_POLICY = "tokenized_longest_first_full_scan"


@dataclass(frozen=True, slots=True)
class ManifestBuildConfig:
    name: str = DEFAULT_MANIFEST_NAME
    dataset_id: str = DEFAULT_DATASET_ID
    dataset_config: str = "train"
    dataset_split: str = "train"
    dataset_revision: str | None = None
    eval_dataset_id: str | None = None
    eval_dataset_config: str = "test"
    eval_dataset_split: str | None = None
    eval_dataset_revision: str | None = None
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = None
    tokenizer_id: str = DEFAULT_MODEL_ID
    tokenizer_revision: str | None = None
    max_length: int = 4096
    fixed_pad_length: int = 4096
    required_examples: int = 1
    source_row_scan_multiplier: int = 64
    scan_limit: int | None = None
    min_observed_pad_length: int | None = 3000
    allow_label_repairs: bool = False
    require_label_json: bool = True
    trust_remote_code: bool = False
    selection_policy: str = DEFAULT_SELECTION_POLICY
    target_slice: str | None = None
    min_exact_span_f1: float = 0.0
    baseline_active_tps_model_only: float | None = None
    created_at_utc: str | None = None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_hf_revision(api: HfApi, repo_id: str, *, repo_type: str, revision: str | None) -> str:
    if repo_type == "dataset":
        sha = getattr(api.dataset_info(repo_id=repo_id, revision=revision), "sha", None)
    elif repo_type == "model":
        sha = getattr(api.model_info(repo_id=repo_id, revision=revision), "sha", None)
    else:  # pragma: no cover - defensive guard for future repo types
        msg = f"Unsupported Hugging Face repo_type: {repo_type}"
        raise ValueError(msg)
    if not sha:
        msg = f"Could not resolve immutable revision for {repo_type} repo {repo_id!r}"
        raise RuntimeError(msg)
    return str(sha)


def _scan_count(config: ManifestBuildConfig) -> int:
    if config.required_examples <= 0:
        msg = "required_examples must be positive"
        raise ValueError(msg)
    if config.source_row_scan_multiplier < 1:
        msg = "source_row_scan_multiplier must be >= 1"
        raise ValueError(msg)
    if config.scan_limit is not None and config.scan_limit <= 0:
        msg = "scan_limit must be positive when set"
        raise ValueError(msg)
    return config.scan_limit or (config.required_examples * config.source_row_scan_multiplier)


def _load_scanned_rows(config: ManifestBuildConfig, *, dataset_revision: str) -> list[dict[str, Any]]:
    ds = load_dataset(
        config.dataset_id,
        config.dataset_config,
        split=config.dataset_split,
        revision=dataset_revision,
    )
    scan_count = min(_scan_count(config), len(ds))
    return [dict(ds[index]) for index in range(scan_count)]


def _id_to_label(entity_labels: Sequence[str]) -> dict[int, str]:
    label_vocab = build_bioes_label_space(entity_labels)
    return {idx: label for label, idx in build_label_to_id(label_vocab).items()}


# reason: the tokenizer arrives from `AutoTokenizer.from_pretrained`, which the pinned transformers declares as
# reason: `Unknown | TokenizersBackend | None | SentencePieceBackend`. There is no single honest static type to write,
# reason: and spelling the union out would add an `Unknown` arm and a `None` the pinned version never returns.
def _candidate_for_audited_row(
    row: AuditedSourceRow,
    tokenizer: Any,  # ruff: ignore[any-type]
    *,
    max_length: int,
    entity_labels: Sequence[str],
    id_to_label: Mapping[int, str],
) -> WorkloadCandidate:
    target_spans = tuple(row.parsed.spans)
    row_hash = row_hash_from_parts(uid=row.uid, raw=row.parsed.raw, spans=target_spans)
    if not target_spans:
        return WorkloadCandidate(
            uid=row.uid,
            row_hash=row_hash,
            raw_chars=len(row.parsed.raw),
            token_length=0,
            accepted=False,
            skip_reason="no_mapped_spans",
        )
    try:
        tokenized = tokenize_and_align(
            tokenizer,
            row.parsed.raw,
            target_spans,
            max_length=max_length,
            entity_labels=entity_labels,
        )
    except ValueError:
        return WorkloadCandidate(
            uid=row.uid,
            row_hash=row_hash,
            raw_chars=len(row.parsed.raw),
            token_length=0,
            accepted=False,
            skip_reason="alignment",
        )

    token_length = len(tokenized.input_ids)
    if tokenized.truncated:
        return WorkloadCandidate(
            uid=row.uid,
            row_hash=row_hash,
            raw_chars=len(row.parsed.raw),
            token_length=token_length,
            truncated=True,
            accepted=False,
            skip_reason="truncated",
        )

    decoded = decode_bioes_from_offsets(row.parsed.raw, tokenized.offset_mapping, tokenized.labels, dict(id_to_label))
    if _span_signature(decoded) != _span_signature(target_spans):
        return WorkloadCandidate(
            uid=row.uid,
            row_hash=row_hash,
            raw_chars=len(row.parsed.raw),
            token_length=token_length,
            accepted=False,
            skip_reason="round_trip",
        )

    return WorkloadCandidate(
        uid=row.uid,
        row_hash=row_hash,
        raw_chars=len(row.parsed.raw),
        token_length=token_length,
    )


# reason: the tokenizer arrives from `AutoTokenizer.from_pretrained`, which the pinned transformers declares as
# reason: `Unknown | TokenizersBackend | None | SentencePieceBackend`. There is no single honest static type to write,
# reason: and spelling the union out would add an `Unknown` arm and a `None` the pinned version never returns.
def _candidate_audit(
    source_rows: Sequence[AuditedSourceRow],
    tokenizer: Any,  # ruff: ignore[any-type]
    *,
    max_length: int,
    entity_labels: Sequence[str],
    id_to_label: Mapping[int, str],
) -> tuple[list[WorkloadCandidate], PreparationStats]:
    stats = PreparationStats(candidates=len(source_rows))
    candidates: list[WorkloadCandidate] = []
    for row in source_rows:
        candidate = _candidate_for_audited_row(
            row,
            tokenizer,
            max_length=max_length,
            entity_labels=entity_labels,
            id_to_label=id_to_label,
        )
        candidates.append(candidate)
        if candidate.accepted:
            stats.accepted += 1
        elif candidate.skip_reason == "no_mapped_spans":
            stats.skipped_no_spans += 1
        elif candidate.skip_reason == "alignment":
            stats.skipped_alignment += 1
        elif candidate.skip_reason == "truncated":
            stats.skipped_truncated += 1
        elif candidate.skip_reason == "round_trip":
            stats.skipped_round_trip += 1
    return candidates, stats


# reason: build manifest coordinates hf revision with scanned rows; extra seams would fragment diagnostics.
def build_manifest(config: ManifestBuildConfig) -> dict[str, Any]:  # ruff: ignore[too-many-locals]
    if config.max_length <= 0:
        msg = "max_length must be positive"
        raise ValueError(msg)
    if config.fixed_pad_length < config.max_length:
        msg = "fixed_pad_length must be >= max_length"
        raise ValueError(msg)
    if config.min_observed_pad_length is not None and config.min_observed_pad_length <= 0:
        msg = "min_observed_pad_length must be positive when set"
        raise ValueError(msg)
    if config.target_slice is not None and config.target_slice not in ADVERSARIAL_SLICE_NAMES:
        msg = f"target_slice must be one of {ADVERSARIAL_SLICE_NAMES}; got {config.target_slice!r}"
        raise ValueError(msg)
    entity_labels = ENTITY_LABELS

    api = HfApi()
    dataset_revision = _resolve_hf_revision(api, config.dataset_id, repo_type="dataset", revision=config.dataset_revision)
    eval_dataset_id = config.eval_dataset_id or config.dataset_id
    eval_dataset_revision = (
        dataset_revision
        if eval_dataset_id == config.dataset_id and config.eval_dataset_revision is None
        else _resolve_hf_revision(
            api,
            eval_dataset_id,
            repo_type="dataset",
            revision=config.eval_dataset_revision,
        )
    )
    model_revision = _resolve_hf_revision(api, config.model_id, repo_type="model", revision=config.model_revision)
    tokenizer_revision = _resolve_hf_revision(
        api,
        config.tokenizer_id,
        repo_type="model",
        revision=config.tokenizer_revision,
    )

    scanned_rows = _load_scanned_rows(config, dataset_revision=dataset_revision)
    source_rows, source_stats = _select_source_rows(
        scanned_rows,
        limit=None,
        allow_label_repairs=config.allow_label_repairs,
        require_label_json=config.require_label_json,
        sort_by_length=False,
    )
    candidate_source_rows = (
        [row for row in source_rows if config.target_slice is not None and config.target_slice in row.slices]
        if config.target_slice is not None
        else source_rows
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_id,
        revision=tokenizer_revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer is None:
        msg = f"tokenizer {config.tokenizer_id!r} could not be loaded"
        raise ValueError(msg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    candidates, preparation_stats = _candidate_audit(
        candidate_source_rows,
        tokenizer,
        max_length=config.max_length,
        entity_labels=entity_labels,
        id_to_label=_id_to_label(entity_labels),
    )
    objective: dict[str, Any] = {
        "phase": "speed_probe",
        "min_exact_span_f1": config.min_exact_span_f1,
    }
    if config.target_slice is not None:
        objective["target_slice"] = config.target_slice
    if config.baseline_active_tps_model_only is not None:
        objective["baseline_active_tps_model_only"] = config.baseline_active_tps_model_only

    return build_workload_manifest(
        name=config.name,
        dataset_id=config.dataset_id,
        dataset_config=config.dataset_config,
        dataset_split=config.dataset_split,
        dataset_revision=dataset_revision,
        eval_dataset_id=eval_dataset_id,
        eval_dataset_config=config.eval_dataset_config,
        eval_dataset_split=config.eval_dataset_split or config.dataset_split,
        eval_dataset_revision=eval_dataset_revision,
        model_id=config.model_id,
        model_revision=model_revision,
        tokenizer_id=config.tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        max_length=config.max_length,
        fixed_pad_length=config.fixed_pad_length,
        selection_policy=config.selection_policy,
        required_examples=config.required_examples,
        candidates=candidates,
        min_observed_pad_length=config.min_observed_pad_length,
        objective=objective,
        scan={
            "source_row_scan_multiplier": config.source_row_scan_multiplier,
            "scan_limit": config.scan_limit,
            "source_rows_loaded": len(scanned_rows),
            "source_selection_stats": asdict(source_stats),
            "target_slice": config.target_slice,
            "target_slice_source_candidates": len(candidate_source_rows),
            "tokenized_candidate_stats": asdict(preparation_stats),
            "allow_label_repairs": config.allow_label_repairs,
            "require_label_json": config.require_label_json,
        },
        created_at_utc=config.created_at_utc or _utc_now(),
    )


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    destination.write_text(rendered, encoding="utf-8")
    return destination


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a locked workload manifest from a fully tokenized scanned pool.")
    parser.add_argument("--name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-config", default="train")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--eval-dataset-id")
    parser.add_argument("--eval-dataset-config", default="test")
    parser.add_argument("--eval-dataset-split")
    parser.add_argument("--eval-dataset-revision")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--fixed-pad-length", type=int, default=4096)
    parser.add_argument("--required-examples", type=int, default=1)
    parser.add_argument("--source-row-scan-multiplier", type=int, default=64)
    parser.add_argument("--scan-limit", type=int)
    parser.add_argument("--min-observed-pad-length", type=int, default=3000)
    parser.add_argument("--allow-label-repairs", action="store_true")
    parser.add_argument("--no-require-label-json", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--selection-policy", default=DEFAULT_SELECTION_POLICY)
    parser.add_argument("--target-slice", choices=ADVERSARIAL_SLICE_NAMES)
    parser.add_argument("--min-exact-span-f1", type=float, default=0.0)
    parser.add_argument("--baseline-active-tps-model-only", type=float)
    parser.add_argument("--created-at-utc")
    parser.add_argument("--out", help="Optional destination path. Defaults to stdout.")
    return parser


def _config_from_args(args: argparse.Namespace) -> ManifestBuildConfig:
    return ManifestBuildConfig(
        name=args.name,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_revision=args.dataset_revision,
        eval_dataset_id=args.eval_dataset_id,
        eval_dataset_config=args.eval_dataset_config,
        eval_dataset_split=args.eval_dataset_split,
        eval_dataset_revision=args.eval_dataset_revision,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        max_length=args.max_length,
        fixed_pad_length=args.fixed_pad_length,
        required_examples=args.required_examples,
        source_row_scan_multiplier=args.source_row_scan_multiplier,
        scan_limit=args.scan_limit,
        min_observed_pad_length=args.min_observed_pad_length,
        allow_label_repairs=args.allow_label_repairs,
        require_label_json=not args.no_require_label_json,
        trust_remote_code=args.trust_remote_code,
        selection_policy=args.selection_policy,
        target_slice=args.target_slice,
        min_exact_span_f1=args.min_exact_span_f1,
        baseline_active_tps_model_only=args.baseline_active_tps_model_only,
        created_at_utc=args.created_at_utc,
    )


def parse_args(argv: Sequence[str] | None = None) -> ManifestBuildConfig:
    return _config_from_args(_arg_parser().parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    config = _config_from_args(args)
    manifest = build_manifest(config)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        write_manifest(manifest, args.out)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
