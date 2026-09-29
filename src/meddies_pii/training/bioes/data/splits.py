from __future__ import annotations

# ruff: file-ignore[print]
# reason: results and progress travel back through the streamed run log, because Modal's large-result blob path is
# reason: unimplemented in this workspace.
import argparse
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from meddies_pii.languages import language_bucket
from meddies_pii.taxonomy import PII_LABELS
from meddies_pii.training.bioes.data.augmentation_artifacts import text_hash
from meddies_pii.training.bioes.data.record_io import write_jsonl

MAX_BALANCE_SCORE = 3

LANGUAGE_BALANCE_TARGETS = {"vi": 0.30, "en": 0.30, "other": 0.40}
"""vi=0.30/en=0.30/other=0.40 per ADR 0006 (vi is already the model's strongest language.

Diversity of the 15 weak non-vi/en languages is the Run-2 target, not more vi).

"""


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def _labels(row: Mapping[str, object]) -> set[str]:
    labels = row.get("label")
    out: set[str] = set()
    if not isinstance(labels, Sequence):
        return out
    for label in labels:
        if isinstance(label, Mapping):
            category = label.get("category")
            if isinstance(category, str):
                out.add(category)
    return out


def _info_value(row: Mapping[str, object], key: str, default: str) -> str:
    info = row.get("info")
    if not isinstance(info, Mapping):
        return default
    value = info.get(key)
    return str(value) if value is not None else default


def _validation_label_doc_counts(rows: Sequence[Mapping[str, object]], selected: set[int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for idx in selected:
        counts.update(_labels(rows[idx]))
    return counts


# reason: select validation keeps labels beside info value; splitting would fragment diagnostics.
def _select_validation_indices(  # ruff: ignore[complex-structure,too-many-branches,too-many-arguments,too-many-statements]
    rows: Sequence[dict[str, object]],
    *,
    validation_rows: int,
    min_docs_per_label: int,
    medical_ratio: float,
    vi_ratio: float,
    en_ratio: float,
    seed: int,
) -> set[int]:
    """Pick rare labels first so private_url/secret cannot be starved by common labels.

    Returns:
        The chosen row indices: first enough per label to meet ``min_docs_per_label``, rarest
        label first, then filled toward the language and domain ratios by greedy best score.

    Raises:
        ValueError: If the validation size is not positive or not smaller than the input, if
            some label has fewer candidate docs than the per-label minimum, or if meeting that
            minimum for every label would need more rows than the validation size allows.

    """
    if validation_rows <= 0:
        msg = "--validation-rows must be positive"
        raise ValueError(msg)
    if validation_rows >= len(rows):
        msg = "--validation-rows must be smaller than the input row count"
        raise ValueError(msg)

    # reason: the label-stratified validation split must be reproducible from its seed, or two runs of the same
    # reason: config would score against different validation rows. Reproducibility is the contract; this picks
    # reason: row indices, never a secret, a token, or a key.
    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    by_label: dict[str, list[int]] = {label: [] for label in PII_LABELS}
    for idx, row in enumerate(rows):
        for label in _labels(row):
            if label in by_label:
                by_label[label].append(idx)
    for label, indices in by_label.items():
        if len(indices) < min_docs_per_label:
            msg = f"label {label!r} only has {len(indices)} candidate docs; cannot require {min_docs_per_label}"
            raise ValueError(
                msg,
            )
        rng.shuffle(indices)

    selected: set[int] = set()
    for label in sorted(PII_LABELS, key=lambda item: len(by_label[item])):
        while _validation_label_doc_counts(rows, selected).get(label, 0) < min_docs_per_label:
            for idx in by_label[label]:
                if idx not in selected:
                    selected.add(idx)
                    break
            else:  # pragma: no cover - guarded by candidate count check above.
                msg = f"could not satisfy label support for {label!r}"
                raise ValueError(msg)
            if len(selected) > validation_rows:
                msg = (
                    "label support target exceeds validation size; "
                    "increase --validation-rows or lower --min-docs-per-label"
                )
                raise ValueError(
                    msg,
                )

    target_language = {
        "vi": round(validation_rows * vi_ratio),
        "en": round(validation_rows * en_ratio),
    }
    target_language["other"] = validation_rows - target_language["vi"] - target_language["en"]
    target_domain = {
        "medical": round(validation_rows * medical_ratio),
    }
    target_domain["general"] = validation_rows - target_domain["medical"]

    candidates = [idx for idx in range(len(rows)) if idx not in selected]
    rng.shuffle(candidates)

    def language_counts() -> Counter[str]:
        return Counter(_info_value(rows[idx], "language_bucket", "other") for idx in selected)

    def domain_counts() -> Counter[str]:
        return Counter(_info_value(rows[idx], "domain_bucket", "UNKNOWN") for idx in selected)

    while len(selected) < validation_rows:
        lang_counts = language_counts()
        dom_counts = domain_counts()

        best_pos = 0
        best_score = -1
        for pos, idx in enumerate(candidates):
            lang = _info_value(rows[idx], "language_bucket", "other")
            domain = _info_value(rows[idx], "domain_bucket", "UNKNOWN")
            score = 0
            if lang_counts[lang] < target_language.get(lang, 0):
                score += 2
            if dom_counts[domain] < target_domain.get(domain, 0):
                score += 1
            if score > best_score:
                best_score = score
                best_pos = pos
                if score == MAX_BALANCE_SCORE:
                    break

        selected.add(candidates.pop(best_pos))

    return selected


def _summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    source_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    label_doc_counts: Counter[str] = Counter()
    label_span_counts: Counter[str] = Counter()
    for row in rows:
        source_counts[_info_value(row, "source_dataset", "UNKNOWN")] += 1
        domain_counts[_info_value(row, "domain_bucket", "UNKNOWN")] += 1
        language_counts[_info_value(row, "language_bucket", "other")] += 1
        label_doc_counts.update(_labels(row))
        labels = row.get("label")
        if isinstance(labels, Sequence):
            for label in labels:
                if isinstance(label, Mapping):
                    category = label.get("category")
                    if isinstance(category, str):
                        label_span_counts[category] += 1
    return {
        "rows": len(rows),
        "source_counts": dict(source_counts),
        "domain_bucket_counts": dict(domain_counts),
        "language_bucket_counts": dict(language_counts),
        "label_doc_counts": {label: label_doc_counts.get(label, 0) for label in PII_LABELS},
        "label_span_counts": {label: label_span_counts.get(label, 0) for label in PII_LABELS},
    }


def _validate_split(
    *,
    train_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    train_ids = {_info_value(row, "id", "") for row in train_rows}
    validation_ids = {_info_value(row, "id", "") for row in validation_rows}
    train_text_hashes = {text_hash(normalize_text(str(row.get("text") or ""))) for row in train_rows}
    validation_text_hashes = {text_hash(normalize_text(str(row.get("text") or ""))) for row in validation_rows}
    return {
        "train_validation_id_overlap": len(train_ids & validation_ids),
        "train_validation_text_overlap": len(train_text_hashes & validation_text_hashes),
        "duplicate_train_ids": len(train_rows) - len(train_ids),
        "duplicate_validation_ids": len(validation_rows) - len(validation_ids),
    }


def normalize_text(text: str) -> str:
    """Leakage-normalization: strip, lowercase, collapse internal whitespace.

    Two rows whose text differs only by case/whitespace must map to one key so a
    near-duplicate cannot leak across the train / held-out boundary.

    Returns:
        The text stripped, lowercased, and with every internal whitespace run collapsed to one
        space -- the key both sides of the split are compared on.

    """
    return " ".join(text.strip().lower().split())


def row_language_bucket(row: Mapping[str, object]) -> str:
    """Bucket a row into vi / en / other via the shared mixed.py classifier.

    Returns:
        ``"vi"``, ``"en"`` or ``"other"``, decided by ``mixed.py`` from the row's recorded
        language and source. A row missing both reads as ``"other"`` rather than failing.

    """
    return language_bucket(
        language=_info_value(row, "language", ""),
        source=_info_value(row, "source", ""),
    )


def _balanced_targets(heldout_rows: int) -> dict[str, int]:
    targets = {
        "vi": round(heldout_rows * LANGUAGE_BALANCE_TARGETS["vi"]),
        "en": round(heldout_rows * LANGUAGE_BALANCE_TARGETS["en"]),
    }
    targets["other"] = heldout_rows - targets["vi"] - targets["en"]
    return targets


def carve_heldout(
    rows: Sequence[Mapping[str, object]],
    *,
    heldout_rows: int,
    seed: int,
    tolerance: float = 0.05,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Carve a held-out set balanced to 30/30/40 vi/en/other, disjoint from train.

    Returns ``(train, heldout)``. The held-out set is filled greedily toward the
    per-bucket targets within ``tolerance``; train is then the remainder with every
    row that collides with the held-out set by ``id`` OR normalized-text-hash removed,
    guaranteeing 0 overlap on both keys.

    Returns:
        ``(train, heldout)`` -- the remainder after collision removal, and the balanced
        held-out set.

    Raises:
        ValueError: If the held-out size is not positive or not smaller than the input, if a
            language bucket has fewer candidate rows than its target, or if an achieved bucket
            share ends up further from its target than ``tolerance``.

    """
    if heldout_rows <= 0:
        msg = "heldout_rows must be positive"
        raise ValueError(msg)
    if heldout_rows >= len(rows):
        msg = "heldout_rows must be smaller than the input row count"
        raise ValueError(msg)

    targets = _balanced_targets(heldout_rows)

    by_bucket: dict[str, list[int]] = {"vi": [], "en": [], "other": []}
    for idx, row in enumerate(rows):
        by_bucket[row_language_bucket(row)].append(idx)

    # reason: the language-balanced heldout draw must be reproducible from its seed so the same config yields the
    # reason: same held rows across runs. Reproducibility is the contract; this shuffles per-language index
    # reason: buckets, never a secret, a token, or a key.
    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    for indices in by_bucket.values():
        rng.shuffle(indices)

    selected: set[int] = set()
    for bucket, target in targets.items():
        available = by_bucket[bucket]
        if len(available) < target:
            msg = f"bucket {bucket!r} has {len(available)} candidate rows; cannot fill held-out target of {target}"
            raise ValueError(
                msg,
            )
        selected.update(available[:target])

    heldout = [dict(rows[idx]) for idx in sorted(selected)]

    heldout_ids = {_info_value(row, "id", "") for row in heldout}
    heldout_hashes = {text_hash(normalize_text(str(row.get("text") or ""))) for row in heldout}

    train = [
        dict(row)
        for idx, row in enumerate(rows)
        if idx not in selected
        and _info_value(row, "id", "") not in heldout_ids
        and text_hash(normalize_text(str(row.get("text") or ""))) not in heldout_hashes
    ]

    achieved = Counter(row_language_bucket(row) for row in heldout)
    for bucket, fraction in LANGUAGE_BALANCE_TARGETS.items():
        if abs(achieved.get(bucket, 0) / heldout_rows - fraction) > tolerance:
            msg = (
                f"bucket {bucket!r} balance {achieved.get(bucket, 0) / heldout_rows:.3f} "
                f"exceeds tolerance {tolerance} around target {fraction}"
            )
            raise ValueError(
                msg,
            )

    return train, heldout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a Meddies Labels JSONL bundle into no-leakage train/validation splits.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--validation-rows", type=int, default=500)
    parser.add_argument("--min-docs-per-label", type=int, default=50)
    parser.add_argument("--medical-ratio", type=float, default=0.70)
    parser.add_argument("--vi-ratio", type=float, default=0.50)
    parser.add_argument("--en-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    validation_indices = _select_validation_indices(
        rows,
        validation_rows=args.validation_rows,
        min_docs_per_label=args.min_docs_per_label,
        medical_ratio=args.medical_ratio,
        vi_ratio=args.vi_ratio,
        en_ratio=args.en_ratio,
        seed=args.seed,
    )
    train_rows = [row for idx, row in enumerate(rows) if idx not in validation_indices]
    validation_rows = [row for idx, row in enumerate(rows) if idx in validation_indices]

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.validation_output, validation_rows)

    summary: dict[str, Any] = {
        "input_rows": len(rows),
        "train": _summarize(train_rows),
        "validation": _summarize(validation_rows),
        "validation_policy": {
            "validation_rows": args.validation_rows,
            "min_docs_per_label": args.min_docs_per_label,
            "medical_ratio": args.medical_ratio,
            "vi_ratio": args.vi_ratio,
            "en_ratio": args.en_ratio,
            "seed": args.seed,
        },
        "split_validation": _validate_split(train_rows=train_rows, validation_rows=validation_rows),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
