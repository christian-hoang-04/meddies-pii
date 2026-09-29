from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.training.bioes.data.augmentation_artifacts import text_hash
from meddies_pii.training.bioes.data.corpus_gate import (
    Cell,
    CorpusCounts,
    GateReport,
    count_corpus,
    evaluate_counts,
    merge_counts,
)
from meddies_pii.training.bioes.data.corpus_manifest import (
    MANIFEST_FILENAME,
    write_manifest,
)
from meddies_pii.training.bioes.data.mixed_build import (
    DEFAULT_ENTROPY_TARGETS,
    DEFAULT_LABEL_FLOORS,
)
from meddies_pii.training.bioes.data.mixed_sources import load_local_external_rows
from meddies_pii.training.bioes.data.span_quality import clean_spans
from meddies_pii.training.bioes.data.splits import normalize_text

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

MAX_DROPPED_LABEL_RATIO = 0.4

DEFAULT_MIX_PATH = Path(f"mix/train.{LEGACY_ARTIFACT_TOKEN}.jsonl")

_SOURCE_PATTERNS: tuple[str, ...] = (
    f"base/*.train.{LEGACY_ARTIFACT_TOKEN}.jsonl",
    "hf_configs/pii-bioes.train.jsonl",
    f"internal/hf_configs/*.{LEGACY_ARTIFACT_TOKEN}.jsonl",
    f"internal/nvidia/*.{LEGACY_ARTIFACT_TOKEN}.jsonl",
    f"internal/ai4privacy/*.{LEGACY_ARTIFACT_TOKEN}.jsonl",
    "internal/grpo/*.jsonl",
    "synthetic/*.jsonl",
    "synthetic/**/accepted*.jsonl",
    "synthetic/**/pm_accepted*.jsonl",
)


class AssemblyGateError(RuntimeError):
    """Raised when a corpus fails the hard ADR 0008 assembly gate."""


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    output_path: Path
    manifest_path: Path
    mix_rows: int
    total_before_dedup: int
    duplicate_text_dropped: int
    rows_missing_text_dropped: int
    bad_spans_dropped: int
    docs_quarantined: int
    per_source_rows: dict[str, int]
    external_dropped: dict[str, int]
    label_span_counts: dict[str, int]
    raw_gate: GateReport
    gate: GateReport
    mix_recipe: dict[str, Any] = field(repr=False)
    manifest: dict[str, Any] = field(repr=False)

    def to_summary(self) -> dict[str, Any]:
        return {
            "output": str(self.output_path),
            "manifest": str(self.manifest_path),
            "mix_rows": self.mix_rows,
            "total_before_dedup": self.total_before_dedup,
            "duplicate_text_dropped": self.duplicate_text_dropped,
            "rows_missing_text_dropped": self.rows_missing_text_dropped,
            "bad_spans_dropped": self.bad_spans_dropped,
            "docs_quarantined": self.docs_quarantined,
            "per_source_rows": self.per_source_rows,
            "external_dropped": self.external_dropped,
            "label_span_counts": self.label_span_counts,
            "gate_passed": self.gate.passed,
            "gate_label_floor_failures": self.gate.label_floor_failures,
            "gate_zero_cells": len(self.gate.zero_cells),
            "gate_sparse_cells": len(self.gate.sparse_cells),
            "gate_waived_cells": [list(cell) for cell in self.gate.waived_cells],
            "gate_entropy": self.gate.entropy,
            "gate_blocking_reasons": self.gate.blocking_reasons,
        }


def discover_assembly_source_files(corpus_root: Path) -> list[Path]:
    """Return local BIOES source JSONL files in deterministic assembly order.

    Returns:
        The source files, pattern order first and sorted within each pattern, with duplicates
        removed by resolved path. Order is fixed rather than filesystem-dependent because the
        mix is deduped as it is built, so which of two identical rows survives -- and therefore
        the assembled corpus -- would otherwise depend on directory iteration order.

    """
    root = Path(corpus_root)
    source_files: list[Path] = []
    seen: set[Path] = set()
    for pattern in _SOURCE_PATTERNS:
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            source_files.append(path)
    return source_files


def parse_waiver_cells(values: Iterable[str]) -> frozenset[Cell]:
    waivers: set[Cell] = set()
    for value in values:
        language, sep, label = value.partition(":")
        if sep and language and label:
            waivers.add((language, label))
    return frozenset(waivers)


# reason: assemble bioes v2 combines kept rows and write; splitting would split cleanup from writes.
def assemble_bioes_v2_corpus(  # ruff: ignore[too-many-arguments,too-many-locals]
    corpus_root: Path,
    *,
    output_path: Path | None = None,
    label_floors: Mapping[str, int] | None = None,
    entropy_targets: Mapping[str, float] | None = None,
    cell_min: int = 1,
    waivers: Iterable[Cell] = (),
    clean_bad_spans: bool = True,
    enforce_gate: bool = True,
) -> AssemblyResult:
    """Assemble local BIOES sources into one deduped mix plus MANIFEST.

    The package owns the invariant path: discover local files, convert staged
    external rows, dedupe by ``text_hash(normalize_text(text))``, apply the
    acceptance-side span filter, evaluate the ADR 0008 gate, then write the mix
    and rich manifest. Scripts and Modal only adapt arguments and storage.

    Returns:
        The assembly result: the written mix path, the per-source and dedup counts, and the
        gate verdict with its blocking reasons. The verdict is returned whether or not it
        passed, so a report-only run still sees exactly what an enforcing run would have
        refused.

    Raises:
        AssemblyGateError: When ``enforce_gate`` is set and the ADR 0008 gate did not pass,
            with every blocking reason joined into the message. The temporary output is
            unlinked before the raise and the real output is only replaced after it, so a
            blocked assembly leaves the previous mix in place rather than a partial new one.

    """
    root = Path(corpus_root)
    root = root if root.is_absolute() else Path.cwd() / root
    output = Path(output_path) if output_path is not None else DEFAULT_MIX_PATH
    output = output if output.is_absolute() else root / output
    output_relative = _relative_to_root(output, root)
    tmp_output = output.with_suffix(f"{output.suffix}.tmp")

    source_files = discover_assembly_source_files(root)
    seen_text_hashes: set[str] = set()
    per_source_rows: dict[str, int] = {}
    external_dropped: Counter[str] = Counter()
    total_before_dedup = 0
    duplicate_text_dropped = 0
    rows_missing_text_dropped = 0
    bad_spans_dropped = 0
    docs_quarantined = 0
    running_counts = CorpusCounts()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tmp_output.open("w", encoding="utf-8") as out:
        for source_path in source_files:
            source_key = _source_key(source_path, root)
            source_rows = list(_iter_jsonl_records(source_path))
            (
                kept_rows,
                total_before_dedup,
                duplicate_text_dropped,
                rows_missing_text_dropped,
                bad_spans_dropped,
                docs_quarantined,
            ) = _write_kept_rows(
                out=out,
                rows=source_rows,
                seen_text_hashes=seen_text_hashes,
                total_before_dedup=total_before_dedup,
                duplicate_text_dropped=duplicate_text_dropped,
                rows_missing_text_dropped=rows_missing_text_dropped,
                bad_spans_dropped=bad_spans_dropped,
                docs_quarantined=docs_quarantined,
                clean_bad_spans=clean_bad_spans,
            )
            per_source_rows[source_key] = len(kept_rows)
            running_counts = merge_counts([running_counts, count_corpus(kept_rows)])

        external_dir = root / "external"
        if external_dir.exists():
            external_rows, dropped = load_local_external_rows(external_dir)
            external_dropped.update(dropped)
            (
                kept_rows,
                total_before_dedup,
                duplicate_text_dropped,
                rows_missing_text_dropped,
                bad_spans_dropped,
                docs_quarantined,
            ) = _write_kept_rows(
                out=out,
                rows=external_rows,
                seen_text_hashes=seen_text_hashes,
                total_before_dedup=total_before_dedup,
                duplicate_text_dropped=duplicate_text_dropped,
                rows_missing_text_dropped=rows_missing_text_dropped,
                bad_spans_dropped=bad_spans_dropped,
                docs_quarantined=docs_quarantined,
                clean_bad_spans=clean_bad_spans,
            )
            per_source_rows["external"] = len(kept_rows)
            running_counts = merge_counts([running_counts, count_corpus(kept_rows)])

    raw_gate = evaluate_counts(
        running_counts,
        label_floors=dict(label_floors or DEFAULT_LABEL_FLOORS),
        cell_min=cell_min,
        entropy_targets=dict(entropy_targets or DEFAULT_ENTROPY_TARGETS),
    )
    waived_cells = frozenset(waivers)
    gate = evaluate_counts(
        running_counts,
        label_floors=dict(label_floors or DEFAULT_LABEL_FLOORS),
        cell_min=cell_min,
        entropy_targets=dict(entropy_targets or DEFAULT_ENTROPY_TARGETS),
        waivers=waived_cells,
    )
    if enforce_gate and not gate.passed:
        tmp_output.unlink(missing_ok=True)
        raise AssemblyGateError("Assembly gate BLOCKED (ADR 0008 §2/§3): " + "; ".join(gate.blocking_reasons))

    tmp_output.replace(output)
    mix_recipe: dict[str, Any] = {
        "assembled": "union+dedup+gate (base is balanced; net-new added)",
        "per_source_rows": per_source_rows,
        "total_before_dedup": total_before_dedup,
        "duplicate_text_dropped": duplicate_text_dropped,
        "rows_missing_text_dropped": rows_missing_text_dropped,
        "external_dropped": dict(external_dropped),
        "dedup": "text_hash(normalize_text(text))",
        "bad_spans_dropped": bad_spans_dropped,
        "docs_quarantined": docs_quarantined,
        "waived_cells": [list(cell) for cell in sorted(waived_cells)],
        "gate_raw": asdict(raw_gate),
        "gate": asdict(gate),
    }
    manifest = write_manifest(
        root,
        datasets=[
            {
                "id": "bioes-v2-mix",
                "role": "mix",
                "current_path": output_relative.as_posix(),
                "row_count": sum(per_source_rows.values()),
                "hash_file": True,
            },
        ],
        mix_recipe=mix_recipe,
    )
    return AssemblyResult(
        output_path=output,
        manifest_path=root / MANIFEST_FILENAME,
        mix_rows=sum(per_source_rows.values()),
        total_before_dedup=total_before_dedup,
        duplicate_text_dropped=duplicate_text_dropped,
        rows_missing_text_dropped=rows_missing_text_dropped,
        bad_spans_dropped=bad_spans_dropped,
        docs_quarantined=docs_quarantined,
        per_source_rows=per_source_rows,
        external_dropped=dict(external_dropped),
        label_span_counts=dict(running_counts.label_span_counts),
        raw_gate=raw_gate,
        gate=gate,
        mix_recipe=mix_recipe,
        manifest=manifest,
    )


# reason: write kept rows keeps out/clean bad at its adapter seam; bundling would hide required inputs.
def _write_kept_rows(  # ruff: ignore[too-many-arguments]
    *,
    out: TextIO,
    rows: Sequence[Mapping[str, object]],
    seen_text_hashes: set[str],
    total_before_dedup: int,
    duplicate_text_dropped: int,
    rows_missing_text_dropped: int,
    bad_spans_dropped: int,
    docs_quarantined: int,
    clean_bad_spans: bool,
) -> tuple[list[dict[str, object]], int, int, int, int, int]:
    kept_rows: list[dict[str, object]] = []
    for raw_row in rows:
        total_before_dedup += 1
        row = dict(raw_row)
        text = row.get("text")
        if not isinstance(text, str):
            rows_missing_text_dropped += 1
            continue
        digest = text_hash(normalize_text(text))
        if digest in seen_text_hashes:
            duplicate_text_dropped += 1
            continue
        seen_text_hashes.add(digest)
        if clean_bad_spans:
            labels = row.get("label")
            if isinstance(labels, list) and labels:
                # reason: the list holds unvalidated corpus spans. `clean_spans` already assumes
                # reason: every item is a mapping and raises on anything else, and the quarantine
                # reason: ratio below divides by this exact length, so filtering the list to
                # reason: satisfy the annotation would both swallow a malformed row and change
                # reason: the denominator that decides whether the document is quarantined.
                cleaned, dropped = clean_spans(labels)
                if dropped:
                    if dropped / len(labels) > MAX_DROPPED_LABEL_RATIO:
                        docs_quarantined += 1
                        continue
                    row["label"] = [dict(span) for span in cleaned]
                    bad_spans_dropped += dropped
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept_rows.append(row)
    return (
        kept_rows,
        total_before_dedup,
        duplicate_text_dropped,
        rows_missing_text_dropped,
        bad_spans_dropped,
        docs_quarantined,
    )


def _iter_jsonl_records(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _relative_to_root(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        msg = "--output must be inside the corpus root"
        raise ValueError(msg) from exc


def _source_key(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
