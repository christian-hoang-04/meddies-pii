"""Source-row auditing and tokenized-row preparation for BIOES training.

Splits the BIOES training pipeline's data layer from the trainer:

- `_audit_source_row`: gates each raw HF row through label-repair, span
  presence, and label-JSON consistency checks.
- `_select_source_rows`: drives auditing across a row sequence, applies
  optional length sorting and limit, raises if the limit can't be met.
- `_prepare_rows`: tokenizes audited rows, aligns BIOES tags via
  `tokenize_and_align`, decodes the round-trip, rejects mismatches, and treats
  `limit` as an upper bound on prepared rows.

The dataclasses (`PreparedRow`, `PreparationStats`, `AuditedSourceRow`)
and helpers (`_record_skip`, `_row_uid`, `_span_signature`,
`MAX_SKIP_EXAMPLES`) live here too — they're tightly bound to the
preparation pipeline and have no other consumers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, TypeGuard

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    TokenizedExample,
    decode_bioes_from_offsets,
    tokenize_and_align,
)
from meddies_pii.annotations.tagged_text import (
    ParsedTaggedDocument,
    label_json_from_string,
    normalize_label_json,
    parse_tagged_text,
    spans_to_label_json,
)
from meddies_pii.spans import CharSpan
from meddies_pii.training.bioes.eval.harness import (
    SliceFilterReport,
    classify_adversarial_slices,
    record_slice_event,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from meddies_pii.training.bioes.data.record_schema import Record

MAX_SKIP_EXAMPLES = 10


@dataclass(slots=True)
class PreparedRow:
    uid: str
    raw: str
    provided_label_json: dict[str, list[str]]
    parsed_label_json: dict[str, list[str]]
    parsed: ParsedTaggedDocument
    tokenized: TokenizedExample
    slices: tuple[str, ...] = ("all",)


@dataclass(slots=True)
class PreparationStats:
    candidates: int = 0
    accepted: int = 0
    skipped_empty_text: int = 0
    skipped_no_spans: int = 0
    skipped_label_repairs: int = 0
    skipped_missing_label_json: int = 0
    skipped_empty_label_json: int = 0
    skipped_label_mismatch: int = 0
    skipped_truncated: int = 0
    skipped_alignment: int = 0
    skipped_round_trip: int = 0
    skip_examples: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class AuditedSourceRow:
    uid: str
    source: Record
    parsed: ParsedTaggedDocument
    provided_label_json: dict[str, list[str]]
    parsed_label_json: dict[str, list[str]]
    slices: tuple[str, ...] = ("all",)


def _record_skip(stats: PreparationStats, *, uid: str, stage: str, reason: str) -> None:
    if len(stats.skip_examples) >= MAX_SKIP_EXAMPLES:
        return
    stats.skip_examples.append({
        "uid": uid,
        "stage": stage,
        "reason": reason,
    })


def _span_signature(spans: Sequence[CharSpan]) -> list[tuple[str, int, int, str]]:
    return [(span.label, span.start, span.end, span.text) for span in spans]


def _row_uid(row: Mapping[str, object], idx: int) -> str:
    return str(row.get("uid") or row.get("source") or f"row-{idx}")


def _looks_like_tagged_text(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and "]<" in value


def _source_tagged_text(row: Mapping[str, object]) -> str | None:
    """Return the tagged source document for a Meddies HF-style row.

    Current `Meddies/meddies-pii` rows store inline-tagged text in `text` and
    plain de-tagged text in `raw`. Older/synthetic rows may use only `raw`.
    Prefer the field that visibly contains inline tags so selection does not
    silently train on the plain text column.

    Returns:
        The field that visibly carries inline tags, preferring ``text`` over ``raw``; failing
        that, whichever of the two is a string; ``None`` when neither is. Preferring the tagged
        field by inspection rather than by name is what stops a row whose columns are swapped
        from training the tagger on de-tagged text with no spans to find.

    """
    text = row.get("text")
    raw = row.get("raw")
    if _looks_like_tagged_text(text):
        return text
    if _looks_like_tagged_text(raw):
        return raw
    if isinstance(text, str):
        return text
    if isinstance(raw, str):
        return raw
    return None


def _label_list_spans(text: str, label_entries: object) -> tuple[CharSpan, ...] | None:
    """Parse OPF/Meddies Labels JSONL label-list records into character spans.

    Returns:
        The parsed spans, or ``None`` when the entries are not a non-string sequence. ``None``
        means this row does not use the label-list shape at all, which the caller distinguishes
        from an empty tuple -- a row that uses the shape and legitimately carries no span.

    """
    if not isinstance(label_entries, Sequence) or isinstance(label_entries, (str, bytes)):
        return None
    spans: list[CharSpan] = []
    for entry in label_entries:
        if not isinstance(entry, Mapping):
            continue
        category = entry.get("category")
        start = entry.get("start")
        end = entry.get("end")
        # reason: label list spans keeps category/entity in one gate; helper predicates would scatter the rule.
        if (
            not isinstance(category, str)  # ruff: ignore[too-many-boolean-expressions]
            or category not in ENTITY_LABELS
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(text)
        ):
            continue
        expected_text = entry.get("text")
        span_text = text[start:end]
        if isinstance(expected_text, str) and expected_text != span_text:
            continue
        spans.append(CharSpan(start=start, end=end, text=span_text, label=category))
    return tuple(spans)


def _audit_label_list_row(
    row: Record,
    *,
    uid: str,
    stats: PreparationStats,
    slice_filter_report: SliceFilterReport | None = None,
) -> AuditedSourceRow | None:
    text = row.get("text")
    if not isinstance(text, str) or not text:
        return None
    label_spans = _label_list_spans(text, row.get("label"))
    if label_spans is None:
        return None

    slices = classify_adversarial_slices(text, label_spans)
    record_slice_event(slice_filter_report, slices, stage="attempted")
    if not label_spans:
        record_slice_event(slice_filter_report, slices, stage="source_rejected", reason="no_spans")
        stats.skipped_no_spans += 1
        _record_skip(stats, uid=uid, stage="source_selection", reason="no_spans")
        return None

    parsed = ParsedTaggedDocument(
        text=text,
        normalized_text=text,
        had_label_repairs=False,
        raw=text,
        spans=label_spans,
    )
    parsed_label_json = normalize_label_json(spans_to_label_json(label_spans))
    record_slice_event(slice_filter_report, slices, stage="source_accepted")
    return AuditedSourceRow(
        uid=uid,
        source=dict(row),
        parsed=parsed,
        provided_label_json=parsed_label_json,
        parsed_label_json=parsed_label_json,
        slices=slices,
    )


# reason: audit source row combines tagged text and row uid; splitting would fragment diagnostics.
def _audit_source_row(  # ruff: ignore[too-many-return-statements,too-many-arguments]
    row: Record,
    *,
    idx: int,
    stats: PreparationStats,
    allow_label_repairs: bool,
    require_label_json: bool,
    slice_filter_report: SliceFilterReport | None = None,
) -> AuditedSourceRow | None:
    uid = _row_uid(row, idx)
    label_list_row = _audit_label_list_row(row, uid=uid, stats=stats, slice_filter_report=slice_filter_report)
    if label_list_row is not None:
        return label_list_row

    text = _source_tagged_text(row)
    if not text:
        record_slice_event(slice_filter_report, ("all",), stage="attempted")
        record_slice_event(slice_filter_report, ("all",), stage="source_rejected", reason="empty_text")
        stats.skipped_empty_text += 1
        _record_skip(stats, uid=uid, stage="source_selection", reason="empty_text")
        return None

    repaired_parse = parse_tagged_text(text, normalize_labels=True)
    slices = classify_adversarial_slices(str(text), repaired_parse.spans)
    record_slice_event(slice_filter_report, slices, stage="attempted")
    if repaired_parse.had_label_repairs and not allow_label_repairs:
        record_slice_event(
            slice_filter_report,
            slices,
            stage="source_rejected",
            reason="label_repairs",
        )
        stats.skipped_label_repairs += 1
        _record_skip(stats, uid=uid, stage="source_selection", reason="label_repairs")
        return None

    parsed = repaired_parse if allow_label_repairs else parse_tagged_text(text, normalize_labels=False)
    slices = classify_adversarial_slices(str(text), parsed.spans)
    if not parsed.spans:
        record_slice_event(slice_filter_report, slices, stage="source_rejected", reason="no_spans")
        stats.skipped_no_spans += 1
        _record_skip(stats, uid=uid, stage="source_selection", reason="no_spans")
        return None

    raw_label = row.get("label")
    if raw_label is None:
        if require_label_json:
            record_slice_event(
                slice_filter_report,
                slices,
                stage="source_rejected",
                reason="missing_label_json",
            )
            stats.skipped_missing_label_json += 1
            _record_skip(stats, uid=uid, stage="source_selection", reason="missing_label_json")
            return None
        provided_label_json = {}
    elif isinstance(raw_label, str):
        provided_label_json = normalize_label_json(label_json_from_string(raw_label))
        if not provided_label_json and require_label_json:
            record_slice_event(
                slice_filter_report,
                slices,
                stage="source_rejected",
                reason="empty_label_json",
            )
            stats.skipped_empty_label_json += 1
            _record_skip(stats, uid=uid, stage="source_selection", reason="empty_label_json")
            return None
    else:
        msg = "record label must be a JSON string when it is present"
        raise TypeError(msg)

    parsed_label_json = normalize_label_json(spans_to_label_json(parsed.spans))
    if provided_label_json and provided_label_json != parsed_label_json:
        record_slice_event(
            slice_filter_report,
            slices,
            stage="source_rejected",
            reason="label_mismatch",
        )
        stats.skipped_label_mismatch += 1
        _record_skip(stats, uid=uid, stage="source_selection", reason="label_mismatch")
        return None

    record_slice_event(slice_filter_report, slices, stage="source_accepted")
    return AuditedSourceRow(
        uid=uid,
        source=dict(row),
        parsed=parsed,
        provided_label_json=provided_label_json,
        parsed_label_json=parsed_label_json,
        slices=slices,
    )


# reason: select source keeps rows/slice filter at its adapter seam; bundling would hide required inputs.
def _select_source_rows(  # ruff: ignore[too-many-arguments]
    rows: Sequence[Record],
    *,
    limit: int | None,
    allow_label_repairs: bool,
    require_label_json: bool,
    sort_by_length: bool = False,
    slice_filter_report: SliceFilterReport | None = None,
) -> tuple[list[AuditedSourceRow], PreparationStats]:
    stats = PreparationStats(candidates=len(rows))
    audited: list[AuditedSourceRow] = []
    for idx, row in enumerate(rows):
        audited_row = _audit_source_row(
            row,
            idx=idx,
            stats=stats,
            allow_label_repairs=allow_label_repairs,
            require_label_json=require_label_json,
            slice_filter_report=slice_filter_report,
        )
        if audited_row is not None:
            audited.append(audited_row)
            stats.accepted += 1

    if sort_by_length:
        audited.sort(key=lambda row: (-len(row.parsed.raw), row.uid))

    selected = audited if limit is None else audited[:limit]
    if limit is not None and len(selected) < limit:
        msg = f"Only selected {len(selected)} source rows for limit={limit}; stats={asdict(stats)}"
        raise RuntimeError(msg)
    return selected, stats


# reason: prepare rows keeps rows/slice filter at its adapter seam; bundling would hide required inputs.
def _prepare_rows(  # ruff: ignore[too-many-arguments]
    rows: Sequence[AuditedSourceRow],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    limit: int | None,
    id_to_label: dict[int, str],
    entity_labels: Sequence[str] = ENTITY_LABELS,
    slice_filter_report: SliceFilterReport | None = None,
) -> tuple[list[PreparedRow], PreparationStats]:
    """Tokenize audited rows and align BIOES tags.

    Returns:
        The prepared rows and the statistics describing what became of the candidates. A row
        that survives auditing can still be dropped here -- no target spans, or tags that will
        not align to the tokenization -- so the prepared count is a lower bound on the audited
        count, and the stats are what explain the gap.

    """
    prepared: list[PreparedRow] = []
    stats = PreparationStats(candidates=len(rows))

    for row in rows:
        record_slice_event(slice_filter_report, row.slices, stage="preparation_attempted")
        target_spans = tuple(row.parsed.spans)
        if not target_spans:
            record_slice_event(
                slice_filter_report,
                row.slices,
                stage="preparation_rejected",
                reason="no_mapped_spans",
            )
            stats.skipped_no_spans += 1
            _record_skip(stats, uid=row.uid, stage="preparation", reason="no_mapped_spans")
            continue
        parsed = ParsedTaggedDocument(
            text=row.parsed.text,
            normalized_text=row.parsed.normalized_text,
            had_label_repairs=row.parsed.had_label_repairs,
            raw=row.parsed.raw,
            spans=target_spans,
        )
        parsed_label_json = normalize_label_json(spans_to_label_json(target_spans))
        try:
            tokenized = tokenize_and_align(
                tokenizer,
                row.parsed.raw,
                target_spans,
                max_length=max_length,
                entity_labels=entity_labels,
            )
        except ValueError:
            record_slice_event(
                slice_filter_report,
                row.slices,
                stage="preparation_rejected",
                reason="alignment",
            )
            stats.skipped_alignment += 1
            _record_skip(stats, uid=row.uid, stage="preparation", reason="alignment")
            continue

        if tokenized.truncated:
            record_slice_event(
                slice_filter_report,
                row.slices,
                stage="preparation_rejected",
                reason="truncated",
            )
            stats.skipped_truncated += 1
            _record_skip(stats, uid=row.uid, stage="preparation", reason="truncated")
            continue

        decoded = decode_bioes_from_offsets(row.parsed.raw, tokenized.offset_mapping, tokenized.labels, id_to_label)
        if _span_signature(decoded) != _span_signature(target_spans):
            record_slice_event(
                slice_filter_report,
                row.slices,
                stage="preparation_rejected",
                reason="round_trip",
            )
            stats.skipped_round_trip += 1
            _record_skip(stats, uid=row.uid, stage="preparation", reason="round_trip")
            continue

        prepared.append(
            PreparedRow(
                uid=row.uid,
                raw=row.parsed.raw,
                provided_label_json=row.provided_label_json,
                parsed_label_json=parsed_label_json,
                parsed=parsed,
                tokenized=tokenized,
                slices=row.slices,
            ),
        )
        record_slice_event(slice_filter_report, row.slices, stage="prepared")
        stats.accepted += 1
        if limit is not None and len(prepared) >= limit:
            break

    return prepared, stats
