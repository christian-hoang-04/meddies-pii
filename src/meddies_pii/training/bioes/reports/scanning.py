from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import json
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.annotations.span_rendering import highlight_spans, snippet_html

from .models import LabelExample, RowPreview, TrainingReportOptions, TrainingScan

if TYPE_CHECKING:
    from pathlib import Path


def scan_training_jsonl(options: TrainingReportOptions) -> TrainingScan:
    """Scan training JSONL once for counts, examples, and row previews.

    Returns:
        The scan, forwarded from ``scan_labeled_jsonl`` with the options unpacked. One pass
        is the point: counts, per-label examples and previews all come from the same read.

    """
    return scan_labeled_jsonl(
        options.train_jsonl,
        rows_per_label=options.rows_per_label,
        row_table_limit=options.row_table_limit,
        include_all_rows=options.include_all_rows,
        snippet_context=options.snippet_context,
    )


def scan_labeled_jsonl(
    jsonl_path: Path,
    *,
    rows_per_label: int,
    row_table_limit: int,
    include_all_rows: bool,
    snippet_context: int,
) -> TrainingScan:
    """Scan a Meddies Labels label-list JSONL once for counts, examples, and previews.

    Returns:
        The populated scan: rows seen, span count, per-label counts, the row previews up to
        the resolved limit, and one entry in ``invalid_rows`` per record that failed to
        parse. A malformed line is recorded and skipped, never fatal, so one bad row cannot
        cost the whole scan.

    Raises:
        ValueError: Raised internally when a decoded record is not a JSON object, and caught
            in the same loop -- it is how a bad row reaches ``invalid_rows`` rather than the
            caller. Nothing propagates out of this function for a malformed record.

    """
    scan = TrainingScan()
    resolved_row_table_limit = None if include_all_rows else max(0, row_table_limit)
    with jsonl_path.open(encoding="utf-8") as fh:
        for row_index, line in enumerate(fh):
            if not line.strip():
                continue
            scan.rows_seen += 1
            try:
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    msg = "record must be a JSON object"
                    # reason: the raise and its catch are one mechanism, and the docstring above already states
                    # reason: it: this is how a bad row reaches `invalid_rows` instead of the caller. Moving the
                    # reason: raise into a helper would put it outside the try that gives it its meaning.
                    raise ValueError(msg)  # ruff: ignore[raise-within-try]
                example_id, text, spans = parse_labeled_record(record, default_id=f"row-{row_index}")
            except ValueError as exc:
                scan.invalid_rows.append(f"line {row_index + 1}: {exc}")
                continue

            labels = tuple(sorted({span.label for span in spans}))
            scan.span_count += len(spans)
            scan.label_counts.update(span.label for span in spans)

            if resolved_row_table_limit is None or len(scan.row_previews) < resolved_row_table_limit:
                scan.row_previews.append(
                    RowPreview(
                        row_index=row_index,
                        example_id=example_id,
                        labels=labels,
                        text_len=len(text),
                        span_count=len(spans),
                        snippet_html=snippet_html(
                            text,
                            spans,
                            focus_span=spans[0] if spans else None,
                            context=snippet_context,
                        ),
                    ),
                )

            for span in spans:
                examples = scan.label_examples.setdefault(span.label, [])
                if len(examples) >= rows_per_label:
                    continue
                examples.append(
                    LabelExample(
                        row_index=row_index,
                        example_id=example_id,
                        label=span.label,
                        span_text=span.text,
                        snippet_html=snippet_html(
                            text,
                            spans,
                            focus_span=span,
                            context=snippet_context,
                        ),
                    ),
                )
    return scan


def scan_audit_jsonl(audit_jsonl: Path, samples_per_rule: int) -> dict[str, list[dict[str, object]]]:
    """Collect bounded samples per migration decision rule.

    Returns:
        Samples keyed ``<decision>:<rule>``, sorted by key, each bucket capped at
        ``samples_per_rule``. The cap is what keeps an audit of any size reviewable; a line
        that will not decode, or whose decisions are not a list, is skipped silently because
        this is a sampler rather than a validator.

    """
    samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    with audit_jsonl.open(encoding="utf-8") as fh:
        for line_index, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            decisions = record.get("decisions")
            if not isinstance(decisions, list):
                continue
            for decision in decisions:
                if not isinstance(decision, Mapping):
                    continue
                key = f"{decision.get('decision')}:{decision.get('rule')}"
                bucket = samples[key]
                if len(bucket) >= samples_per_rule:
                    continue
                bucket.append({
                    "uid": str(record.get("uid", f"audit-line-{line_index + 1}")),
                    "status": str(record.get("status", "")),
                    "old_label": str(decision.get("old_label", "")),
                    "new_label": str(decision.get("new_label", "")),
                    "text": str(decision.get("text", "")),
                    "start": decision.get("start"),
                    "end": decision.get("end"),
                })
    return dict(sorted(samples.items()))


def scan_adversarial_jsonl(path: Path) -> list[RowPreview]:
    """Read all adversarial rows as highlighted previews.

    Returns:
        One preview per row, in file order, with its spans highlighted. Unlike the training
        scan this applies no row limit -- the adversarial set is meant to be read whole.

    """
    previews: list[RowPreview] = []
    with path.open(encoding="utf-8") as fh:
        for row_index, line in enumerate(fh):
            if not line.strip():
                continue
            record = json.loads(line)
            example_id, text, spans = parse_labeled_record(record, default_id=f"adversarial-{row_index}")
            previews.append(
                RowPreview(
                    row_index=row_index,
                    example_id=example_id,
                    labels=tuple(sorted({span.label for span in spans})),
                    text_len=len(text),
                    span_count=len(spans),
                    snippet_html=highlight_spans(text, spans),
                ),
            )
    return previews
