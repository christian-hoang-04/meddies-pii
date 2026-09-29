"""BIOES schema — the contract for the meddies BIOES output space.

The schema sits on top of the entity vocabulary defined in
`annotations.bioes.ENTITY_LABELS` and exposes the 37-class BIOES output classes
plus the JSONL record contract that the BIOES training pipeline consumes
(`parse_labeled_record`) and emits (`prediction_record`, `redaction_record`).

| Concept | Value |
|---|---|
| Entity labels | 9 (address, company_name, date, email_address, human_name, id_number,
phone_number, private_url, secret) |
| Boundary tags | 4 (B, I, E, S) |
| Background | 1 (O) |
| Total output classes | 37 = 9 × 4 + 1 |

Ordering is alphabetical (stable) so the index of any class is deterministic
across model checkpoints — the entity ordering in `annotations.bioes.ENTITY_LABELS`
is the canonical one and must not be reordered without retraining.
"""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
# ruff: file-ignore[ambiguous-unicode-character-docstring]
# reason: the module docstring states the class count as an arithmetic product, so U+00D7 MULTIPLICATION
# reason: SIGN is the operator it means and not a confusable for the letter x. Substituting the letter
# reason: would misstate the derivation this table exists to record.
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Literal

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    build_label_to_id,
)
from meddies_pii.json_types import is_str_mapping
from meddies_pii.spans import CharSpan

__all__ = [
    "BACKGROUND_LABEL",
    "BIOES_LABELS",
    "ENTITY_LABELS",
    "OUTPUT_SCHEMA_VERSION",
    "SPAN_CLASS_NAMES",
    "build_bioes_label_space",
    "build_label_to_id",
    "parse_labeled_record",
    "placeholder",
    "prediction_record",
    "redaction_record",
    "validate_label",
]


BACKGROUND_LABEL: str = "O"
SPAN_CLASS_NAMES: tuple[str, ...] = (BACKGROUND_LABEL, *ENTITY_LABELS)
BIOES_LABELS: tuple[str, ...] = build_bioes_label_space(ENTITY_LABELS)
OUTPUT_SCHEMA_VERSION: int = 1


def placeholder(label: str) -> str:
    """Return the placeholder for redacted output, e.g. `<HUMAN_NAME>`.

    Returns:
        The bracketed placeholder for this label, falling back to ``<REDACTED>`` when the label normalizes to nothing.

    """
    normalized = "".join(ch if ch.isalnum() else "_" for ch in label.upper()).strip("_")
    return f"<{normalized or 'REDACTED'}>"


def validate_label(label: str, *, field: str) -> None:
    if label not in ENTITY_LABELS:
        msg = f"{field} label {label!r} is not in the meddies taxonomy; valid labels: {ENTITY_LABELS}"
        raise ValueError(msg)


# reason: parse labeled keeps check label beside from offsets; splitting would fragment diagnostics.
def parse_labeled_record(  # ruff: ignore[complex-structure,too-many-branches]
    record: Mapping[str, object],
    *,
    default_id: str = "record-0",
    eval_mode: Literal["typed", "untyped"] = "typed",
) -> tuple[str, str, tuple[CharSpan, ...]]:
    """Parse a JSONL record into (example_id, text, spans).

    The record schema accepts either:
    - `text` plus `spans` mapping like `"human_name: Alice": [[0, 5]]`
    - `text` plus `label` list like `[{"category": "human_name", "start": 0, "end": 5}]`

    `eval_mode="typed"` validates each label against the meddies taxonomy;
    `"untyped"` accepts any label string for cases where label-agnostic
    span recall is being measured.

    Returns:
        The record's example id, its text, and its spans as a tuple.

    Raises:
        ValueError: if the text is not a string, the eval mode is unrecognised, the spans are not in one of the two
            accepted shapes, or a label is outside the Meddies taxonomy under ``eval_mode='typed'``.

    """
    text_raw = record.get("text")
    if not isinstance(text_raw, str):
        msg = "record text must be a string"
        raise ValueError(msg)
    info = record.get("info")
    example_id = default_id
    if is_str_mapping(info):
        raw_id = info.get("id")
        if raw_id is not None:
            example_id = str(raw_id)

    if eval_mode not in {"typed", "untyped"}:
        msg = f"unsupported eval_mode: {eval_mode!r}"
        raise ValueError(msg)

    spans_raw = record.get("spans") or {}
    parsed: list[CharSpan] = []
    if spans_raw:
        if not isinstance(spans_raw, Mapping):
            msg = "record spans must be a mapping"
            raise ValueError(msg)
        for raw_key, raw_offsets in spans_raw.items():
            key = str(raw_key)
            label = key.split(": ", 1)[0] if ": " in key else key
            if not label:
                msg = f"spans key {key!r} is missing a label"
                raise ValueError(msg)
            if eval_mode == "typed":
                validate_label(label, field=f"spans.{key}")
            if not isinstance(raw_offsets, Sequence) or isinstance(raw_offsets, (str, bytes)):
                msg = f"spans offsets for {key!r} must be a sequence"
                raise ValueError(msg)
            for offset_idx, raw_pair in enumerate(raw_offsets):
                if not isinstance(raw_pair, Sequence) or isinstance(raw_pair, (str, bytes)) or len(raw_pair) != 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
                    msg = f"span {key!r}[{offset_idx}] must be [start, end]"
                    raise ValueError(msg)
                parsed.append(
                    _span_from_offsets(
                        text=text_raw,
                        label=label,
                        start=raw_pair[0],
                        end=raw_pair[1],
                        field=f"spans.{key}[{offset_idx}]",
                    ),
                )
        return example_id, text_raw, tuple(parsed)

    labels_raw = record.get("label") or []
    if not isinstance(labels_raw, Sequence) or isinstance(labels_raw, (str, bytes)):
        msg = "record label must be a sequence"
        raise ValueError(msg)
    for label_idx, raw_entry in enumerate(labels_raw):
        if not isinstance(raw_entry, Mapping):
            msg = f"label[{label_idx}] must be an object"
            raise ValueError(msg)
        category = raw_entry.get("category")
        if not isinstance(category, str) or not category:
            msg = f"label[{label_idx}] must have a category"
            raise ValueError(msg)
        if eval_mode == "typed":
            validate_label(category, field=f"label[{label_idx}]")
        parsed.append(
            _span_from_offsets(
                text=text_raw,
                label=category,
                start=raw_entry.get("start"),
                end=raw_entry.get("end"),
                field=f"label[{label_idx}]",
            ),
        )
    return example_id, text_raw, tuple(parsed)


def prediction_record(
    *,
    example_id: str,
    text: str,
    spans: Sequence[CharSpan],
) -> OrderedDict[str, object]:
    """Emit a prediction record grouping spans by `"label: text"` keys.

    Returns:
        The record carrying the example id, its text, and the predicted spans grouped under ``"label: text"`` keys.

    """
    predicted_spans: OrderedDict[str, list[list[int]]] = OrderedDict()
    for span in spans:
        key = f"{span.label}: {span.text}"
        predicted_spans.setdefault(key, []).append([span.start, span.end])
    return OrderedDict([
        ("example_id", example_id),
        ("text", text),
        ("predicted_spans", predicted_spans),
    ])


def redaction_record(
    *,
    text: str,
    spans: Sequence[CharSpan],
    output_mode: Literal["typed", "redacted"] = "typed",
    decoded_mismatch: bool = False,
) -> OrderedDict[str, object]:
    """Emit a redaction record with detected spans + redacted text.

    `output_mode="typed"` uses per-label placeholders (e.g. `<HUMAN_NAME>`);
    `"redacted"` collapses every span to the literal `<REDACTED>` placeholder.

    Returns:
        The record carrying the schema version, a summary of the output mode and span count, the detected spans, and the
        redacted text.

    Raises:
        ValueError: if a span carries a label outside the Meddies taxonomy.

    """
    if output_mode not in {"typed", "redacted"}:
        msg = f"unsupported output_mode: {output_mode!r}"
        raise ValueError(msg)

    redacted_text_parts: list[str] = []
    detected_spans: list[OrderedDict[str, object]] = []
    cursor = 0
    labels_for_summary: list[str] = []
    for span in sorted(spans, key=lambda item: (item.start, -(item.end - item.start), item.label)):
        if span.start < cursor or span.end <= span.start:
            continue
        output_label = "redacted" if output_mode == "redacted" else span.label
        redaction_placeholder = "<REDACTED>" if output_mode == "redacted" else placeholder(span.label)
        redacted_text_parts.extend((text[cursor : span.start], redaction_placeholder))
        cursor = span.end
        labels_for_summary.append(output_label)
        detected_spans.append(
            OrderedDict([
                ("label", output_label),
                ("start", span.start),
                ("end", span.end),
                ("text", span.text),
                ("placeholder", redaction_placeholder),
            ]),
        )
    redacted_text_parts.append(text[cursor:])

    by_label = {label: labels_for_summary.count(label) for label in sorted(set(labels_for_summary))}
    return OrderedDict([
        ("schema_version", OUTPUT_SCHEMA_VERSION),
        (
            "summary",
            OrderedDict([
                ("output_mode", output_mode),
                ("span_count", len(detected_spans)),
                ("by_label", by_label),
                ("decoded_mismatch", decoded_mismatch),
            ]),
        ),
        ("text", text),
        ("detected_spans", detected_spans),
        ("redacted_text", "".join(redacted_text_parts)),
    ])


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field} must be an integer character offset"
        raise ValueError(msg)
    return value


def _span_from_offsets(
    *,
    text: str,
    label: str,
    start: object,
    end: object,
    field: str,
) -> CharSpan:
    start_int = _require_int(start, field=f"{field}.start")
    end_int = _require_int(end, field=f"{field}.end")
    if not (0 <= start_int <= end_int <= len(text)):
        msg = f"{field} span ({start_int}, {end_int}) is invalid for text length {len(text)}"
        raise ValueError(msg)
    return CharSpan(
        start=start_int,
        end=end_int,
        text=text[start_int:end_int],
        label=label,
    )
