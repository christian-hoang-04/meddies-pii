from __future__ import annotations

# ruff: file-ignore[import-private-name]
# reason: targeted evaluation must reuse `_select_source_rows` and `_prepare_rows` so its adversarial slices apply
# reason: the same source filtering, BIOES parsing, and statistics contract as training. These are private data
# reason: preparation mechanics, intentionally consumed only by the adjacent BIOES evaluation Module.
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from meddies_pii.annotations.bioes import ENTITY_LABELS
from meddies_pii.annotations.tagged_text import (
    normalize_label_json,
    parse_tagged_text,
    spans_to_label_json,
)
from meddies_pii.training.bioes.data.preparation import (
    PreparationStats,
    PreparedRow,
    _prepare_rows,
    _select_source_rows,
)

from .harness import ADVERSARIAL_SLICE_NAMES, new_slice_filter_report

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class TargetedSliceSelection:
    slice_name: str
    prepared_rows: tuple[PreparedRow, ...]
    source_support_docs: int
    prepared_support_docs: int
    source_stats: PreparationStats
    preparation_stats: PreparationStats
    slice_filter_report: dict[str, dict[str, Any]]

    def to_report(self) -> dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "source_support_docs": self.source_support_docs,
            "prepared_support_docs": self.prepared_support_docs,
            "prepared_row_uids": [row.uid for row in self.prepared_rows],
            "source_stats": asdict(self.source_stats),
            "preparation_stats": asdict(self.preparation_stats),
            "slice_filter_report": self.slice_filter_report,
        }


def _obfuscate_email_surface_at_dot(surface: str) -> str | None:
    if "@" not in surface:
        return None
    local, domain = surface.split("@", 1)
    if not local or "." not in domain:
        return None
    return f"{local} (at) {domain.replace('.', ' (dot) ')}"


def at_dot_obfuscate_email_rows(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Derive parseable at/dot adversarial rows from real tagged email rows.

    The tagged-text format uses square brackets for entity boundaries, so a
    gold email span cannot safely contain literal ``[at]`` or ``[dot]`` tokens.
    We use the parenthesized variant instead: ``name (at) example (dot) com``.
    This keeps the row parseable while still exercising the same adversarial
    slice family.

    Returns:
        The derived rows, at most ``limit`` when one is given. Only rows carrying a real gold
        email span can be derived from, so the result is usually shorter than the input and
        an empty list means the input held no usable email rows rather than that derivation
        failed. The parenthesized ``(at)``/``(dot)`` form is load-bearing: the square-bracket
        variant would collide with the tagged-text entity delimiters and stop parsing.

    """
    derived: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or "email_address" not in text:
            continue
        parsed = parse_tagged_text(text)
        updated_text = text
        changed = False
        for span in parsed.spans:
            if span.label != "email_address":
                continue
            obfuscated = _obfuscate_email_surface_at_dot(span.text)
            if obfuscated is None:
                continue
            original_tag = f"[{span.text}]<{span.label}>"
            obfuscated_tag = f"[{obfuscated}]<{span.label}>"
            if original_tag not in updated_text:
                continue
            updated_text = updated_text.replace(original_tag, obfuscated_tag, 1)
            changed = True

        if not changed:
            continue
        reparsed = parse_tagged_text(updated_text)
        derived_row = dict(row)
        source_uid = str(
            row.get("uid")
            or row.get("source")
            or (f"dataset-index-{row['_dataset_index']}" if "_dataset_index" in row else f"row-{index}"),
        )
        derived_row["uid"] = f"{source_uid}:at-dot-obfuscation"
        derived_row["source_uid"] = source_uid
        derived_row["text"] = updated_text
        derived_row["label"] = json.dumps(
            normalize_label_json(spans_to_label_json(reparsed.spans)),
            ensure_ascii=False,
            sort_keys=True,
        )
        derived_row["adversarial_derivation"] = {
            "kind": "email_at_dot_obfuscation",
            "source_uid": source_uid,
        }
        derived.append(derived_row)
        if limit is not None and len(derived) >= limit:
            break
    return derived


# reason: select prepared exposes rows/require as its public contract; bundling would break callers.
def select_prepared_rows_for_adversarial_slice(  # ruff: ignore[too-many-arguments]
    rows: Sequence[dict[str, Any]],
    # reason: the tokenizer arrives from `AutoTokenizer.from_pretrained`, which the pinned transformers declares as
    # reason: `Unknown | TokenizersBackend | None | SentencePieceBackend`. There is no single honest static type to
    # reason: write, and spelling the union out would add an `Unknown` arm and a `None` the pin never returns.
    tokenizer: Any,  # ruff: ignore[any-type]
    *,
    slice_name: str,
    max_length: int,
    id_to_label: dict[int, str],
    limit: int | None = None,
    entity_labels: Sequence[str] = ENTITY_LABELS,
    allow_label_repairs: bool = False,
    require_label_json: bool = True,
) -> TargetedSliceSelection:
    """Select prepared eval rows carrying a source-derived adversarial slice.

    This is a CPU/data-prep gate, not a training benchmark. It reuses the same
    source audit and tokenization/round-trip checks as smoke/speed training so
    a non-zero result means the row is actually usable by the BIOES eval path.

    Returns:
        The selected rows together with the filter report showing where rows were dropped.
        Selection runs the full source audit and round-trip checks before filtering to the
        slice, so a row that survives is usable by the real eval path -- an empty selection
        is a data finding worth reading the report over, not a caller error.

    Raises:
        ValueError: When ``slice_name`` is not a known adversarial slice. Checked FIRST,
            before any row work, so a typo'd slice fails immediately rather than after a full
            audit pass that would have returned an empty selection and looked like a data
            problem.

    """
    if slice_name not in ADVERSARIAL_SLICE_NAMES:
        msg = f"slice_name must be one of {ADVERSARIAL_SLICE_NAMES}; got {slice_name!r}"
        raise ValueError(msg)

    slice_filter_report = new_slice_filter_report()
    source_rows, source_stats = _select_source_rows(
        rows,
        limit=None,
        allow_label_repairs=allow_label_repairs,
        require_label_json=require_label_json,
        sort_by_length=False,
        slice_filter_report=slice_filter_report,
    )
    slice_source_rows = [row for row in source_rows if slice_name in row.slices]
    prepared_rows, preparation_stats = _prepare_rows(
        slice_source_rows,
        tokenizer,
        max_length=max_length,
        limit=None,
        id_to_label=id_to_label,
        entity_labels=entity_labels,
        slice_filter_report=slice_filter_report,
    )
    selected_rows = tuple(prepared_rows if limit is None else prepared_rows[:limit])
    return TargetedSliceSelection(
        slice_name=slice_name,
        prepared_rows=selected_rows,
        source_support_docs=len(slice_source_rows),
        prepared_support_docs=len(prepared_rows),
        source_stats=source_stats,
        preparation_stats=preparation_stats,
        slice_filter_report=slice_filter_report,
    )
