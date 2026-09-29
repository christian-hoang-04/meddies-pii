"""Fail-closed company-name support accounting for the packed BIOES corpus."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: the render subcommands write their result to standard output; that output is this module's product.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import argparse
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from meddies_pii.annotations.bioes.vocabulary import (
    ENTITY_LABELS,
    IGNORE_INDEX,
    build_bioes_label_space,
    build_label_to_id,
)
from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.trainers.pins import (
    EVAL_CONFIG,
    EVAL_DATASET_ID,
    EVAL_DATASET_REVISION,
    EVAL_ROWS,
    EVAL_SPLIT,
    PACKED_DATASET_ID,
    PACKED_DATASET_REVISION,
    PACKED_UNIT_COUNT,
)

COMPANY_NAME_AUDIT_CONFIRMATION = "AUDIT_COMPANY_NAME_EXPOSURE"
DEFAULT_MODAL_PROFILE = "meddies-pii"
PREFIX_PACKED_UNITS = 7_680
AUDIT_PINS = {
    "packed": {
        "id": PACKED_DATASET_ID,
        "revision": PACKED_DATASET_REVISION,
        "expected_units": PACKED_UNIT_COUNT,
    },
    "evaluation": {
        "id": EVAL_DATASET_ID,
        "revision": EVAL_DATASET_REVISION,
        "config": EVAL_CONFIG,
        "split": EVAL_SPLIT,
        "expected_rows": EVAL_ROWS,
    },
}
"""The dataset identities and row counts a company-name audit run is pinned to.

Read this before launching an audit: it names each dataset, the revision the audit is valid for,
and the unit or row count the run must observe. The individual constants it collects are the ones
the audit functions take as defaults, so this is the one place to check that a pin still holds.

"""


@dataclass(frozen=True, slots=True)
class PackedCompanyNameAudit:
    label2id: dict[str, int]
    id2label: dict[int, str]
    company_name_token_count: int
    company_name_span_count: int
    company_name_packed_unit_count: int
    company_name_original_row_range_count: int
    malformed_transition_count: int
    processed_packed_units: int
    source_membership_status: str
    observed_schema_fields: tuple[str, ...]
    observed_row_range_fields: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "label2id": self.label2id,
            "id2label": self.id2label,
            "company_name_token_count": self.company_name_token_count,
            "company_name_span_count": self.company_name_span_count,
            "company_name_packed_unit_count": self.company_name_packed_unit_count,
            "company_name_original_row_range_count": self.company_name_original_row_range_count,
            "malformed_transition_count": self.malformed_transition_count,
            "processed_packed_units": self.processed_packed_units,
            "source_membership_status": self.source_membership_status,
            "observed_schema_fields": self.observed_schema_fields,
            "observed_row_range_fields": self.observed_row_range_fields,
        }


@dataclass(frozen=True, slots=True)
class EvalCompanyNameAudit:
    expected_rows: int
    processed_rows: int
    company_name_document_count: int
    company_name_span_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "expected_rows": self.expected_rows,
            "processed_rows": self.processed_rows,
            "company_name_document_count": self.company_name_document_count,
            "company_name_span_count": self.company_name_span_count,
        }


def authoritative_label_vocabulary() -> tuple[str, ...]:
    """Use the same vocabulary source as BIOES training and final evaluation.

    Returns:
        The BIOES label space built from the fixed entity labels. Sharing the builder is the
        point: an audit against a privately-built vocabulary would count a different corpus
        than the one training and evaluation see.

    """
    return build_bioes_label_space(ENTITY_LABELS)


def authoritative_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    label2id = build_label_to_id(authoritative_label_vocabulary())
    return label2id, {label_id: label for label, label_id in label2id.items()}


def _validate_label_maps(
    label2id: Mapping[str, int],
    id2label: Mapping[int, str],
) -> tuple[dict[str, int], dict[int, str]]:
    expected_label2id, expected_id2label = authoritative_label_maps()
    if dict(label2id) != expected_label2id:
        msg = "label2id is missing or inconsistent with BIOES vocabulary"
        raise RuntimeError(msg)
    if dict(id2label) != expected_id2label:
        msg = "id2label is missing or inconsistent with BIOES vocabulary"
        raise RuntimeError(msg)
    return expected_label2id, expected_id2label


def _sequence_of_ints(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        msg = f"packed schema field {field} must be an integer sequence"
        raise RuntimeError(msg)
    narrowed = tuple(item for item in value if isinstance(item, int) and not isinstance(item, bool))
    if len(narrowed) != len(value):
        msg = f"packed schema field {field} contains a non-integer"
        raise RuntimeError(msg)
    return narrowed


def _row_ranges(value: object, label_count: int) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        msg = "packed schema has no truthful row_ranges support"
        raise RuntimeError(msg)
    ranges: list[Mapping[str, object]] = []
    previous_end = 0
    for index, item in enumerate(value):
        if not is_str_mapping(item):
            msg = "packed row range is not an object"
            raise RuntimeError(msg)
        start, end, uid = item.get("start"), item.get("end"), item.get("uid")
        # reason: row ranges keeps start/end in one gate; helper predicates would scatter the rule.
        if (
            isinstance(start, bool)  # ruff: ignore[too-many-boolean-expressions]
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(uid, str)
            or start < previous_end
            or end <= start
            or end > label_count
        ):
            msg = f"packed row range {index} has invalid prefix geometry"
            raise RuntimeError(msg)
        previous_end = end
        ranges.append(item)
    if not ranges:
        msg = "packed schema has no original row ranges"
        raise RuntimeError(msg)
    return tuple(ranges)


class CompanyNameCounter:
    """Streaming counter. It never retains packed units or source text."""

    def __init__(self, label2id: Mapping[str, int], id2label: Mapping[int, str]) -> None:
        self.label2id, self.id2label = _validate_label_maps(label2id, id2label)
        self.company_ids = {self.label2id[f"{prefix}-company_name"] for prefix in ("B", "I", "E", "S")}
        self.company_name_token_count = 0
        self.company_name_span_count = 0
        self.company_name_packed_unit_count = 0
        self.company_name_original_row_range_count = 0
        self.malformed_transition_count = 0
        self.processed_packed_units = 0
        self.observed_schema_fields: set[str] = set()
        self.observed_row_range_fields: set[str] = set()

    def consume(self, unit: Mapping[str, object]) -> None:
        self.observed_schema_fields.update(str(key) for key in unit)
        labels = _sequence_of_ints(unit.get("labels"), "labels")
        ranges = _row_ranges(unit.get("row_ranges"), len(labels))
        self.processed_packed_units += 1
        company_positions = {index for index, label_id in enumerate(labels) if label_id in self.company_ids}
        covered_positions: set[int] = set()
        unit_has_company_name = False
        for row_range in ranges:
            self.observed_row_range_fields.update(str(key) for key in row_range)
            start, end = row_range["start"], row_range["end"]
            if not isinstance(start, int) or not isinstance(end, int):
                msg = "validated packed row range changed shape"
                raise RuntimeError(msg)
            range_ids = labels[start:end]
            covered_positions.update(range(start, end))
            company_in_range = any(label_id in self.company_ids for label_id in range_ids)
            if company_in_range:
                unit_has_company_name = True
                self.company_name_original_row_range_count += 1
            self._consume_original_row_range(range_ids)
        if company_positions - covered_positions:
            msg = "company_name labels fall outside original row-range geometry"
            raise RuntimeError(msg)
        self.company_name_token_count += len(company_positions)
        if unit_has_company_name:
            self.company_name_packed_unit_count += 1

    # reason: CompanyNameCounter combines active and prefix; splitting would split parity state.
    def _consume_original_row_range(self, labels: Sequence[int]) -> None:  # ruff: ignore[complex-structure,too-many-branches]
        active = False
        for label_id in labels:
            if label_id == IGNORE_INDEX:
                if active:
                    self.malformed_transition_count += 1
                    active = False
                continue
            try:
                label = self.id2label[label_id]
            except KeyError as error:
                msg = f"packed labels contain unknown label ID {label_id}"
                raise RuntimeError(msg) from error
            if label == "O" or not label.endswith("-company_name"):
                if active:
                    self.malformed_transition_count += 1
                    active = False
                continue
            prefix = label[0]
            if prefix == "B":
                if active:
                    self.malformed_transition_count += 1
                active = True
            elif prefix == "I":
                if not active:
                    self.malformed_transition_count += 1
            elif prefix == "E":
                if not active:
                    self.malformed_transition_count += 1
                else:
                    self.company_name_span_count += 1
                    active = False
            elif prefix == "S":
                if active:
                    self.malformed_transition_count += 1
                    active = False
                self.company_name_span_count += 1
            else:
                msg = f"unexpected company_name BIOES prefix {prefix!r}"
                raise RuntimeError(msg)
        if active:
            self.malformed_transition_count += 1

    def finish(self, *, required_units: int | None = None) -> PackedCompanyNameAudit:
        """Packed row ranges carry original-row identity.

        Not a source-dataset attribution field with defined semantics. Do not infer provenance.

        Returns:
            The receipt: the label maps, the company_name token, span, packed-unit and
            original-row-range counts, the malformed-transition count, and the schema fields
            actually observed. ``source_membership_status`` is fixed at ``"unavailable"``
            rather than guessed, which is what the note above is about.

        Raises:
            RuntimeError: If a required unit count was given and the run processed a
                different number. A short prefix is refused rather than reported, because
                the receipt's value is that it covers exactly the geometry it names.

        """
        if required_units is not None and self.processed_packed_units != required_units:
            msg = f"packed prefix geometry is short: expected {required_units}, got {self.processed_packed_units}"
            raise RuntimeError(
                msg,
            )
        return PackedCompanyNameAudit(
            label2id=dict(self.label2id),
            id2label=dict(self.id2label),
            company_name_token_count=self.company_name_token_count,
            company_name_span_count=self.company_name_span_count,
            company_name_packed_unit_count=self.company_name_packed_unit_count,
            company_name_original_row_range_count=self.company_name_original_row_range_count,
            malformed_transition_count=self.malformed_transition_count,
            processed_packed_units=self.processed_packed_units,
            source_membership_status="unavailable",
            observed_schema_fields=tuple(sorted(self.observed_schema_fields)),
            observed_row_range_fields=tuple(sorted(self.observed_row_range_fields)),
        )


def audit_packed_units(
    units: Iterable[Mapping[str, object]],
    *,
    label2id: Mapping[str, int],
    id2label: Mapping[int, str],
    max_units: int | None = None,
    required_units: int | None = None,
) -> PackedCompanyNameAudit:
    if max_units is not None and max_units < 1:
        msg = "max_units must be positive"
        raise ValueError(msg)
    counter = CompanyNameCounter(label2id, id2label)
    for unit in units:
        if max_units is not None and counter.processed_packed_units >= max_units:
            break
        counter.consume(unit)
    return counter.finish(required_units=required_units)


def merge_packed_company_name_audits(
    audits: Iterable[PackedCompanyNameAudit],
) -> PackedCompanyNameAudit:
    """Merge complete shard receipts without retaining packed units.

    Returns:
        One receipt summing every shard's counts, with the label maps and schema
        observations carried across. Shard receipts are merged rather than re-parsed, which
        is what keeps a full-corpus audit inside memory.

    Raises:
        RuntimeError: If there are no receipts to merge, chained from the underlying error;
            or if any receipt claims a source-membership status other than unavailable,
            which would assert provenance the packed corpus does not define.

    """
    reports = iter(audits)
    try:
        first = next(reports)
    except StopIteration as error:
        msg = "cannot merge an empty packed company_name audit"
        raise RuntimeError(msg) from error
    label2id, id2label = _validate_label_maps(first.label2id, first.id2label)
    token_count = first.company_name_token_count
    span_count = first.company_name_span_count
    packed_unit_count = first.company_name_packed_unit_count
    row_range_count = first.company_name_original_row_range_count
    malformed_count = first.malformed_transition_count
    processed_units = first.processed_packed_units
    schema_fields = set(first.observed_schema_fields)
    row_range_fields = set(first.observed_row_range_fields)
    for report in reports:
        _validate_label_maps(report.label2id, report.id2label)
        if report.source_membership_status != "unavailable":
            msg = "packed source membership lacks defined dataset semantics"
            raise RuntimeError(msg)
        token_count += report.company_name_token_count
        span_count += report.company_name_span_count
        packed_unit_count += report.company_name_packed_unit_count
        row_range_count += report.company_name_original_row_range_count
        malformed_count += report.malformed_transition_count
        processed_units += report.processed_packed_units
        schema_fields.update(report.observed_schema_fields)
        row_range_fields.update(report.observed_row_range_fields)
    if first.source_membership_status != "unavailable":
        msg = "packed source membership lacks defined dataset semantics"
        raise RuntimeError(msg)
    return PackedCompanyNameAudit(
        label2id=label2id,
        id2label=id2label,
        company_name_token_count=token_count,
        company_name_span_count=span_count,
        company_name_packed_unit_count=packed_unit_count,
        company_name_original_row_range_count=row_range_count,
        malformed_transition_count=malformed_count,
        processed_packed_units=processed_units,
        source_membership_status="unavailable",
        observed_schema_fields=tuple(sorted(schema_fields)),
        observed_row_range_fields=tuple(sorted(row_range_fields)),
    )


def audit_eval_records(records: Iterable[Mapping[str, object]], *, expected_rows: int = EVAL_ROWS) -> EvalCompanyNameAudit:
    processed_rows = 0
    documents = 0
    spans = 0
    for index, record in enumerate(records):
        _example_id, _text, gold_spans = parse_labeled_record(record, default_id=f"eval-{index}", eval_mode="typed")
        company_spans = [span for span in gold_spans if span.label == "company_name"]
        processed_rows += 1
        spans += len(company_spans)
        documents += int(bool(company_spans))
    if processed_rows != expected_rows:
        msg = f"pinned eval inventory changed: expected {expected_rows}, got {processed_rows}"
        raise RuntimeError(msg)
    return EvalCompanyNameAudit(expected_rows, processed_rows, documents, spans)


def render_modal_command(profile: str = DEFAULT_MODAL_PROFILE) -> str:
    if not profile.strip():
        msg = "profile must not be empty"
        raise ValueError(msg)
    return (
        f"MODAL_PROFILE={shlex.quote(profile)} uv run modal run -m "
        "meddies_pii.training.bioes.modal.company_name_audit "
        f"--confirmation {COMPANY_NAME_AUDIT_CONFIRMATION}"
    )


def require_execution(confirmation: str) -> None:
    if confirmation != COMPANY_NAME_AUDIT_CONFIRMATION:
        msg = "company_name audit requires the exact confirmation token"
        raise RuntimeError(msg)


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-command", action="store_true")
    parser.add_argument("--profile", default=DEFAULT_MODAL_PROFILE)
    arguments = parser.parse_args(argv)
    if not arguments.render_command:
        parser.error("--render-command is required; run the rendered command explicitly")
    print(render_modal_command(arguments.profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(__import__("sys").argv[1:]))
