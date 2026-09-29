"""Canonical, immutable identities for reproducible evaluation artifacts."""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

from meddies_pii.json_types import (
    JsonObject,
    as_json_object,
    is_json_value,
    is_str_list,
    is_str_mapping,
)
from meddies_pii.spans import CharSpan, char_span_from_value, char_span_to_dict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


SHA256_HEX_LENGTH = 64
GIT_SHA_HEX_LENGTH = 40


def canonical_json_bytes(value: object) -> bytes:
    """Return the only canonical JSON encoding used by evaluation identities.

    Returns:
        UTF-8 bytes with keys sorted, no inserted whitespace and non-ASCII left as-is. One
        value must encode to one byte string for a digest over it to mean anything, which is
        why every identity in this module goes through here rather than calling ``json.dumps``.

    Raises:
        ValueError: If the value is not JSON-compatible. ``allow_nan`` is off as well, so a
            float infinity or NaN is refused rather than encoded as the non-standard token that
            another decoder would reject or read differently.

    """
    if not is_json_value(value):
        msg = "canonical JSON requires JSON-compatible values"
        raise ValueError(msg)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> TypeGuard[str]:
    """Return whether one value is a lowercase hexadecimal SHA-256 digest.

    Returns:
        ``True`` only for a string of exactly the digest length made entirely of lowercase hex
        characters, narrowing the value to ``str`` for the caller. Uppercase is rejected rather
        than folded, so two spellings of one digest can never both pass and compare unequal.

    """
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: str | Path) -> str:
    path_object = Path(path)
    path_stat_before = path_object.stat()
    digest = hashlib.sha256()
    # reason: the streaming contract is pinned by a test that observes `builtins.open`; `Path.open`
    # reason: bypasses that seam and the bounded-chunk guarantee stops being checked.
    with open(path, "rb") as source:  # ruff: ignore[builtin-open]
        descriptor_stat_before = os.fstat(source.fileno())
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        descriptor_stat_after = os.fstat(source.fileno())
    path_stat_after = path_object.stat()
    if (
        _stable_file_signature(path_stat_before) != _stable_file_signature(descriptor_stat_before)
        or _stable_file_signature(descriptor_stat_before) != _stable_file_signature(descriptor_stat_after)
        or _stable_file_signature(descriptor_stat_after) != _stable_file_signature(path_stat_after)
    ):
        msg = f"artifact changed while hashing: {path}"
        raise RuntimeError(msg)
    return digest.hexdigest()


def manifest_artifact(reference: str, revision: str) -> ArtifactIdentity:
    """Identify an immutable commit-addressed repository by its manifest digest.

    Returns:
        An identity over the reference and revision pair alone. Nothing is downloaded: a commit
        address is already immutable, so hashing the pair is enough to pin it, and two runs
        naming the same commit produce the same identity without fetching anything.

    """
    return ArtifactIdentity(
        reference,
        revision,
        canonical_sha256({"reference": reference, "revision": revision}),
    )


def source_artifact(reference: str, path: str) -> ArtifactIdentity:
    digest = file_sha256(path)
    return ArtifactIdentity(reference, f"sha256:{digest}", digest)


def payload_artifact(reference: str, payload: object) -> ArtifactIdentity:
    digest = canonical_sha256(payload)
    return ArtifactIdentity(reference, f"sha256:{digest}", digest)


# reason: evaluation exposes model/result as its public contract; bundling would break callers.
def evaluation_contract(  # ruff: ignore[too-many-arguments]
    *,
    model: ArtifactIdentity,
    vendor_inference_source: ArtifactIdentity,
    local_adapter_source: ArtifactIdentity,
    applied_label_prediction_contract: ArtifactIdentity,
    decoder_contract: ArtifactIdentity,
    resolved_runtime_environment: ArtifactIdentity,
    scorer_contract: ArtifactIdentity,
    supported_labels: Sequence[str],
    result_schema: Sequence[str],
) -> EvaluationContract:
    return EvaluationContract(
        model=model,
        vendor_inference_source=vendor_inference_source,
        local_adapter_source=local_adapter_source,
        applied_label_prediction_contract=applied_label_prediction_contract,
        decoder_contract=decoder_contract,
        resolved_runtime_environment=resolved_runtime_environment,
        scorer_contract=scorer_contract,
        supported_labels=tuple(sorted(supported_labels)),
        result_schema=tuple(result_schema),
    )


def checkpoint_tree_artifact(reference: str, root: str | Path) -> ArtifactIdentity:
    root_path = Path(root)
    if not root_path.is_dir():
        msg = f"checkpoint tree is missing: {root_path}"
        raise ValueError(msg)
    files = [
        {
            "path": path.relative_to(root_path).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(str(path)),
        }
        for path in sorted(candidate for candidate in root_path.rglob("*") if candidate.is_file())
    ]
    if not files:
        msg = f"checkpoint tree has no files: {root_path}"
        raise ValueError(msg)
    digest = canonical_sha256(files)
    return ArtifactIdentity(reference, f"tree-sha256:{digest}", digest)


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    reference: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        _non_empty("artifact reference", self.reference)
        _immutable_revision(self.revision)
        _sha256("artifact digest", self.sha256)

    def to_payload(self) -> JsonObject:
        return {
            "reference": self.reference,
            "revision": self.revision,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class EvaluationContract:
    model: ArtifactIdentity
    vendor_inference_source: ArtifactIdentity
    local_adapter_source: ArtifactIdentity
    applied_label_prediction_contract: ArtifactIdentity
    decoder_contract: ArtifactIdentity
    resolved_runtime_environment: ArtifactIdentity
    scorer_contract: ArtifactIdentity
    supported_labels: tuple[str, ...]
    result_schema: tuple[str, ...]

    def __post_init__(self) -> None:
        _sorted_unique("supported labels", self.supported_labels)
        _unique("result schema", self.result_schema)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_payload())

    def to_payload(self) -> JsonObject:
        return _json_object({
            "schema_version": 1,
            "model": self.model.to_payload(),
            "vendor_inference_source": self.vendor_inference_source.to_payload(),
            "local_adapter_source": self.local_adapter_source.to_payload(),
            "applied_label_prediction_contract": self.applied_label_prediction_contract.to_payload(),
            "decoder_contract": self.decoder_contract.to_payload(),
            "resolved_runtime_environment": self.resolved_runtime_environment.to_payload(),
            "scorer_contract": self.scorer_contract.to_payload(),
            "supported_labels": list(self.supported_labels),
            "result_schema": list(self.result_schema),
        })


@dataclass(frozen=True, slots=True)
class FixtureRowIdentity:
    stable_id: str
    text_sha256: str
    gold_spans: tuple[CharSpan, ...]
    language: str
    slices: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty("fixture stable id", self.stable_id)
        _sha256("fixture text digest", self.text_sha256)
        _non_empty("fixture language", self.language)
        if tuple(sorted(self.slices)) != self.slices or len(set(self.slices)) != len(self.slices):
            msg = "fixture slices must be sorted and unique"
            raise ValueError(msg)

    def to_payload(self) -> JsonObject:
        return _json_object({
            "stable_id": self.stable_id,
            "text_sha256": self.text_sha256,
            "gold_spans": [char_span_to_dict(span) for span in self.gold_spans],
            "language": self.language,
            "slices": list(self.slices),
        })


@dataclass(frozen=True, slots=True)
class DatasetShardFixtureIdentity:
    """Identify the ordered data and labels scored in one dataset shard."""

    dataset: str
    shard: str
    fixture: tuple[FixtureRowIdentity, ...]
    row_count: int

    def __post_init__(self) -> None:
        _non_empty("dataset", self.dataset)
        _non_empty("shard", self.shard)
        if self.row_count != len(self.fixture):
            msg = "fixture row count must equal fixture length"
            raise ValueError(msg)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_payload())

    def to_payload(self) -> JsonObject:
        return _json_object({
            "schema_version": 1,
            "dataset": self.dataset,
            "shard": self.shard,
            "fixture": [row.to_payload() for row in self.fixture],
            "row_count": self.row_count,
        })


@dataclass(frozen=True, slots=True)
class DatasetShardIdentity:
    """Identify the evaluation contract and fixture expected for one shard."""

    evaluation_contract: EvaluationContract
    dataset: str
    shard: str
    fixture: tuple[FixtureRowIdentity, ...]
    row_count: int

    def __post_init__(self) -> None:
        DatasetShardFixtureIdentity(
            dataset=self.dataset,
            shard=self.shard,
            fixture=self.fixture,
            row_count=self.row_count,
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_payload())

    @property
    def fixture_identity(self) -> DatasetShardFixtureIdentity:
        """Data-only identity used to compare benchmark fixtures."""
        return DatasetShardFixtureIdentity(
            dataset=self.dataset,
            shard=self.shard,
            fixture=self.fixture,
            row_count=self.row_count,
        )

    def to_payload(self) -> JsonObject:
        return _json_object({
            "schema_version": 1,
            "evaluation_contract": self.evaluation_contract.to_payload(),
            "dataset": self.dataset,
            "shard": self.shard,
            "fixture": [row.to_payload() for row in self.fixture],
            "row_count": self.row_count,
        })


def dataset_shard_identity(
    evaluation_contract: EvaluationContract,
    *,
    dataset: str,
    shard: str,
    rows: Sequence[object],
) -> DatasetShardIdentity:
    fixture: list[FixtureRowIdentity] = []
    for row in rows:
        stable_id = getattr(row, "stable_id", None)
        text_sha256 = getattr(row, "text_sha256", None)
        gold_spans = getattr(row, "gold_spans", None)
        language = getattr(row, "language", None)
        slices = getattr(row, "slices", None)
        # reason: dataset shard keeps stable id/text in one gate; helper predicates would scatter the rule.
        if (
            not isinstance(stable_id, str)  # ruff: ignore[too-many-boolean-expressions]
            or not isinstance(text_sha256, str)
            or not isinstance(gold_spans, tuple)
            or not all(isinstance(span, CharSpan) for span in gold_spans)
            or not isinstance(language, str)
            or not isinstance(slices, frozenset)
            or not all(isinstance(item, str) for item in slices)
        ):
            msg = "rows must provide complete EvalRow fixture fields"
            raise ValueError(msg)
        fixture.append(
            FixtureRowIdentity(
                stable_id=stable_id,
                text_sha256=text_sha256,
                gold_spans=gold_spans,
                language=language,
                slices=tuple(sorted(slices)),
            ),
        )
    return DatasetShardIdentity(evaluation_contract, dataset, shard, tuple(fixture), len(fixture))


def dataset_shard_identity_from_result_rows(
    evaluation_contract: EvaluationContract,
    *,
    dataset: str,
    shard: str,
    rows: Sequence[Mapping[str, object]],
) -> DatasetShardIdentity:
    fixture: list[FixtureRowIdentity] = []
    for row in rows:
        spans: list[CharSpan] = []
        raw_spans = row.get("gold_spans")
        raw_slices = row.get("slice")
        if not isinstance(raw_spans, list) or not isinstance(raw_slices, list):
            msg = "result rows must include gold spans and slices"
            raise ValueError(msg)
        for raw_span in raw_spans:
            span = char_span_from_value(raw_span)
            if span is None:
                msg = "result rows contain an invalid gold span"
                raise ValueError(msg)
            spans.append(span)
        stable_id = row.get("doc_id")
        text_sha256 = row.get("id")
        language = row.get("language")
        if (
            not isinstance(stable_id, str)
            or not isinstance(text_sha256, str)
            or not isinstance(language, str)
            or not is_str_list(raw_slices)
        ):
            msg = "result rows must include complete fixture fields"
            raise ValueError(msg)
        fixture.append(
            FixtureRowIdentity(
                stable_id=stable_id,
                text_sha256=text_sha256,
                gold_spans=tuple(spans),
                language=language,
                slices=tuple(sorted(raw_slices)),
            ),
        )
    return DatasetShardIdentity(evaluation_contract, dataset, shard, tuple(fixture), len(fixture))


def parse_evaluation_contract(value: object) -> EvaluationContract:
    payload = _object(value, "evaluation contract")
    if payload.get("schema_version") != 1:
        msg = "evaluation contract schema version must be 1"
        raise ValueError(msg)
    return EvaluationContract(
        model=_artifact(payload, "model"),
        vendor_inference_source=_artifact(payload, "vendor_inference_source"),
        local_adapter_source=_artifact(payload, "local_adapter_source"),
        applied_label_prediction_contract=_artifact(payload, "applied_label_prediction_contract"),
        decoder_contract=_artifact(payload, "decoder_contract"),
        resolved_runtime_environment=_artifact(payload, "resolved_runtime_environment"),
        scorer_contract=_artifact(payload, "scorer_contract"),
        supported_labels=_strings(payload, "supported_labels"),
        result_schema=_strings(payload, "result_schema"),
    )


def parse_dataset_shard_identity(value: object) -> DatasetShardIdentity:
    payload = _object(value, "dataset shard identity")
    if payload.get("schema_version") != 1:
        msg = "dataset shard identity schema version must be 1"
        raise ValueError(msg)
    fixture_rows: list[FixtureRowIdentity] = []
    for raw in _array(payload, "fixture"):
        row = _object(raw, "fixture row")
        spans: list[CharSpan] = []
        for raw_span in _array(row, "gold_spans"):
            span = char_span_from_value(raw_span)
            if span is None:
                msg = "fixture gold spans must be valid CharSpan values"
                raise ValueError(msg)
            spans.append(span)
        fixture_rows.append(
            FixtureRowIdentity(
                stable_id=_string(row, "stable_id"),
                text_sha256=_string(row, "text_sha256"),
                gold_spans=tuple(spans),
                language=_string(row, "language"),
                slices=_strings(row, "slices"),
            ),
        )
    row_count = payload.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        msg = "fixture row_count must be an integer"
        raise ValueError(msg)
    return DatasetShardIdentity(
        evaluation_contract=parse_evaluation_contract(payload.get("evaluation_contract")),
        dataset=_string(payload, "dataset"),
        shard=_string(payload, "shard"),
        fixture=tuple(fixture_rows),
        row_count=row_count,
    )


def _json_object(value: object) -> JsonObject:
    object_value = as_json_object(value)
    if object_value is None:
        msg = "identity payload must be a JSON object"
        raise ValueError(msg)
    return object_value


def _stable_file_signature(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _non_empty(name: str, value: str) -> None:
    if not value.strip():
        msg = f"{name} must not be empty"
        raise ValueError(msg)


def _immutable_revision(value: str) -> None:
    _non_empty("artifact revision", value)
    if len(value) == GIT_SHA_HEX_LENGTH and all(character in "0123456789abcdef" for character in value):
        return
    for prefix in ("sha256:", "tree-sha256:"):
        if value.startswith(prefix):
            _sha256("artifact revision digest", value.removeprefix(prefix))
            return
    msg = "artifact revision must be an immutable lowercase git commit or SHA-256"
    raise ValueError(msg)


def _sha256(name: str, value: str) -> None:
    if len(value) != SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in value):
        msg = f"{name} must be lowercase 64-hex"
        raise ValueError(msg)


def _sorted_unique(name: str, values: tuple[str, ...]) -> None:
    if not values or tuple(sorted(values)) != values or len(set(values)) != len(values):
        msg = f"{name} must be non-empty, sorted, and unique"
        raise ValueError(msg)


def _unique(name: str, values: tuple[str, ...]) -> None:
    if not values or len(set(values)) != len(values) or any(not value for value in values):
        msg = f"{name} must be non-empty and unique"
        raise ValueError(msg)


def _object(value: object, name: str) -> Mapping[str, object]:
    if not is_str_mapping(value):
        msg = f"{name} must be an object with string keys"
        raise ValueError(msg)
    return value


def _string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        msg = f"{name} must be a string"
        raise ValueError(msg)
    return value


def _strings(payload: Mapping[str, object], name: str) -> tuple[str, ...]:
    values = _array(payload, name)
    if not all(isinstance(value, str) for value in values):
        msg = f"{name} must contain strings"
        raise ValueError(msg)
    return tuple(value for value in values if isinstance(value, str))


def _array(payload: Mapping[str, object], name: str) -> tuple[object, ...]:
    value = payload.get(name)
    if not isinstance(value, list):
        msg = f"{name} must be an array"
        raise ValueError(msg)
    return tuple(value)


def _artifact(payload: Mapping[str, object], name: str) -> ArtifactIdentity:
    raw = _object(payload.get(name), name)
    return ArtifactIdentity(
        reference=_string(raw, "reference"),
        revision=_string(raw, "revision"),
        sha256=_string(raw, "sha256"),
    )
