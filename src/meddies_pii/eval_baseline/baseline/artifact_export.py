"""Local-only streaming export of safe scoreable artifacts for later Hub upload."""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.evaluation.identity import canonical_json_bytes, file_sha256, is_sha256
from meddies_pii.json_types import is_str_list, is_str_mapping

if TYPE_CHECKING:
    from meddies_pii.json_types import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class HubSafeShardPackage:
    """A deterministic local package with no raw document or span surfaces."""

    manifest_path: Path
    rows_path: Path
    row_count: int
    sha256: str


def export_hub_safe_shard_package(
    *,
    result_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
) -> HubSafeShardPackage:
    """Stream a local package; this function never imports or mutates the Hub.

    Returns:
        The package description: the safe-rows filename, its digest, the package digest and the
        row count. Rows and manifest are written to temporary siblings and only moved into
        place once both succeed, so a failed export leaves no half-written package behind.

    Raises:
        ValueError: If the metadata carries no ``result_sha256`` string or the result file no
            longer hashes to it, which means the result changed since it was bound; if there is
            no integer ``rows_in``; or if the rows actually written do not match that count. A
            digest is checked before any row is read, so a changed result never reaches the
            output at all.

    """
    result = Path(result_path)
    metadata = _read_json_object(Path(metadata_path))
    expected_digest = metadata.get("result_sha256")
    if not isinstance(expected_digest, str) or file_sha256(str(result)) != expected_digest:
        msg = "cannot export result with an unbound or changed digest"
        raise ValueError(msg)
    expected_rows = metadata.get("rows_in")
    if not isinstance(expected_rows, int):
        msg = "cannot export result without an integer row count"
        raise ValueError(msg)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    temporary_rows = _temporary_sibling(destination / "results.safe.jsonl")
    temporary_manifest = _temporary_sibling(destination / "manifest.json")
    try:
        row_count = _write_safe_rows(result, temporary_rows)
        if row_count != expected_rows:
            msg = "cannot export result with an inconsistent row count"
            raise ValueError(msg)
        safe_rows_sha256 = file_sha256(str(temporary_rows))
        rows_path = destination / f"results.safe.{safe_rows_sha256}.jsonl"
        Path(temporary_rows).replace(rows_path)
        _fsync_directory(destination)
        manifest = _safe_manifest(
            metadata,
            row_count=row_count,
            safe_rows_file=rows_path.name,
            safe_rows_sha256=safe_rows_sha256,
        )
        package_digest = _package_sha256(manifest, rows_path)
        manifest["package_sha256"] = package_digest
        manifest_path = destination / "manifest.json"
        _write_fsync(temporary_manifest, canonical_json_bytes(manifest) + b"\n")
        Path(temporary_manifest).replace(manifest_path)
        _fsync_directory(destination)
        return validate_hub_safe_shard_package(manifest_path)
    finally:
        temporary_rows.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def validate_hub_safe_shard_package(
    manifest_path: str | Path,
) -> HubSafeShardPackage:
    """Validate the manifest pointer and immutable safe-row generation it names.

    Returns:
        The same package description the export produced, re-derived from the manifest on disk
        and checked against the rows beside it.

    Raises:
        ValueError: If the manifest's ``safe_rows_file`` is not a bare basename, which is the
            check that stops a manifest from pointing outside its own directory; if either
            digest field is missing or malformed; if ``row_count`` is not a non-negative
            integer, with ``bool`` excluded so ``True`` cannot pass as 1; if the rows file does
            not hash to the recorded digest; or if the rows counted do not match the recorded
            count. Both the bytes and the count are checked, so a file that was truncated and
            re-hashed still fails on the count.

    """
    path = Path(manifest_path)
    manifest = _read_json_object(path)
    safe_rows_file = _required_string(manifest, "safe_rows_file")
    if Path(safe_rows_file).name != safe_rows_file:
        msg = "safe rows file must be a basename"
        raise ValueError(msg)
    safe_rows_sha256 = _required_sha256(manifest, "safe_rows_sha256")
    package_sha256 = _required_sha256(manifest, "package_sha256")
    row_count = manifest.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        msg = "manifest row_count must be a non-negative integer"
        raise ValueError(msg)
    rows_path = path.parent / safe_rows_file
    if file_sha256(str(rows_path)) != safe_rows_sha256:
        msg = "safe rows digest does not match manifest"
        raise ValueError(msg)
    if _count_rows(rows_path) != row_count:
        msg = "safe rows row count does not match manifest"
        raise ValueError(msg)
    if _package_sha256(manifest, rows_path) != package_sha256:
        msg = "package digest does not match manifest"
        raise ValueError(msg)
    return HubSafeShardPackage(
        manifest_path=path,
        rows_path=rows_path,
        row_count=row_count,
        sha256=package_sha256,
    )


def _write_safe_rows(result_path: Path, destination: Path) -> int:
    row_count = 0
    with result_path.open(encoding="utf-8") as source, destination.open("wb") as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"result artifact has malformed row {line_number}"
                raise ValueError(msg) from exc
            if not isinstance(value, dict):
                msg = f"result artifact row {line_number} is not an object"
                raise ValueError(msg)
            output.write(canonical_json_bytes(_safe_row(value)) + b"\n")
            row_count += 1
        output.flush()
        os.fsync(output.fileno())
    return row_count


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"cannot read export metadata: {path}"
        raise ValueError(msg) from exc
    if not is_str_mapping(value):
        msg = "export metadata must be a JSON object"
        raise ValueError(msg)
    return dict(value)


def _safe_row(row: Mapping[str, object]) -> JsonObject:
    identity = _required_string(row, "id")
    doc_id = _required_string(row, "doc_id")
    language = _required_string(row, "language")
    view = _required_string(row, "evaluation_view")
    inference_identity = _required_string(row, "shared_inference_identity")
    slices = _string_list(row.get("slice"), "slice")
    safe_slices: list[JsonValue] = [*sorted(slices)]
    return {
        "id": identity,
        "doc_id_sha256": hashlib.sha256(doc_id.encode("utf-8")).hexdigest(),
        "pred_spans": _safe_spans(row.get("pred_spans"), "pred_spans"),
        "gold_spans": _safe_spans(row.get("gold_spans"), "gold_spans"),
        "language": language,
        "slice": safe_slices,
        "evaluation_view": view,
        "shared_inference_identity": inference_identity,
    }


def _safe_spans(value: object, field: str) -> list[JsonValue]:
    if not isinstance(value, list):
        msg = f"{field} must be a list"
        raise ValueError(msg)
    safe: list[JsonValue] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            msg = f"{field} must contain objects"
            raise ValueError(msg)
        start, end = raw.get("start"), raw.get("end")
        label = raw.get("label")
        # reason: safe spans keeps start/end in one gate; helper predicates would scatter the rule.
        if (
            isinstance(start, bool)  # ruff: ignore[too-many-boolean-expressions]
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or not isinstance(label, str)
            or not label
        ):
            msg = f"{field} contains an invalid span"
            raise ValueError(msg)
        safe.append({"start": start, "end": end, "label": label})
    return safe


def _safe_manifest(
    metadata: Mapping[str, object],
    *,
    row_count: int,
    safe_rows_file: str,
    safe_rows_sha256: str,
) -> JsonObject:
    view = _required_string(metadata, "evaluation_view")
    shared_inference_identity = _required_string(metadata, "shared_inference_identity")
    contract = metadata.get("evaluation_contract")
    if not isinstance(contract, dict):
        msg = "metadata is missing the evaluation contract"
        raise ValueError(msg)
    contract_digest = _required_string(metadata, "evaluation_contract_sha256")
    fixture_digest = _required_string(metadata, "fixture_sha256")
    result_digest = _required_string(metadata, "result_sha256")
    spans_out = metadata.get("spans_out")
    elapsed_seconds = metadata.get("elapsed_seconds")
    failure_state = metadata.get("failure_state")
    shard_identity = metadata.get("dataset_shard_identity")
    if not is_str_mapping(shard_identity):
        msg = "metadata is missing the dataset shard identity"
        raise ValueError(msg)
    dataset = _required_string(shard_identity, "dataset")
    shard = _required_string(shard_identity, "shard")
    if not isinstance(spans_out, int) or not isinstance(elapsed_seconds, int | float):
        msg = "metadata is missing scoreable result counts"
        raise ValueError(msg)
    if failure_state is not None and not isinstance(failure_state, str):
        msg = "metadata failure_state must be a string or null"
        raise ValueError(msg)
    return {
        "schema_version": 1,
        "package": "meddies-pii-hub-safe-evaluation-shard",
        "row_count": row_count,
        "dataset": dataset,
        "shard": shard,
        "spans_out": spans_out,
        "elapsed_seconds": elapsed_seconds,
        "result_sha256": result_digest,
        "fixture_sha256": fixture_digest,
        "evaluation_view": view,
        "shared_inference_identity": shared_inference_identity,
        "evaluation_contract": contract,
        "evaluation_contract_sha256": contract_digest,
        "failure_state": failure_state,
        "safe_rows_file": safe_rows_file,
        "safe_rows_sha256": safe_rows_sha256,
    }


def _package_sha256(manifest: Mapping[str, object], rows_path: Path) -> str:
    manifest_without_digest = dict(manifest)
    manifest_without_digest.pop("package_sha256", None)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(manifest_without_digest))
    with rows_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def _temporary_sibling(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _required_string(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        msg = f"missing required string field {key!r}"
        raise ValueError(msg)
    return result


def _required_sha256(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not is_sha256(result):
        msg = f"missing required SHA-256 field {key!r}"
        raise ValueError(msg)
    return result


def _string_list(value: object, field: str) -> Sequence[str]:
    if not is_str_list(value):
        msg = f"{field} must be a string list"
        raise ValueError(msg)
    return value
