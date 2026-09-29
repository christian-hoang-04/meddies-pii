from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from meddies_pii.spans import CharSpan
    from meddies_pii.training.bioes.data.record_schema import Record

MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class WorkloadCandidate:
    uid: str
    row_hash: str
    raw_chars: int
    token_length: int
    truncated: bool = False
    accepted: bool = True
    skip_reason: str | None = None

    def to_record(self) -> Record:
        return {
            "uid": self.uid,
            "row_hash": self.row_hash,
            "raw_chars": self.raw_chars,
            "token_length": self.token_length,
            "truncated": self.truncated,
            "accepted": self.accepted,
            "skip_reason": self.skip_reason,
        }


def stable_json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _span_payload(span: CharSpan) -> Record:
    return {
        "label": str(span.label),
        "start": int(span.start),
        "end": int(span.end),
        "text": str(span.text),
    }


def row_hash_from_parts(*, uid: str, raw: str, spans: Sequence[CharSpan]) -> str:
    payload = {
        "uid": uid,
        "raw": raw,
        "spans": [_span_payload(span) for span in spans],
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


def _candidate_token_length(candidate: Mapping[str, object]) -> int:
    value = candidate.get("token_length")
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "candidate token_length must be an integer"
        raise ValueError(msg)
    return value


def select_candidates_by_token_length(
    candidates: Sequence[WorkloadCandidate | Mapping[str, object]],
    *,
    required: int,
    min_observed_pad_length: int | None = None,
) -> list[Record]:
    if required <= 0:
        msg = "required must be positive"
        raise ValueError(msg)
    normalized: list[Record] = [
        candidate.to_record() if isinstance(candidate, WorkloadCandidate) else dict(candidate) for candidate in candidates
    ]
    missing_accepted = [
        str(candidate.get("uid", index)) for index, candidate in enumerate(normalized) if "accepted" not in candidate
    ]
    if missing_accepted:
        msg = f"candidate accepted flag is required before manifest selection; missing for: {', '.join(missing_accepted)}"
        raise ValueError(
            msg,
        )
    eligible = [
        candidate
        for candidate in normalized
        if candidate.get("accepted") is True and candidate.get("truncated", False) is False
    ]
    eligible.sort(
        key=lambda candidate: (
            -_candidate_token_length(candidate),
            str(candidate["uid"]),
            str(candidate["row_hash"]),
        ),
    )
    if len(eligible) < required:
        msg = f"Need {required} eligible candidates, found {len(eligible)}"
        raise RuntimeError(msg)
    selected = eligible[:required]
    observed_pad_length = max(_candidate_token_length(candidate) for candidate in selected)
    if min_observed_pad_length is not None and observed_pad_length < min_observed_pad_length:
        msg = f"Observed selected max length {observed_pad_length} is below \
min_observed_pad_length={min_observed_pad_length}"
        raise RuntimeError(
            msg,
        )
    return selected


def _manifest_id_payload(manifest: Mapping[str, object]) -> Record:
    payload = dict(manifest)
    payload.pop("manifest_id", None)
    payload.pop("created_at_utc", None)
    return payload


def manifest_fingerprint(manifest: Mapping[str, object]) -> str:
    return stable_sha256(_manifest_id_payload(manifest))[:16]


# reason: build workload exposes name/created at as its public contract; bundling would break callers.
def build_workload_manifest(  # ruff: ignore[too-many-arguments]
    *,
    name: str,
    dataset_id: str,
    dataset_config: str,
    dataset_split: str,
    dataset_revision: str,
    eval_dataset_id: str | None = None,
    eval_dataset_config: str | None = None,
    eval_dataset_split: str | None = None,
    eval_dataset_revision: str | None = None,
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    max_length: int,
    fixed_pad_length: int,
    selection_policy: str,
    required_examples: int,
    candidates: Sequence[WorkloadCandidate | Mapping[str, object]],
    min_observed_pad_length: int | None = None,
    objective: Mapping[str, object] | None = None,
    scan: Mapping[str, object] | None = None,
    created_at_utc: str | None = None,
) -> Record:
    if max_length <= 0:
        msg = "max_length must be positive"
        raise ValueError(msg)
    if fixed_pad_length < max_length:
        msg = "fixed_pad_length must be >= max_length for fixed-shape research manifests"
        raise ValueError(msg)
    normalized_candidates: list[Record] = [
        candidate.to_record() if isinstance(candidate, WorkloadCandidate) else dict(candidate) for candidate in candidates
    ]
    selected = select_candidates_by_token_length(
        normalized_candidates,
        required=required_examples,
        min_observed_pad_length=min_observed_pad_length,
    )
    manifest: Record = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": name,
        "created_at_utc": created_at_utc,
        "dataset": {
            "id": dataset_id,
            "config": dataset_config,
            "split": dataset_split,
            "revision": dataset_revision,
        },
        "eval_dataset": {
            "id": eval_dataset_id or dataset_id,
            "config": eval_dataset_config or "test",
            "split": eval_dataset_split or dataset_split,
            "revision": eval_dataset_revision or dataset_revision,
        },
        "model": {
            "id": model_id,
            "revision": model_revision,
        },
        "tokenizer": {
            "id": tokenizer_id,
            "revision": tokenizer_revision,
        },
        "shape": {
            "max_length": max_length,
            "fixed_pad_length": fixed_pad_length,
            "min_observed_pad_length": min_observed_pad_length,
        },
        "selection_policy": selection_policy,
        "required_examples": required_examples,
        "selected": selected,
        "candidates": normalized_candidates,
        "objective": dict(objective or {}),
    }
    if scan is not None:
        manifest["scan"] = dict(scan)
    manifest["manifest_id"] = f"{name}-{manifest_fingerprint(manifest)}"
    return manifest
