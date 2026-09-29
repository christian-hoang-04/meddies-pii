"""Meddies BIOES packing Adapter for Unsloth-backed token classification.

This is not native SFTTrainer packing.  The Adapter keeps the external
Interface small for BIOES: callers hand it already-aligned ``PreparedRow``
objects and receive fixed-size packed units plus a collator that emits
``packed_seq_lengths`` and ``position_ids`` for Unsloth's packed-sequence
metadata contract.  ADR 0003 requires a packed-attention contamination probe
before packed training launches are trusted.
"""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from meddies_pii.annotations.bioes import IGNORE_INDEX

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from meddies_pii.training.bioes.data.preparation import PreparedRow


@dataclass(frozen=True, slots=True)
class PackedRowRange:
    uid: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PackedTrainingUnit:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    seq_lengths: tuple[int, ...]
    position_ids: tuple[int, ...]
    row_uids: tuple[str, ...]
    row_ranges: tuple[PackedRowRange, ...]
    real_token_count: int
    padded_token_count: int
    boundary_token_count: int = 0

    @property
    def padding_token_count(self) -> int:
        """Physical padding tokens, excluding source and guard tokens."""
        return self.padded_token_count - self.real_token_count - self.boundary_token_count


def _positions_for(length: int) -> list[int]:
    return list(range(length))


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "packer continuation integer is invalid"
        raise ValueError(msg)
    return value


def _as_string(value: object) -> str:
    if not isinstance(value, str):
        msg = "packer continuation string is invalid"
        raise ValueError(msg)
    return value


class StatefulPacker:
    """Bounded continuation-safe packer; call ``finish`` exactly once at EOF."""

    def __init__(
        self,
        *,
        max_length: int,
        pad_token_id: int,
        boundary_token_count: int = 0,
        boundary_token_id: int | None = None,
    ) -> None:
        if max_length <= 0:
            msg = "max_length must be positive"
            raise ValueError(msg)
        if boundary_token_count < 0:
            msg = "boundary_token_count must be >= 0"
            raise ValueError(msg)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.boundary_token_count = boundary_token_count
        self.boundary_token_id = pad_token_id if boundary_token_id is None else boundary_token_id
        self.ids: list[int] = []
        self.labels: list[int] = []
        self.lengths: list[int] = []
        self.positions: list[int] = []
        self.uids: list[str] = []
        self.ranges: list[PackedRowRange] = []
        self.real = 0
        self.boundaries = 0
        self._finished = False

    def _flush(self) -> PackedTrainingUnit | None:
        if not self.ids:
            return None
        pad = self.max_length - len(self.ids)
        if pad < 0:
            msg = "packed unit length exceeded max_length"
            raise ValueError(msg)
        if pad:
            self.ids.extend([self.pad_token_id] * pad)
            self.labels.extend([IGNORE_INDEX] * pad)
            self.lengths.append(pad)
            self.positions.extend(_positions_for(pad))
        unit = PackedTrainingUnit(
            tuple(self.ids),
            tuple(self.labels),
            tuple(self.lengths),
            tuple(self.positions),
            tuple(self.uids),
            tuple(self.ranges),
            self.real,
            self.max_length,
            self.boundaries,
        )
        self.ids = []
        self.labels = []
        self.lengths = []
        self.positions = []
        self.uids = []
        self.ranges = []
        self.real = 0
        self.boundaries = 0
        return unit

    def push(self, rows: Iterable[PreparedRow]) -> list[PackedTrainingUnit]:
        if self._finished:
            msg = "cannot push after finish; restore a pre-EOF state"
            raise RuntimeError(msg)
        output: list[PackedTrainingUnit] = []
        for row in rows:
            ids, labels = list(row.tokenized.input_ids), list(row.tokenized.labels)
            if not ids:
                continue
            if len(ids) != len(labels):
                msg = f"row {row.uid!r} has mismatched input/label lengths"
                raise ValueError(msg)
            if len(ids) > self.max_length:
                msg = f"row {row.uid!r} length {len(ids)} exceeds max_length={self.max_length}"
                raise ValueError(msg)
            boundary = self.boundary_token_count if self.ids else 0
            if self.ids and len(self.ids) + boundary + len(ids) > self.max_length:
                unit = self._flush()
                if unit is not None:
                    output.append(unit)
                boundary = 0
            if boundary:
                self.ids.extend([self.boundary_token_id] * boundary)
                self.labels.extend([IGNORE_INDEX] * boundary)
                self.lengths.append(boundary)
                self.positions.extend(_positions_for(boundary))
                self.boundaries += boundary
            start = len(self.ids)
            self.ids.extend(ids)
            self.labels.extend(labels)
            self.lengths.append(len(ids))
            self.positions.extend(_positions_for(len(ids)))
            self.uids.append(row.uid)
            self.ranges.append(PackedRowRange(row.uid, start, start + len(ids)))
            self.real += len(ids)
        return output

    def finish(self) -> list[PackedTrainingUnit]:
        if self._finished:
            msg = "finish may only be called once"
            raise RuntimeError(msg)
        self._finished = True
        unit = self._flush()
        return [unit] if unit is not None else []

    def snapshot(self) -> dict[str, object]:
        """JSON-safe continuation state. It is valid only before ``finish``.

        Returns:
            The packer's in-flight buffers copied into plain JSON-safe lists, so the caller
            can serialize it and the packer can keep mutating its own state afterwards.

        Raises:
            RuntimeError: After ``finish``. A finished packer has flushed its partial unit,
                so a snapshot taken then would describe an empty buffer and resuming from it
                would silently restart mid-corpus rather than continue -- refused instead of
                returning state that looks valid.

        """
        if self._finished:
            msg = "finished packers have no resumable continuation state"
            raise RuntimeError(msg)
        return {
            "max_length": self.max_length,
            "pad_token_id": self.pad_token_id,
            "boundary_token_count": self.boundary_token_count,
            "boundary_token_id": self.boundary_token_id,
            "ids": list(self.ids),
            "labels": list(self.labels),
            "lengths": list(self.lengths),
            "positions": list(self.positions),
            "uids": list(self.uids),
            "ranges": [{"uid": item.uid, "start": item.start, "end": item.end} for item in self.ranges],
            "real": self.real,
            "boundaries": self.boundaries,
        }

    @classmethod
    def from_snapshot(cls, state: dict[str, object]) -> StatefulPacker:
        required = {
            "max_length",
            "pad_token_id",
            "boundary_token_count",
            "boundary_token_id",
            "ids",
            "labels",
            "lengths",
            "positions",
            "uids",
            "ranges",
            "real",
            "boundaries",
        }
        if set(state) != required:
            msg = "invalid packer continuation state"
            raise ValueError(msg)
        packer = cls(
            max_length=_as_int(state["max_length"]),
            pad_token_id=_as_int(state["pad_token_id"]),
            boundary_token_count=_as_int(state["boundary_token_count"]),
            boundary_token_id=_as_int(state["boundary_token_id"]),
        )
        ids = cast("list[object]", state["ids"])
        labels = cast("list[object]", state["labels"])
        lengths = cast("list[object]", state["lengths"])
        positions = cast("list[object]", state["positions"])
        uids = cast("list[object]", state["uids"])
        ranges = cast("list[Mapping[str, object]]", state["ranges"])
        packer.ids = [_as_int(item) for item in ids]
        packer.labels = [_as_int(item) for item in labels]
        packer.lengths = [_as_int(item) for item in lengths]
        packer.positions = [_as_int(item) for item in positions]
        packer.uids = [_as_string(item) for item in uids]
        packer.ranges = [
            PackedRowRange(_as_string(item["uid"]), _as_int(item["start"]), _as_int(item["end"])) for item in ranges
        ]
        packer.real = _as_int(state["real"])
        packer.boundaries = _as_int(state["boundaries"])
        if not (
            len(packer.ids) == len(packer.labels) == len(packer.positions) < packer.max_length
            and sum(packer.lengths) == len(packer.ids)
            and len(packer.uids) == len(packer.ranges)
            and packer.real + packer.boundaries <= len(packer.ids)
        ):
            msg = "invalid packer continuation invariants"
            raise ValueError(msg)
        return packer


def pack_prepared_rows(
    rows: Sequence[PreparedRow],
    *,
    max_length: int,
    pad_token_id: int,
    boundary_token_count: int = 0,
    boundary_token_id: int | None = None,
) -> list[PackedTrainingUnit]:
    """Pack aligned BIOES rows without crossing attention boundaries.

    Each returned unit is exactly ``max_length`` tokens.  Padding, when needed,
    is represented as its own final packed segment with ``IGNORE_INDEX`` labels
    so that the sum of ``packed_seq_lengths`` equals the tensor token count.
    That invariant is required by Unsloth's varlen/block-diagonal attention
    path.

    Returns:
        The packed units, each EXACTLY ``max_length`` tokens with no short final unit --
        padding becomes its own trailing segment carrying ``IGNORE_INDEX`` labels rather than
        being appended invisibly. That is what keeps the sum of ``packed_seq_lengths`` equal
        to the tensor token count, which Unsloth's varlen attention path requires; a padding
        scheme that skipped the segment would break attention rather than merely waste
        tokens.

    """
    packer = StatefulPacker(
        max_length=max_length,
        pad_token_id=pad_token_id,
        boundary_token_count=boundary_token_count,
        boundary_token_id=boundary_token_id,
    )
    return [*packer.push(rows), *packer.finish()]


def collate_packed_units(
    units: Sequence[PackedTrainingUnit],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    if not units:
        msg = "cannot collate an empty packed-unit batch"
        raise ValueError(msg)
    expected_length = len(units[0].input_ids)
    if expected_length == 0:
        msg = "packed units must be non-empty"
        raise ValueError(msg)
    for unit in units:
        if (
            len(unit.input_ids) != expected_length
            or len(unit.labels) != expected_length
            or len(unit.position_ids) != expected_length
        ):
            msg = "packed units in a batch must have equal token length"
            raise ValueError(msg)
    packed_seq_lengths = [int(length) for unit in units for length in unit.seq_lengths if length > 0]
    if sum(packed_seq_lengths) != expected_length * len(units):
        msg = "sum(packed_seq_lengths) must match flattened packed token count"
        raise ValueError(msg)
    return {
        "input_ids": torch.tensor([unit.input_ids for unit in units], dtype=torch.long, device=device),
        "labels": torch.tensor([unit.labels for unit in units], dtype=torch.long, device=device),
        "packed_seq_lengths": torch.tensor(packed_seq_lengths, dtype=torch.int32, device=device),
        "position_ids": torch.tensor([unit.position_ids for unit in units], dtype=torch.int32, device=device),
    }


def packing_utilization(units: Sequence[PackedTrainingUnit]) -> float | None:
    if not units:
        return None
    real_tokens = sum(unit.real_token_count for unit in units)
    padded_tokens = sum(unit.padded_token_count for unit in units)
    if padded_tokens <= 0:
        return None
    return real_tokens / padded_tokens
