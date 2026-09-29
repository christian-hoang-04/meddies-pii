"""Validated boundary types for persisted and downloaded BIOES data rows.

The dataset and JSON libraries return dynamically typed values.  This module is
the one boundary where those values become the ``Record`` objects used by the
BIOES data pipeline.  Individual converters still validate the source-specific
fields they consume.
"""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
from collections.abc import Iterable, Mapping
from typing import Protocol, TypedDict

type Record = dict[str, object]
type ReadOnlyRecord = Mapping[str, object]


class LabelValue(TypedDict):
    """Persisted character-span value in a PII-label record."""

    category: str
    start: int
    end: int
    text: str


class NormalizedRecord(TypedDict):
    """Canonical augmentation record emitted after source validation."""

    text: str
    label: list[LabelValue]
    info: dict[str, str]


class BatchLengthTokenizer(Protocol):
    """The small tokenizer surface needed for augmentation length filtering."""

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
        padding: bool,
    ) -> Mapping[str, object]: ...


def record_from_object(value: object, *, source: str) -> Record:
    """Validate an untyped decoded row before any source converter reads it.

    Returns:
        The row as a string-keyed record, copied rather than aliased, so a converter cannot
        mutate the caller's decoded object.

    Raises:
        ValueError: If the value is not a mapping, or if any key is not a string. Both messages
            name the source, because these rows arrive from several dataset converters and the
            failure is only actionable once you know which one produced it.

    """
    if not isinstance(value, Mapping):
        msg = f"{source} row must be an object"
        raise ValueError(msg)
    record: Record = {}
    for key, field_value in value.items():
        if not isinstance(key, str):
            msg = f"{source} row keys must be strings"
            raise ValueError(msg)
        record[key] = field_value
    return record


def record_sequence_from_object(value: object, *, source: str) -> Iterable[object]:
    """Validate a dataset split before iterating its dynamically typed rows.

    Returns:
        The split unchanged, once proved iterable. It is not materialized, so a streaming split
        stays lazy through this check.

    Raises:
        ValueError: If the value is not iterable, or is a ``str`` or ``bytes``. Those two are
            excluded explicitly because they satisfy the iterable test and would silently
            iterate character by character, producing thousands of one-character "rows"
            instead of a type error.

    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        msg = f"{source} rows must be iterable"
        raise ValueError(msg)
    return value
