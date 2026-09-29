"""Validated JSON value types for untrusted provider and dataset payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeGuard

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    """Return whether a value is a dict, claiming nothing about its key or value types.

    Returns:
        True for any dict, narrowing the value to ``dict[object, object]``. Use
        ``is_str_mapping`` where the caller needs string keys; this guard does not check them.

    """
    return isinstance(value, dict)


def is_str_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    """Return whether an untrusted value is a mapping whose keys are all strings.

    A bare `isinstance(value, Mapping)` leaves the key type unknown, and `Mapping` is invariant in
    it, so the result stays unusable where `Mapping[str, object]` is required. Checking the keys
    here states the guarantee once instead of at each call site.

    Returns:
        True when the value is a Mapping and every key is a str, which narrows it to
        ``Mapping[str, object]`` so string subscripts downstream typecheck.

    """
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def is_str_list(value: object) -> TypeGuard[list[str]]:
    """Return whether an untrusted value is a list holding only strings.

    Returns:
        True when the value is a list and every item is a str, narrowing it to ``list[str]``.
        An empty list satisfies this, since it holds no non-string item.

    """
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def is_json_value(value: object) -> TypeGuard[JsonValue]:
    """Return whether an untrusted value can be safely written as JSON.

    Returns:
        True when the value is a JSON scalar, or a list or dict whose every member is itself
        one -- checked recursively, and requiring str keys at every dict level, so a caller
        that gets True can serialize without a custom encoder.

    """
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if _is_object_list(value):
        return all(is_json_value(item) for item in value)
    if is_object_dict(value):
        return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
    return False


def is_json_object(value: object) -> TypeGuard[JsonObject]:
    """Return whether an untrusted value is a JSON object with string keys.

    Returns:
        True when the value is a dict and serializable all the way down, narrowing it to
        ``JsonObject``. This is the dict-rooted case of ``is_json_value``.

    """
    return is_object_dict(value) and is_json_value(value)


def as_json_object(value: object) -> JsonObject | None:
    """Narrow a system-boundary value to a JSON object, or reject it.

    Returns:
        The value itself typed as ``JsonObject``, or None if it is not one. Returning None
        rather than raising lets the caller decide whether a rejected payload is fatal.

    """
    return value if is_json_object(value) else None


def as_object_list(value: object) -> list[object] | None:
    """Narrow an untrusted JSON top-level value to a list of opaque records.

    Returns:
        The value itself typed as ``list[object]``, or None if it is not a list. The items
        stay opaque: this says the container is a list, nothing about what it holds.

    """
    return value if _is_object_list(value) else None


def required_int(value: object, *, field: str) -> int:
    """Narrow an untrusted JSON field to an integer, refusing a bool rather than coercing it.

    `bool` is a subclass of `int`, so a bare `isinstance(value, int)` accepts `True` and
    silently stores it as 1. Rejecting it here states the boundary rule once instead of at
    each decoder.

    Returns:
        The value itself typed as ``int``.

    Raises:
        ValueError: When the value is a bool or is not an int, naming the field so a
            malformed persisted payload says which key was wrong.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field} must be an integer"
        # reason: the contract is malformed persisted payload -> ValueError, and every caller of this
        # reason: decoder matches that type; a TypeError here would split one failure class on guard shape.
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    return value
