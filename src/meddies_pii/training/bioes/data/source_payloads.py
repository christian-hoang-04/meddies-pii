"""Controlled decoding of historical source collection fields."""

from __future__ import annotations

# ruff: file-ignore[try-except-in-loop]
# reason: the try IS this loop's per-item parse decision -- the failing item is skipped, recorded or
# reason: partitioned by name. Hoisting it would discard which item failed and abort the rest.
import ast
import json


def parse_serialized_collection(value: object) -> object:
    """Decode a source list or mapping without executing source-provided code.

    Returns:
        A ``dict`` or ``list`` unchanged, and whatever the first successful parser yields for a
        string -- ``ast.literal_eval`` is tried before ``json.loads``, since these sources
        serialize with Python's own ``repr`` more often than with JSON. Everything else becomes
        ``None``: a real null, the literal string ``"None"`` these datasets write for one, a
        string neither parser accepts, and any other type. It never raises, so an unparseable
        payload reads as an absent one.

        ``literal_eval`` rather than ``eval`` is the point of the function. The payloads come
        from external datasets, and ``literal_eval`` evaluates literals only, so a crafted row
        cannot execute anything.

    """
    if value is None or value == "None":
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        for parser in (ast.literal_eval, json.loads):
            try:
                return parser(value)
            except (SyntaxError, ValueError, TypeError):
                continue
    return None
