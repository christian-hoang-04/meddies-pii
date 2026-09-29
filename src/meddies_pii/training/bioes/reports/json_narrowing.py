from __future__ import annotations

# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from meddies_pii.json_types import is_str_mapping as is_str_map

if TYPE_CHECKING:
    from pathlib import Path

JsonMap = Mapping[str, object]
EMPTY_MAP: JsonMap = {}


def load_optional_json_object(path: Path | None) -> dict[str, object]:
    """Load an optional JSON object, returning an empty object when absent.

    Returns:
        The decoded object, or an empty dict when the path is ``None`` or does not exist.
        Absence and an empty object are deliberately indistinguishable here -- the callers
        treat a missing optional report input as "no data" rather than an error.

    Raises:
        ValueError: When the file exists and holds valid JSON that is not an object. A
            present-but-wrong-shaped file is a real defect and is refused, unlike an absent
            one, so the empty return never hides a malformed input.

    """
    if path is None or not path.exists():
        return {}
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not is_str_map(value):
        msg = f"expected JSON object in {path}"
        raise ValueError(msg)
    return dict(value)


def parse_json_object_line(line: str) -> dict[str, object] | None:
    """Parse one JSON object line, skipping blank, malformed, and non-object values.

    Returns:
        The decoded object, or ``None`` for a blank line, a line that is not valid JSON, and
        valid JSON that is not an object. All three collapse to ``None`` on purpose: this
        reads report JSONL where a partial trailing write is expected, so the caller skips
        the line rather than failing the report. Contrast ``load_optional_json_object``,
        which REFUSES a wrong-shaped whole file.

    """
    if not line.strip():
        return None
    try:
        value: object = json.loads(line)
    except json.JSONDecodeError:
        return None
    return dict(value) if is_str_map(value) else None


def map_at(source: Mapping[str, object], key: str) -> JsonMap:
    value = source.get(key)
    return value if is_str_map(value) else EMPTY_MAP


def int_like(value: object, default: int = 0) -> int:
    # reason: `value` is `object`, not `str`, so the suggested `not value` would also swallow 0,
    # reason: False, 0.0 and every empty container -- `int_like(0)` would return the DEFAULT rather
    # reason: than 0. Only the empty STRING may fall through to the default here.
    if value is None or value == "":  # ruff: ignore[compare-to-empty-string]
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        return int(value)
    return default


def int_map(value: object) -> dict[str, int]:
    if not is_str_map(value):
        return {}
    return {key: int_like(raw) for key, raw in value.items()}


def list_of_maps_at(source: Mapping[str, object], key: str) -> list[JsonMap]:
    value = source.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [item for item in value if is_str_map(item)]
