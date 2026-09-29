from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.reports.json_narrowing import (
    load_optional_json_object,
    parse_json_object_line,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_optional_json_object_loader_preserves_missing_and_invalid_shape_policy(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")

    assert load_optional_json_object(None) == {}
    assert load_optional_json_object(missing) == {}
    with pytest.raises(ValueError, match=f"expected JSON object in {invalid}"):
        load_optional_json_object(invalid)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", None),
        ("{not-json", None),
        ("[]", None),
        ('{"uid": "row-1"}', {"uid": "row-1"}),
    ],
)
def test_tolerant_json_object_line_parser_skips_non_records(
    line: str,
    expected: dict[str, object] | None,
) -> None:
    assert parse_json_object_line(line) == expected
