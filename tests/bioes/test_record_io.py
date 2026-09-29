from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.data.record_io import write_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


@pytest.mark.parametrize(
    ("rows", "expected", "expected_sha256"),
    [
        (
            (),
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            ({"z": 1, "a": "alpha"},),
            b'{"a": "alpha", "z": 1}\n',
            "dad43f82214e383d244f97ee2b10272a6bf043fc25f98439a5817658518a07a7",
        ),
        (
            ({"b": 2, "a": 1}, {"text": "xin chào", "label": "tên"}),
            '{"a": 1, "b": 2}\n{"label": "tên", "text": "xin chào"}\n'.encode(),
            "45c3e826667f3cca2c25c3bdf2780f1cfccc1d57d603d3bf838c4524630f1a7c",
        ),
        (
            (
                {
                    # reason: the EN DASH is the fixture, not a typo. This case exists to prove the
                    # reason: writer round-trips non-ASCII punctuation, and the sha256 three lines
                    # reason: below is pinned to these exact bytes — a hyphen would change the digest.
                    "text": "Hà Nội – Việt Nam",  # ruff: ignore[ambiguous-unicode-character-string]
                    "emoji": "🧪",
                    "nested": {"z": "ế", "a": 1},
                },
            ),
            # reason: same EN DASH as the input above — this is the expected serialization, so it has
            # reason: to match byte for byte.
            ('{"emoji": "🧪", "nested": {"a": 1, "z": "ế"}, "text": "Hà Nội – Việt Nam"}\n').encode(),  # ruff: ignore[ambiguous-unicode-character-string]
            "4dd8498f47ac34acc74debd0557e225f8853b2c17ff729802409341789eaab3e",
        ),
    ],
)
def test_write_jsonl_preserves_canonical_artifact_bytes(
    tmp_path: Path,
    rows: Iterable[Mapping[str, object]],
    expected: bytes,
    expected_sha256: str,
) -> None:
    path = tmp_path / "nested" / "records.jsonl"

    write_jsonl(path, (row for row in rows))

    payload = path.read_bytes()
    assert payload == expected
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
