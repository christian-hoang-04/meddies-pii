from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.spans import CharSpan

SHA256_HEX_LENGTH = 64


def validate_sha256(value: str, *, name: str) -> None:
    if len(value) != SHA256_HEX_LENGTH:
        msg = f"{name} must be a 64-character hex digest"
        raise ValueError(msg)
    try:
        bytes.fromhex(value)
    except ValueError as error:
        msg = f"{name} must be a 64-character hex digest"
        raise ValueError(msg) from error


@dataclass(frozen=True, slots=True)
class ExpectedFileIdentity:
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            msg = "size_bytes must be a positive integer"
            raise ValueError(msg)
        validate_sha256(self.sha256, name="sha256")


@dataclass(frozen=True, slots=True)
class SpanDetection:
    spans: tuple[CharSpan, ...]
    bucket: int
    num_tokens: int
    truncated: bool


class PiiSpanDetector(Protocol):
    def load(self) -> None: ...

    def detect(self, texts: Sequence[str]) -> list[SpanDetection]: ...
