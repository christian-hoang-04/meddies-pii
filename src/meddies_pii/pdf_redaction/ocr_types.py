from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from meddies_pii.pdf_redaction.contracts import GeometryPage, Point

if TYPE_CHECKING:
    import subprocess


class OcrAdapterError(RuntimeError):
    def __init__(self, engine: str, stage: str, *, page_index: int | None = None) -> None:
        location = "document" if page_index is None else f"page {page_index}"
        super().__init__(f"{engine} OCR failed at {location}: {stage}")
        self.engine = engine
        self.stage = stage
        self.page_index = page_index


@dataclass(frozen=True, slots=True)
class PixelToPageTransform:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.a, self.b, self.c, self.d, self.e, self.f)):
            msg = "pixel transform values must be finite"
            raise ValueError(msg)

    def apply(self, x: float, y: float) -> Point:
        return Point(
            x=self.a * x + self.c * y + self.e,
            y=self.b * x + self.d * y + self.f,
        )


@dataclass(frozen=True, slots=True)
class RasterPage:
    page_index: int
    image: object
    width_px: int
    height_px: int
    width_pt: float
    height_pt: float
    pixel_to_page: PixelToPageTransform | None = None

    def __post_init__(self) -> None:
        if self.page_index < 0:
            msg = "page_index must be non-negative"
            raise ValueError(msg)
        if self.width_px <= 0 or self.height_px <= 0:
            msg = "raster dimensions must be positive"
            raise ValueError(msg)
        if not math.isfinite(self.width_pt) or self.width_pt <= 0:
            msg = "page width must be finite and positive"
            raise ValueError(msg)
        if not math.isfinite(self.height_pt) or self.height_pt <= 0:
            msg = "page height must be finite and positive"
            raise ValueError(msg)
        if self.pixel_to_page is None:
            object.__setattr__(
                self,
                "pixel_to_page",
                PixelToPageTransform(
                    a=self.width_pt / self.width_px,
                    b=0.0,
                    c=0.0,
                    d=self.height_pt / self.height_px,
                    e=0.0,
                    f=0.0,
                ),
            )

    def map_pixel(self, x: float, y: float) -> Point:
        transform = self.pixel_to_page
        if transform is None:
            msg = "pixel transform was not initialized"
            raise RuntimeError(msg)
        return transform.apply(x, y)


class GeometryOcr(Protocol):
    def extract(self, raster: RasterPage) -> GeometryPage: ...


class RapidEngine(Protocol):
    def __call__(
        self,
        image: object,
        *,
        return_word_box: bool,
        return_single_char_box: bool,
    ) -> object: ...


class TesseractRunner(Protocol):
    def run(self, image: object) -> str: ...


class CommandRunner(Protocol):
    # reason: CommandRunner mirrors subprocess.run so injected runners and production share one callable protocol.
    def __call__(  # ruff: ignore[too-many-arguments]
        self,
        args: tuple[str, ...],
        *,
        input: bytes,  # ruff: ignore[builtin-argument-shadowing] reason: mirrors subprocess.run's keyword contract
        stdout: int,
        stderr: int,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]: ...


RapidOcrBackend = Literal["openvino", "onnxruntime"]
RapidOcrModelTier = Literal["small", "tiny"]
OpenVinoPerformanceHint = Literal["LATENCY", "THROUGHPUT"]


@dataclass(frozen=True, slots=True)
class RapidOcrRuntimeConfig:
    backend: RapidOcrBackend
    model_tier: RapidOcrModelTier
    threads: int = 4
    openvino_num_streams: int = 1
    openvino_performance_hint: OpenVinoPerformanceHint = "LATENCY"
    ort_inter_op_threads: int = 1

    def __post_init__(self) -> None:
        if self.backend not in {"openvino", "onnxruntime"}:
            msg = "unsupported RapidOCR backend"
            raise ValueError(msg)
        if self.model_tier not in {"small", "tiny"}:
            msg = "unsupported PP-OCRv6 model tier"
            raise ValueError(msg)
        _require_positive_integer(self.threads, "threads")
        _require_positive_integer(self.openvino_num_streams, "openvino_num_streams")
        if self.openvino_performance_hint not in {"LATENCY", "THROUGHPUT"}:
            msg = "unsupported OpenVINO performance hint"
            raise ValueError(msg)
        _require_positive_integer(self.ort_inter_op_threads, "ort_inter_op_threads")


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
