from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the redaction runtime is an optional extra, and several backends are resolved by name at call time.
import importlib
import math
import operator
from contextlib import ExitStack, closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from meddies_pii.pdf_redaction.writers import RasterRebuildWriter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL.Image import Image as PillowImage

    from meddies_pii.pdf_redaction.contracts import PageRegion, Point

MAX_CHANNEL_VALUE = 255
QUARTER_TURN_DEGREES = 90
HALF_TURN_DEGREES = 180
THREE_QUARTER_TURN_DEGREES = 270

DEFAULT_RENDER_DPI = 72
DEFAULT_PIXEL_DIFFERENCE_THRESHOLD = 16
DEFAULT_MASK_DILATION_PIXELS = 2


class _PdfiumBitmap(Protocol):
    def to_pil(self) -> PillowImage: ...


class _PdfiumPage(Protocol):
    def render(self, *, scale: float) -> _PdfiumBitmap: ...

    def get_size(self) -> tuple[float, float]: ...

    def get_rotation(self) -> int: ...


class _PdfiumDocument(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _PdfiumPage: ...

    def close(self) -> None: ...


class _PdfiumModule(Protocol):
    PdfDocument: Callable[[Path], _PdfiumDocument]


@dataclass(frozen=True, slots=True)
class UnexplainedChangeRegion:
    page_index: int
    x0_px: int
    y0_px: int
    x1_px: int
    y1_px: int
    pixel_count: int


@dataclass(frozen=True, slots=True)
class RenderFidelity:
    page_count: int
    render_dpi: int
    pixel_difference_threshold: int
    mask_dilation_pixels: int
    total_outside_mask_pixels: int
    changed_outside_mask_pixels: int
    changed_outside_mask_rate: float
    unexplained_changed_regions: tuple[UnexplainedChangeRegion, ...]


def write_raster_reference(
    source: Path,
    destination: Path,
    *,
    dpi: int,
) -> Path:
    """Build the benchmark's unredacted raster baseline.

    Returns:
        The path the unredacted raster baseline was written to. It is produced with no redaction
        regions so later comparisons measure rendering, not masking.

    """
    # reason: the benchmark needs an unredacted reference, but the product writer must not expose that capability.
    writer = RasterRebuildWriter(dpi=dpi)
    result = writer._write(  # ruff: ignore[private-member-access]
        source,
        (),
        destination,
        allow_empty_regions=True,
        safety="reference",
    )
    return result.output_path


# reason: measure render orders validate before render rgb; helper seams would let sample setup drift.
def measure_render_fidelity(  # ruff: ignore[too-many-arguments,too-many-locals]
    source: Path,
    output: Path,
    oracle_regions: Sequence[PageRegion],
    *,
    mask_geometry_source: Path | None = None,
    dpi: int = DEFAULT_RENDER_DPI,
    pixel_difference_threshold: int = DEFAULT_PIXEL_DIFFERENCE_THRESHOLD,
    mask_dilation_pixels: int = DEFAULT_MASK_DILATION_PIXELS,
) -> RenderFidelity:
    """Compare PDFium renders outside dilated oracle masks.

    A pixel changes when its largest absolute RGB-channel difference exceeds the
    configured 0-255 threshold. Change regions use eight-pixel connectivity and
    half-open pixel bounds.

    Returns:
        The `RenderFidelity` record: page count, render DPI, the pixel-difference threshold and the
        measured differences outside the dilated oracle masks.

    Raises:
        ValueError: if the source and output PDFs disagree on page count; if the mask-geometry PDF
            disagrees with them; if an oracle region references a page that does not exist; or if a
            source and output page render at different sizes. Each would make the outside-mask
            comparison meaningless rather than merely inaccurate, so it refuses instead of scoring.

    """
    _validate_configuration(dpi, pixel_difference_threshold, mask_dilation_pixels)

    pdfium = cast("_PdfiumModule", importlib.import_module("pypdfium2"))

    with ExitStack() as stack:
        source_document = stack.enter_context(closing(pdfium.PdfDocument(source)))
        output_document = stack.enter_context(closing(pdfium.PdfDocument(output)))
        geometry_document = (
            source_document
            if mask_geometry_source is None
            else stack.enter_context(closing(pdfium.PdfDocument(mask_geometry_source)))
        )
        if len(source_document) != len(output_document):
            msg = "source and output PDFs must have the same page count"
            raise ValueError(msg)
        if len(geometry_document) != len(source_document):
            msg = "mask geometry PDF must have the same page count"
            raise ValueError(msg)
        if any(region.page_index >= len(source_document) for region in oracle_regions):
            msg = "oracle region references a missing page"
            raise ValueError(msg)

        total_pixels = 0
        changed_pixels = 0
        changed_regions: list[UnexplainedChangeRegion] = []
        scale = dpi / 72
        for page_index in range(len(source_document)):
            source_page = source_document[page_index]
            output_page = output_document[page_index]
            geometry_page = geometry_document[page_index]
            _require_same_display_size(source_page.get_size(), output_page.get_size())
            _require_same_display_size(source_page.get_size(), geometry_page.get_size())
            source_rgb = _render_rgb(source_page, scale)
            output_rgb = _render_rgb(output_page, scale)
            if source_rgb.shape != output_rgb.shape:
                msg = "source and output page renders must have the same size"
                raise ValueError(msg)

            page_regions = tuple(region for region in oracle_regions if region.page_index == page_index)
            outside_mask = _outside_mask(
                geometry_page,
                source_rgb.shape[:2],
                page_regions,
                mask_dilation_pixels,
            )
            changed = (
                np.max(
                    np.abs(source_rgb.astype(np.int16) - output_rgb.astype(np.int16)),
                    axis=2,
                )
                > pixel_difference_threshold
            ) & outside_mask
            total_pixels += int(np.count_nonzero(outside_mask))
            changed_pixels += int(np.count_nonzero(changed))
            changed_regions.extend(_connected_regions(changed, page_index))

        changed_rate = changed_pixels / total_pixels if total_pixels else 0.0
        return RenderFidelity(
            page_count=len(source_document),
            render_dpi=dpi,
            pixel_difference_threshold=pixel_difference_threshold,
            mask_dilation_pixels=mask_dilation_pixels,
            total_outside_mask_pixels=total_pixels,
            changed_outside_mask_pixels=changed_pixels,
            changed_outside_mask_rate=changed_rate,
            unexplained_changed_regions=tuple(changed_regions),
        )


def _validate_configuration(
    dpi: int,
    pixel_difference_threshold: int,
    mask_dilation_pixels: int,
) -> None:
    if dpi <= 0:
        msg = "render DPI must be positive"
        raise ValueError(msg)
    if not 0 <= pixel_difference_threshold <= MAX_CHANNEL_VALUE:
        msg = "pixel difference threshold must be between 0 and 255"
        raise ValueError(msg)
    if mask_dilation_pixels < 0:
        msg = "mask dilation must be non-negative"
        raise ValueError(msg)


def _require_same_display_size(
    source_size: tuple[float, float],
    output_size: tuple[float, float],
) -> None:
    if not all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=0.01) for left, right in zip(source_size, output_size, strict=True)
    ):
        msg = "source and output PDF pages must have the same display size"
        raise ValueError(msg)


def _render_rgb(page: _PdfiumPage, scale: float) -> NDArray[np.uint8]:
    image = page.render(scale=scale).to_pil().convert("RGB")
    return np.asarray(image, dtype=np.uint8).copy()


def _outside_mask(
    page: _PdfiumPage,
    image_shape: tuple[int, int],
    regions: Sequence[PageRegion],
    dilation_pixels: int,
) -> NDArray[np.bool_]:
    from PIL import Image, ImageDraw, ImageFilter

    height_px, width_px = image_shape
    display_width, display_height = page.get_size()
    rotation = page.get_rotation()
    if rotation in {90, 270}:
        unrotated_width, unrotated_height = display_height, display_width
    else:
        unrotated_width, unrotated_height = display_width, display_height

    mask = Image.new("L", (width_px, height_px), 0)
    draw = ImageDraw.Draw(mask)
    for region in regions:
        points = tuple(
            _point_to_pixel(
                point,
                rotation=rotation,
                unrotated_width=unrotated_width,
                unrotated_height=unrotated_height,
                display_width=display_width,
                display_height=display_height,
                width_px=width_px,
                height_px=height_px,
            )
            for point in region.quad.points
        )
        draw.polygon(points, fill=255)
    if dilation_pixels and regions:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilation_pixels + 1))
    return cast("NDArray[np.bool_]", np.asarray(mask, dtype=np.uint8) == 0)


# reason: point to pixel keeps point/height px at its adapter seam; bundling would hide required inputs.
def _point_to_pixel(  # ruff: ignore[too-many-arguments]
    point: Point,
    *,
    rotation: int,
    unrotated_width: float,
    unrotated_height: float,
    display_width: float,
    display_height: float,
    width_px: int,
    height_px: int,
) -> tuple[float, float]:
    if not 0 <= point.x <= unrotated_width or not 0 <= point.y <= unrotated_height:
        msg = "oracle region must be inside its source page"
        raise ValueError(msg)
    if rotation == 0:
        display_x, display_y = point.x, point.y
    elif rotation == QUARTER_TURN_DEGREES:
        display_x, display_y = unrotated_height - point.y, point.x
    elif rotation == HALF_TURN_DEGREES:
        display_x = unrotated_width - point.x
        display_y = unrotated_height - point.y
    elif rotation == THREE_QUARTER_TURN_DEGREES:
        display_x, display_y = point.y, unrotated_width - point.x
    else:
        msg = "PDF page rotation must be a multiple of 90 degrees"
        raise ValueError(msg)
    return (
        display_x * width_px / display_width,
        display_y * height_px / display_height,
    )


# reason: connected regions owns find and unexplained together; splitting would let sample setup drift.
def _connected_regions(changed: NDArray[np.bool_], page_index: int) -> tuple[UnexplainedChangeRegion, ...]:  # ruff: ignore[complex-structure]
    if not np.any(changed):
        return ()

    parents: list[int] = []
    runs: list[tuple[int, int, int, int]] = []
    previous_runs: list[tuple[int, int, int]] = []

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = int(parents[label])
        return label

    def union(left: int, right: int) -> int:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root
        return left_root

    for y, row in enumerate(changed):
        transitions = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        current_runs: list[tuple[int, int, int]] = []
        for start_value, end_value in zip(starts, ends, strict=True):
            start = int(start_value)
            end = int(end_value)
            overlapping_labels = [
                label
                for previous_start, previous_end, label in previous_runs
                if previous_end >= start and end >= previous_start
            ]
            if overlapping_labels:
                label = find(overlapping_labels[0])
                for overlapping_label in overlapping_labels[1:]:
                    label = union(label, overlapping_label)
            else:
                label = len(parents)
                parents.append(label)
            current_runs.append((start, end, label))
            runs.append((y, start, end, label))
        previous_runs = current_runs

    component_bounds: dict[int, list[int]] = {}
    for y, start, end, label in runs:
        root = find(label)
        if root not in component_bounds:
            component_bounds[root] = [start, y, end, y + 1, end - start]
            continue
        bounds = component_bounds[root]
        bounds[0] = min(bounds[0], start)
        bounds[1] = min(bounds[1], y)
        bounds[2] = max(bounds[2], end)
        bounds[3] = max(bounds[3], y + 1)
        bounds[4] += end - start

    return tuple(
        UnexplainedChangeRegion(
            page_index=page_index,
            x0_px=bounds[0],
            y0_px=bounds[1],
            x1_px=bounds[2],
            y1_px=bounds[3],
            pixel_count=bounds[4],
        )
        for bounds in sorted(component_bounds.values(), key=operator.itemgetter(1, 0))
    )
