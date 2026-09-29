from __future__ import annotations

from typing import TYPE_CHECKING

import pymupdf
import pytest

from meddies_pii.pdf_redaction.extraction import PdfiumExtractor, PymupdfExtractor

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.pdf_redaction.contracts import Point


def _signed_double_area(points: tuple[Point, Point, Point, Point]) -> float:
    return sum(
        current.x * following.y - following.x * current.y
        for current, following in zip(points, points[1:] + points[:1], strict=True)
    )


def _write_rotated_native_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=200.0, height=300.0)
    page.insert_text((20.0, 50.0), "José ID-42", fontsize=12.0)
    page.insert_text((140.0, 180.0), "ROT", fontsize=12.0, rotate=90)
    page.set_rotation(90)
    document.save(path)
    document.close()


def test_pymupdf_extracts_exact_offsets_in_unrotated_page_space(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rotated.pdf"
    _write_rotated_native_pdf(source)

    (page,) = PymupdfExtractor().extract(source)

    assert (page.width_pt, page.height_pt) == (200.0, 300.0)
    assert page.text == "José ID-42\n\nROT"
    assert all(page.text[token.start : token.end] == token.text for token in page.tokens)
    assert all(token.confidence is None for token in page.tokens)
    rotated_r = next(token for token in page.tokens if token.text == "R")
    first, second, *_ = rotated_r.quad.points
    assert first.x == pytest.approx(second.x)
    assert first.y != second.y
    assert _signed_double_area(rotated_r.quad.points) > 0.0


def test_pdfium_extracts_unicode_and_ignores_whitespace_geometry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rotated.pdf"
    _write_rotated_native_pdf(source)

    (page,) = PdfiumExtractor().extract(source)

    assert (page.width_pt, page.height_pt) == (200.0, 300.0)
    assert page.text == "José ID-42\r\nROT"
    assert "".join(token.text for token in page.tokens) == "JoséID-42ROT"
    assert all(page.text[token.start : token.end] == token.text for token in page.tokens)
    assert all(token.confidence is None for token in page.tokens)
    assert all(_signed_double_area(token.quad.points) > 0.0 for token in page.tokens)
