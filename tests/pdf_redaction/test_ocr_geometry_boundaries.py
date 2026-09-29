from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
from types import SimpleNamespace

import pytest

from meddies_pii.pdf_redaction.ocr_geometry import RapidOcrAdapter, TesseractTsvAdapter
from meddies_pii.pdf_redaction.ocr_types import OcrAdapterError, RasterPage


def _raster(page_index: int = 7) -> RasterPage:
    return RasterPage(page_index, object(), 100, 100, 200.0, 200.0)


class _RapidEngine:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls = 0

    def __call__(
        self,
        image: object,
        *,
        return_word_box: bool,
        return_single_char_box: bool,
    ) -> object:
        del image
        assert return_word_box is True
        assert return_single_char_box is False
        self.calls += 1
        return self.output


def test_rapidocr_preserves_rotated_and_overlapping_word_geometry() -> None:
    engine = _RapidEngine(
        SimpleNamespace(
            txts=("alpha beta",),
            word_results=(
                (
                    ("alpha", 0.8, ((4, 4), (24, 4), (20, 12), (0, 12))),
                    ("beta", 0.9, ((12, 2), (28, 2), (24, 10), (8, 10))),
                ),
            ),
        ),
    )
    adapter = RapidOcrAdapter(engine_factory=lambda: engine)

    page = adapter.extract(_raster())

    assert page.text == "alpha beta"
    assert tuple(token.text for token in page.tokens) == ("alpha", "beta")
    assert page.tokens[0].quad.points[2].x == 40.0
    assert page.tokens[0].quad.points[2].y == 24.0
    assert engine.calls == 1


def test_rapidocr_fails_closed_for_rotated_points_in_wrong_perimeter_order() -> None:
    engine = _RapidEngine(
        SimpleNamespace(
            txts=("alpha",),
            word_results=((("alpha", 0.8, ((0, 0), (-4, 8), (16, 8), (20, 0))),),),
        ),
    )

    with pytest.raises(OcrAdapterError) as raised:
        RapidOcrAdapter(engine_factory=lambda: engine).extract(_raster(3))

    assert raised.value.engine == "rapidocr"
    assert raised.value.stage == "invalid_output"
    assert raised.value.page_index == 3


def test_tesseract_accepts_empty_valid_tsv_without_invoking_geometry_mapping() -> None:
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tleft\ttop\twidth\theight\tconf\ttext\n"

    class Runner:
        # reason: this mirrors the `TesseractRunner` protocol, which `ocr_geometry.py:79` calls as
        # reason: `self._get_runner().run(raster.image)` on the instance the factory returns.
        def run(self, image: object) -> str:  # ruff: ignore[no-self-use]
            del image
            return header

    page = TesseractTsvAdapter(runner_factory=Runner).extract(_raster())

    # reason: the empty string is the exact value proven here; `not page.text` would also accept `None`, so the
    # reason: equality is what pins the header-only TSV to a text of `""` rather than a missing text.
    assert page.text == ""  # ruff: ignore[compare-to-empty-string]
    assert page.tokens == ()


def test_tesseract_rejects_nonfinite_geometry_without_exposing_tsv_text() -> None:
    canary = "private-canary"
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tleft\ttop\twidth\t"
        "height\tconf\ttext\n"
        f"5\t1\t1\t1\t1\tnan\t0\t10\t10\t95\t{canary}\n"
    )

    class Runner:
        # reason: this mirrors the `TesseractRunner` protocol, which `ocr_geometry.py:79` calls as
        # reason: `self._get_runner().run(raster.image)` on the instance the factory returns.
        def run(self, image: object) -> str:  # ruff: ignore[no-self-use]
            del image
            return tsv

    with pytest.raises(OcrAdapterError) as raised:
        TesseractTsvAdapter(runner_factory=Runner).extract(_raster(4))

    assert raised.value.engine == "tesseract"
    assert raised.value.stage == "invalid_tsv"
    assert raised.value.page_index == 4
    assert canary not in str(raised.value)


@pytest.mark.parametrize(
    "output",
    [
        SimpleNamespace(txts=7, word_results=()),
        SimpleNamespace(txts=(1,), word_results=((),)),
        SimpleNamespace(txts=("word",), word_results=()),
        SimpleNamespace(
            txts=("word",),
            word_results=((("", 0.8, ((0, 0), (1, 0), (1, 1), (0, 1))),),),
        ),
        SimpleNamespace(
            txts=("word",),
            word_results=((("word", True, ((0, 0), (1, 0), (1, 1), (0, 1))),),),
        ),
        SimpleNamespace(
            txts=("word",),
            word_results=((("word", float("nan"), ((0, 0), (1, 0), (1, 1), (0, 1))),),),
        ),
        SimpleNamespace(txts=("word",), word_results=((("word", 0.8, ((0, 0), (1, 0), (1, 1))),),)),
        SimpleNamespace(
            txts=("word",),
            word_results=((("word", 0.8, ((0, 0), (1, 0), (1, float("inf")), (0, 1))),),),
        ),
    ],
)
def test_rapidocr_rejects_malformed_word_payload_shapes(output: object) -> None:
    with pytest.raises(OcrAdapterError, match="invalid_output"):
        RapidOcrAdapter(engine_factory=lambda: _RapidEngine(output)).extract(_raster())


@pytest.mark.parametrize(
    "row",
    [
        "4\t1\t1\t1\t1\t0\t0\t10\t10\t95\tignored\n",
        "5\t1\t1\t1\t1\t0\t0\t0\t10\t95\tzero-width\n",
        "5\t1\t1\t1\t1\tbogus\t0\t10\t10\t95\tbad-number\n",
    ],
)
def test_tesseract_rejects_bad_or_nonword_rows_at_the_public_adapter(
    row: str,
) -> None:
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tleft\ttop\twidth\theight\tconf\ttext\n"

    class Runner:
        # reason: this mirrors the `TesseractRunner` protocol, which `ocr_geometry.py:79` calls as
        # reason: `self._get_runner().run(raster.image)` on the instance the factory returns.
        def run(self, image: object) -> str:  # ruff: ignore[no-self-use]
            del image
            return header + row

    adapter = TesseractTsvAdapter(runner_factory=Runner)
    if row.startswith("4"):
        assert adapter.extract(_raster()).tokens == ()
    else:
        with pytest.raises(OcrAdapterError, match="invalid_tsv"):
            adapter.extract(_raster())


def test_tesseract_rejects_a_tsv_with_missing_required_columns() -> None:
    class Runner:
        # reason: this mirrors the `TesseractRunner` protocol, which `ocr_geometry.py:79` calls as
        # reason: `self._get_runner().run(raster.image)` on the instance the factory returns.
        def run(self, image: object) -> str:  # ruff: ignore[no-self-use]
            del image
            return "level\ttext\n5\tvisible\n"

    with pytest.raises(OcrAdapterError, match="invalid_tsv"):
        TesseractTsvAdapter(runner_factory=Runner).extract(_raster())


def test_rapidocr_rejects_polygon_points_without_two_coordinates() -> None:
    output = SimpleNamespace(
        txts=("word",),
        word_results=((("word", 0.8, ((0,), (1, 0), (1, 1), (0, 1))),),),
    )

    with pytest.raises(OcrAdapterError, match="invalid_output"):
        RapidOcrAdapter(engine_factory=lambda: _RapidEngine(output)).extract(_raster())
