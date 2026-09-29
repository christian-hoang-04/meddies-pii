from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

import pytest
from PIL import Image

import meddies_pii.pdf_redaction.document_pdfium as document_module
import meddies_pii.pdf_redaction.document_pdfium_support as support_module
from meddies_pii.pdf_redaction.document import (
    DocumentAdapterError,
    PdfiumDocumentAdapter,
)

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Iterable


class _ByrefUnsignedInt(Protocol):
    """What `ctypes.byref(c_uint())` exposes to the colour getters this module fakes."""

    _obj: ctypes.c_uint


@dataclass
class _RawText:
    mode: int
    fill_available: bool = False
    fill_alpha: int = 0
    stroke_available: bool = False
    stroke_alpha: int = 0


class _TextObject:
    def __init__(self, value: object, raw: _RawText) -> None:
        self._value = value
        self.raw = raw

    def extract(self) -> object:
        return self._value


class _ImageObject:
    def __init__(self, bounds: object) -> None:
        self._bounds = bounds

    def get_bounds(self) -> object:
        return self._bounds


class _TextPage:
    raw = object()

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Bitmap:
    def __init__(self) -> None:
        self.closed = False

    # reason: this mirrors `pypdfium2`'s `PdfBitmap.to_pil`, which the code under test calls on the bitmap
    # reason: instance returned by `render`, so the bound-method form is the API being stood in for.
    def to_pil(self) -> Image.Image:  # ruff: ignore[no-self-use]
        return Image.new("RGB", (2, 2), "white")

    def close(self) -> None:
        self.closed = True


class _Page:
    def __init__(
        self,
        *,
        cropbox: object = (0.0, 0.0, 100.0, 100.0),
        size: tuple[float, float] = (100.0, 100.0),
        rotation: int = 0,
        objects: Iterable[object] = (),
    ) -> None:
        self.cropbox = cropbox
        self.size = size
        self.rotation = rotation
        self.objects = tuple(objects)
        self.text_page = _TextPage()
        self.bitmap = _Bitmap()
        self.closed = False

    def get_cropbox(self) -> object:
        return self.cropbox

    def get_size(self) -> tuple[float, float]:
        return self.size

    def get_rotation(self) -> int:
        return self.rotation

    def get_textpage(self) -> _TextPage:
        return self.text_page

    def get_objects(self, *, max_depth: int, textpage: _TextPage) -> Iterable[object]:
        assert max_depth == 15
        assert textpage is self.text_page
        return self.objects

    def render(self, **_: object) -> _Bitmap:
        return self.bitmap

    def close(self) -> None:
        self.closed = True


class _Document:
    def __init__(self, pages: tuple[_Page, ...]) -> None:
        self.pages = pages
        self.forms_initialized = False
        self.closed = False

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> _Page:
        return self.pages[index]

    def init_forms(self) -> None:
        self.forms_initialized = True

    def close(self) -> None:
        self.closed = True


class _Annotation:
    def __init__(self, subtype: str) -> None:
        self.subtype = subtype

    def get(self, key: str, default: object) -> object:
        assert key == "/Subtype"
        return self.subtype or default


class _AnnotationReference:
    def __init__(self, subtype: str) -> None:
        self.annotation = _Annotation(subtype)

    def get_object(self) -> _Annotation:
        return self.annotation


class _PypdfPage:
    def __init__(self, annotations: object = ()) -> None:
        self.annotations = annotations

    def get(self, key: str, default: object) -> object:
        assert key == "/Annots"
        return self.annotations if self.annotations is not None else default


class _Reader:
    def __init__(self, pages: tuple[_PypdfPage, ...]) -> None:
        self.pages = pages
        self.attachments: dict[str, bytes] = {}
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _raw_module() -> SimpleNamespace:
    fill_modes = (1, 2, 3, 4)
    stroke_modes = (5, 6, 3, 4)

    def fill_color(
        raw: _RawText,
        red: object,
        green: object,
        blue: object,
        alpha: _ByrefUnsignedInt,
    ) -> int:
        del red, green, blue
        alpha._obj.value = raw.fill_alpha
        return int(raw.fill_available)

    def stroke_color(
        raw: _RawText,
        red: object,
        green: object,
        blue: object,
        alpha: _ByrefUnsignedInt,
    ) -> int:
        del red, green, blue
        alpha._obj.value = raw.stroke_alpha
        return int(raw.stroke_available)

    return SimpleNamespace(
        FPDF_TEXTRENDERMODE_FILL=fill_modes[0],
        FPDF_TEXTRENDERMODE_FILL_CLIP=fill_modes[1],
        FPDF_TEXTRENDERMODE_FILL_STROKE=fill_modes[2],
        FPDF_TEXTRENDERMODE_FILL_STROKE_CLIP=fill_modes[3],
        FPDF_TEXTRENDERMODE_STROKE=stroke_modes[0],
        FPDF_TEXTRENDERMODE_STROKE_CLIP=stroke_modes[1],
        FPDFTextObj_GetTextRenderMode=lambda raw: raw.mode,
        FPDFPageObj_GetFillColor=fill_color,
        FPDFPageObj_GetStrokeColor=stroke_color,
    )


def _install_pdf_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pages: tuple[_Page, ...],
    pypdf_pages: tuple[_PypdfPage, ...] | None = None,
) -> tuple[_Document, _Reader]:
    document = _Document(pages)
    reader = _Reader(pypdf_pages or tuple(_PypdfPage() for _ in pages))
    pdfium = SimpleNamespace(
        PdfDocument=lambda _: document,
        PdfTextObj=_TextObject,
        PdfImage=_ImageObject,
    )
    pypdf = SimpleNamespace(PdfReader=lambda *_args, **_kwargs: reader)
    raw = _raw_module()

    def import_module(name: str) -> object:
        if name == "pypdfium2":
            return pdfium
        if name == "pypdfium2.raw":
            return raw
        if name == "pypdf":
            return pypdf
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(document_module.importlib, "import_module", import_module)
    monkeypatch.setattr(support_module.importlib, "import_module", import_module)
    return document, reader


@pytest.mark.parametrize("operation", ["inspect", "render"])
def test_pdfium_adapter_fails_closed_when_dependency_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    def missing_dependency(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        document_module.importlib,
        "import_module",
        missing_dependency,
    )
    adapter = PdfiumDocumentAdapter()

    # reason: the branch only selects which call to make, and both are the raising call. Hoisting the `if` out
    # reason: of the block would put the raising call outside `raises`, letting the exception escape the test.
    with pytest.raises(DocumentAdapterError) as raised:  # ruff: ignore[pytest-raises-with-multiple-statements]
        if operation == "inspect":
            adapter.inspect(b"%PDF")
        else:
            adapter.render_page(b"%PDF", page_index=0)

    assert raised.value.stage == "dependency_unavailable"


@pytest.mark.parametrize(
    ("indices", "stage"),
    [
        ((True,), "invalid_page_index"),
        ((-1,), "page_index_out_of_range"),
    ],
)
def test_pdfium_adapter_validates_indices_before_loading_native_runtime(
    monkeypatch: pytest.MonkeyPatch,
    indices: tuple[int, ...],
    stage: str,
) -> None:
    imported = False

    def reject_import(_: str) -> object:
        nonlocal imported
        imported = True
        msg = "native runtime must not load"
        raise AssertionError(msg)

    monkeypatch.setattr(document_module.importlib, "import_module", reject_import)

    with pytest.raises(DocumentAdapterError) as raised:
        PdfiumDocumentAdapter().render_pages(b"%PDF", page_indices=indices)

    assert raised.value.stage == stage
    assert imported is False


def test_pdfium_adapter_rejects_out_of_range_page_and_closes_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, _ = _install_pdf_fakes(monkeypatch, pages=(_Page(),))

    with pytest.raises(DocumentAdapterError) as raised:
        PdfiumDocumentAdapter().render_page(b"%PDF", page_index=1)

    assert raised.value.stage == "page_index_out_of_range"
    assert document.forms_initialized is True
    assert document.closed is True


def test_pdfium_adapter_rejects_unsupported_rotation_and_cleans_render_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _Page(rotation=45)
    document, _ = _install_pdf_fakes(monkeypatch, pages=(page,))

    with pytest.raises(DocumentAdapterError) as raised:
        PdfiumDocumentAdapter(dpi=72).render_page(b"%PDF", page_index=0)

    assert raised.value.stage == "page_render"
    assert page.bitmap.closed is True
    assert page.closed is True
    assert document.closed is True


def test_pdfium_inspection_classifies_text_visibility_images_and_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = (
        _TextObject("VISIBLE", _RawText(1, fill_available=True, fill_alpha=255)),
        _TextObject("HIDDEN", _RawText(1)),
        _TextObject("STROKE", _RawText(5, stroke_available=True, stroke_alpha=255)),
        _TextObject("NONE", _RawText(5)),
        _ImageObject((0.0, 0.0, 60.0, 40.0)),
        _ImageObject((0.0, 60.0, 60.0, 100.0)),
        _ImageObject((30.0, 20.0, 90.0, 80.0)),
        _ImageObject((200.0, 200.0, 300.0, 300.0)),
    )
    page = _Page(objects=objects)
    annotations = (
        _AnnotationReference("/Widget"),
        _AnnotationReference("/Text"),
    )
    document, reader = _install_pdf_fakes(
        monkeypatch,
        pages=(page,),
        pypdf_pages=(_PypdfPage(annotations),),
    )

    inspection = PdfiumDocumentAdapter().inspect(b"%PDF")

    inspected = inspection.pages[0]
    assert inspected.signals.visible_native_characters == 13
    assert inspected.invisible_native_characters == 10
    assert inspected.displayed_image_count == 3
    assert inspected.raster_area_ratio == pytest.approx(0.72)
    assert inspected.annotation_count == 1
    assert inspected.widget_count == 1
    assert set(inspected.signals.trust_diagnostics) == {
        "annotation_object",
        "form_widget",
        "invisible_text_layer",
    }
    assert page.text_page.closed is True
    assert page.closed is True
    assert document.closed is True
    assert reader.closed is True


@pytest.mark.parametrize(
    ("page", "stage"),
    [
        (_Page(cropbox=(0.0, 0.0, 0.0, 100.0)), "page_inspection"),
        (_Page(cropbox=(0.0, 0.0, 100.0)), "page_inspection"),
        (
            _Page(objects=(_TextObject(7, _RawText(1)),)),
            "page_inspection",
        ),
    ],
)
def test_pdfium_inspection_wraps_invalid_native_output_and_cleans_resources(
    monkeypatch: pytest.MonkeyPatch,
    page: _Page,
    stage: str,
) -> None:
    document, reader = _install_pdf_fakes(monkeypatch, pages=(page,))

    with pytest.raises(DocumentAdapterError) as raised:
        PdfiumDocumentAdapter().inspect(b"%PDF")

    assert raised.value.stage == stage
    assert page.text_page.closed is True
    assert page.closed is True
    assert document.closed is True
    assert reader.closed is True
