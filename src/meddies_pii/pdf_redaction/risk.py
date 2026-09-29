"""pypdf exposes operators, not computed opacity and visibility.

MuPDF trace supplies the rendered-state signal without returning text in the result.

"""

from __future__ import annotations

# ruff: file-ignore[invalid-function-name]
# reason: the PyMuPDF module protocol must spell its upstream `Rect` constructor exactly for structural typing.
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast

import pymupdf
from pypdf import PdfReader
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    IndirectObject,
    NullObject,
)
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from meddies_pii.pdf_redaction.contracts import DocumentRisk

PdfRiskEvidenceCode = Literal[
    "encrypted_document",
    "parse_failure",
    "no_pages",
    "render_inspection_failure",
    "unsupported_active_content",
    "acroform",
    "widget",
    "embedded_files",
    "javascript",
    "action",
    "xfa",
    "annotation",
    "uri_action",
    "info_metadata",
    "xmp_metadata",
    "invisible_text",
    "optional_content",
]

INVISIBLE_TEXT_SPAN_TYPE = 3
MAX_UNICODE_CODEPOINT = 0x10FFFF

_EVIDENCE_ORDER: tuple[PdfRiskEvidenceCode, ...] = (
    "encrypted_document",
    "parse_failure",
    "no_pages",
    "render_inspection_failure",
    "unsupported_active_content",
    "acroform",
    "widget",
    "embedded_files",
    "javascript",
    "action",
    "xfa",
    "annotation",
    "uri_action",
    "info_metadata",
    "xmp_metadata",
    "invisible_text",
    "optional_content",
)
_FLATTENABLE_ANNOTATIONS = frozenset({
    "/Caret",
    "/Circle",
    "/FileAttachment",
    "/FreeText",
    "/Highlight",
    "/Ink",
    "/Line",
    "/Link",
    "/PolyLine",
    "/Polygon",
    "/Popup",
    "/PrinterMark",
    "/Redact",
    "/Square",
    "/Squiggly",
    "/Stamp",
    "/StrikeOut",
    "/Text",
    "/TrapNet",
    "/Underline",
    "/Watermark",
    "/Widget",
})


class _PymupdfRect(Protocol):
    is_empty: bool
    is_infinite: bool

    def intersects(self, other: _PymupdfRect) -> bool: ...


class _PymupdfPage(Protocol):
    cropbox: _PymupdfRect

    def get_texttrace(self) -> object: ...


class _PymupdfDocument(Protocol):
    needs_pass: bool

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def __iter__(self) -> Iterator[_PymupdfPage]: ...


class _PymupdfModule(Protocol):
    def open(self, path: Path) -> _PymupdfDocument: ...

    def Rect(self, bbox: tuple[float, float, float, float]) -> _PymupdfRect: ...


_PYMUPDF = cast("_PymupdfModule", pymupdf)


class _TraceSpan(TypedDict, total=False):
    type: int
    opacity: float
    bbox: tuple[float, float, float, float]
    chars: Sequence[tuple[object, ...]]


@dataclass(frozen=True, slots=True)
class PdfRiskResult:
    risk: DocumentRisk
    evidence_codes: tuple[PdfRiskEvidenceCode, ...]


def classify_pdf_risk(path: Path) -> PdfRiskResult:
    try:
        reader = PdfReader(path, strict=True)
        if reader.is_encrypted:
            return _result("unsupported", {"encrypted_document"})
        pages = tuple(reader.pages)
        if not pages:
            return _result("unsupported", {"no_pages"})
        dynamic, unsupported = _inspect_pdf_objects(reader, pages)
    # reason: risk classification fails closed for any parser failure rather than leaking an unknown PDF into a safe route.
    except Exception:  # ruff: ignore[blind-except]
        return _result("unsupported", {"parse_failure"})

    try:
        if _contains_invisible_text(path):
            dynamic.add("invisible_text")
    # reason: render inspection also fails closed because an unknown transparency state cannot be treated as safe.
    except Exception:  # ruff: ignore[blind-except]
        unsupported.add("render_inspection_failure")

    if unsupported:
        return _result("unsupported", dynamic | unsupported)
    if dynamic:
        return _result("dynamic_renderable", dynamic)
    return _result("static", set())


# reason: inspect pdf orders resolved before action slots; helper seams would mix coordinate frames.
def _inspect_pdf_objects(  # ruff: ignore[complex-structure,too-many-branches]
    reader: PdfReader,
    pages: Sequence[DictionaryObject],
) -> tuple[set[PdfRiskEvidenceCode], set[PdfRiskEvidenceCode]]:
    dynamic: set[PdfRiskEvidenceCode] = set()
    unsupported: set[PdfRiskEvidenceCode] = set()
    root = reader.root_object

    info = _resolved_dictionary(reader.trailer.get("/Info"))
    if info:
        dynamic.add("info_metadata")
    if _has_value(root, "/Metadata"):
        dynamic.add("xmp_metadata")
    if _has_value(root, "/AF"):
        dynamic.add("embedded_files")
    if _has_value(root, "/OCProperties"):
        dynamic.add("optional_content")
    if _has_value(root, "/Collection") or _has_value(root, "/Requirements"):
        unsupported.add("unsupported_active_content")

    names = _resolved_dictionary(root.get("/Names"))
    if _has_value(root, "/Names") and names is None:
        unsupported.add("unsupported_active_content")
    elif names is not None:
        if _has_value(names, "/EmbeddedFiles"):
            dynamic.add("embedded_files")
        if _has_value(names, "/JavaScript"):
            dynamic.update(("action", "javascript"))

    form = _resolved_dictionary(root.get("/AcroForm"))
    if _has_value(root, "/AcroForm"):
        dynamic.add("acroform")
    if _has_value(root, "/AcroForm") and form is None:
        unsupported.add("unsupported_active_content")
    elif form is not None:
        if _has_value(form, "/XFA"):
            dynamic.add("xfa")
        _inspect_action_slots(form, dynamic)
        _inspect_form_fields(form.get("/Fields"), dynamic)

    _inspect_action_slots(root, dynamic)
    _inspect_outline_actions(root.get("/Outlines"), dynamic)
    for page in pages:
        _inspect_page(page, dynamic, unsupported)
    return dynamic, unsupported


def _inspect_page(
    page: DictionaryObject,
    dynamic: set[PdfRiskEvidenceCode],
    unsupported: set[PdfRiskEvidenceCode],
) -> None:
    if _has_value(page, "/Metadata"):
        dynamic.add("xmp_metadata")
    if _has_value(page, "/AF"):
        dynamic.add("embedded_files")
    _inspect_action_slots(page, dynamic)

    annotations_value = _resolve(page.get("/Annots"))
    if _has_value(page, "/Annots") and not isinstance(
        annotations_value,
        ArrayObject,
    ):
        unsupported.add("unsupported_active_content")
    annotations = tuple(annotations_value) if isinstance(annotations_value, ArrayObject) else ()
    for annotation_value in annotations:
        annotation = _resolved_dictionary(annotation_value)
        if annotation is None:
            unsupported.add("unsupported_active_content")
            continue
        dynamic.add("annotation")
        subtype = str(annotation.get("/Subtype", ""))
        if subtype == "/Widget":
            dynamic.add("widget")
        elif subtype == "/FileAttachment":
            dynamic.add("embedded_files")
        elif subtype not in _FLATTENABLE_ANNOTATIONS:
            unsupported.add("unsupported_active_content")
        _inspect_action_slots(annotation, dynamic)


def _inspect_action_slots(
    container: DictionaryObject,
    evidence: set[PdfRiskEvidenceCode],
) -> None:
    for key in ("/A", "/AA", "/OpenAction"):
        if not _has_value(container, key):
            continue
        evidence.add("action")
        value = container.get(key)
        if key == "/AA":
            additional_actions = _resolved_dictionary(value)
            if additional_actions is not None:
                for action in additional_actions.values():
                    _inspect_action_value(action, evidence)
                continue
        _inspect_action_value(value, evidence)


def _inspect_action_value(
    value: object,
    evidence: set[PdfRiskEvidenceCode],
) -> None:
    resolved = _resolve(value)
    if isinstance(resolved, DictionaryObject):
        action_type = str(resolved.get("/S", ""))
        if action_type == "/JavaScript" or _has_value(resolved, "/JS"):
            evidence.add("javascript")
        if action_type == "/URI" or _has_value(resolved, "/URI"):
            evidence.add("uri_action")
        if _has_value(resolved, "/Next"):
            _inspect_action_value(resolved.get("/Next"), evidence)
    elif isinstance(resolved, ArrayObject):
        for child in resolved:
            _inspect_action_value(child, evidence)


def _inspect_form_fields(
    value: object,
    evidence: set[PdfRiskEvidenceCode],
) -> None:
    stack = list(_resolved_array(value))
    seen: set[int] = set()
    while stack:
        field = _resolved_dictionary(stack.pop())
        if field is None or id(field) in seen:
            continue
        seen.add(id(field))
        _inspect_action_slots(field, evidence)
        stack.extend(_resolved_array(field.get("/Kids")))


def _inspect_outline_actions(
    value: object,
    evidence: set[PdfRiskEvidenceCode],
) -> None:
    root = _resolved_dictionary(value)
    if root is None:
        return
    stack = [root]
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        _inspect_action_slots(item, evidence)
        for key in ("/First", "/Next"):
            child = _resolved_dictionary(item.get(key))
            if child is not None:
                stack.append(child)


def _resolved_array(value: object) -> Iterator[object]:
    resolved = _resolve(value)
    if isinstance(resolved, ArrayObject):
        yield from resolved


def _resolved_dictionary(value: object) -> DictionaryObject | None:
    resolved = _resolve(value)
    return resolved if isinstance(resolved, DictionaryObject) else None


def _resolve(value: object) -> object:
    return value.get_object() if isinstance(value, IndirectObject) else value


def _has_value(container: DictionaryObject, key: str) -> bool:
    if key not in container:
        return False
    return not isinstance(_resolve(container.get(key)), NullObject)


def _contains_invisible_text(path: Path) -> bool:
    with _PYMUPDF.open(path) as document:
        if document.needs_pass:
            msg = "encrypted document cannot be inspected"
            raise ValueError(msg)
        for page in document:
            visible_page = page.cropbox
            spans = cast("Sequence[_TraceSpan]", page.get_texttrace())
            for span in spans:
                if not _trace_has_non_whitespace(span):
                    continue
                if span.get("type") == INVISIBLE_TEXT_SPAN_TYPE or span.get("opacity", 1.0) <= 0.0:
                    return True
                bbox = span.get("bbox")
                if bbox is not None:
                    rectangle = _PYMUPDF.Rect(bbox)
                    if rectangle.is_empty or rectangle.is_infinite or not rectangle.intersects(visible_page):
                        return True
    return False


def _trace_has_non_whitespace(span: _TraceSpan) -> bool:
    for character in span.get("chars", ()):
        codepoint = character[0]
        if isinstance(codepoint, int) and 0 <= codepoint <= MAX_UNICODE_CODEPOINT and not chr(codepoint).isspace():
            return True
    return False


def _result(
    risk: DocumentRisk,
    evidence: set[PdfRiskEvidenceCode],
) -> PdfRiskResult:
    return PdfRiskResult(
        risk=risk,
        evidence_codes=tuple(code for code in _EVIDENCE_ORDER if code in evidence),
    )
