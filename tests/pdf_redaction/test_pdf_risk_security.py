from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, TextStringObject

from meddies_pii.pdf_redaction.risk import classify_pdf_risk

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_risk_classifier_fails_closed_for_a_parseable_pdf_without_pages(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no-pages.pdf"
    with path.open("wb") as stream:
        PdfWriter().write(stream)

    result = classify_pdf_risk(path)

    assert result.risk == "unsupported"
    assert result.evidence_codes == ("no_pages",)


def _write_static_pdf(path: Path) -> PdfWriter:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.metadata = None
    with path.open("wb") as stream:
        writer.write(stream)
    return writer


def test_risk_classifier_escalates_malformed_names_and_annotations(
    tmp_path: Path,
) -> None:
    names_path = tmp_path / "bad-names.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.metadata = None
    writer.root_object[NameObject("/Names")] = TextStringObject("not-a-dictionary")
    with names_path.open("wb") as stream:
        writer.write(stream)

    annotations_path = tmp_path / "bad-annotations.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.metadata = None
    writer.pages[0][NameObject("/Annots")] = TextStringObject("not-an-array")
    with annotations_path.open("wb") as stream:
        writer.write(stream)

    for path in (names_path, annotations_path):
        result = classify_pdf_risk(path)
        assert result.risk == "unsupported"
        assert "unsupported_active_content" in result.evidence_codes


def test_risk_classifier_collects_nested_javascript_and_uri_actions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested-actions.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.metadata = None
    javascript = DictionaryObject({
        NameObject("/S"): NameObject("/JavaScript"),
        NameObject("/JS"): TextStringObject("x"),
    })
    uri = DictionaryObject({
        NameObject("/S"): NameObject("/URI"),
        NameObject("/URI"): TextStringObject("https://example.invalid"),
        NameObject("/Next"): ArrayObject([writer._add_object(javascript)]),
    })
    writer.root_object[NameObject("/OpenAction")] = writer._add_object(uri)
    with path.open("wb") as stream:
        writer.write(stream)

    result = classify_pdf_risk(path)

    assert result.risk == "dynamic_renderable"
    assert {"action", "javascript", "uri_action"}.issubset(result.evidence_codes)


def test_risk_classifier_marks_render_inspection_failures_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "static.pdf"
    _write_static_pdf(path)

    def unavailable(_: Path) -> object:
        msg = "native renderer unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr("meddies_pii.pdf_redaction.risk.pymupdf.open", unavailable)

    result = classify_pdf_risk(path)

    assert result.risk == "unsupported"
    assert result.evidence_codes == ("render_inspection_failure",)


def test_risk_classifier_marks_optional_content_and_embedded_associations_dynamic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "semantic.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.metadata = None
    writer.root_object[NameObject("/AF")] = ArrayObject()
    writer.root_object[NameObject("/OCProperties")] = DictionaryObject()
    with path.open("wb") as stream:
        writer.write(stream)

    result = classify_pdf_risk(path)

    assert result.risk == "dynamic_renderable"
    assert {"embedded_files", "optional_content"}.issubset(result.evidence_codes)


def test_risk_classifier_escalates_active_root_and_page_structures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "active-structures.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.metadata = None
    writer.root_object[NameObject("/Collection")] = DictionaryObject()
    writer.root_object[NameObject("/AcroForm")] = TextStringObject("not-a-form")
    writer.pages[0][NameObject("/Metadata")] = TextStringObject("metadata")
    writer.pages[0][NameObject("/AF")] = ArrayObject()
    attachment = DictionaryObject({NameObject("/Subtype"): NameObject("/FileAttachment")})
    writer.pages[0][NameObject("/Annots")] = ArrayObject([writer._add_object(attachment)])
    with path.open("wb") as stream:
        writer.write(stream)

    result = classify_pdf_risk(path)

    assert result.risk == "unsupported"
    assert {
        "acroform",
        "embedded_files",
        "unsupported_active_content",
        "xmp_metadata",
    }.issubset(result.evidence_codes)


def test_risk_classifier_traverses_additional_action_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "additional-actions.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.metadata = None
    javascript = DictionaryObject({
        NameObject("/S"): NameObject("/JavaScript"),
        NameObject("/JS"): TextStringObject("x"),
    })
    additional = DictionaryObject({NameObject("/E"): writer._add_object(javascript)})
    writer.pages[0][NameObject("/AA")] = writer._add_object(additional)
    with path.open("wb") as stream:
        writer.write(stream)

    result = classify_pdf_risk(path)

    assert result.risk == "dynamic_renderable"
    assert {"action", "javascript"}.issubset(result.evidence_codes)


def test_risk_classifier_checks_native_trace_opacity_and_outside_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "static.pdf"
    _write_static_pdf(path)

    class Page:
        cropbox = (0.0, 0.0, 100.0, 100.0)

        # reason: this mirrors `pymupdf`'s `Page.get_texttrace`, which the code under test calls on a page
        # reason: instance, so the bound-method form is the API being stood in for.
        def get_texttrace(self) -> tuple[dict[str, object], ...]:  # ruff: ignore[no-self-use]
            return (
                {
                    "type": 0,
                    "opacity": 0.0,
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "chars": ((ord("x"),),),
                },
            )

    class Document:
        needs_pass = False

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self) -> object:
            return iter((Page(),))

    monkeypatch.setattr("meddies_pii.pdf_redaction.risk.pymupdf.open", lambda _: Document())

    result = classify_pdf_risk(path)

    assert result.risk == "dynamic_renderable"
    assert result.evidence_codes == ("invisible_text",)
