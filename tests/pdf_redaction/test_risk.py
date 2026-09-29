from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    RectangleObject,
)

from meddies_pii.pdf_redaction.benchmark.corpus import (
    CorpusFixture,
    generate_challenge_corpus,
)
from meddies_pii.pdf_redaction.risk import classify_pdf_risk

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(scope="module")
def challenge_fixture() -> CorpusFixture:
    return generate_challenge_corpus()


def _write(writer: PdfWriter, path: Path) -> None:
    with path.open("wb") as stream:
        writer.write(stream)


def _static_writer() -> PdfWriter:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.metadata = None
    return writer


def _corpus_page_paths(
    fixture: CorpusFixture,
    directory: Path,
) -> Iterator[Path]:
    reader = PdfReader(BytesIO(fixture.pdf_bytes), strict=True)
    for page_index in range(9):
        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])
        writer.metadata = None
        path = directory / f"page-{page_index + 1}.pdf"
        _write(writer, path)
        yield path


def test_static_pdf_has_no_risk_evidence(tmp_path: Path) -> None:
    path = tmp_path / "static.pdf"
    _write(_static_writer(), path)

    result = classify_pdf_risk(path)

    assert result.risk == "static"
    assert result.evidence_codes == ()


def test_challenge_pages_do_not_treat_raster_content_as_dynamic(
    tmp_path: Path,
    challenge_fixture: CorpusFixture,
) -> None:
    results = tuple(classify_pdf_risk(path) for path in _corpus_page_paths(challenge_fixture, tmp_path))

    assert tuple(result.risk for result in results) == (
        "static",
        "static",
        "static",
        "static",
        "dynamic_renderable",
        "static",
        "static",
        "static",
        "static",
    )
    assert results[4].evidence_codes == ("invisible_text",)
    assert all(results[index].risk == "static" for index in (5, 6, 7, 8))


def test_full_challenge_corpus_reports_semantic_object_risks_without_values(
    tmp_path: Path,
    challenge_fixture: CorpusFixture,
) -> None:
    path = tmp_path / "challenge.pdf"
    path.write_bytes(challenge_fixture.pdf_bytes)

    result = classify_pdf_risk(path)

    assert result.risk == "dynamic_renderable"
    assert set(result.evidence_codes) >= {
        "acroform",
        "annotation",
        "embedded_files",
        "info_metadata",
        "invisible_text",
        "uri_action",
        "widget",
        "xmp_metadata",
    }
    assert not any(canary.raw_value in repr(result) for canary in challenge_fixture.gold.canaries)


def test_javascript_action_is_dynamic_and_never_returned(tmp_path: Path) -> None:
    canary = "SYNTH-JAVASCRIPT-SECRET-771"
    writer = _static_writer()
    writer.add_js(f'app.alert("{canary}")')
    path = tmp_path / "javascript.pdf"
    _write(writer, path)

    result = classify_pdf_risk(path)

    assert result.risk == "dynamic_renderable"
    assert set(result.evidence_codes) >= {"action", "javascript"}
    assert canary not in repr(result)


def test_xfa_is_dynamic_without_reading_its_raw_payload(tmp_path: Path) -> None:
    canary = "SYNTH-XFA-SECRET-882"
    writer = _static_writer()
    xfa = DecodedStreamObject()
    xfa.set_data(f"<xfa>{canary}</xfa>".encode())
    form = DictionaryObject({
        NameObject("/Fields"): ArrayObject(),
        NameObject("/XFA"): writer._add_object(xfa),
    })
    writer.root_object[NameObject("/AcroForm")] = writer._add_object(form)
    path = tmp_path / "xfa.pdf"
    _write(writer, path)

    result = classify_pdf_risk(path)

    assert result.risk == "dynamic_renderable"
    assert set(result.evidence_codes) >= {"acroform", "xfa"}
    assert canary not in repr(result)


def test_attachment_uri_and_info_metadata_are_dynamic(tmp_path: Path) -> None:
    canary = "SYNTH-OBJECT-SECRET-993"
    writer = _static_writer()
    writer.add_attachment("synthetic.txt", canary.encode())
    writer.add_uri(
        page_number=0,
        uri=f"https://example.invalid/{canary}",
        rect=RectangleObject((72, 72, 180, 96)),
    )
    writer.add_metadata({"/Subject": canary})
    path = tmp_path / "objects.pdf"
    _write(writer, path)

    result = classify_pdf_risk(path)

    assert result.risk == "dynamic_renderable"
    assert set(result.evidence_codes) >= {
        "action",
        "annotation",
        "embedded_files",
        "info_metadata",
        "uri_action",
    }
    assert canary not in repr(result)


def test_encrypted_pdf_is_unsupported_without_attempting_decryption(
    tmp_path: Path,
) -> None:
    writer = _static_writer()
    writer.encrypt("synthetic-password")
    path = tmp_path / "encrypted.pdf"
    _write(writer, path)

    result = classify_pdf_risk(path)

    assert result.risk == "unsupported"
    assert result.evidence_codes == ("encrypted_document",)


def test_parse_failure_is_unsupported_and_does_not_leak_input(tmp_path: Path) -> None:
    raw = b"not-a-pdf SYNTH-PARSE-SECRET-441"
    path = tmp_path / "malformed.pdf"
    path.write_bytes(raw)

    result = classify_pdf_risk(path)

    assert result.risk == "unsupported"
    assert result.evidence_codes == ("parse_failure",)
    assert raw.decode() not in repr(result)


def test_ambiguous_multimedia_annotation_fails_closed(tmp_path: Path) -> None:
    writer = _static_writer()
    annotation = DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/RichMedia"),
        NameObject("/Rect"): ArrayObject([FloatObject(72), FloatObject(72), FloatObject(180), FloatObject(120)]),
    })
    writer.pages[0][NameObject("/Annots")] = ArrayObject([writer._add_object(annotation)])
    path = tmp_path / "rich-media.pdf"
    _write(writer, path)

    result = classify_pdf_risk(path)

    assert result.risk == "unsupported"
    assert "unsupported_active_content" in result.evidence_codes
