from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pypdf import PdfWriter

import meddies_pii.pdf_redaction.writer_raster as raster_module
from meddies_pii.pdf_redaction import writer_common
from meddies_pii.pdf_redaction.contracts import PageRegion, Point, Quad
from meddies_pii.pdf_redaction.errors import (
    InputDocumentError,
    OutputPathError,
    RedactionApplyError,
)
from meddies_pii.pdf_redaction.writers import (
    PyMuPdfRedactionWriter,
    RasterRebuildWriter,
)


def _write_pdf(path: Path, *, rotation: int = 0) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    if rotation:
        page.rotate(rotation)
    writer.write(path)


def _region(
    *,
    page_index: int = 0,
    x0: float = 10,
    y0: float = 10,
    x1: float = 20,
    y1: float = 20,
) -> PageRegion:
    return PageRegion(
        page_index=page_index,
        quad=Quad(
            points=(
                Point(x0, y0),
                Point(x1, y0),
                Point(x1, y1),
                Point(x0, y1),
            ),
        ),
        label="secret",
        span_start=0,
        span_end=1,
    )


def test_writer_rejects_missing_input_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "output.pdf"

    with pytest.raises(InputDocumentError) as raised:
        PyMuPdfRedactionWriter().write(
            tmp_path / "missing.pdf",
            (_region(),),
            output,
        )

    assert raised.value.code == "input_not_file"
    assert not output.exists()


def test_writer_rejects_in_place_and_missing_parent_destinations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    _write_pdf(source)

    with pytest.raises(OutputPathError) as in_place:
        PyMuPdfRedactionWriter().write(source, (_region(),), source)
    with pytest.raises(OutputPathError) as missing_parent:
        PyMuPdfRedactionWriter().write(
            source,
            (_region(),),
            tmp_path / "missing" / "output.pdf",
        )

    assert in_place.value.code == "in_place_output"
    assert missing_parent.value.code == "output_parent_missing"


@pytest.mark.parametrize(
    ("region", "code"),
    [
        (_region(page_index=1), "page_out_of_range"),
        (_region(x0=-1), "region_outside_page"),
    ],
)
def test_writer_rejects_regions_outside_document_geometry(
    tmp_path: Path,
    region: PageRegion,
    code: str,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source)

    with pytest.raises(InputDocumentError) as raised:
        PyMuPdfRedactionWriter().write(source, (region,), output)

    assert raised.value.code == code
    assert not output.exists()
    assert not tuple(tmp_path.glob(".meddies-pdf-*.working"))


def test_writer_rejects_corrupted_nonfinite_region_at_runtime(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source)
    region = _region()
    # reason: `Point` is `@dataclass(frozen=True, slots=True)` (`contracts.py:19-22`), so `setattr` raises
    # reason: `FrozenInstanceError`. Bypassing the freeze to plant a NaN IS the condition under test.
    object.__setattr__(region.quad.points[0], "x", float("nan"))  # ruff: ignore[unnecessary-dunder-call]

    with pytest.raises(InputDocumentError) as raised:
        PyMuPdfRedactionWriter().write(source, (region,), output)

    assert raised.value.code == "invalid_region"
    assert not output.exists()


@pytest.mark.parametrize("rotation", [180, 270])
def test_raster_writer_redacts_pages_with_inverse_rotation_geometry(
    tmp_path: Path,
    rotation: int,
) -> None:
    source = tmp_path / f"source-{rotation}.pdf"
    output = tmp_path / f"output-{rotation}.pdf"
    _write_pdf(source, rotation=rotation)

    result = RasterRebuildWriter(dpi=72).write(source, (_region(),), output)

    assert result.output_path == output
    assert output.is_file()


def test_raster_writer_rejects_native_rotation_outside_pdf_quadrants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source)

    class _Bitmap:
        # reason: this mirrors `pypdfium2`'s `PdfBitmap.to_pil`, which the code under test calls on the bitmap
        # reason: instance returned by `render`, so the bound-method form is the API being stood in for.
        def to_pil(self) -> Image.Image:  # ruff: ignore[no-self-use]
            return Image.new("RGB", (100, 100), "white")

    class _Page:
        # reason: this mirrors `pypdfium2`'s `PdfPage.get_size`, which the code under test calls on the page
        # reason: instance, so the bound-method form is the API being stood in for.
        def get_size(self) -> tuple[float, float]:  # ruff: ignore[no-self-use]
            return 100.0, 100.0

        # reason: this mirrors `pypdfium2`'s `PdfPage.get_rotation`, which the code under test calls on the page
        # reason: instance, so the bound-method form is the API being stood in for.
        def get_rotation(self) -> int:  # ruff: ignore[no-self-use]
            return 45

        # reason: this mirrors `pypdfium2`'s `PdfPage.render`, which the code under test calls on the page
        # reason: instance, so the bound-method form is the API being stood in for.
        def render(self, *, scale: float) -> _Bitmap:  # ruff: ignore[no-self-use]
            assert scale == 1.0
            return _Bitmap()

    class _Document:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> _Page:
            assert index == 0
            return _Page()

        def close(self) -> None:
            pass

    pdfium = SimpleNamespace(
        PdfDocument=lambda _: _Document(),
        PdfiumError=RuntimeError,
    )
    real_import = raster_module.importlib.import_module

    def import_module(name: str) -> object:
        if name == "pypdfium2":
            return pdfium
        return real_import(name)

    monkeypatch.setattr(raster_module.importlib, "import_module", import_module)

    with pytest.raises(InputDocumentError) as raised:
        RasterRebuildWriter(dpi=72).write(source, (_region(),), output)

    assert raised.value.code == "unsupported_page_rotation"
    assert not output.exists()


def test_writer_reports_private_staging_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    _write_pdf(source)
    real_mkdir = Path.mkdir

    # reason: mirrors `Path.mkdir` so the spy forwards its real contract; `parents` and `exist_ok`
    # reason: are keyword-only because every `.mkdir(` call in this tree passes them by keyword.
    def fail_private_workspace(
        path: Path,
        mode: int = 0o777,
        *,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if path.name.startswith(".meddies-pdf-"):
            msg = "synthetic denial"
            raise PermissionError(msg)
        real_mkdir(path, mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", fail_private_workspace)

    with pytest.raises(OutputPathError) as raised:
        PyMuPdfRedactionWriter().write(source, (_region(),), tmp_path / "output.pdf")

    assert raised.value.code == "output_staging_failed"
    assert raised.value.cause_type == "PermissionError"


def test_writer_reports_source_open_failure_without_leaking_path_contents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "private-source.pdf"
    _write_pdf(source)
    real_open = writer_common.os.open

    def fail_source_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is None and Path(os.fsdecode(path)) == source:
            msg = "synthetic denial"
            raise PermissionError(msg)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(writer_common.os, "open", fail_source_open)

    with pytest.raises(InputDocumentError) as raised:
        PyMuPdfRedactionWriter().write(source, (_region(),), tmp_path / "output.pdf")

    assert raised.value.code == "source_unreadable"
    assert raised.value.cause_type == "PermissionError"


def test_writer_rejects_source_that_changes_type_after_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    _write_pdf(source)
    real_fstat = writer_common.os.fstat
    first_fstat = True

    def report_fifo_once(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal first_fstat
        if first_fstat:
            first_fstat = False
            return SimpleNamespace(st_mode=stat.S_IFIFO, st_size=source.stat().st_size)
        return real_fstat(descriptor)

    monkeypatch.setattr(writer_common.os, "fstat", report_fifo_once)

    with pytest.raises(InputDocumentError) as raised:
        PyMuPdfRedactionWriter().write(source, (_region(),), tmp_path / "output.pdf")

    assert raised.value.code == "source_not_regular"
    assert not tuple(tmp_path.glob(".meddies-pdf-*.working"))


@pytest.mark.parametrize("change", ["truncated", "grown"])
def test_writer_rejects_source_identity_change_during_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
) -> None:
    source = tmp_path / "source.pdf"
    _write_pdf(source)
    real_read = writer_common.os.read
    reads = 0

    def changed_read(descriptor: int, size: int) -> bytes:
        nonlocal reads
        reads += 1
        if change == "truncated" and reads == 1:
            return b""
        payload = real_read(descriptor, size)
        if change == "grown" and reads == 2:
            return b"x"
        return payload

    monkeypatch.setattr(writer_common.os, "read", changed_read)

    with pytest.raises(InputDocumentError) as raised:
        PyMuPdfRedactionWriter().write(source, (_region(),), tmp_path / "output.pdf")

    assert raised.value.code == "source_identity_changed"
    assert not tuple(tmp_path.glob(".meddies-pdf-*.working"))


def test_writer_rejects_private_snapshot_write_without_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    _write_pdf(source)
    monkeypatch.setattr(writer_common.os, "write", lambda *_: 0)

    with pytest.raises(InputDocumentError) as raised:
        PyMuPdfRedactionWriter().write(source, (_region(),), tmp_path / "output.pdf")

    assert raised.value.code == "source_staging_failed"
    assert raised.value.cause_type == "OSError"


def test_writer_preserves_racing_destination_and_cleans_private_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "output.pdf"
    _write_pdf(source)

    def race_destination(_: Path, output: Path) -> None:
        output.write_bytes(b"other-process")
        raise FileExistsError(output)

    monkeypatch.setattr(writer_common.os, "link", race_destination)

    with pytest.raises(OutputPathError) as raised:
        PyMuPdfRedactionWriter().write(source, (_region(),), destination)

    assert raised.value.code == "output_exists"
    assert destination.read_bytes() == b"other-process"
    assert not tuple(tmp_path.glob(".meddies-pdf-*.working"))


def test_writer_surfaces_private_stage_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "output.pdf"
    _write_pdf(source)
    real_rmdir = Path.rmdir
    blocked_workspace: Path | None = None

    def fail_private_workspace_removal(path: Path) -> None:
        nonlocal blocked_workspace
        if path.name.startswith(".meddies-pdf-"):
            blocked_workspace = path
            msg = "synthetic denial"
            raise PermissionError(msg)
        real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", fail_private_workspace_removal)

    with pytest.raises(RedactionApplyError) as raised:
        PyMuPdfRedactionWriter().write(source, (_region(),), destination)

    assert raised.value.code == "output_staging_cleanup_failed"
    assert destination.is_file()
    assert blocked_workspace is not None
    real_rmdir(blocked_workspace)
