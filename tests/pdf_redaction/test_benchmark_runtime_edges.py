from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast, override

import pytest

from meddies_pii.bioes_inference import SpanDetection
from meddies_pii.bioes_inference.detector import BioesSpanDetector
from meddies_pii.pdf_redaction.benchmark.runtime import (
    ModelCandidateGateError,
    ModelGate,
    detection_signature,
    extract_ocr_pages,
    manifest_bytes,
    manifest_sha256,
    measure_model_gate,
    measure_ocr_candidate,
    merge_prepared_pages,
    prepare_pages,
    select_ocr_model_files,
)
from meddies_pii.pdf_redaction.benchmark.spatial import OracleRegion
from meddies_pii.pdf_redaction.contracts import (
    GeometryPage,
    PageRegion,
    Point,
    Quad,
)
from meddies_pii.pdf_redaction.document import (
    DocumentInspection,
    PageInspection,
    RasterArtifact,
)
from meddies_pii.pdf_redaction.ocr import RasterPage
from meddies_pii.pdf_redaction.routing import PageRouteDecision, PageRouteSignals
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.bioes_inference.detector import BioesInferenceBackend
    from meddies_pii.pdf_redaction.document_types import PdfSource
    from meddies_pii.pdf_redaction.extraction import NativeGeometryPrecision


def _quad() -> Quad:
    return Quad(points=(Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)))


def _page(index: int, text: str = "canary") -> GeometryPage:
    return GeometryPage(index, 100, 100, text, ())


def _artifact(index: int) -> RasterArtifact:
    return RasterArtifact(
        raster=RasterPage(index, object(), 10, 10, 100, 100),
        png_bytes=b"tiny",
        sha256="a" * 64,
        dpi=72,
    )


class _Ocr:
    def __init__(self, pages: dict[int, GeometryPage]) -> None:
        self.pages = pages
        self.calls: list[int] = []

    def extract(self, raster: RasterPage) -> GeometryPage:
        self.calls.append(raster.page_index)
        return self.pages[raster.page_index]


def test_ocr_measurement_scores_visible_targets_and_preserves_page_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meddies_pii.pdf_redaction.benchmark import runtime

    adapter = _Ocr({0: _page(0, "found"), 5: _page(5, "other")})
    oracle = (
        OracleRegion(PageRegion(0, _quad(), "human_name", 0, 1), 0, "visible_raster"),
        OracleRegion(PageRegion(5, _quad(), "phone_number", 0, 1), 1, "visible_raster"),
        OracleRegion(PageRegion(0, _quad(), "email_address", 0, 1), 0, "visible_native"),
    )
    monkeypatch.setattr(
        runtime,
        "discover_ocr_model_manifest",
        lambda _: ({"role": "recognizer", "bytes": 7, "sha256": "b" * 64},),
    )

    measured = measure_ocr_candidate(
        candidate_id="rapidocr-ppocrv6-small-openvino",
        adapter=adapter,
        artifacts=(_artifact(0), _artifact(5)),
        oracle_regions=oracle,
        forbidden_values=("found", "missing"),
    )

    assert tuple(page.page_index for page in measured.pages) == (0, 5)
    assert measured.exact_visible_recall == 0.5
    assert measured.clean_page_misses == 1
    assert measured.model_manifest[0]["bytes"] == 7
    assert adapter.calls == [0, 5] * 4


def test_ocr_page_index_change_is_rejected() -> None:
    adapter = _Ocr({0: _page(1)})

    with pytest.raises(RuntimeError, match="page indices"):
        extract_ocr_pages(adapter, (_artifact(0),))


def test_merge_and_prepare_pages_route_only_ocr_pages() -> None:
    native_pages = (_page(0, "native"), _page(1, "native"))
    ocr_pages = (_page(1, "ocr"),)
    assert tuple(page.text for page in merge_prepared_pages(native_pages, ocr_pages)) == (
        "native",
        "ocr",
    )

    trusted = PageRouteSignals(
        page_index=0,
        visible_native_characters=3,
        has_meaningful_raster=False,
        native_text_trusted=True,
    )
    image = PageRouteSignals(
        page_index=1,
        visible_native_characters=0,
        has_meaningful_raster=True,
        native_text_trusted=True,
    )
    inspection = DocumentInspection(
        pages=(
            PageInspection(
                trusted,
                PageRouteDecision(
                    page_index=0,
                    content_class="native_text",
                    content_route="native_trusted",
                    requires_full_page_ocr=False,
                    diagnostics=(),
                ),
                0.0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            PageInspection(
                image,
                PageRouteDecision(
                    page_index=1,
                    content_class="image_only",
                    content_route="image_only",
                    requires_full_page_ocr=True,
                    diagnostics=(),
                ),
                1.0,
                1,
                0,
                0,
                0,
                0,
                0,
            ),
        ),
        attachment_count=0,
    )

    class Document:
        # reason: this mirrors the `DocumentAdapter` protocol, whose `inspect` the code under test calls on the
        # reason: adapter instance it is handed, so the bound-method form is the API being stood in for.
        def inspect(self, source: PdfSource) -> DocumentInspection:  # ruff: ignore[no-self-use]
            del source
            return inspection

        # reason: this mirrors the `DocumentAdapter` protocol, whose `render_pages` the code under test calls on
        # reason: the adapter instance it is handed, so the bound-method form is the API being stood in for.
        def render_pages(  # ruff: ignore[no-self-use]
            self,
            source: PdfSource,
            *,
            page_indices: Sequence[int],
        ) -> tuple[RasterArtifact, ...]:
            del source
            assert tuple(page_indices) == (1,)
            return (_artifact(1),)

    class Extractor:
        geometry_precision: NativeGeometryPrecision = "character_quad"

        # reason: this mirrors the `NativeExtractor` protocol, whose `extract` the code under test calls on the
        # reason: extractor instance it is handed, so the bound-method form is the API being stood in for.
        def extract(self, source: PdfSource) -> tuple[GeometryPage, ...]:  # ruff: ignore[no-self-use]
            del source
            return native_pages

    prepared = prepare_pages(
        Path("synthetic.pdf"),
        Document(),
        Extractor(),
        _Ocr({1: _page(1, "ocr")}),
    )
    assert tuple(page.text for page in prepared) == ("native", "ocr")


@pytest.mark.parametrize(
    ("native_pages", "ocr_pages", "message"),
    [
        ((_page(0),), (_page(0), _page(0)), "duplicate indices"),
        ((_page(1),), (), "not contiguous"),
    ],
)
def test_merge_prepared_pages_rejects_ambiguous_or_noncontiguous_pages(
    native_pages: tuple[GeometryPage, ...],
    ocr_pages: tuple[GeometryPage, ...],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        merge_prepared_pages(native_pages, ocr_pages)


class _Backend:
    def __init__(self, name: str, threads: int | None = None) -> None:
        self.name = name
        self.threads = threads

    # reason: `_Detector` below answers from `outcomes` and never runs inference, so the gate reaching
    # reason: either of these means the fake stopped standing in for the real detector.
    def load(self, model_path: Path) -> None:
        msg = f"the fake detector never loads backend {self.name}: {model_path}"
        raise AssertionError(msg)

    def infer(self, input_ids: list[int], attention_mask: list[int], *, bucket: int) -> object:
        msg = f"the fake detector never infers on {self.name}: {len(input_ids)}/{len(attention_mask)} at {bucket}"
        raise AssertionError(msg)


class _Detector(BioesSpanDetector):
    outcomes: ClassVar[dict[str, tuple[SpanDetection, ...] | Exception]] = {}

    # reason: the real __init__ verifies model bytes and a tokenizer manifest this fake has none of,
    # reason: so it deliberately does not delegate; every method the gate calls is overridden below.
    def __init__(self, *, backend: BioesInferenceBackend, **_: object) -> None:
        self.backend = backend
        self.loaded = False

    @override
    def load(self) -> None:
        self.loaded = True

    @override
    def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
        del texts
        outcome = self.outcomes[self.backend.name]
        if isinstance(outcome, Exception):
            raise outcome
        return list(outcome)


def _detection(*, truncated: bool = False, label: str = "human_name") -> SpanDetection:
    return SpanDetection(
        spans=(CharSpan(0, 1, "x", label),),
        bucket=1,
        num_tokens=1,
        truncated=truncated,
    )


def _measure_gate() -> ModelGate:
    return measure_model_gate(
        model_path=Path("model.onnx"),
        tokenizer_root=Path("tokenizer"),
        expected_model_size_bytes=1,
        expected_sha256="a" * 64,
        expected_tokenizer_files={},
        pages=(_page(0, "x"),),
        openvino_backends=(_Backend("openvino", 6), _Backend("openvino-fast", 8)),
        ort_backend=_Backend("onnxruntime", 6),
        detector_class=_Detector,
    )


def test_model_gate_requires_exact_consensus_and_publishes_safe_measurements() -> None:
    same = (_detection(),)
    _Detector.outcomes = {"openvino": same, "openvino-fast": same, "onnxruntime": same}

    gate = _measure_gate()

    assert gate.selected_runtime_id in {"openvino-t6", "openvino-fast-t8"}
    assert len(gate.component_rows) == 3
    assert all(row["status"] == "measured" for row in gate.component_rows)
    assert all("detected_spans" in row and "error_digest" not in row for row in gate.component_rows)


@pytest.mark.parametrize(
    "outcomes",
    [
        {
            "openvino": (_detection(),),
            "openvino-fast": (_detection(label="phone_number"),),
            "onnxruntime": (_detection(),),
        },
        {
            "openvino": (_detection(),),
            "openvino-fast": RuntimeError("synthetic canary"),
            "onnxruntime": (_detection(),),
        },
        {
            "openvino": (_detection(truncated=True),),
            "openvino-fast": (_detection(),),
            "onnxruntime": (_detection(),),
        },
    ],
)
def test_model_gate_failure_rows_are_structured_and_do_not_expose_error_text(
    outcomes: dict[str, tuple[SpanDetection, ...] | Exception],
) -> None:
    _Detector.outcomes = outcomes

    with pytest.raises(ModelCandidateGateError) as raised:
        _measure_gate()

    rows = raised.value.candidate_rows
    assert rows
    assert all("candidate_id" in row or "runtime_id" in row for row in rows)
    failed = [row for row in rows if row["status"] == "failed"]
    assert all(len(cast("str", row["error_digest"])) == 64 for row in failed)
    assert all("error_message" not in row for row in rows)


def test_model_gate_rejects_empty_pages_or_missing_openvino_backend() -> None:
    _Detector.outcomes = {"onnxruntime": (_detection(),)}
    with pytest.raises(RuntimeError, match="empty"):
        measure_model_gate(
            model_path=Path("model.onnx"),
            tokenizer_root=Path("tokenizer"),
            expected_model_size_bytes=1,
            expected_sha256="a" * 64,
            expected_tokenizer_files={},
            pages=(_page(0, ""),),
            openvino_backends=(),
            ort_backend=_Backend("onnxruntime"),
            detector_class=_Detector,
        )
    with pytest.raises(ValueError, match="OpenVINO"):
        measure_model_gate(
            model_path=Path("model.onnx"),
            tokenizer_root=Path("tokenizer"),
            expected_model_size_bytes=1,
            expected_sha256="a" * 64,
            expected_tokenizer_files={},
            pages=(_page(0, "x"),),
            openvino_backends=(),
            ort_backend=_Backend("onnxruntime"),
            detector_class=_Detector,
        )


def test_manifest_contracts_are_deterministic_and_validate_integer_bytes(
    tmp_path: Path,
) -> None:
    manifest = ({"role": "detector", "bytes": 3, "sha256": "a" * 64},)
    assert manifest_bytes(manifest) == 3
    assert manifest_sha256(manifest) == manifest_sha256(manifest)
    with pytest.raises(TypeError, match="integers"):
        manifest_bytes(({"bytes": True},))

    model = tmp_path / "detector.onnx"
    model.write_bytes(b"tiny")
    selected = select_ocr_model_files(
        "tesseract-fast-eng-vie",
        ((tmp_path / "eng.traineddata", 1), (tmp_path / "vie.traineddata", 1)),
    )
    assert tuple(role for role, _ in selected) == ("recognizer_eng", "recognizer_vie")
    with pytest.raises(ValueError, match="unknown"):
        select_ocr_model_files("unknown", ((model, 4),))


def test_detection_signature_omits_sensitive_span_text() -> None:
    signature = detection_signature((_detection(),))
    assert signature == (((0, 1, "human_name"),),)
