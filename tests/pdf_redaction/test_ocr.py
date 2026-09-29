from __future__ import annotations

# reason: this module never spawns anything. It imports subprocess to BUILD the `CompletedProcess`
# reason: and `TimeoutExpired` values the Tesseract runner doubles return, which is why it carries no
# reason: `subprocess-without-shell-equals-true` finding anywhere.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from meddies_pii.pdf_redaction.contracts import Point
from meddies_pii.pdf_redaction.ocr import (
    OcrAdapterError,
    PixelToPageTransform,
    RapidOcrAdapter,
    RapidOcrEngineFactory,
    RapidOcrModelTier,
    RapidOcrRuntimeConfig,
    RasterPage,
    SubprocessTesseractRunner,
    TesseractTsvAdapter,
)


@dataclass(frozen=True)
class _RapidResult:
    txts: tuple[str, ...]
    word_results: tuple[tuple[tuple[str, float, tuple[tuple[float, float], ...]], ...], ...]


# reason: every unused parameter in this module's OCR doubles is contract, not leftover. The test is
# reason: the predecessor's: does a production call site WRITE the name? `ocr_geometry.py:55-56` calls
# reason: the engine with `return_word_box=True, return_single_char_box=False` as keywords, and
# reason: `ocr_types.py:97-98` declares the protocol the doubles stand in for, so a double that
# reason: dropped or renamed one would stop being callable by the code under test.
class _FakeRapidEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.options: tuple[bool, bool] | None = None

    def __call__(
        self,
        image: object,  # ruff: ignore[unused-method-argument]
        *,
        return_word_box: bool,
        return_single_char_box: bool,
    ) -> _RapidResult:
        self.calls += 1
        self.options = (return_word_box, return_single_char_box)
        return _RapidResult(
            txts=("Maya Sato", "ID-42"),
            word_results=(
                (
                    ("Maya", 0.91, ((0, 0), (40, 0), (40, 10), (0, 10))),
                    ("Sato", 0.88, ((50, 0), (90, 0), (90, 10), (50, 10))),
                ),
                (("ID-42", 0.97, ((0, 20), (50, 20), (50, 30), (0, 30))),),
            ),
        )


def test_rapidocr_loads_lazily_and_preserves_line_text_offsets() -> None:
    engine = _FakeRapidEngine()
    factory_calls = 0

    def factory() -> _FakeRapidEngine:
        nonlocal factory_calls
        factory_calls += 1
        return engine

    adapter = RapidOcrAdapter(engine_factory=factory)
    raster = RasterPage(
        page_index=2,
        image=object(),
        width_px=100,
        height_px=200,
        width_pt=50.0,
        height_pt=100.0,
    )
    assert factory_calls == 0

    page = adapter.extract(raster)
    adapter.extract(raster)

    assert factory_calls == 1
    assert engine.calls == 2
    assert engine.options == (True, False)
    assert page.text == "Maya Sato\nID-42"
    assert tuple((token.start, token.end) for token in page.tokens) == (
        (0, 4),
        (5, 9),
        (10, 15),
    )
    assert page.tokens[0].quad.points == (
        Point(0.0, 0.0),
        Point(20.0, 0.0),
        Point(20.0, 5.0),
        Point(0.0, 5.0),
    )
    assert tuple(token.confidence for token in page.tokens) == (0.91, 0.88, 0.97)


def test_rapidocr_default_is_lazy_openvino_tiny_without_loading_real_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_params: list[dict[str, object]] = []
    engine = _FakeRapidEngine()
    module = SimpleNamespace(
        EngineType=SimpleNamespace(OPENVINO="openvino-enum", ONNXRUNTIME="ort-enum"),
        ModelType=SimpleNamespace(SMALL="small-enum", TINY="tiny-enum"),
        OCRVersion=SimpleNamespace(PPOCRV6="v6-enum"),
        RapidOCR=lambda *, params: received_params.append(params) or engine,
    )
    monkeypatch.setitem(sys.modules, "rapidocr", module)
    adapter = RapidOcrAdapter()
    raster = RasterPage(0, object(), 100, 200, 50.0, 100.0)

    assert received_params == []

    adapter.extract(raster)

    assert received_params == [
        {
            "Global.log_level": "critical",
            "Det.engine_type": "openvino-enum",
            "Cls.engine_type": "openvino-enum",
            "Rec.engine_type": "openvino-enum",
            "Det.model_type": "tiny-enum",
            "Rec.model_type": "tiny-enum",
            "Det.ocr_version": "v6-enum",
            "Rec.ocr_version": "v6-enum",
            "EngineConfig.openvino.inference_num_threads": 4,
            "EngineConfig.openvino.performance_hint": "LATENCY",
            "EngineConfig.openvino.num_streams": 1,
        },
    ]


def test_rapidocr_applies_inverse_rotation_at_the_adapter_boundary() -> None:
    raster = RasterPage(
        page_index=2,
        image=object(),
        width_px=100,
        height_px=50,
        width_pt=50.0,
        height_pt=100.0,
        pixel_to_page=PixelToPageTransform(0.0, -1.0, 1.0, 0.0, 0.0, 100.0),
    )

    page = RapidOcrAdapter(engine_factory=_FakeRapidEngine).extract(raster)

    assert page.tokens[0].quad.points == (
        Point(0.0, 100.0),
        Point(0.0, 60.0),
        Point(10.0, 60.0),
        Point(10.0, 100.0),
    )


class _MisalignedRapidEngine:
    def __call__(
        self,
        image: object,  # ruff: ignore[unused-method-argument]
        *,
        return_word_box: bool,  # ruff: ignore[unused-method-argument]
        return_single_char_box: bool,  # ruff: ignore[unused-method-argument]
    ) -> _RapidResult:
        return _RapidResult(
            txts=("visible",),
            word_results=((("private-canary", 0.9, ((0, 0), (5, 0), (5, 5), (0, 5))),),),
        )


def test_rapidocr_fails_closed_without_echoing_recognized_text() -> None:
    raster = RasterPage(0, object(), 10, 10, 10.0, 10.0)

    with pytest.raises(OcrAdapterError, match="invalid_output") as caught:
        RapidOcrAdapter(engine_factory=_MisalignedRapidEngine).extract(raster)

    assert "private-canary" not in str(caught.value)


class _DegenerateRapidEngine:
    def __call__(
        self,
        image: object,  # ruff: ignore[unused-method-argument]
        *,
        return_word_box: bool,  # ruff: ignore[unused-method-argument]
        return_single_char_box: bool,  # ruff: ignore[unused-method-argument]
    ) -> _RapidResult:
        return _RapidResult(
            txts=("visible",),
            word_results=((("visible", 0.9, ((0, 0), (5, 0), (10, 0), (0, 0))),),),
        )


def test_rapidocr_rejects_degenerate_polygons() -> None:
    raster = RasterPage(0, object(), 10, 10, 10.0, 10.0)

    with pytest.raises(OcrAdapterError, match="invalid_output"):
        RapidOcrAdapter(engine_factory=_DegenerateRapidEngine).extract(raster)


def test_rapidocr_accepts_official_empty_output_shape() -> None:
    class EmptyEngine:
        def __call__(
            self,
            image: object,  # ruff: ignore[unused-method-argument]
            *,
            return_word_box: bool,  # ruff: ignore[unused-method-argument]
            return_single_char_box: bool,  # ruff: ignore[unused-method-argument]
        ) -> object:
            return SimpleNamespace(txts=None, word_results=(("", 1.0, None),))

    raster = RasterPage(0, object(), 10, 10, 10.0, 10.0)

    page = RapidOcrAdapter(engine_factory=EmptyEngine).extract(raster)

    # reason: an engine that returns nothing must still yield a page whose text is the empty STRING.
    # reason: The rule's `not page.text` would also accept a `None` the redaction pipeline would then
    # reason: carry into its span search, which is the failure this assertion exists to catch.
    assert page.text == ""  # ruff: ignore[compare-to-empty-string]
    assert page.tokens == ()


def test_rapidocr_never_borrows_repeated_words_across_lines() -> None:
    class CrossLineEngine:
        def __call__(
            self,
            image: object,  # ruff: ignore[unused-method-argument]
            *,
            return_word_box: bool,  # ruff: ignore[unused-method-argument]
            return_single_char_box: bool,  # ruff: ignore[unused-method-argument]
        ) -> _RapidResult:
            return _RapidResult(
                txts=("ID ID", "ID"),
                word_results=(
                    (("ID", 0.9, ((0, 0), (2, 0), (2, 2), (0, 2))),),
                    (
                        ("ID", 0.9, ((3, 0), (5, 0), (5, 2), (3, 2))),
                        ("ID", 0.9, ((0, 3), (2, 3), (2, 5), (0, 5))),
                    ),
                ),
            )

    raster = RasterPage(0, object(), 10, 10, 10.0, 10.0)

    with pytest.raises(OcrAdapterError, match="invalid_output"):
        RapidOcrAdapter(engine_factory=CrossLineEngine).extract(raster)


@pytest.mark.parametrize(
    ("model_tier", "expected_model"),
    [("small", "small-enum"), ("tiny", "tiny-enum")],
)
def test_rapidocr_factory_configures_ppocrv6_openvino_without_eager_import(
    model_tier: RapidOcrModelTier,
    expected_model: str,
) -> None:
    imports: list[str] = []
    received_params: list[dict[str, object]] = []
    engine = _FakeRapidEngine()
    module = SimpleNamespace(
        EngineType=SimpleNamespace(OPENVINO="openvino-enum", ONNXRUNTIME="ort-enum"),
        ModelType=SimpleNamespace(SMALL="small-enum", TINY="tiny-enum"),
        OCRVersion=SimpleNamespace(PPOCRV6="v6-enum"),
        RapidOCR=lambda *, params: received_params.append(params) or engine,
    )

    def load_module(name: str) -> object:
        imports.append(name)
        return module

    factory = RapidOcrEngineFactory(
        RapidOcrRuntimeConfig(
            backend="openvino",
            model_tier=model_tier,
            threads=4,
            openvino_num_streams=1,
            openvino_performance_hint="LATENCY",
        ),
        module_loader=load_module,
    )
    assert imports == []

    assert factory() is engine

    assert imports == ["rapidocr"]
    assert received_params == [
        {
            "Global.log_level": "critical",
            "Det.engine_type": "openvino-enum",
            "Cls.engine_type": "openvino-enum",
            "Rec.engine_type": "openvino-enum",
            "Det.model_type": expected_model,
            "Rec.model_type": expected_model,
            "Det.ocr_version": "v6-enum",
            "Rec.ocr_version": "v6-enum",
            "EngineConfig.openvino.inference_num_threads": 4,
            "EngineConfig.openvino.performance_hint": "LATENCY",
            "EngineConfig.openvino.num_streams": 1,
        },
    ]


def test_rapidocr_factory_configures_the_single_ort_control() -> None:
    received_params: list[dict[str, object]] = []
    engine = _FakeRapidEngine()
    module = SimpleNamespace(
        EngineType=SimpleNamespace(OPENVINO="openvino-enum", ONNXRUNTIME="ort-enum"),
        ModelType=SimpleNamespace(SMALL="small-enum", TINY="tiny-enum"),
        OCRVersion=SimpleNamespace(PPOCRV6="v6-enum"),
        RapidOCR=lambda *, params: received_params.append(params) or engine,
    )
    factory = RapidOcrEngineFactory(
        RapidOcrRuntimeConfig(
            backend="onnxruntime",
            model_tier="small",
            threads=4,
            ort_inter_op_threads=1,
        ),
        module_loader=lambda _: module,
    )

    assert factory() is engine
    assert received_params == [
        {
            "Global.log_level": "critical",
            "Det.engine_type": "ort-enum",
            "Cls.engine_type": "ort-enum",
            "Rec.engine_type": "ort-enum",
            "Det.model_type": "small-enum",
            "Rec.model_type": "small-enum",
            "Det.ocr_version": "v6-enum",
            "Rec.ocr_version": "v6-enum",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
        },
    ]


def test_rapidocr_factory_missing_capability_is_sanitized() -> None:
    module = SimpleNamespace(
        EngineType=SimpleNamespace(OPENVINO="openvino-enum"),
        ModelType=SimpleNamespace(SMALL="small-enum"),
        OCRVersion=SimpleNamespace(PPOCRV6="v6-enum"),
        RapidOCR=lambda *, params: params,
    )

    factory = RapidOcrEngineFactory(
        RapidOcrRuntimeConfig(backend="openvino", model_tier="tiny", threads=4),
        module_loader=lambda _: module,
    )
    raster = RasterPage(6, object(), 10, 10, 10.0, 10.0)

    with pytest.raises(OcrAdapterError, match="unsupported_configuration") as caught:
        RapidOcrAdapter(engine_factory=factory).extract(raster)

    assert caught.value.page_index == 6
    assert "TINY" not in str(caught.value)


def test_rapidocr_factory_sanitizes_module_import_failure() -> None:
    private_marker = "SYNTH-PRIVATE-RAPIDOCR-IMPORT"

    def fail_import(_: str) -> object:
        raise RuntimeError(private_marker)

    factory = RapidOcrEngineFactory(
        RapidOcrRuntimeConfig(backend="openvino", model_tier="small", threads=4),
        module_loader=fail_import,
    )
    raster = RasterPage(3, object(), 10, 10, 10.0, 10.0)

    with pytest.raises(OcrAdapterError, match="dependency_import") as caught:
        RapidOcrAdapter(engine_factory=factory).extract(raster)

    assert caught.value.page_index == 3
    assert private_marker not in str(caught.value)


class _FakeTesseractRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, image: object) -> str:  # ruff: ignore[unused-method-argument]
        self.calls += 1
        return (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t40\t10\t92.5\tMaya\n"
            "5\t1\t1\t1\t1\t2\t50\t0\t40\t10\t88\tSato\n"
            "5\t1\t1\t1\t2\t1\t0\t20\t50\t10\t97\tID-42"
        )


def test_tesseract_parses_tsv_words_and_loads_runner_lazily() -> None:
    runner = _FakeTesseractRunner()
    factory_calls = 0

    def factory() -> _FakeTesseractRunner:
        nonlocal factory_calls
        factory_calls += 1
        return runner

    adapter = TesseractTsvAdapter(runner_factory=factory)
    raster = RasterPage(0, object(), 100, 200, 50.0, 100.0)
    assert factory_calls == 0

    page = adapter.extract(raster)
    adapter.extract(raster)

    assert factory_calls == 1
    assert runner.calls == 2
    assert page.text == "Maya Sato\nID-42"
    assert tuple((token.start, token.end) for token in page.tokens) == (
        (0, 4),
        (5, 9),
        (10, 15),
    )
    assert tuple(token.confidence for token in page.tokens) == (0.925, 0.88, 0.97)


def test_subprocess_tesseract_uses_structured_bounded_tsv_command() -> None:
    calls: list[dict[str, object]] = []
    tsv = _FakeTesseractRunner().run(object())

    # reason: This recorder mirrors subprocess.run so the OCR adapter call is checked without an argument shim.
    # reason: `input` shadows the builtin because that IS subprocess.run's keyword, and
    # reason: `ocr_backends.py:134` writes `input=bytes(image)` — renaming it here would make the
    # reason: double uncallable by the code under test, which is the opposite of what the rule wants.
    def run_command(  # ruff: ignore[too-many-arguments]
        args: tuple[str, ...],
        *,
        input: bytes,  # ruff: ignore[builtin-argument-shadowing]
        stdout: int,
        stderr: int,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append({
            "args": args,
            "input": input,
            "stdout": stdout,
            "stderr": stderr,
            "timeout": timeout,
            "check": check,
        })
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=tsv.encode(),
            stderr=b"ignored diagnostic output",
        )

    runner = SubprocessTesseractRunner(
        language="eng+vie",
        psm=3,
        timeout_seconds=12.5,
        command_runner=run_command,
    )

    assert runner.run(b"synthetic PNG bytes") == tsv
    assert calls == [
        {
            "args": (
                "tesseract",
                "stdin",
                "stdout",
                "-l",
                "eng+vie",
                "--psm",
                "3",
                "tsv",
            ),
            "input": b"synthetic PNG bytes",
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "timeout": 12.5,
            "check": False,
        },
    ]


def test_subprocess_tesseract_failure_is_page_scoped_and_never_echoes_stderr() -> None:
    private_marker = "SYNTH-PRIVATE-TESSERACT-STDERR"

    # reason: This failure fake mirrors subprocess.run so private-stderr sanitization uses the production call seam.
    def fail_command(  # ruff: ignore[too-many-arguments]
        args: tuple[str, ...],
        *,
        input: bytes,  # ruff: ignore[builtin-argument-shadowing,unused-function-argument]
        stdout: int,  # ruff: ignore[unused-function-argument]
        stderr: int,  # ruff: ignore[unused-function-argument]
        timeout: float,  # ruff: ignore[unused-function-argument]
        check: bool,  # ruff: ignore[unused-function-argument]
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout=b"",
            stderr=private_marker.encode(),
        )

    runner = SubprocessTesseractRunner(command_runner=fail_command)
    raster = RasterPage(7, b"synthetic PNG bytes", 10, 10, 10.0, 10.0)

    with pytest.raises(OcrAdapterError, match="process_exit") as caught:
        TesseractTsvAdapter(runner_factory=lambda: runner).extract(raster)

    assert caught.value.page_index == 7
    assert private_marker not in str(caught.value)


def test_subprocess_tesseract_timeout_is_page_scoped_and_phi_free() -> None:
    private_marker = "SYNTH-PRIVATE-TESSERACT-TIMEOUT"

    # reason: This timeout fake mirrors subprocess.run so timeout sanitization uses the production call seam.
    def timeout_command(  # ruff: ignore[too-many-arguments]
        args: tuple[str, ...],
        *,
        input: bytes,  # ruff: ignore[builtin-argument-shadowing,unused-function-argument]
        stdout: int,  # ruff: ignore[unused-function-argument]
        stderr: int,  # ruff: ignore[unused-function-argument]
        timeout: float,
        check: bool,  # ruff: ignore[unused-function-argument]
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(
            cmd=args,
            timeout=timeout,
            output=b"",
            stderr=private_marker.encode(),
        )

    runner = SubprocessTesseractRunner(timeout_seconds=0.25, command_runner=timeout_command)
    raster = RasterPage(4, b"synthetic PNG bytes", 10, 10, 10.0, 10.0)

    with pytest.raises(OcrAdapterError, match="timeout") as caught:
        TesseractTsvAdapter(runner_factory=lambda: runner).extract(raster)

    assert caught.value.page_index == 4
    assert private_marker not in str(caught.value)
