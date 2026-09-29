from __future__ import annotations

import importlib
import math
import re

# reason: Tesseract execution uses list-form argv through an injectable runner; image bytes are stdin, never command text.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from numbers import Real
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from meddies_pii.pdf_redaction.ocr_types import (
    CommandRunner,
    OcrAdapterError,
    RapidEngine,
    RapidOcrRuntimeConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _OcrVersionEnum(Protocol):
    PPOCRV6: object


@runtime_checkable
class _RapidOcrModule(Protocol):
    """The rapidocr module surface this factory requires.

    Every attribute is guarded by an `AttributeError` handler at the call site, so a build missing
    one raises `unsupported_configuration` rather than failing here.
    """

    RapidOCR: Callable[..., object]
    EngineType: object
    ModelType: object
    OCRVersion: _OcrVersionEnum


class RapidOcrEngineFactory:
    def __init__(
        self,
        config: RapidOcrRuntimeConfig,
        *,
        module_loader: Callable[[str], object] = importlib.import_module,
    ) -> None:
        self._config = config
        self._module_loader = module_loader

    def __call__(self) -> RapidEngine:
        try:
            loaded = self._module_loader("rapidocr")
        except ImportError:
            msg = "rapidocr"
            raise OcrAdapterError(msg, "dependency_unavailable") from None
        # reason: plugin import hooks may raise arbitrary loader errors; the adapter preserves
        # reason: a stable import-failure code.
        except Exception:  # ruff: ignore[blind-except]
            msg = "rapidocr"
            raise OcrAdapterError(msg, "dependency_import") from None

        if not isinstance(loaded, _RapidOcrModule):
            msg = "rapidocr"
            raise OcrAdapterError(msg, "unsupported_configuration")
        try:
            engine_class = loaded.RapidOCR
            params = _rapidocr_params(loaded, self._config)
        except AttributeError:
            msg = "rapidocr"
            raise OcrAdapterError(msg, "unsupported_configuration") from None

        try:
            return cast("RapidEngine", engine_class(params=params))
        except ImportError:
            msg = "rapidocr"
            raise OcrAdapterError(msg, "dependency_unavailable") from None
        # reason: RapidOCR engine construction crosses versioned native and Python APIs with no stable exception set.
        except Exception:  # ruff: ignore[blind-except]
            msg = "rapidocr"
            raise OcrAdapterError(msg, "engine_initialization") from None


MAX_TESSERACT_PAGE_SEGMENTATION_MODE = 13

_TESSERACT_LANGUAGE_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*\Z")


class SubprocessTesseractRunner:
    def __init__(
        self,
        *,
        language: str = "eng+vie",
        psm: int = 3,
        timeout_seconds: float = 30.0,
        executable: str = "tesseract",
        command_runner: CommandRunner | None = None,
    ) -> None:
        if not _TESSERACT_LANGUAGE_PATTERN.fullmatch(language):
            msg = "language must contain valid Tesseract language codes"
            raise ValueError(msg)
        if isinstance(psm, bool) or not isinstance(psm, int) or not 0 <= psm <= MAX_TESSERACT_PAGE_SEGMENTATION_MODE:
            msg = "psm must be an integer between zero and thirteen"
            raise ValueError(msg)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, Real)
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0.0
        ):
            msg = "timeout_seconds must be finite and positive"
            raise ValueError(msg)
        if not executable or "\x00" in executable:
            msg = "executable must be a non-empty command path"
            raise ValueError(msg)
        self._language: str = language
        self._psm: int = psm
        self._timeout_seconds: float = float(timeout_seconds)
        self._executable: str = executable
        self._command_runner: CommandRunner = command_runner or cast("CommandRunner", subprocess.run)

    def run(self, image: object) -> str:
        if not isinstance(image, (bytes, bytearray, memoryview)):
            msg = "tesseract"
            raise OcrAdapterError(msg, "unsupported_image")
        command = (
            self._executable,
            "stdin",
            "stdout",
            "-l",
            self._language,
            "--psm",
            str(self._psm),
            "tsv",
        )
        try:
            completed = self._command_runner(
                command,
                input=bytes(image),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            msg = "tesseract"
            raise OcrAdapterError(msg, "dependency_unavailable") from None
        except subprocess.TimeoutExpired:
            msg = "tesseract"
            raise OcrAdapterError(msg, "timeout") from None
        except OSError:
            msg = "tesseract"
            raise OcrAdapterError(msg, "process_start") from None

        if completed.returncode != 0:
            msg = "tesseract"
            raise OcrAdapterError(msg, "process_exit")
        try:
            return completed.stdout.decode("utf-8")
        except UnicodeDecodeError:
            msg = "tesseract"
            raise OcrAdapterError(msg, "invalid_encoding") from None


def _rapidocr_params(module: _RapidOcrModule, config: RapidOcrRuntimeConfig) -> dict[str, object]:
    engine_values = module.EngineType
    model_values = module.ModelType
    version_values = module.OCRVersion
    engine = getattr(
        engine_values,
        "OPENVINO" if config.backend == "openvino" else "ONNXRUNTIME",
    )
    model = getattr(model_values, config.model_tier.upper())
    version = version_values.PPOCRV6
    params: dict[str, object] = {
        "Global.log_level": "critical",
        "Det.engine_type": engine,
        "Cls.engine_type": engine,
        "Rec.engine_type": engine,
        "Det.model_type": model,
        "Rec.model_type": model,
        "Det.ocr_version": version,
        "Rec.ocr_version": version,
    }
    if config.backend == "openvino":
        params.update({
            "EngineConfig.openvino.inference_num_threads": config.threads,
            "EngineConfig.openvino.performance_hint": (config.openvino_performance_hint),
            "EngineConfig.openvino.num_streams": config.openvino_num_streams,
        })
    else:
        params.update({
            "EngineConfig.onnxruntime.intra_op_num_threads": config.threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": (config.ort_inter_op_threads),
            "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
        })
    return params
