from __future__ import annotations

from typing import Literal

ErrorStage = Literal["input", "output", "apply", "verify", "tool"]


class PdfRedactionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: ErrorStage,
        code: str,
        cause_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.cause_type = cause_type


class InputDocumentError(PdfRedactionError):
    pass


class OutputPathError(PdfRedactionError):
    pass


class RedactionApplyError(PdfRedactionError):
    pass


class VerificationError(PdfRedactionError):
    pass
