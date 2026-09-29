from __future__ import annotations

from typing import Any, override


# reason: renaming this is a public-API change, not a lint fix. It is the package's exported base exception,
# reason: subclassed by three siblings here and imported by name from label_corpus.runner and two test modules,
# reason: so the rename belongs on the owner ledger with its consumer table, not in a lint branch.
class MeddiesException(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.retry_after = retry_after

    @override
    def __str__(self) -> str:
        return self.message


class ConfigurationError(MeddiesException):
    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, context=context)


class DailyBudgetExceeded(MeddiesException):
    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, context=context)


class APIError(MeddiesException):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message=message, retry_after=retry_after)
        self.status_code = status_code
