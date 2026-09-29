from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pytest
from google.genai import types

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class FakeFiles:
    """Stands in for `GeminiFiles`, producing the same `types.File` the provider returns."""

    def __init__(
        self,
        *,
        upload_result: types.File | None = None,
        download_result: bytes = b"",
        upload_error: Exception | None = None,
        download_error: Exception | None = None,
    ) -> None:
        self.upload_result = upload_result or types.File(name="files/request")
        self.download_result = download_result
        self.upload_error = upload_error
        self.download_error = download_error
        self.upload_calls: list[dict[str, object]] = []
        self.download_calls: list[str] = []

    def upload(self, *, file: str, config: types.UploadFileConfigOrDict | None = None) -> types.File:
        self.upload_calls.append({"file": file, "config": config})
        if self.upload_error is not None:
            raise self.upload_error
        return self.upload_result

    def download(self, *, file: str) -> bytes:
        self.download_calls.append(file)
        if self.download_error is not None:
            raise self.download_error
        return self.download_result


class FakeBatches:
    """Stands in for `GeminiBatches`, producing the same `types.BatchJob` the provider returns."""

    # reason: Independent create, get, list, cancel, and delete outcomes keep fake batch lifecycle branches isolated.
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        create_result: types.BatchJob | None = None,
        get_events: Sequence[types.BatchJob | Exception] | None = None,
        listed_jobs: Sequence[types.BatchJob] | None = None,
        create_error: Exception | None = None,
        list_error: Exception | None = None,
        cancel_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.create_result: types.BatchJob = create_result or types.BatchJob(name="batches/test")
        self.get_events: list[types.BatchJob | Exception] = list(get_events) if get_events is not None else []
        self.listed_jobs: list[types.BatchJob] = list(listed_jobs) if listed_jobs is not None else []
        self.create_error = create_error
        self.list_error = list_error
        self.cancel_error = cancel_error
        self.delete_error = delete_error
        self.create_calls: list[dict[str, object]] = []
        self.get_calls: list[str] = []
        self.list_calls: list[object] = []
        self.cancel_calls: list[str] = []
        self.delete_calls: list[str] = []

    def create(
        self,
        *,
        model: str,
        src: types.BatchJobSourceUnionDict,
        config: types.CreateBatchJobConfigOrDict | None = None,
    ) -> types.BatchJob:
        self.create_calls.append({"model": model, "src": src, "config": config})
        if self.create_error is not None:
            raise self.create_error
        return self.create_result

    def get(self, *, name: str) -> types.BatchJob:
        self.get_calls.append(name)
        if not self.get_events:
            msg = "unexpected batch status request"
            raise AssertionError(msg)
        event = self.get_events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    def list(self, *, config: types.ListBatchJobsConfigOrDict | None = None) -> Iterable[types.BatchJob]:
        self.list_calls.append(config)
        if self.list_error is not None:
            raise self.list_error
        return self.listed_jobs

    def cancel(self, *, name: str) -> None:
        self.cancel_calls.append(name)
        if self.cancel_error is not None:
            raise self.cancel_error

    def delete(self, *, name: str) -> None:
        self.delete_calls.append(name)
        if self.delete_error is not None:
            raise self.delete_error


# reason: `FakeBatches` and `FakeFiles` above are plain classes because they carry behaviour, and this
# reason: double stands beside them as the third of one set. A dataclass here would make the trio read
# reason: as two kinds of thing for a saving of two lines.
class FakeClient:  # ruff: ignore[class-as-data-structure]
    def __init__(self, batches: FakeBatches, files: FakeFiles) -> None:
        self.batches = batches
        self.files = files


def batch_job(
    name: str = "batches/test",
    state_name: str = "JOB_STATE_SUCCEEDED",
    *,
    dest: types.BatchJobDestination | None = None,
    error: types.JobError | None = None,
) -> types.BatchJob:
    """Build the `types.BatchJob` the provider would return for one lifecycle state.

    Returns:
        A real `BatchJob`, not a stand-in, so a test that reads `job.state.name` or
        `job.dest.file_name` is reading the same model the provider returns.

    """
    return types.BatchJob(
        name=name,
        state=types.JobState(state_name),
        dest=dest,
        error=error,
        create_time="2026-07-29T12:00:00Z",
    )


class ClientFactory(Protocol):
    """The shape `fake_client_factory` yields: build a client and hand back both surfaces."""

    def __call__(
        self,
        *,
        batches: FakeBatches | None = None,
        files: FakeFiles | None = None,
    ) -> tuple[FakeClient, FakeBatches, FakeFiles]: ...


class BatchJobFactory(Protocol):
    """The shape `batch_job_factory` yields, matching `batch_job` above."""

    def __call__(
        self,
        name: str = ...,
        state_name: str = ...,
        *,
        dest: types.BatchJobDestination | None = None,
        error: types.JobError | None = None,
    ) -> types.BatchJob: ...


@pytest.fixture
def fake_client_factory() -> ClientFactory:
    def make_client(
        *,
        batches: FakeBatches | None = None,
        files: FakeFiles | None = None,
    ) -> tuple[FakeClient, FakeBatches, FakeFiles]:
        batch_surface = batches or FakeBatches()
        file_surface = files or FakeFiles()
        return FakeClient(batch_surface, file_surface), batch_surface, file_surface

    return make_client


@pytest.fixture
def batch_job_factory() -> BatchJobFactory:
    return batch_job
