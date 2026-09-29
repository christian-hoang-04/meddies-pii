"""Typed lazy seam around the optional Hugging Face datasets package."""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import importlib
from collections.abc import Iterable, Iterator
from typing import Protocol, TypeGuard, runtime_checkable


@runtime_checkable
class _DatasetsModule(Protocol):
    def load_dataset(self, *args: object, **kwargs: object) -> object: ...


@runtime_checkable
class _DatasetSplits(Protocol):
    def __getitem__(self, name: str) -> object: ...


@runtime_checkable
class DatasetRows(Protocol):
    def __iter__(self) -> Iterator[object]: ...

    def __len__(self) -> int: ...


def _is_object_iterable(value: object) -> TypeGuard[Iterable[object]]:
    return isinstance(value, Iterable)


def _load_module() -> _DatasetsModule:
    module = importlib.import_module("datasets")
    if not isinstance(module, _DatasetsModule):
        msg = "datasets does not expose load_dataset"
        raise RuntimeError(msg)
    return module


def load_streaming_train(repo: str) -> Iterable[object]:
    """Return the train split of a streaming dataset after runtime validation.

    Returns:
        The train split as an iterable of objects, checked at this boundary rather than trusted
        from the library's own annotations.

    Raises:
        RuntimeError: If ``datasets`` returns something that is not a split mapping, or if its
            train split is not iterable. Both are checked here because the failure would
            otherwise surface deep in a consuming loop, after the caller has already committed
            to a long streaming read.

    """
    dataset = _load_module().load_dataset(repo, streaming=True)
    if not isinstance(dataset, _DatasetSplits):
        msg = "datasets returned an invalid streaming dataset"
        raise RuntimeError(msg)
    train_split = dataset["train"]
    if not _is_object_iterable(train_split):
        msg = "datasets train split is not iterable"
        raise RuntimeError(msg)
    return train_split


def load_train_rows(repo_id: str, config: str) -> DatasetRows:
    """Return a materialized train split after runtime validation.

    Returns:
        The fully materialized train split for the named repo and config, checked at this
        boundary to be the row-collection shape callers index into.

    Raises:
        RuntimeError: If ``datasets`` returns something other than that shape. Unlike the
            streaming path this has already paid the download, so the check exists to give the
            failure a name at the seam rather than an attribute error at first use.

    """
    dataset = _load_module().load_dataset(repo_id, config, split="train")
    if not isinstance(dataset, DatasetRows):
        msg = "datasets returned an invalid train split"
        raise RuntimeError(msg)
    return dataset
