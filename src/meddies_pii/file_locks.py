"""Small cross-platform file-locking primitives for local coordination."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import BinaryIO

_lock_backend: Any
if os.name == "nt":
    import msvcrt as _lock_backend
else:
    import fcntl as _lock_backend


@contextmanager
def exclusive_file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` until the context exits.

    The lock file is separate from the data it protects so callers can
    atomically replace the data file without releasing the lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        _ensure_lock_byte(handle)
        _acquire(handle, blocking=blocking)
        try:
            yield
        finally:
            _release(handle)


def _ensure_lock_byte(handle: BinaryIO) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)


def _acquire(handle: BinaryIO, *, blocking: bool) -> None:
    if os.name == "nt":
        mode = _lock_backend.LK_LOCK if blocking else _lock_backend.LK_NBLCK
        try:
            _lock_backend.locking(handle.fileno(), mode, 1)
        except OSError as error:
            if not blocking:
                msg = "file lock is already held"
                raise BlockingIOError(msg) from error
            raise
        return

    flags = _lock_backend.LOCK_EX
    if not blocking:
        flags |= _lock_backend.LOCK_NB
    _lock_backend.flock(handle.fileno(), flags)


def _release(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        _lock_backend.locking(handle.fileno(), _lock_backend.LK_UNLCK, 1)
        return
    _lock_backend.flock(handle.fileno(), _lock_backend.LOCK_UN)
