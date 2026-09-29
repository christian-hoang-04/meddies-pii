from __future__ import annotations

from typing import TYPE_CHECKING

from meddies_pii.file_locks import exclusive_file_lock
from meddies_pii.runtime_memory import current_rss_bytes, peak_rss_bytes

if TYPE_CHECKING:
    from pathlib import Path


def test_exclusive_file_lock_creates_a_stable_lock_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache.lock"

    with exclusive_file_lock(lock_path):
        assert lock_path.is_file()

    assert lock_path.read_bytes() == b"\0"


def test_current_rss_reads_proc_style_status(tmp_path: Path) -> None:
    status_path = tmp_path / "status"
    status_path.write_text("Name:\tpython\nVmRSS:\t123 kB\n", encoding="utf-8")

    assert current_rss_bytes(status_path=status_path) == 123 * 1024


def test_peak_rss_is_positive() -> None:
    assert peak_rss_bytes() > 0
