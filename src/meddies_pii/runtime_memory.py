"""Portable process-memory measurements used by benchmark reports."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Any

_resource: Any = None
try:
    import resource as _resource
except ImportError:
    _resource = None

_PROC_STATUS_FIELDS = 3
_BYTES_PER_KIB = 1024


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_rss_bytes() -> int:
    """Return process peak resident bytes, or a safe positive fallback.

    Returns:
        A positive byte count, or ``1`` when the host does not expose a
        measurement API.

    """
    if _resource is not None:
        try:
            value = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
        except (OSError, ValueError):
            pass
        else:
            multiplier = 1 if sys.platform == "darwin" else _BYTES_PER_KIB
            return max(1, value * multiplier)

    if sys.platform == "win32":
        windows_value = _windows_memory_bytes("PeakWorkingSetSize")
        if windows_value is not None:
            return windows_value
    return 1


def current_rss_bytes(*, status_path: Path = Path("/proc/self/status")) -> int:
    """Return current resident bytes using procfs or a platform fallback.

    Returns:
        A positive byte count.

    """
    try:
        rows = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        value = _windows_memory_bytes("WorkingSetSize") if sys.platform == "win32" else None
        return value or peak_rss_bytes()
    for row in rows:
        if not row.startswith("VmRSS:"):
            continue
        fields = row.split()
        if len(fields) == _PROC_STATUS_FIELDS and fields[2] == "kB" and fields[1].isdigit():
            return max(1, int(fields[1]) * _BYTES_PER_KIB)
        break
    return peak_rss_bytes()


def _windows_memory_bytes(field: str) -> int | None:
    if sys.platform != "win32":
        return None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    try:
        process = ctypes.windll.kernel32.GetCurrentProcess()
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        success = get_info(process, ctypes.byref(counters), counters.cb)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if not success:
        return None
    return max(1, int(getattr(counters, field)))
