"""Neutral Modal image configuration shared by operational entrypoints."""

from __future__ import annotations

from typing import Protocol, TypeVar

from typing_extensions import Self

MODAL_SOURCE_ROOT = "/root/src"
MODAL_PYTHONPATH = MODAL_SOURCE_ROOT
DEBIAN_SNAPSHOT_SOURCES = (
    "deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260729T000000Z/ bookworm main",
    (
        "deb [check-valid-until=no] "
        "https://snapshot.debian.org/archive/debian-security/20260729T000000Z/ "
        "bookworm-security main"
    ),
)


def _snapshot_sources_command() -> str:
    quoted_sources = " ".join(repr(source) for source in DEBIAN_SNAPSHOT_SOURCES)
    return f"printf '%s\\n' {quoted_sources} > /etc/apt/sources.list"


_DEBIAN_SNAPSHOT_SETUP_COMMANDS = (
    "rm -f /etc/apt/sources.list.d/debian.sources",
    _snapshot_sources_command(),
    "apt-get -o Acquire::Check-Valid-Until=false update",
)


class _EnvironmentImage(Protocol):
    # reason: positional-only, because a protocol matches a parameter by name and modal.Image calls it `vars`.
    def env(self, variables: dict[str, str], /) -> Self: ...


_EnvironmentImageT = TypeVar("_EnvironmentImageT", bound=_EnvironmentImage)


class _BuildCommandImage(Protocol):
    def run_commands(self, *commands: str) -> Self: ...


_BuildImageT = TypeVar("_BuildImageT", bound=_BuildCommandImage)


def add_source_pythonpath(image: _EnvironmentImageT) -> _EnvironmentImageT:
    return image.env({"PYTHONPATH": MODAL_PYTHONPATH})


def use_pinned_debian_snapshot(image: _BuildImageT) -> _BuildImageT:
    return image.run_commands(*_DEBIAN_SNAPSHOT_SETUP_COMMANDS)
