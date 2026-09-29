"""Typed contract for the Modal decorators used by BIOES jobs."""

from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: `modal/` is the only subpackage of `bioes/` without an `__init__.py` — assembly, data, eval,
# reason: reports and trainers all have one — so the asymmetry reads as an oversight, and adding the
# reason: file is likely inert under hatchling's src-layout discovery.
# reason: Deliberately deferred rather than fixed here: this directory holds every spend-authorization
# reason: gate, and its Modal-remote import paths have only fake-mediated local coverage, so adding
# reason: `__init__.py` is a post-merge change whose proof is a real GPU smoke run — owner ledger item.
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

P = ParamSpec("P")
ResultT_co = TypeVar("ResultT_co", covariant=True)


class RemoteFunction(Protocol[P, ResultT_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ResultT_co: ...

    def remote(self, *args: P.args, **kwargs: P.kwargs) -> ResultT_co: ...

    def local(self, *args: P.args, **kwargs: P.kwargs) -> ResultT_co: ...

    def map(self, inputs: Iterable[object]) -> Iterable[ResultT_co]: ...


class TypedModalApp(Protocol):
    def function(self, **options: object) -> Callable[[Callable[P, ResultT_co]], RemoteFunction[P, ResultT_co]]: ...

    def local_entrypoint(self, **options: object) -> Callable[[Callable[P, ResultT_co]], Callable[P, ResultT_co]]: ...
