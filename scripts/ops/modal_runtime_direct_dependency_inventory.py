#!/usr/bin/env python
"""Enumerate active Modal image direct dependencies and reject mutable inputs.

This is not a transitive dependency lock or a resolved environment inventory. Python
requirements must parse as a single exact package version. APT requirements must use
``package=version``. Evaluation records capture their resolved environment separately.
"""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, cast

from packaging.requirements import InvalidRequirement, Requirement

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class ModalImageDirectDependencies:
    path: str
    requirements: tuple[str, ...]
    apt_requirements: tuple[str, ...] = ()
    base_references: tuple[str, ...] = ()
    apt_snapshot_references: tuple[str, ...] = ()


def _string_tuple_assignments(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    values: dict[str, tuple[str, ...]] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        target = statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
        value = statement.value
        if not isinstance(target, ast.Name) or not isinstance(value, (ast.Tuple, ast.List)):
            continue
        if all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts):
            # reason: the isinstance filter narrows each element, and the all() guard above proves the
            # reason: filter drops nothing and that every .value is a str.
            values[target.id] = cast(
                "tuple[str, ...]",
                tuple(item.value for item in value.elts if isinstance(item, ast.Constant)),
            )
    return values


def _requirements_from_call(call: ast.Call, constants: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    requirements: list[str] = []
    for argument in call.args:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            requirements.append(argument.value)
        elif isinstance(argument, ast.Starred) and isinstance(argument.value, ast.Name):
            requirements.extend(constants.get(argument.value.id, (f"*{argument.value.id}",)))
        else:
            requirements.append(ast.unparse(argument))
    return tuple(requirements)


def _is_modal_image_constructor(node: ast.AST) -> TypeGuard[ast.Call]:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"debian_slim", "from_registry"}
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "Image"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "modal"
    )


def _base_reference(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute) and call.func.attr == "debian_slim":
        return "modal.Image.debian_slim"
    if not call.args:
        return "<missing registry reference>"
    argument = call.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    return ast.unparse(argument)


def _snapshot_references(tree: ast.Module) -> tuple[str, ...]:
    return tuple(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "https://snapshot.debian.org/archive/" in node.value
    )


def _active_modal_paths(root: Path) -> Iterable[Path]:
    for base in (root / "scripts" / "ops", root / "src" / "meddies_pii" / "training"):
        yield from sorted(base.rglob("*.py"))


def collect_modal_images(root: Path) -> tuple[ModalImageDirectDependencies, ...]:
    images: list[ModalImageDirectDependencies] = []
    for path in _active_modal_paths(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        image_calls = tuple(node for node in ast.walk(tree) if _is_modal_image_constructor(node))
        if not image_calls:
            continue
        constants = _string_tuple_assignments(tree)
        requirements = tuple(
            requirement
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pip_install"
            for requirement in _requirements_from_call(node, constants)
        )
        apt_requirements = tuple(
            requirement
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "apt_install"
            for requirement in _requirements_from_call(node, constants)
        )
        images.append(
            ModalImageDirectDependencies(
                str(path.relative_to(root)),
                requirements,
                apt_requirements,
                tuple(_base_reference(call) for call in image_calls),
                _snapshot_references(tree),
            ),
        )
    return tuple(images)


def assert_fully_pinned(images: Iterable[ModalImageDirectDependencies]) -> None:
    mutable = [
        f"{image.path}: {requirement}"
        for image in images
        for requirement in image.requirements
        if not _is_exact_python_requirement(requirement)
    ]
    mutable_apt = [
        f"{image.path}: {requirement}"
        for image in images
        for requirement in image.apt_requirements
        if not _is_exact_apt_requirement(requirement)
    ]
    mutable_bases = [
        f"{image.path}: {reference}"
        for image in images
        for reference in image.base_references
        if not _is_exact_base_reference(reference)
    ]
    missing = [image.path for image in images if not image.requirements]
    missing_base = [image.path for image in images if not image.base_references]
    missing_snapshot = [
        image.path
        for image in images
        if image.apt_requirements and not _has_snapshot_provenance(image.apt_snapshot_references)
    ]
    if missing:
        raise ValueError("Modal images without audited dependencies: " + "; ".join(missing))
    if mutable:
        raise ValueError("Mutable Modal dependency declarations: " + "; ".join(mutable))
    if mutable_apt:
        raise ValueError("Unpinned APT dependencies: " + "; ".join(mutable_apt))
    if missing_base:
        raise ValueError("Modal images without pinned base references: " + "; ".join(missing_base))
    if mutable_bases:
        raise ValueError("Floating Modal base images: " + "; ".join(mutable_bases))
    if missing_snapshot:
        raise ValueError("APT images without Debian snapshot provenance: " + "; ".join(missing_snapshot))


def _is_exact_python_requirement(requirement: str) -> bool:
    if requirement.startswith("*"):
        return False
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return False
    specifiers = tuple(parsed.specifier)
    return (
        parsed.url is None and len(specifiers) == 1 and specifiers[0].operator == "==" and "*" not in specifiers[0].version
    )


def _is_exact_apt_requirement(requirement: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9+.-]*=[^=\s*]+", requirement))


def _is_exact_base_reference(reference: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", reference))


def _has_snapshot_provenance(references: tuple[str, ...]) -> bool:
    sources = {
        match.group(1)
        for reference in references
        for match in re.finditer(
            r"https://snapshot\.debian\.org/archive/(debian|debian-security)/"
            r"\d{8}T\d{6}Z/",
            reference,
        )
    }
    return sources == {"debian", "debian-security"}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    images = collect_modal_images(root)
    assert_fully_pinned(images)
    print(json.dumps([asdict(image) for image in images], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
