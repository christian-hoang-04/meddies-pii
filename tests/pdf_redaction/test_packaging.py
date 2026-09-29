from __future__ import annotations

import ast
from pathlib import Path

import tomllib

MODAL_SCRIPT = Path("scripts/ops/run_pdf_redaction_benchmark.py")


def _exact_pypdfium2_requirement(requirements: tuple[str, ...]) -> str:
    matches = tuple(requirement for requirement in requirements if requirement.startswith("pypdfium2=="))
    assert len(matches) == 1
    return matches[0]


def _modal_pdf_packages() -> tuple[str, ...]:
    module = ast.parse(MODAL_SCRIPT.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PDF_PACKAGES" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Tuple)
    packages = tuple(
        element.value
        for element in assignment.value.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )
    assert len(packages) == len(assignment.value.elts)
    return packages


def test_pdf_redaction_extra_is_runtime_complete_and_permissive() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = tuple(item.lower() for item in project["project"]["optional-dependencies"]["pdf-redaction"])

    assert any(item.startswith("numpy") for item in dependencies)
    assert any(item.startswith("openvino") for item in dependencies)
    assert any(item.startswith("pypdfium2") for item in dependencies)
    assert any(item.startswith("rapidocr") for item in dependencies)
    assert not any(item.startswith("torch") for item in dependencies)
    assert not any(item.startswith("pymupdf") for item in dependencies)
    assert not any(item.startswith("onnxruntime") for item in dependencies)


def test_modal_pdf_image_matches_pdf_redaction_pypdfium2_pin() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    local_requirement = _exact_pypdfium2_requirement(tuple(project["project"]["optional-dependencies"]["pdf-redaction"]))
    modal_requirement = _exact_pypdfium2_requirement(_modal_pdf_packages())

    assert modal_requirement == local_requirement
