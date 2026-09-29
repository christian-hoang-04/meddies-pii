from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
import ast
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Self

import pytest
import tomllib
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ops/modal_runtime_direct_dependency_inventory.py"
PYTHON_BASE_IMAGE = "python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d"
EXPECTED_DEBIAN_SNAPSHOT_URLS = (
    "https://snapshot.debian.org/archive/debian/20260729T000000Z/",
    "https://snapshot.debian.org/archive/debian-security/20260729T000000Z/",
)
spec = importlib.util.spec_from_file_location("modal_runtime_direct_dependency_inventory", SCRIPT)
assert spec is not None
assert spec.loader is not None
inventory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = inventory
spec.loader.exec_module(inventory)


def test_project_dependency_groups_preserve_runtime_capability_boundaries() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = {Requirement(specifier).name for specifier in project["project"]["dependencies"]}
    training = {Requirement(specifier).name for specifier in project["project"]["optional-dependencies"]["training"]}
    development = tuple(project["dependency-groups"]["dev"])

    assert {"packaging", "typing-extensions"} <= runtime
    assert "typer" not in runtime
    assert training == {"onnx", "onnxruntime", "peft", "torch", "transformers"}
    assert "meddies-pii[pdf-redaction,training]" in development
    assert training.isdisjoint({Requirement(specifier).name for specifier in development})


def imported_names(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


class FakeImage:
    def __init__(self) -> None:
        self.environment_calls: list[dict[str, str]] = []

    def env(self, variables: dict[str, str]) -> Self:
        self.environment_calls.append(variables)
        return self


class FakeBuildImage:
    def __init__(self) -> None:
        self.command_calls: list[tuple[str, ...]] = []

    def run_commands(self, *commands: str) -> Self:
        self.command_calls.append(commands)
        return self


def test_source_pythonpath_configures_a_modal_compatible_image() -> None:
    from meddies_pii.modal_runtime import (
        MODAL_PYTHONPATH,
        add_source_pythonpath,
    )

    image = FakeImage()

    configured = add_source_pythonpath(image)

    assert configured is image
    assert image.environment_calls == [{"PYTHONPATH": MODAL_PYTHONPATH}]


def test_debian_snapshot_setup_commands_preserve_the_pinned_image_contract() -> None:
    from meddies_pii.modal_runtime import use_pinned_debian_snapshot

    image = FakeBuildImage()

    configured = use_pinned_debian_snapshot(image)

    assert configured is image
    assert image.command_calls == [
        (
            "rm -f /etc/apt/sources.list.d/debian.sources",
            (
                "printf '%s\\n' "
                "'deb [check-valid-until=no] "
                "https://snapshot.debian.org/archive/debian/20260729T000000Z/ "
                "bookworm main' "
                "'deb [check-valid-until=no] "
                "https://snapshot.debian.org/archive/debian-security/"
                "20260729T000000Z/ bookworm-security main' > /etc/apt/sources.list"
            ),
            "apt-get -o Acquire::Check-Valid-Until=false update",
        ),
    ]


def test_debian_snapshot_sources_have_one_runtime_owner() -> None:
    active_python = (
        *(ROOT / "src/meddies_pii").rglob("*.py"),
        *(path for path in (ROOT / "scripts").rglob("*.py") if not path.is_relative_to(ROOT / "scripts/archive")),
    )
    owners = {
        path.relative_to(ROOT)
        for path in active_python
        if any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("deb [check-valid-until=no] ")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        )
    }

    assert owners == {Path("src/meddies_pii/modal_runtime.py")}


def test_every_active_modal_image_is_in_the_machine_readable_inventory() -> None:
    from meddies_pii.modal_runtime import (
        DEBIAN_SNAPSHOT_SOURCES as RUNTIME_SNAPSHOT_SOURCES,
    )

    images = inventory.collect_modal_images(ROOT)
    paths = {image.path for image in images}
    assert paths
    pdf_image = next(image for image in images if image.path == "scripts/ops/run_pdf_redaction_benchmark.py")
    assert pdf_image.apt_requirements == (
        "fonts-dejavu-core=2.37-6",
        "libgl1=1.6.0-1",
        "libglib2.0-0=2.74.6-2+deb12u9",
        "poppler-utils=22.12.0-2+deb12u2",
        "qpdf=11.3.0-1+deb12u1",
        "tesseract-ocr=5.3.0-2",
        "tesseract-ocr-eng=1:4.1.0-2",
        "tesseract-ocr-vie=1:4.1.0-2",
    )
    assert pdf_image.base_references == (PYTHON_BASE_IMAGE,)
    lfm25_image = next(image for image in images if image.path == "scripts/ops/run_lfm25_pii_baseline.py")
    assert lfm25_image.apt_requirements == ("gcc=4:12.2.0-3",)
    assert lfm25_image.base_references == (PYTHON_BASE_IMAGE,)
    assert "use_pinned_debian_snapshot" in imported_names(ROOT / lfm25_image.path, "meddies_pii.modal_runtime")
    assert all(any(url in source for source in RUNTIME_SNAPSHOT_SOURCES) for url in EXPECTED_DEBIAN_SNAPSHOT_URLS)

    apt_image_paths = {image.path for image in images if image.apt_requirements}
    assert apt_image_paths == {
        "scripts/ops/export_pii350_release.py",
        "scripts/ops/run_lfm25_pii_baseline.py",
        "scripts/ops/run_lfm_bioes_baseline.py",
        "scripts/ops/run_pdf_redaction_benchmark.py",
        "scripts/ops/run_pii350_checkpoint_benchmark.py",
    }
    for relative_path in apt_image_paths:
        assert "use_pinned_debian_snapshot" in imported_names(ROOT / relative_path, "meddies_pii.modal_runtime")

    inventory.assert_fully_pinned(
        tuple(
            replace(
                image,
                apt_snapshot_references=RUNTIME_SNAPSHOT_SOURCES,
            )
            if image.apt_requirements
            else image
            for image in images
        ),
    )


def test_mutable_requirement_is_a_red_gate() -> None:
    image = inventory.ModalImageDirectDependencies("fake.py", ("torch>=2.13",), base_references=(PYTHON_BASE_IMAGE,))
    with pytest.raises(ValueError, match=r"fake\.py: torch>=2\.13"):
        inventory.assert_fully_pinned((image,))


def test_empty_modal_image_dependency_inventory_is_a_red_gate() -> None:
    image = inventory.ModalImageDirectDependencies("fake.py", (), base_references=(PYTHON_BASE_IMAGE,))
    with pytest.raises(ValueError, match=r"fake\.py"):
        inventory.assert_fully_pinned((image,))


def test_exact_python_and_apt_requirements_pass() -> None:
    image = inventory.ModalImageDirectDependencies(
        "fake.py",
        ("torch[cu130]==2.13.0",),
        ("libgl1=1.2.3-1",),
        (PYTHON_BASE_IMAGE,),
        EXPECTED_DEBIAN_SNAPSHOT_URLS,
    )

    inventory.assert_fully_pinned((image,))


@pytest.mark.parametrize(
    "requirement",
    [
        "torch>=2.13",
        "torch==2.*",
        "torch @ https://example.invalid/torch.whl",
        "*PINNED",
    ],
)
def test_non_exact_python_requirement_is_a_red_gate(requirement: str) -> None:
    image = inventory.ModalImageDirectDependencies("fake.py", (requirement,), base_references=(PYTHON_BASE_IMAGE,))

    with pytest.raises(ValueError, match="Mutable Modal dependency declarations"):
        inventory.assert_fully_pinned((image,))


def test_unpinned_apt_requirement_is_a_red_gate() -> None:
    image = inventory.ModalImageDirectDependencies(
        "fake.py",
        ("torch==2.13.0",),
        ("libgl1",),
        (PYTHON_BASE_IMAGE,),
        EXPECTED_DEBIAN_SNAPSHOT_URLS,
    )

    with pytest.raises(ValueError, match=r"Unpinned APT dependencies: fake\.py: libgl1"):
        inventory.assert_fully_pinned((image,))


def test_floating_base_image_is_a_red_gate() -> None:
    image = inventory.ModalImageDirectDependencies("fake.py", ("torch==2.13.0",), base_references=("python:3.12-slim",))

    with pytest.raises(ValueError, match=r"Floating Modal base images: fake\.py"):
        inventory.assert_fully_pinned((image,))


def test_apt_image_without_snapshot_provenance_is_a_red_gate() -> None:
    image = inventory.ModalImageDirectDependencies(
        "fake.py",
        ("torch==2.13.0",),
        ("libgl1=1.2.3-1",),
        (PYTHON_BASE_IMAGE,),
    )

    with pytest.raises(ValueError, match="APT images without Debian snapshot provenance"):
        inventory.assert_fully_pinned((image,))


def _modal_function_secret_names(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    decorator = next(
        decorator
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "function"
    )
    secrets = next(keyword.value for keyword in decorator.keywords if keyword.arg == "secrets")
    return {
        argument.value
        for node in ast.walk(secrets)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_name"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }


def test_lfm25_aggregate_declares_secret_for_gated_eval_fixture() -> None:
    lfm25_script = ROOT / "scripts/ops/run_lfm25_pii_baseline.py"

    assert _modal_function_secret_names(lfm25_script, "aggregate") == {"huggingface-secret"}
