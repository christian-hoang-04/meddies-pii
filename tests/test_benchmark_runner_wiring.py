from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunnerContract:
    functions: tuple[str, ...]
    image: str
    baseline_mount: str


RUNNER_CONTRACTS = {
    "run_model_baseline.py": RunnerContract(
        ("openmed_first_light", "openmed_eval_cell", "aggregate_openmed"),
        "openmed_image",
        "BASELINE_VOLUME_MOUNT",
    ),
    "run_gliner2_baseline.py": RunnerContract(
        ("gliner2_eval_cell", "aggregate_gliner2"),
        "gliner2_image",
        "BASELINE_VOLUME_MOUNT",
    ),
    "run_lfm25_pii_baseline.py": RunnerContract(("eval_cell", "aggregate"), "image", "BASELINE_MOUNT"),
    "run_lfm_bioes_baseline.py": RunnerContract(
        ("lfm_first_light", "lfm_eval_cell", "aggregate_lfm"),
        "image",
        "BASELINE_VOLUME_MOUNT",
    ),
    "run_opf_baseline.py": RunnerContract(
        ("opf_first_light", "opf_eval_cell", "aggregate_opf"),
        "opf_image",
        "BASELINE_VOLUME_MOUNT",
    ),
}


@dataclass(frozen=True)
class RemoteFunctionConfig:
    image: str | None
    volumes: dict[str, str]
    has_gpu: bool


def _name(node: ast.expr | None) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _app_default_image(module: ast.Module) -> str | None:
    for node in module.body:
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "modal"
            and node.value.func.attr == "App"
        ):
            continue
        return next(
            (_name(keyword.value) for keyword in node.value.keywords if keyword.arg == "image"),
            None,
        )
    return None


def _function_config(function: ast.FunctionDef, app_default_image: str | None) -> RemoteFunctionConfig:
    decorator = next(
        (
            decorator
            for decorator in function.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "app"
            and decorator.func.attr == "function"
        ),
        None,
    )
    assert decorator is not None, function.name
    keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
    volumes = keywords.get("volumes")
    assert isinstance(volumes, ast.Dict), function.name
    return RemoteFunctionConfig(
        image=_name(keywords.get("image")) or app_default_image,
        volumes={
            key.id: value.id
            for key, value in zip(volumes.keys, volumes.values, strict=True)
            if isinstance(key, ast.Name) and isinstance(value, ast.Name)
        },
        has_gpu="gpu" in keywords,
    )


def _contract_function_configs(runner_name: str) -> dict[str, RemoteFunctionConfig]:
    runner_path = REPO_ROOT / "scripts" / "ops" / runner_name
    module = ast.parse(runner_path.read_text(encoding="utf-8"))
    app_default_image = _app_default_image(module)
    configs: dict[str, RemoteFunctionConfig] = {}
    for function in (node for node in module.body if isinstance(node, ast.FunctionDef)):
        if not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_evaluation_contract"
            for node in ast.walk(function)
        ):
            continue
        configs[function.name] = _function_config(function, app_default_image)
    return configs


def test_contract_builders_share_their_runner_image_and_baseline_volume() -> None:
    """Sidecar contracts must be rebuilt in the same runtime and result Volume."""
    for runner_name, contract in RUNNER_CONTRACTS.items():
        configs = _contract_function_configs(runner_name)

        assert set(configs) == set(contract.functions), runner_name
        assert {config.image for config in configs.values()} == {contract.image}
        assert all(contract.baseline_mount in config.volumes for config in configs.values())


def test_lfm_bioes_contract_builders_mount_the_checkpoint_artifact_volume() -> None:
    """Every remote function that hashes the checkpoint can read it on Modal."""
    configs = _contract_function_configs("run_lfm_bioes_baseline.py")

    assert all(config.volumes.get("ARTIFACT_VOLUME_MOUNT") == "artifact_volume" for config in configs.values())
    assert configs["aggregate_lfm"].volumes == {
        "BASELINE_VOLUME_MOUNT": "baseline_volume",
        "ARTIFACT_VOLUME_MOUNT": "artifact_volume",
    }
    assert not configs["aggregate_lfm"].has_gpu
