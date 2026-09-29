from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


def install_hf_script_stubs(
    monkeypatch: pytest.MonkeyPatch,
    dotenv_calls: list[str] | None = None,
) -> None:
    def fake_load_dotenv(*_args: object, **_kwargs: object) -> object:
        if dotenv_calls is not None:
            dotenv_calls.append("load_dotenv")
        return None

    class FakeDataset:
        @staticmethod
        def from_list(_rows: object) -> object:
            return object()

    def forbidden_provider_call(*_args: object, **_kwargs: object) -> object:
        msg = "provider work happened during help/import"
        raise AssertionError(msg)

    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(load_dotenv=fake_load_dotenv),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Dataset=FakeDataset,
            get_dataset_config_names=forbidden_provider_call,
            load_dataset=forbidden_provider_call,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(login=forbidden_provider_call),
    )


def import_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_path", "module_name"),
    [
        (
            REPO / "scripts/migrations/convert_meddies_configs.py",
            "_convert_meddies_configs_safety_test",
        ),
        (
            REPO / "scripts/migrations/download_meddies_pii_configs.py",
            "_download_meddies_pii_configs_safety_test",
        ),
        (
            REPO / "scripts/reports/build_training_data_report.py",
            "_build_training_data_report_safety_test",
        ),
    ],
)
def test_data_or_report_scripts_parse_help_before_work(
    script_path: Path,
    module_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = import_script(script_path, module_name)

    with pytest.raises(SystemExit) as raised:
        module.parse_args(["--help"])

    assert raised.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_validate_pushed_import_has_no_provider_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden_call(*_args: object, **_kwargs: object) -> object:
        calls.append("called")
        msg = "provider work happened during import"
        raise AssertionError(msg)

    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(load_dotenv=forbidden_call),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(login=forbidden_call),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=forbidden_call,
            load_dataset=forbidden_call,
        ),
    )

    module = import_script(
        REPO / "scripts/reports/validate_pushed.py",
        "_validate_pushed_safety_test",
    )

    assert calls == []
    assert callable(module.main)


def test_validate_pushed_recognizes_modern_and_legacy_schemas() -> None:
    module = import_script(
        REPO / "scripts/reports/validate_pushed.py",
        "_validate_pushed_schema_test",
    )

    class FakeDataset:
        def __init__(self, rows: list[dict[str, object]], columns: list[str]) -> None:
            self._rows = rows
            self.column_names = columns

        def __getitem__(self, index: int) -> dict[str, object]:
            return self._rows[index]

    clean_modern = FakeDataset(
        [
            {
                "text": "Dr. Lan",
                "label": [{"category": "human_name", "start": 0, "end": 7, "text": "Dr. Lan"}],
                "info": {},
            },
        ],
        ["text", "label", "info"],
    )
    assert module.validate_config_rows(clean_modern, [0]) == []

    dirty_modern = FakeDataset(
        [
            {
                "text": "hello world",
                "label": [{"category": "not_a_label", "start": 0, "end": 3, "text": "xyz"}],
                "info": {},
            },
        ],
        ["text", "label", "info"],
    )
    issues = module.validate_config_rows(dirty_modern, [0])
    assert any("invalid category" in issue for issue in issues)
    assert any("offset mismatch" in issue for issue in issues)

    legacy = FakeDataset(
        [
            {
                "text": "[Lan]<human_name>",
                "raw": "Lan",
                "label": '{"human_name": ["Lan"]}',
            },
        ],
        ["text", "raw", "label"],
    )
    assert module.validate_config_rows(legacy, [0]) == []


@pytest.mark.parametrize(
    ("script_path", "module_name"),
    [
        (
            REPO / "scripts/archive/process_all_subsets.py",
            "_process_all_subsets_import_test",
        ),
        (REPO / "scripts/archive/reformat_cleaned.py", "_reformat_cleaned_import_test"),
    ],
)
def test_legacy_hf_scripts_import_without_dotenv_or_provider_side_effects(
    script_path: Path,
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_calls: list[str] = []
    install_hf_script_stubs(monkeypatch, dotenv_calls)

    module = import_script(script_path, module_name)

    assert dotenv_calls == []
    assert callable(module.main)


@pytest.mark.parametrize(
    ("script_path", "module_name"),
    [
        (
            REPO / "scripts/generation/generate_synthetic_data.py",
            "_synthetic_data_help_test",
        ),
        (
            REPO / "scripts/archive/process_all_subsets.py",
            "_process_all_subsets_help_test",
        ),
        (REPO / "scripts/archive/reformat_cleaned.py", "_reformat_cleaned_help_test"),
        (
            REPO / "scripts/generation/run_mimo_general_20k.py",
            "_mimo_general_help_test",
        ),
        (
            REPO / "scripts/generation/run_opencode_zen_clinical_daily.py",
            "_opencode_zen_help_test",
        ),
    ],
)
def test_script_help_parses_before_dotenv_or_provider_work(
    script_path: Path,
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dotenv_calls: list[str] = []
    install_hf_script_stubs(monkeypatch, dotenv_calls)
    module = import_script(script_path, module_name)

    with pytest.raises(SystemExit) as raised:
        module.main(["--help"])

    assert raised.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert dotenv_calls == []


def test_reformat_cleaned_requires_explicit_publish_before_hf_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_calls: list[str] = []
    install_hf_script_stubs(monkeypatch, dotenv_calls)
    module = import_script(
        REPO / "scripts/archive/reformat_cleaned.py",
        "_reformat_cleaned_publish_guard_test",
    )

    with pytest.raises(SystemExit) as raised:
        module.main(["--configs", "english"])

    assert raised.value.code == 2
    assert dotenv_calls == []


def test_generate_label_corpus_rejects_unknown_required_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_script(
        REPO / "scripts/generation/generate_label_corpus.py",
        "_generate_label_corpus_label_boundary_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_label_corpus.py",
            "--language",
            "English",
            "--target-count",
            "1",
            "--required-label",
            "unknown_label",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        module.parse_args()

    assert raised.value.code == 2
