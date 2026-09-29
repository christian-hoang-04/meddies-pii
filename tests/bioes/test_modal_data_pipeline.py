from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn

import pytest

from meddies_pii.training.bioes import assembly
from meddies_pii.training.bioes.assembly import AssemblyGateError
from meddies_pii.training.bioes.data import grpo_convert, inline_tags, mixed
from meddies_pii.training.bioes.modal import data_pipeline

if TYPE_CHECKING:
    from collections.abc import Mapping


def _local_data_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    commits: list[str] = []
    monkeypatch.setattr(data_pipeline, "MOUNT", str(tmp_path))
    monkeypatch.setattr(
        data_pipeline,
        "volume",
        SimpleNamespace(commit=lambda: commits.append("commit")),
    )
    return commits


def test_convert_inline_local_skips_invalid_and_dropped_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commits = _local_data_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "hf_configs" / "vietnamese-translated.train.jsonl"
    source.parent.mkdir()
    source.write_text('{"text": "keep"}\nnot-json\n{"text": "drop"}\n', encoding="utf-8")
    calls: list[tuple[str, str]] = []

    def fake_convert(
        row: Mapping[str, object],
        *,
        uid: str,
        default_language: str,
    ) -> dict[str, str] | None:
        calls.append((uid, default_language))
        return {"uid": uid} if row["text"] == "keep" else None

    monkeypatch.setattr(inline_tags, "convert_config_row", fake_convert)

    kept = data_pipeline.convert_inline.local("vietnamese-translated")

    destination = next((tmp_path / "internal" / "hf_configs").glob("*.jsonl"))
    assert kept == 1
    assert [json.loads(line) for line in destination.read_text().splitlines()] == [{"uid": "vietnamese-translated-0"}]
    assert calls == [
        ("vietnamese-translated-0", "Vietnamese"),
        ("vietnamese-translated-2", "Vietnamese"),
    ]
    assert commits == ["commit"]


def test_convert_nvidia_and_grpo_local_keep_only_converted_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commits = _local_data_pipeline(monkeypatch, tmp_path)
    nvidia_source = tmp_path / "hf_configs" / "nvidia-health.train.jsonl"
    grpo_source = tmp_path / "hf_configs" / "grpo-train.train.jsonl"
    nvidia_source.parent.mkdir()
    nvidia_source.write_text(
        '{"text": "Ada", "label": "{\\"human_name\\": [\\"Ada\\"]}"}\n{"text": "none", "label": "{}"}\n',
        encoding="utf-8",
    )
    grpo_source.write_text('{"text": "kept"}\n{"text": "dropped"}\n', encoding="utf-8")
    monkeypatch.setattr(
        grpo_convert,
        "find_spans",
        lambda text, _extractions: [{"label": "human_name"}] if text == "Ada" else [],
    )
    monkeypatch.setattr(
        grpo_convert,
        "convert_grpo_hf_row",
        lambda row, *, uid: {"uid": uid} if row["text"] == "kept" else None,
    )

    nvidia_kept = data_pipeline.convert_nvidia.local("nvidia-health")
    grpo_kept = data_pipeline.convert_grpo_hf.local("grpo-train")

    assert nvidia_kept == 1
    assert grpo_kept == 1
    nvidia_output = next((tmp_path / "internal" / "nvidia").glob("*.jsonl"))
    assert json.loads(nvidia_output.read_text()) == {
        "text": "Ada",
        "label": [{"label": "human_name"}],
        "info": {
            "source": "nvidia-health",
            "language": "unknown",
            "uid": "nvidia-health-0",
        },
    }
    assert commits == ["commit", "commit"]


def test_download_convert_ai4privacy_local_uses_faked_stream_and_filters_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commits = _local_data_pipeline(monkeypatch, tmp_path)
    fake_datasets = SimpleNamespace(
        load_dataset=lambda *_args, split, **_kwargs: (
            [{"language": "vi", "text": "keep"}, {"language": "xx", "text": "drop"}] if split == "train" else []
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets)
    monkeypatch.setattr(mixed, "is_supported_meddies_language", lambda language: language == "vi")
    monkeypatch.setattr(
        mixed,
        "convert_ai4privacy_row",
        lambda _row, *, dataset_id, default_uid: (
            {"uid": default_uid, "dataset": dataset_id},
            {},
        ),
    )

    result = data_pipeline.download_convert_ai4privacy.local()

    assert result == {"scanned": 2, "kept": 1, "dropped_lang": 1}
    output = next((tmp_path / "internal" / "ai4privacy").glob("*.jsonl"))
    assert json.loads(output.read_text()) == {
        "uid": "ai4-1",
        "dataset": "ai4privacy/pii-masking-openpii-1.5m",
    }
    assert commits == ["commit"]


def test_count_file_local_discards_non_objects_and_invalid_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _local_data_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text('{"uid": "one"}\n[]\nnot-json\n{"uid": "two"}\n', encoding="utf-8")

    class _Counts:
        @staticmethod
        def to_json() -> dict[str, int]:
            return {"kept": 2}

    monkeypatch.setattr(grpo_convert, "find_spans", grpo_convert.find_spans)
    from meddies_pii.training.bioes.data import corpus_gate

    monkeypatch.setattr(corpus_gate, "count_corpus", lambda rows: _Counts() if len(rows) == 2 else None)

    result = data_pipeline.count_file.local(str(source))

    assert result == {"path": "source.jsonl", "rows": 2, "counts": {"kept": 2}}


def test_list_sources_and_assemble_mix_local_use_package_owned_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commits = _local_data_pipeline(monkeypatch, tmp_path)
    source_paths = [
        tmp_path / "internal" / "a.jsonl",
        tmp_path / "internal" / "b.jsonl",
    ]
    monkeypatch.setattr(assembly, "discover_assembly_source_files", lambda _root: source_paths)

    class _Result:
        @staticmethod
        def to_summary() -> dict[str, int | str]:
            return {"rows": 3, "gate": "pass"}

    monkeypatch.setattr(assembly, "assemble_bioes_v2_corpus", lambda *_args, **_kwargs: _Result())

    assert data_pipeline.list_sources.local() == [str(path) for path in source_paths]
    assert data_pipeline.assemble_mix.local([["vi", "human_name"], ["too-many"]]) == {
        "rows": 3,
        "gate": "pass",
    }
    assert commits == ["commit"]


def test_assemble_mix_local_propagates_gate_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _local_data_pipeline(monkeypatch, tmp_path)

    def fail_assembly(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "coverage gate failed"
        raise AssemblyGateError(msg)

    monkeypatch.setattr(assembly, "assemble_bioes_v2_corpus", fail_assembly)

    with pytest.raises(RuntimeError, match="coverage gate failed"):
        data_pipeline.assemble_mix.local()


def test_download_configs_and_corpus_stats_local_use_faked_dataset_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commits = _local_data_pipeline(monkeypatch, tmp_path)

    class _Split:
        num_rows = 2

        @staticmethod
        def to_json(path: str, **_kwargs: object) -> None:
            Path(path).write_text('{"text": "first"}\n{"text": "second"}\n', encoding="utf-8")

    fake_datasets = SimpleNamespace(
        get_dataset_config_names=lambda _repo: ["train-config"],
        load_dataset=lambda _repo, _config: {"train": _Split()},
    )
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets)

    assert data_pipeline.download_configs.local() == {"train-config.train": 2}

    mix_path = tmp_path / "mix" / "train.jsonl"
    mix_path.parent.mkdir()
    mix_path.write_text(
        '{"text": "Alice", "label": ["human_name"], "info": {"source": "fixture", "language": '
        '"Vietnamese", "text_format": "note", "edge_cases": ["typo"]}}\n'
        "not-json\n"
        '{"text": 5, "label": [], "info": "wrong-shape"}\n',
        encoding="utf-8",
    )
    from meddies_pii.generation import text_formats
    from meddies_pii.training.bioes.data import corpus_gate

    monkeypatch.setattr(corpus_gate, "_resolve_language", lambda language: "vi" if language else None)
    monkeypatch.setattr(corpus_gate, "_row_labels", lambda row: list(row["label"]))
    monkeypatch.setattr(text_formats, "canonical_text_format", lambda value: f"format:{value}")

    stats = data_pipeline.corpus_stats.local("mix/train.jsonl")

    assert stats["total_rows"] == 2
    assert stats["total_spans"] == 1
    assert stats["per_source"] == {"fixture": 1, "unknown": 1}
    assert stats["per_language"] == {"vi": 1, "other": 1}
    assert stats["per_label"] == {"human_name": 1}
    assert stats["per_format"] == {"format:note": 1}
    assert stats["per_edge_case"] == {"typo": 1}
    assert commits == ["commit"]
