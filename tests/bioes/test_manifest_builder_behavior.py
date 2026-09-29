from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.data.build_manifest import (
    ManifestBuildConfig,
    build_manifest,
    write_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


class CharacterTokenizer:
    pad_token: str | None = None
    # reason: these are the tokenizer's sentinel strings, not credentials. The rule keys on a name
    # reason: containing `token`, and a stand-in for a tokenizer has to carry the names the real one does.
    eos_token = "<eos>"  # ruff: ignore[hardcoded-password-string]
    # reason: these are the tokenizer's sentinel strings, not credentials. The rule keys on a name
    # reason: containing `token`, and a stand-in for a tokenizer has to carry the names the real one does.
    unk_token = "<unk>"  # ruff: ignore[hardcoded-password-string]

    def __call__(self, text: str, *, max_length: int, **_: object) -> dict[str, object]:
        kept = min(len(text), max_length)
        return {
            "input_ids": list(range(kept)),
            "attention_mask": [1] * kept,
            "offset_mapping": [(index, index + 1) for index in range(kept)],
            "num_truncated_tokens": len(text) - kept,
        }


def _label_row(uid: str, text: str) -> dict[str, object]:
    return {
        "uid": uid,
        "text": text,
        "label": [{"category": "human_name", "start": 0, "end": 5, "text": text[:5]}],
    }


def test_build_manifest_pins_revisions_and_excludes_truncated_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import meddies_pii.training.bioes.data.build_manifest as manifest_build

    class FakeApi:
        @staticmethod
        def dataset_info(*, repo_id: str, revision: str | None) -> SimpleNamespace:
            return SimpleNamespace(sha=f"dataset-{repo_id}-{revision or 'head'}")

        @staticmethod
        def model_info(*, repo_id: str, revision: str | None) -> SimpleNamespace:
            return SimpleNamespace(sha=f"model-{repo_id}-{revision or 'head'}")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> CharacterTokenizer:
            return CharacterTokenizer()

    scanned = [_label_row("truncated", "Alice too long"), _label_row("good", "Alice")]
    monkeypatch.setattr(manifest_build, "HfApi", FakeApi)
    monkeypatch.setattr(manifest_build, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(manifest_build, "load_dataset", lambda *_args, **_kwargs: scanned)

    manifest = build_manifest(
        ManifestBuildConfig(
            dataset_id="fixture/data",
            model_id="fixture/model",
            tokenizer_id="fixture/tokenizer",
            max_length=6,
            fixed_pad_length=8,
            required_examples=1,
            min_observed_pad_length=None,
            created_at_utc="2026-07-29T00:00:00Z",
        ),
    )

    assert manifest["dataset"]["revision"] == "dataset-fixture/data-head"
    assert manifest["eval_dataset"]["revision"] == "dataset-fixture/data-head"
    assert manifest["selected"][0]["uid"] == "good"
    assert manifest["candidates"][0]["skip_reason"] == "truncated"
    assert manifest["scan"]["tokenized_candidate_stats"]["skipped_truncated"] == 1
    assert manifest["tokenizer"]["revision"] == "model-fixture/tokenizer-head"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (ManifestBuildConfig(max_length=0), "max_length must be positive"),
        (ManifestBuildConfig(max_length=8, fixed_pad_length=7), "fixed_pad_length"),
        (ManifestBuildConfig(min_observed_pad_length=0), "min_observed_pad_length"),
        (ManifestBuildConfig(target_slice="unknown"), "target_slice must be one of"),
    ],
)
def test_build_manifest_rejects_invalid_config_before_network(config: ManifestBuildConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_manifest(config)


def test_write_manifest_creates_parent_and_stable_json(tmp_path: Path) -> None:
    destination = write_manifest({"z": 1, "a": "đ"}, tmp_path / "deep" / "manifest.json")
    assert destination.read_text(encoding="utf-8") == '{\n  "a": "đ",\n  "z": 1\n}\n'


def test_manifest_main_writes_requested_json_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import meddies_pii.training.bioes.data.build_manifest as manifest_build

    class FakeApi:
        @staticmethod
        def dataset_info(**_: object) -> SimpleNamespace:
            return SimpleNamespace(sha="dataset-sha")

        @staticmethod
        def model_info(**_: object) -> SimpleNamespace:
            return SimpleNamespace(sha="model-sha")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> CharacterTokenizer:
            return CharacterTokenizer()

    monkeypatch.setattr(manifest_build, "HfApi", FakeApi)
    monkeypatch.setattr(manifest_build, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        manifest_build,
        "load_dataset",
        lambda *_args, **_kwargs: [_label_row("good", "Alice")],
    )
    destination = tmp_path / "manifest.json"

    assert (
        manifest_build.main([
            "--dataset-id",
            "fixture/data",
            "--model-id",
            "fixture/model",
            "--tokenizer-id",
            "fixture/tokenizer",
            "--max-length",
            "6",
            "--fixed-pad-length",
            "8",
            "--min-observed-pad-length",
            "1",
            "--created-at-utc",
            "2026-07-29T00:00:00Z",
            "--out",
            str(destination),
        ])
        == 0
    )
    assert destination.exists()
