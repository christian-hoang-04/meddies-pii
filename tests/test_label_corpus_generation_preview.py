from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
from typing import TYPE_CHECKING

from meddies_pii.generation.label_corpus.preview import _select_repaired_prefix

if TYPE_CHECKING:
    from pathlib import Path


def test_preview_prefers_clean_repaired_candidates_when_both_languages_exist(tmp_path: Path) -> None:
    (tmp_path / "repaired_candidates.clean.vi.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "repaired_candidates.clean.en.jsonl").write_text("", encoding="utf-8")

    assert _select_repaired_prefix(tmp_path, requested_prefix=None) == "repaired_candidates.clean"


def test_preview_falls_back_to_unfiltered_repaired_candidates_until_clean_pair_exists(
    tmp_path: Path,
) -> None:
    (tmp_path / "repaired_candidates.clean.vi.jsonl").write_text("", encoding="utf-8")

    assert _select_repaired_prefix(tmp_path, requested_prefix=None) == "repaired_candidates"


def test_preview_respects_explicit_repaired_prefix(tmp_path: Path) -> None:
    assert (
        _select_repaired_prefix(tmp_path, requested_prefix="repaired_candidates.flagged") == "repaired_candidates.flagged"
    )


def test_preview_rejects_non_object_repair_summary(tmp_path: Path) -> None:
    import pytest

    from meddies_pii.generation.label_corpus.preview import _read_json

    summary = tmp_path / "repair_summary.en.json"
    summary.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        _read_json(summary)
