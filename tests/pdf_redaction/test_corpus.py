from __future__ import annotations

import hashlib
import json
import os

# reason: the single use at `:38` runs a fixed argv of `[sys.executable, "-c", <literal source>]` with no
# reason: shell and no caller-supplied element, so no untrusted input reaches the spawned process.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import fitz
import pytest

from meddies_pii.pdf_redaction.benchmark.corpus import (
    PAGE_CLASSES,
    CorpusFixture,
    generate_challenge_corpus,
)
from meddies_pii.taxonomy import PII_LABELS

if TYPE_CHECKING:
    from collections.abc import Iterable


class _WidgetLike(Protocol):
    field_value: str | None


@pytest.fixture(scope="module")
def challenge_fixture() -> CorpusFixture:
    return generate_challenge_corpus()


def test_corpus_records_import_without_pdf_provider() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins

real_import = builtins.__import__

def import_without_pdf_provider(name, globals=None, locals=None, fromlist=(), level=0):
    if (
        name == "fitz"
        or name.startswith("fitz.")
        or name == "pymupdf"
        or name.startswith("pymupdf.")
    ):
        raise ImportError(f"blocked PDF provider: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_pdf_provider
from meddies_pii.pdf_redaction.benchmark.corpus_records import CorpusFixture
assert CorpusFixture.__name__ == "CorpusFixture"
""",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr


def test_corpus_has_the_ten_required_page_classes_and_routes(
    challenge_fixture: CorpusFixture,
) -> None:
    fixture = challenge_fixture

    with fitz.open(stream=fixture.pdf_bytes, filetype="pdf") as document:
        assert document.page_count == 10

    assert tuple(page.page_class for page in fixture.gold.pages) == PAGE_CLASSES
    assert tuple(page.route for page in fixture.gold.pages) == (
        "native_trusted",
        "native_trusted",
        "native_trusted",
        "native_trusted",
        "hybrid_untrusted",
        "image_only",
        "image_only",
        "image_only",
        "hybrid_mixed",
        "hybrid_untrusted",
    )


def test_corpus_generation_is_byte_deterministic() -> None:
    first = generate_challenge_corpus()
    second = generate_challenge_corpus()

    assert first.pdf_bytes == second.pdf_bytes
    assert first.safe_manifest()["pdf_sha256"] == hashlib.sha256(first.pdf_bytes).hexdigest()
    assert first.safe_manifest() == second.safe_manifest()
    assert len(first.gold.pages) == 10
    assert len(first.gold.canaries) == 59
    assert len(first.gold.negative_controls) == 10


def test_corpus_gold_covers_labels_routes_degradation_and_hostile_channels(
    challenge_fixture: CorpusFixture,
) -> None:
    gold = challenge_fixture.gold
    pages_by_index = {page.page_index: page for page in gold.pages}

    for label in PII_LABELS:
        matching = tuple(canary for canary in gold.canaries if canary.label == label)
        assert len(matching) >= 3
        assert any(pages_by_index[item.page_index].route == "native_trusted" for item in matching)
        assert any(pages_by_index[item.page_index].route == "image_only" for item in matching)

    assert gold.pages[6].degradations == (
        "150_dpi",
        "skew",
        "blur_noise",
        "low_contrast",
    )
    assert gold.pages[5].source_dpi == 300
    assert {canary.label for canary in gold.canaries if canary.page_index == 3} >= {
        "id_number",
        "phone_number",
    }
    assert {canary.label for canary in gold.canaries if canary.page_index == 6} == {
        "email_address",
        "human_name",
        "id_number",
        "phone_number",
    }
    assert {canary.channel for canary in gold.canaries if canary.page_index == 9} >= {
        "form",
        "annotation",
        "link",
        "hidden_text",
        "metadata",
        "xmp",
        "attachment",
    }
    page_five_channels = {canary.channel for canary in gold.canaries if canary.page_index == 4}
    assert page_five_channels >= {"visible_raster", "hidden_text"}
    assert all(page.negative_controls for page in gold.pages)
    assert len({canary.digest for canary in gold.canaries}) == len(gold.canaries)
    assert all(canary.quads or canary.object_locator for canary in gold.canaries)
    assert all(
        0 <= point.x <= 612 and 0 <= point.y <= 792
        for canary in gold.canaries
        for quad in canary.quads
        for point in quad.points
    )


def test_safe_manifest_and_reprs_never_expose_raw_canaries(
    challenge_fixture: CorpusFixture,
) -> None:
    fixture = challenge_fixture
    safe_json = fixture.safe_manifest_json()
    parsed = json.loads(safe_json)

    assert parsed["seed"] == 20260713
    assert len(parsed["canaries"]) == len(fixture.gold.canaries)
    assert not any(canary.raw_value in safe_json for canary in fixture.gold.canaries)
    assert not any(canary.raw_value in repr(fixture) for canary in fixture.gold.canaries)


def test_hostile_page_contains_every_declared_nonpage_channel(
    challenge_fixture: CorpusFixture,
) -> None:
    by_channel = {canary.channel: canary for canary in challenge_fixture.gold.canaries if canary.page_index == 9}

    with fitz.open(stream=challenge_fixture.pdf_bytes, filetype="pdf") as document:
        page = document[9]
        widgets = cast("Iterable[_WidgetLike]", page.widgets() or ())
        widget_values = {widget.field_value for widget in widgets}
        annotation_contents = {annotation.info["content"] for annotation in page.annots() or ()}
        link_targets = {link["uri"] for link in page.get_links()}

        assert by_channel["hidden_text"].raw_value in page.get_text()
        assert by_channel["form"].raw_value in widget_values
        assert by_channel["annotation"].raw_value in annotation_contents
        assert by_channel["link"].raw_value in link_targets
        metadata = cast("dict[str, str]", document.metadata or {})
        assert metadata["subject"] == by_channel["metadata"].raw_value
        assert by_channel["xmp"].raw_value in document.get_xml_metadata()
        assert by_channel["attachment"].raw_value.encode() in document.embfile_get("synthetic-private-note.txt")
