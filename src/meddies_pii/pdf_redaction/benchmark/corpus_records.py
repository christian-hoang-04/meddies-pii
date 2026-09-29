from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from meddies_pii.pdf_redaction.contracts import ContentRoute, Quad
    from meddies_pii.taxonomy import PiiLabel

PageClass = Literal[
    "native_discharge",
    "native_form",
    "native_two_column",
    "native_rotated",
    "hybrid_hidden_ocr",
    "image_clean",
    "image_degraded",
    "image_complex_clinical",
    "multilingual_hybrid",
    "hostile_objects",
]
GoldChannel = Literal[
    "visible_native",
    "visible_raster",
    "hidden_text",
    "form",
    "annotation",
    "link",
    "metadata",
    "xmp",
    "attachment",
]
Degradation = Literal["150_dpi", "skew", "blur_noise", "low_contrast"]

PAGE_CLASSES: tuple[PageClass, ...] = (
    "native_discharge",
    "native_form",
    "native_two_column",
    "native_rotated",
    "hybrid_hidden_ocr",
    "image_clean",
    "image_degraded",
    "image_complex_clinical",
    "multilingual_hybrid",
    "hostile_objects",
)

_PAGE_ROUTES: tuple[ContentRoute, ...] = (
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

_WIDTH_PT = 612.0
_HEIGHT_PT = 792.0
_FIXED_PDF_DATE = "D:20260713000000Z"
_DOMAIN = b"meddies-pii-pdf-benchmark-v1\0"
_TEXT_WIDGET_TYPE = 7

_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
)


@dataclass(frozen=True, slots=True)
class CanaryGold:
    fixture_id: str
    page_index: int
    label: PiiLabel
    channel: GoldChannel
    digest: str
    quads: tuple[Quad, ...]
    object_locator: str | None = None
    raw_value: str = field(repr=False, default="")


@dataclass(frozen=True, slots=True)
class NegativeControlGold:
    fixture_id: str
    page_index: int
    digest: str
    quads: tuple[Quad, ...]
    raw_value: str = field(repr=False, default="")


@dataclass(frozen=True, slots=True)
class PageGold:
    page_index: int
    page_class: PageClass
    route: ContentRoute
    route_reasons: tuple[str, ...]
    negative_controls: tuple[str, ...]
    rotation_degrees: int = 0
    degradations: tuple[Degradation, ...] = ()
    source_asset_digest: str | None = None
    source_dpi: int | None = None


@dataclass(frozen=True, slots=True)
class CorpusGold:
    seed: int
    pages: tuple[PageGold, ...]
    canaries: tuple[CanaryGold, ...]
    negative_controls: tuple[NegativeControlGold, ...]


def _safe_quad(quad: Quad) -> list[list[float]]:
    return [[round(point.x, 4), round(point.y, 4)] for point in quad.points]


def _safe_canary(canary: CanaryGold) -> dict[str, object]:
    return {
        "fixture_id": canary.fixture_id,
        "page_index": canary.page_index,
        "label": canary.label,
        "channel": canary.channel,
        "digest": canary.digest,
        "quads": [_safe_quad(quad) for quad in canary.quads],
        "object_locator": canary.object_locator,
    }


@dataclass(frozen=True, slots=True)
class CorpusFixture:
    pdf_bytes: bytes = field(repr=False)
    gold: CorpusGold

    def safe_manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "seed": self.gold.seed,
            "pdf_sha256": hashlib.sha256(self.pdf_bytes).hexdigest(),
            "pages": [
                {
                    "page_index": page.page_index,
                    "page_class": page.page_class,
                    "route": page.route,
                    "route_reasons": list(page.route_reasons),
                    "rotation_degrees": page.rotation_degrees,
                    "degradations": list(page.degradations),
                    "source_asset_digest": page.source_asset_digest,
                    "source_dpi": page.source_dpi,
                    "negative_controls": list(page.negative_controls),
                }
                for page in self.gold.pages
            ],
            "canaries": [_safe_canary(canary) for canary in self.gold.canaries],
            "negative_controls": [
                {
                    "fixture_id": control.fixture_id,
                    "page_index": control.page_index,
                    "digest": control.digest,
                    "quads": [_safe_quad(quad) for quad in control.quads],
                }
                for control in self.gold.negative_controls
            ],
        }

    def safe_manifest_json(self) -> str:
        return json.dumps(self.safe_manifest(), sort_keys=True, separators=(",", ":"))
