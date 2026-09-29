"""Liquid LFM2.5 native-label fold and exact Space detector adapter.

The accepted runtime is Liquid's pinned Space ``PiiDetector.detect`` method. Each
input document is detected independently, then its native labels are folded into
the Meddies taxonomy without changing vendor span boundaries.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the adapter loads its model stack inside load(), not when the module is imported.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import hashlib
import importlib.util
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from meddies_pii.eval_baseline.baseline.views import DualViewPrediction
from meddies_pii.json_types import is_str_mapping
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from meddies_pii.taxonomy import PiiLabel

MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M-PII-Detector"
MODEL_REVISION = "b8c9cf3d2d6ae52501b35a27ba46f271449c9ce2"

LIQUID_LABEL_FOLD: dict[str, PiiLabel | None] = {
    "identity.person_name": "human_name",
    "identity.ssn": "id_number",
    "identity.national_id": "id_number",
    "identity.passport": "id_number",
    "identity.drivers_license": "id_number",
    "identity.date_of_birth": "date",
    "identity.tax_id": "id_number",
    "contact.email": "email_address",
    "contact.phone": "phone_number",
    "contact.address": "address",
    "contact.postal_code": "address",
    "contact.ip_address": "id_number",
    "financial.credit_card": "id_number",
    "financial.iban": "id_number",
    "financial.bank_account": "id_number",
    "financial.swift_bic": "id_number",
    "financial.crypto_wallet": "id_number",
    "financial.amount": None,
    "credential.api_key": "secret",
    "credential.password": "secret",
    "credential.private_key": "secret",
    "credential.jwt": "secret",
    "credential.connection_string": "secret",
    "developer.login_credentials": "secret",
    "online.username": "human_name",
    "online.url": "private_url",
    "device.mac_address": "id_number",
    "device.imei": "id_number",
    "developer.device_id": "id_number",
    "location.gps_coordinates": "address",
    "healthcare.medical_record": "id_number",
    "healthcare.condition": None,
    "healthcare.medication": None,
    "healthcare.health_plan_id": "id_number",
    "org.company_name": "company_name",
    "special.religion": None,
    "special.political": None,
    "special.orientation": None,
    "special.health_status": None,
    "legal.case_number": "id_number",
}

LFM25_PII_SUPPORTED_LABELS: frozenset[PiiLabel] = frozenset(
    label for label in LIQUID_LABEL_FOLD.values() if label is not None
)

VendorSpan: TypeAlias = Mapping[str, object]


class SpacePiiDetector(Protocol):
    def detect(self, text: str) -> object: ...


class _LiquidHybridDecoder(Protocol):
    def model_spans(self, text: str, tokenizer: object, model: object) -> list[dict[str, object]]: ...

    def hybrid_spans(self, text: str, model_spans: list[dict[str, object]]) -> list[dict[str, object]]: ...


class _DualViewSpacePiiDetector(SpacePiiDetector, Protocol):
    hd: _LiquidHybridDecoder
    tok: object
    model: object
    lock: AbstractContextManager[object]
    auth_priority_types: frozenset[str]


class _CudaModel(Protocol):
    def to(self, *, device: str, dtype: object) -> _CudaModel: ...


class _PinnedSpaceDetector(SpacePiiDetector, Protocol):
    model: _CudaModel
    auth_priority_types: frozenset[str]


class _PinnedSpaceModule(Protocol):
    MODEL_ID: str
    PiiDetector: Callable[[], _PinnedSpaceDetector]
    hf_hub_download: Callable[..., str]


class _CudaMatmulBackend(Protocol):
    allow_tf32: bool


class _CudaBackend(Protocol):
    matmul: _CudaMatmulBackend


class _TorchBackends(Protocol):
    cuda: _CudaBackend


class _TorchRuntime(Protocol):
    float32: object
    backends: _TorchBackends


_SpaceSourceLoader: TypeAlias = Callable[[str, Path], _PinnedSpaceModule]


def _confined_local_file(root_path: Path, candidate_path: Path, *, role: str) -> Path:
    root = root_path.resolve()
    candidate = candidate_path.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        msg = f"Liquid {role} escapes root: {candidate_path!r}"
        raise RuntimeError(msg) from exc
    if not candidate.is_file():
        msg = f"Liquid {role} is missing: {candidate_path!r}"
        raise RuntimeError(msg)
    return candidate


def _snapshot_file(snapshot_path: Path, filename: str) -> str:
    root = snapshot_path.resolve()
    relative_filename = Path(filename)
    if relative_filename.is_absolute() or ".." in relative_filename.parts:
        msg = f"Liquid snapshot filename escapes root: {filename!r}"
        raise RuntimeError(msg)
    lexical_candidate = root / relative_filename
    try:
        lexical_candidate.relative_to(root)
    except ValueError as exc:
        msg = f"Liquid snapshot filename escapes root: {filename!r}"
        raise RuntimeError(msg) from exc
    candidate = lexical_candidate.resolve()
    if not candidate.is_file():
        msg = f"Liquid snapshot file is missing: {filename!r}"
        raise RuntimeError(msg)
    try:
        candidate.relative_to(root)
    except ValueError:
        model_cache_root = root.parent.parent.resolve()
        if not lexical_candidate.is_symlink():
            msg = f"Liquid snapshot file escapes root: {filename!r}"
            raise RuntimeError(msg) from None
        try:
            candidate.relative_to(model_cache_root)
        except ValueError as exc:
            msg = f"Liquid snapshot file escapes model cache: {filename!r}"
            raise RuntimeError(msg) from exc
    return str(lexical_candidate)


def _verify_space_source(source_path: Path, expected_sha256: str) -> None:
    observed = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if observed != expected_sha256:
        msg = f"Liquid Space source hash changed: expected {expected_sha256}, got {observed}"
        raise RuntimeError(msg)


def _import_pinned_space_source(
    module_name: str,
    source_path: Path,
) -> _PinnedSpaceModule:
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        msg = f"cannot import pinned Liquid Space source: {source_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast("_PinnedSpaceModule", module)


# reason: load pinned space exposes snapshot/torch module as its public contract; bundling would break callers.
def load_pinned_space_detector(  # ruff: ignore[too-many-arguments]
    *,
    snapshot_path: Path,
    source_path: Path,
    space_root: Path,
    space_revision: str,
    expected_source_sha256: str,
    source_loader: _SpaceSourceLoader = _import_pinned_space_source,
    torch_module: _TorchRuntime | None = None,
) -> SpacePiiDetector:
    """Build one exact Liquid Space detector against local immutable inputs.

    The caller owns snapshot acquisition and cache policy. This function binds
    the Space vendor downloader to existing files under that snapshot.

    Returns:
        The vendor detector on CUDA in fp32, with its downloader rebound to read only from the
        local snapshot and its auth-priority types captured. Nothing is fetched: the vendor's
        ``hf_hub_download`` is replaced by a lookup confined to the snapshot, so the detector
        runs against exactly the verified files.

    Raises:
        RuntimeError: If the decoder does not expose its auth-priority type set as a set of
            strings. That set drives the overlap-removal ordering, so reconstructing the
            vendor's own precedence without it would silently score a different resolution
            than the shipped Space produces. The source-confinement and digest checks run
            before this and their refusals reach a caller here too.

    """
    source = _confined_local_file(space_root, source_path, role="Space source")
    _verify_space_source(source, expected_source_sha256)
    module = source_loader(f"liquid_space_pii_{space_revision[:12]}", source)
    module.MODEL_ID = str(snapshot_path)
    module.hf_hub_download = lambda _repo_id, filename, **_kwargs: _snapshot_file(snapshot_path, filename)
    detector = module.PiiDetector()
    if torch_module is None:
        import torch

        torch_module = cast("_TorchRuntime", torch)
    detector.model = detector.model.to(device="cuda", dtype=torch_module.float32)
    torch_module.backends.cuda.matmul.allow_tf32 = True
    decoder = getattr(detector, "hd", None)
    auth_types = getattr(decoder, "_AUTH_TYPES", None)
    if not isinstance(auth_types, set) or not all(isinstance(native_type, str) for native_type in auth_types):
        msg = "Liquid decoder did not expose auth-priority types"
        raise RuntimeError(msg)
    detector.auth_priority_types = frozenset(auth_types)
    return detector


def map_hybrid_spans(
    text: str,
    raw_spans: Sequence[Mapping[str, object]],
) -> list[CharSpan]:
    """Fold native spans without changing their shipped boundaries.

    Returns:
        The vendor spans folded onto this project's label set, boundaries untouched, sorted by
        start, end and label. Four cases are dropped rather than raised: a non-string ``type``,
        a native type the fold map deliberately maps to ``None``, bounds that are not both
        exactly ``int``, and bounds outside the text or non-increasing. Duplicates by start,
        end and folded label are dropped too.

    Raises:
        RuntimeError: If a span carries a string native type that the fold map does not know.
            This is the one unknown that is not skipped, and the distinction is deliberate: a
            type mapped to ``None`` was reviewed and excluded, while an unseen type means the
            vendor added a label since the map was written. Skipping it would silently drop
            real detections and report the baseline as weaker than it is.

    """
    spans: list[CharSpan] = []
    seen: set[tuple[int, int, PiiLabel]] = set()
    for raw in raw_spans:
        native_type = raw.get("type")
        if not isinstance(native_type, str):
            continue
        if native_type not in LIQUID_LABEL_FOLD:
            msg = f"unknown Liquid native label: {native_type!r}"
            raise RuntimeError(msg)
        label = LIQUID_LABEL_FOLD[native_type]
        if label is None:
            continue
        start, end = raw.get("start"), raw.get("end")
        if type(start) is not int or type(end) is not int:
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        spans.append(CharSpan(start=start, end=end, text=text[start:end], label=label))
    return sorted(spans, key=lambda span: (span.start, span.end, span.label))


def _vendor_spans(output: Mapping[str, object]) -> tuple[VendorSpan, ...]:
    raw_spans = output.get("spans")
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
        msg = "Liquid Space detector returned malformed spans"
        raise RuntimeError(msg)
    spans: list[VendorSpan] = []
    for raw in raw_spans:
        if not is_str_mapping(raw):
            msg = "Liquid Space detector returned a malformed span"
            raise RuntimeError(msg)
        start, end, native_type = raw.get("start"), raw.get("end"), raw.get("type")
        if type(start) is not int or type(end) is not int or not isinstance(native_type, str):
            msg = "Liquid Space detector returned a malformed span"
            raise RuntimeError(msg)
        spans.append(raw)
    return tuple(spans)


def _space_final_spans(
    raw_spans: Sequence[Mapping[str, object]],
    *,
    auth_priority_types: frozenset[str],
) -> tuple[VendorSpan, ...]:
    """Apply the pinned Space's auth-priority overlap removal exactly.

    Returns:
        The surviving spans in the vendor's own resolution order, overlaps removed by its
        auth-priority rule rather than by a rule of ours. Reproducing the shipped precedence is
        the point: a different tie-break would score a different system than the one on the Hub.

    """
    ordered = sorted(
        _vendor_spans({"spans": raw_spans}),
        key=lambda span: _space_sort_key(span, auth_priority_types),
    )
    kept: list[VendorSpan] = []
    last_end = -1
    for span in ordered:
        start, end = _vendor_span_bounds(span)
        if start < last_end or end <= start:
            continue
        kept.append(span)
        last_end = end
    return tuple(kept)


def _space_sort_key(span: VendorSpan, auth_priority_types: frozenset[str]) -> tuple[int, int, int]:
    start, end = _vendor_span_bounds(span)
    native_type = _vendor_span_type(span)
    return (
        start,
        0 if native_type in auth_priority_types else 1,
        -(end - start),
    )


def _vendor_span_bounds(span: VendorSpan) -> tuple[int, int]:
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int:
        msg = "Liquid Space detector returned a malformed span"
        raise RuntimeError(msg)
    return start, end


def _vendor_span_type(span: VendorSpan) -> str:
    native_type = span.get("type")
    if not isinstance(native_type, str):
        msg = "Liquid Space detector returned a malformed span"
        raise RuntimeError(msg)
    return native_type


class Lfm25PiiSpaceAdapter:
    """Typed ``PiiAdapter`` façade over one pinned Liquid Space detector."""

    name = "lfm25-pii"
    supported_labels = LFM25_PII_SUPPORTED_LABELS

    def __init__(
        self,
        detector: SpacePiiDetector,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self._detector = detector
        self._on_progress = on_progress

    # reason: `adapter.py:14` declares `load(self)` on the adapter Protocol and six sibling adapters
    # reason: implement it; this one is a no-op only because its detector arrives already loaded.
    def load(self) -> None:  # ruff: ignore[no-self-use]
        return None

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        predictions: list[list[CharSpan]] = []
        total = len(texts)
        for index, text in enumerate(texts, start=1):
            output = self._detector.detect(text)
            if not is_str_mapping(output):
                msg = "Liquid Space detector returned a malformed response"
                raise RuntimeError(msg)
            predictions.append(map_hybrid_spans(text, _vendor_spans(output)))
            if self._on_progress is not None:
                self._on_progress(index, total)
        return predictions

    def predict_views(self, texts: list[str]) -> list[DualViewPrediction]:
        """Return model-only and vendor-hybrid outputs from one forward per text.

        Returns:
            Both views per text: the model's own spans and the vendor's hybrid spans, folded
            through the same label map. Both come from a single forward pass under one lock, so
            the two views describe the same inference rather than two runs that could diverge.

        """
        detector = cast("_DualViewSpacePiiDetector", self._detector)
        predictions: list[DualViewPrediction] = []
        total = len(texts)
        for index, text in enumerate(texts, start=1):
            with detector.lock:
                model_spans = detector.hd.model_spans(text, detector.tok, detector.model)
                hybrid = detector.hd.hybrid_spans(text, model_spans)
            core = map_hybrid_spans(text, _vendor_spans({"spans": model_spans}))
            vendor = map_hybrid_spans(
                text,
                _space_final_spans(
                    hybrid,
                    auth_priority_types=detector.auth_priority_types,
                ),
            )
            predictions.append(DualViewPrediction(tuple(core), tuple(vendor)))
            if self._on_progress is not None:
                self._on_progress(index, total)
        return predictions
