"""Release payload construction for the verified step-250 PII350 checkpoint.

The release claim is decode parity with the benchmark. The published inference
path therefore does not reimplement BIOES decoding: this module vendors the
exact decode sources the evaluation adapter imports, rewriting only their import
prefixes so the payload stands alone, and proves behaviour with a span-level
parity probe against the adapter itself.

Everything here is CPU-only and free of Modal wiring, so the packaging steps are
importable and testable on their own. ``scripts/ops/export_pii350_release.py``
holds the Modal app that drives them.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the sibling is imported inside the call so a test patching it on its own module is seen; binding it at
# reason: import time would bypass that seam.
# ruff: file-ignore[print]
# reason: the upload receipt is emitted on standard output for the operator running the release.
# ruff: file-ignore[hardcoded-sql-expression]
# reason: one f-string renders standalone Python module source; this file neither constructs nor executes SQL.
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from meddies_pii.evaluation.identity import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    is_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from huggingface_hub import HfApi
    from torch import Tensor

    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import VerifiedCheckpoint
    from meddies_pii.spans import CharSpan

GIT_SHA_HEX_LENGTH = 40

RELEASE_MODEL_REPO = "Meddies/meddies-pii-v2"


RELEASE_ONNX_REPO = "Meddies/meddies-pii-v2-onnx"


RELEASE_REPO_TYPE = "model"
"""The identically named dataset repo is unrelated release material.

Every upload and download here pins repo_type="model" so the two can never be confused.

"""


RELEASE_PRIVATE = True


RELEASE_SCHEMA_VERSION = 1


RELEASE_MANIFEST_FILENAME = "manifest.json"


class _ReleaseUploadApi(Protocol):
    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> object: ...


@runtime_checkable
class _PublishedExtractor(Protocol):
    def extract(self, text: str) -> Sequence[CharSpan]: ...


@runtime_checkable
class _PublishedExtractorFactory(Protocol):
    def from_pretrained(
        self,
        path: str,
        /,
        *,
        device: str,
        weights: str,
    ) -> _PublishedExtractor: ...


STEP250_REPO_ID = "HoangHa/pii350-trajectories-private"


STEP250_REVISION = "ceef989794477051a6c17b06d75e17c3b442041c"


STEP250_ARTIFACT_PATH = (
    "trajectories/542478f63b1726aa863c9a0ecec1e84f6347855fa9b6776b451b40c37781dac4/checkpoints/step-00000250"
)


STEP250_CHECKPOINT_DIGEST = "b175a02edf0fac399f6713980f2b22aaa95f5ae96371f02e392d152c78840d70"


RUNTIME_DTYPE = "bfloat16"


RELEASE_KIND_TORCH = "adapter_form_release"


RELEASE_KIND_ONNX = "adapter_form_onnx"


ONNX_OPSET = 17


ONNX_FP32_FILENAME = "model.onnx"


ONNX_INT8_FILENAME = "model.int8.onnx"


POSTPROCESS_FILENAME = "meddies_pii_postprocess.py"


USAGE_SNIPPET_FILENAME = "usage_onnxruntime.py"


REMOTE_CODE_FILENAME = "modeling_meddies_pii.py"


MERGED_WEIGHTS_FILENAME = "model.safetensors"


MERGED_WEIGHTS_FORMAT = "meddies-pii-v2-merged"


MERGED_ENCODER_PREFIX = "encoder."


MERGED_CLASSIFIER_PREFIX = "classifier."


LICENSE_FILENAME = "LICENSE"
"""The base licence permits redistributing a derivative work on three conditions.

The licence travels with it, the changes are stated prominently, and the attribution stays. The merged weights are a
derivative, so all three ship.

"""


CHANGE_NOTICE = (
    "LoRA adapter (r128) and a 37-tag BIOES token-classification head trained by "
    "Meddies were applied to the base encoder; merged weights bake the adapter "
    "into the base."
)


VENDORED_DECODE_PACKAGE = "meddies_pii_decode"


PARITY_ROW_MINIMUM = 200


PARITY_FIXTURE_CELL = "v2-eval"


VENDORED_DECODE_MODULES: tuple[tuple[str, str], ...] = (
    ("meddies_pii/taxonomy.py", "taxonomy.py"),
    ("meddies_pii/spans.py", "char_spans.py"),
    ("meddies_pii/annotations/bioes/vocabulary.py", "vocabulary.py"),
    ("meddies_pii/annotations/bioes/viterbi.py", "viterbi.py"),
    ("meddies_pii/annotations/bioes/spans.py", "bioes_spans.py"),
)
"""The decode closure the evaluation adapter uses, mapped to its vendored name.

Nothing here imports a third-party package at module scope, which is what lets the payload ship the same source without
dragging in the training stack.

"""


VENDORED_IMPORT_REWRITES: tuple[tuple[str, str], ...] = (
    (
        "from meddies_pii.annotations.bioes.vocabulary import",
        f"from {VENDORED_DECODE_PACKAGE}.vocabulary import",
    ),
    (
        "from meddies_pii.taxonomy import",
        f"from {VENDORED_DECODE_PACKAGE}.taxonomy import",
    ),
    (
        "from meddies_pii.spans import",
        f"from {VENDORED_DECODE_PACKAGE}.char_spans import",
    ),
)
"""Applied in order; the only edit the vendoring performs."""


def bioes_label_map() -> dict[int, str]:
    """Return the pinned 37-tag BIOES vocabulary as an index to tag mapping.

    Returns:
        Index to tag for the whole vocabulary, in the order the label space builds it. That
        order IS the published id2label, so a reordering here silently repoints every
        released logit.

    """
    from meddies_pii.annotations.bioes import ENTITY_LABELS, build_bioes_label_space

    vocabulary = build_bioes_label_space(ENTITY_LABELS)
    return dict(enumerate(vocabulary))


def entity_label_for_tag(tag: str) -> str | None:
    """Map one BIOES tag back to its fixed-nine Meddies label, or None for O.

    Returns:
        The entity label after the prefix, or None for the outside tag ``O``.

    Raises:
        ValueError: If the tag is neither ``O`` nor ``<B|I|E|S>-<label>`` with a non-empty
            label. An unrecognised tag is refused rather than mapped to None, which would
            silently drop it from the entity set the release publishes.

    """
    if tag == "O":
        return None
    prefix, separator, label = tag.partition("-")
    if separator != "-" or prefix not in {"B", "I", "E", "S"} or not label:
        msg = f"tag is not a BIOES label: {tag}"
        raise ValueError(msg)
    return label


def release_label_config() -> dict[str, Any]:
    """Build the label block the published config and remote code both read.

    Returns:
        ``id2label``, ``label2id``, the sorted entity labels and the label count -- the four
        fields the published config carries and the remote code reads back.

    Raises:
        RuntimeError: If the entity labels recovered from the tag set are not exactly the
            fixed nine, in order. This is a round-trip check: it fails a release whose
            vocabulary drifted from the label space rather than publishing the drift.

    """
    from meddies_pii.annotations.bioes import ENTITY_LABELS

    id_to_tag = bioes_label_map()
    entities = sorted({entity for tag in id_to_tag.values() if (entity := entity_label_for_tag(tag)) is not None})
    if tuple(entities) != tuple(ENTITY_LABELS):
        msg = "BIOES tag set does not round trip to the fixed-nine labels"
        raise RuntimeError(msg)
    return {
        "id2label": {str(index): tag for index, tag in id_to_tag.items()},
        "label2id": {tag: index for index, tag in id_to_tag.items()},
        "entity_labels": entities,
        "num_labels": len(id_to_tag),
    }


def package_source_root() -> Path:
    import meddies_pii

    package_file = meddies_pii.__file__
    if package_file is None:
        msg = "cannot locate the meddies_pii source root"
        raise RuntimeError(msg)
    return Path(package_file).resolve().parent.parent


def rewrite_vendored_source(text: str) -> str:
    for original, replacement in VENDORED_IMPORT_REWRITES:
        text = text.replace(original, replacement)
    return text


def vendored_decode_payload(source_root: Path | None = None) -> dict[str, bytes]:
    """Copy the adapter's decode closure into a standalone package payload.

    The only edit is the fixed import-prefix substitution above, so the shipped
    decode stays the benchmark's decode rather than a parallel implementation.

    Returns:
        The vendored package as a path-to-bytes mapping, ready to write: a generated
        ``__init__.py`` plus each listed module with its import prefix rewritten.

    Raises:
        RuntimeError: If any decode source cannot be read, chained from the underlying
            OSError. A release that silently shipped a short vendored package would decode
            differently from the benchmark it claims parity with.

    """
    root = source_root if source_root is not None else package_source_root()
    payload: dict[str, bytes] = {
        f"{VENDORED_DECODE_PACKAGE}/__init__.py": b'"""Vendored Meddies BIOES decode, byte-for-byte '
        b'from the evaluated source."""\n',
    }
    for relative, vendored_name in VENDORED_DECODE_MODULES:
        source = root / relative
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as error:
            msg = f"decode source is unavailable: {relative}"
            raise RuntimeError(msg) from error
        payload[f"{VENDORED_DECODE_PACKAGE}/{vendored_name}"] = rewrite_vendored_source(text).encode("utf-8")
    return payload


def decode_spans_from_logits(
    text: str,
    logits: Tensor,
    offsets: Sequence[tuple[int, int]],
    id_to_label: Mapping[int, str],
) -> list[CharSpan]:
    """Decode one row exactly as the evaluation adapter does.

    Returns:
        The decoded spans for the row, produced by the same offset decode the adapter calls,
        so a parity comparison against it measures the weights rather than two decoders.

    """
    from meddies_pii.annotations.bioes import (
        decode_bioes_from_offsets,
        viterbi_decode_logits,
    )

    mapping = dict(id_to_label)
    tags = viterbi_decode_logits(logits, mapping, offsets)
    return list(decode_bioes_from_offsets(text, offsets, tags, mapping))


def required_payload_files() -> tuple[str, ...]:
    """Name every file the torch payload must hold before it may be published.

    Tokenizer files are excluded because their names come from the upstream
    tokenizer; everything here is a file this script is responsible for writing.

    Returns:
        Every path the payload must contain: the adapter config and weights, the classifier,
        ``config.json``, the licence, the merged weights, the remote code module, and each
        vendored decode file. This tuple is the contract ``verify_payload_complete`` checks.

    """
    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        ADAPTER_CONFIG_FILENAME,
        ADAPTER_WEIGHTS_FILENAME,
        CLASSIFIER_FILENAME,
    )

    return (
        ADAPTER_CONFIG_FILENAME,
        ADAPTER_WEIGHTS_FILENAME,
        CLASSIFIER_FILENAME,
        "config.json",
        LICENSE_FILENAME,
        MERGED_WEIGHTS_FILENAME,
        REMOTE_CODE_FILENAME,
        *sorted(vendored_decode_payload()),
    )


def verify_payload_complete(root: Path) -> tuple[str, ...]:
    """Refuse to publish a payload that lost a file it is contracted to carry.

    Returns:
        The verified plan -- the same tuple ``required_payload_files`` names, returned only
        once every entry exists on disk.

    Raises:
        RuntimeError: If any contracted file is absent, naming all of them. It checks
            presence only; the bytes are verified at download time by
            ``verify_manifest_tree``.

    """
    plan = required_payload_files()
    missing = [relative for relative in plan if not (root / relative).is_file()]
    if missing:
        msg = f"release payload is missing required files: {missing}"
        raise RuntimeError(msg)
    return plan


@dataclass(frozen=True, slots=True)
class UploadedFile:
    path_in_repo: str
    sha256: str
    bytes: int


def release_manifest(files: Sequence[UploadedFile], *, kind: str) -> dict[str, Any]:
    body = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "kind": kind,
        "checkpoint_digest": STEP250_CHECKPOINT_DIGEST,
        "source_repo_id": STEP250_REPO_ID,
        "source_revision": STEP250_REVISION,
        "files": [
            {"path": entry.path_in_repo, "sha256": entry.sha256, "bytes": entry.bytes}
            for entry in sorted(files, key=lambda item: item.path_in_repo)
        ],
        "total_bytes": sum(entry.bytes for entry in files),
    }
    return {**body, "manifest_digest": canonical_sha256(body)}


def hf_api() -> HfApi:
    from huggingface_hub import HfApi

    return HfApi()


def upload_release_tree(root: Path, *, repo_id: str, kind: str, api: _ReleaseUploadApi | None = None) -> dict[str, Any]:
    """Upload every payload file, then the manifest, and return the receipt.

    The manifest commits last so a partially uploaded release cannot be read as
    a complete one; its commit is the only revision worth importing.

    Returns:
        The receipt: repository, kind, the immutable commit SHA of the manifest upload, and
        the manifest itself with a digest over every uploaded file's path, SHA-256 and size.
        It is also printed as a single ``PII350_RELEASE_UPLOAD::`` line so the operator has
        the receipt even if the caller drops the return value.

    Raises:
        RuntimeError: If the payload directory holds no files to upload, or if the manifest
            upload returns no immutable commit SHA -- checked as 40 lowercase hexadecimal
            characters, because the receipt's whole value is that the revision it names
            cannot move.

    """
    client = api if api is not None else hf_api()
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != RELEASE_MANIFEST_FILENAME)
    if not paths:
        msg = "release payload is empty"
        raise RuntimeError(msg)
    uploaded: list[UploadedFile] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        client.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=relative,
            repo_id=repo_id,
            repo_type=RELEASE_REPO_TYPE,
            commit_message=f"meddies-pii-v2 release file {relative}",
        )
        uploaded.append(UploadedFile(relative, file_sha256(str(path)), path.stat().st_size))
    manifest = release_manifest(uploaded, kind=kind)
    commit = client.upload_file(
        path_or_fileobj=canonical_json_bytes(manifest),
        path_in_repo=RELEASE_MANIFEST_FILENAME,
        repo_id=repo_id,
        repo_type=RELEASE_REPO_TYPE,
        commit_message=f"meddies-pii-v2 release manifest {manifest['manifest_digest']}",
    )
    oid = getattr(commit, "oid", None)
    if (
        not isinstance(oid, str)
        or len(oid) != GIT_SHA_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in oid)
    ):
        msg = "release upload did not return an immutable commit SHA"
        raise RuntimeError(msg)
    receipt = {
        "repo_id": repo_id,
        "repo_type": RELEASE_REPO_TYPE,
        "private": RELEASE_PRIVATE,
        "kind": kind,
        "hf_commit_sha": oid,
        "manifest": manifest,
    }
    print(f"PII350_RELEASE_UPLOAD::{json.dumps(receipt, sort_keys=True)}", flush=True)
    return receipt


def verified_checkpoint_root(cache_root: Path) -> VerifiedCheckpoint:
    """Verify every checkpoint byte before any deserializer runs.

    Returns:
        The verified checkpoint root. The ordering is the point: nothing is deserialized
        until the artifact's bytes have been checked against the pinned digest.

    """
    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        CheckpointArtifact,
        verify_checkpoint_artifact,
    )

    artifact = CheckpointArtifact(
        STEP250_REPO_ID,
        STEP250_REVISION,
        STEP250_ARTIFACT_PATH,
        STEP250_CHECKPOINT_DIGEST,
    )
    return verify_checkpoint_artifact(cache_root / STEP250_ARTIFACT_PATH, artifact)


def parity_rows(limit: int) -> list[Any]:
    """Take a language-spread slice rather than the head, which is single-language.

    Returns:
        Exactly ``limit`` rows, taken at a fixed stride through the fixture so the slice
        spans languages. Taking the head would measure parity on one language only.

    Raises:
        RuntimeError: If the fixture holds fewer rows than the limit, or if the strided
            selection still falls short of it. The parity probe is refused rather than run
            on a smaller sample than the release claims.

    """
    from meddies_pii.eval_baseline.baseline.datasets import (
        EXTERNAL_DATASET_REVISION,
        V2_DATASET_REVISION,
        load_eval_cell,
    )

    rows = load_eval_cell(
        PARITY_FIXTURE_CELL,
        v2_limit=None,
        external_limit=None,
        v2_revision=V2_DATASET_REVISION,
        external_revision=EXTERNAL_DATASET_REVISION,
    )
    if len(rows) < limit:
        msg = f"parity fixture has {len(rows)} rows, fewer than the required {limit}"
        raise RuntimeError(msg)
    step = max(1, len(rows) // limit)
    selected = [rows[index] for index in range(0, len(rows), step)][:limit]
    if len(selected) < limit:
        msg = "parity row selection did not reach the required minimum"
        raise RuntimeError(msg)
    return selected


def span_key(span: CharSpan) -> tuple[int, int, str]:
    return (int(span.start), int(span.end), str(span.label))


def compare_span_sets(reference: Sequence[Sequence[Any]], candidate: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Compare decoded spans row by row; equality is the release gate.

    Returns:
        The row count, the number of mismatched rows, the first ten mismatches with their
        expected and observed spans, and ``identical``. Spans are compared as sorted
        ``(start, end, label)`` triples, so ordering differences are not mismatches but any
        boundary or label difference is. The mismatch list is truncated; the count is not.

    Raises:
        RuntimeError: If the two sides have different row counts, which would make a
            per-row comparison meaningless rather than merely unequal.

    """
    if len(reference) != len(candidate):
        msg = "parity probe compared a different number of rows"
        raise RuntimeError(msg)
    mismatches: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        expected = sorted(span_key(span) for span in left)
        observed = sorted(span_key(span) for span in right)
        if expected != observed:
            mismatches.append({
                "row": index,
                "expected": [list(key) for key in expected],
                "observed": [list(key) for key in observed],
            })
    return {
        "rows": len(reference),
        "mismatched_rows": len(mismatches),
        "mismatches": mismatches[:10],
        "identical": not mismatches,
    }


def release_parity_record(parity: Mapping[str, Any]) -> dict[str, Any]:
    """Project the parity probe onto the fields the published config carries.

    Both deltas are read without a default: a published release that quietly
    omitted one would read as a clean parity claim it never made.

    Returns:
        The published parity block: row count, reference shape, and the three mismatch
        counts. Every field is read without a default, so a missing one raises here rather
        than being published as a zero.

    """
    return {
        "rows": int(parity["rows"]),
        "reference_shape": "batch_of_1",
        "mismatched_rows": int(parity["mismatched_rows"]),
        "batched_reference_mismatched_rows": int(parity["batched_reference_mismatched_rows"]),
        "merged_mismatched_rows": int(parity["merged_mismatched_rows"]),
    }


def remote_code_module_source() -> str:
    return f'''"""Meddies PII v2 inference with the evaluated BIOES decode."""

from __future__ import annotations

import sys
from pathlib import Path

import json

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from {VENDORED_DECODE_PACKAGE}.bioes_spans import decode_bioes_from_offsets
from {VENDORED_DECODE_PACKAGE}.viterbi import viterbi_decode_logits

MAX_SEQUENCE_LENGTH = 8192
_DTYPES = {{"bfloat16": torch.bfloat16, "float32": torch.float32}}
_WEIGHT_FORMS = ("adapter", "merged")


def _namespace(state, prefix):
    """Select one prefixed namespace out of the merged weights file."""
    return {{
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }}


class MeddiesPiiExtractor:
    """The step-250 BIOES PII extractor, in either published weight form.

    ``weights="adapter"`` is the default and the benchmarked path: the shipped
    adapter and head bytes are verbatim copies of the evaluated checkpoint, and
    the release gate proved this loader decodes the benchmark's spans exactly.
    It downloads the base weights and applies the adapter live, never merging.

    ``weights="merged"`` is a convenience form. It reads one self-contained
    {MERGED_WEIGHTS_FILENAME} whose encoder already has W+BA baked in, so it
    needs no PEFT and no base weight download. Baking rounds the LoRA delta, so
    this form is recorded rather than gated: its span delta against the adapter
    form is published in config.json at
    meddies_release.parity.merged_mismatched_rows.

    The forward pass mirrors HiddenStateTokenTagger.forward from
    src/meddies_pii/training/bioes/data/tagger.py, which is the module the
    evaluation adapter uses.
    """

    def __init__(self, backbone, classifier, tokenizer, id_to_label):
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = torch.nn.Dropout(0.1).eval()
        self.tokenizer = tokenizer
        self.id_to_label = id_to_label

    @classmethod
    def from_pretrained(cls, path, dtype=None, device="cpu", weights="adapter"):
        if weights not in _WEIGHT_FORMS:
            raise ValueError(
                f"weights must be one of {{_WEIGHT_FORMS}}, not {{weights!r}}"
            )
        root = Path(path).resolve()
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        release = config["meddies_release"]
        torch_dtype = _DTYPES[dtype or release["runtime_dtype"]]
        if weights == "merged":
            # from_config builds the skeleton from the base config and remote code
            # alone. from_pretrained here would download the full base weights that
            # the merged file already contains.
            wrapper = AutoModelForTokenClassification.from_config(
                AutoConfig.from_pretrained(
                    release["base_model_id"],
                    revision=release["base_model_revision"],
                    trust_remote_code=True,
                ),
                trust_remote_code=True,
            )
            body = getattr(wrapper, "lfm2", None)
            if body is None:
                raise RuntimeError("base wrapper does not expose its lfm2 encoder body")
            merged = load_file(str(root / "{MERGED_WEIGHTS_FILENAME}"))
            body.load_state_dict(
                _namespace(merged, "{MERGED_ENCODER_PREFIX}"), strict=True
            )
            backbone = body.to(device=device, dtype=torch_dtype).eval()
            head = _namespace(merged, "{MERGED_CLASSIFIER_PREFIX}")
            classifier = torch.nn.Linear(
                head["weight"].shape[1], head["weight"].shape[0]
            )
            classifier.load_state_dict(head, strict=True)
        else:
            wrapper = AutoModelForTokenClassification.from_pretrained(
                release["base_model_id"],
                revision=release["base_model_revision"],
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            body = getattr(wrapper, "lfm2", None)
            if body is None:
                raise RuntimeError("base wrapper does not expose its lfm2 encoder body")
            backbone = (
                PeftModel.from_pretrained(
                    body, str(root / "adapter"), is_trainable=False
                )
                # The full cast, dtype included: the evaluated path casts every
                # parameter AND floating buffer to bf16, and a device-only move
                # leaves buffers like rotary inv_freq in fp32 — measured as
                # 8/200 span flips against the reference.
                .to(device=device, dtype=torch_dtype)
                .eval()
            )
            classifier = torch.nn.Linear(
                config["hidden_size"] if "hidden_size" in config else 1024,
                config["num_labels"],
            )
            state = torch.load(
                root / "classifier.pt", map_location="cpu", weights_only=True
            )
            classifier.load_state_dict(state, strict=True)
        classifier = classifier.to(device=device, dtype=torch_dtype).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
        id_to_label = {{
            int(key): value for key, value in config["id2label"].items()
        }}
        return cls(backbone, classifier, tokenizer, id_to_label)

    def logits(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state
        target = self.classifier.weight.dtype
        if getattr(hidden, "dtype", None) != target:
            hidden = hidden.to(target)
        return self.classifier(self.dropout(hidden))

    def extract(self, text):
        """Return the labeled character spans this model finds in one string."""
        encoded = self.tokenizer(
            [text],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        attention = encoded["attention_mask"]
        device = self.classifier.weight.device
        with torch.inference_mode():
            logits = self.logits(encoded["input_ids"].to(device), attention.to(device))
        active = int(attention[0].sum().item())
        row_offsets = [
            (int(pair[0]), int(pair[1])) for pair in offsets[0][:active].tolist()
        ]
        tags = viterbi_decode_logits(
            logits[0][:active], self.id_to_label, row_offsets
        )
        return list(
            decode_bioes_from_offsets(text, row_offsets, tags, self.id_to_label)
        )


def extract(text, model):
    """Convenience wrapper matching the documented one-call entry point."""
    return model.extract(text)
'''


def postprocess_module_source() -> str:
    return f'''"""Decode Meddies PII v2 ONNX logits with the evaluated BIOES decode."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from {VENDORED_DECODE_PACKAGE}.bioes_spans import decode_bioes_from_offsets
from {VENDORED_DECODE_PACKAGE}.viterbi import viterbi_decode_logits


def decode(text, logits, offsets, id_to_label):
    """Turn one row of ONNX logits into labeled character spans."""
    import torch

    mapping = {{int(key): value for key, value in id_to_label.items()}}
    tensor = logits if hasattr(logits, "dim") else torch.as_tensor(logits)
    pairs = [(int(start), int(end)) for start, end in offsets]
    tags = viterbi_decode_logits(tensor, mapping, pairs)
    return list(decode_bioes_from_offsets(text, pairs, tags, mapping))
'''


def usage_snippet_source() -> str:
    return f'''"""Run Meddies PII v2 with onnxruntime."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from {POSTPROCESS_FILENAME[:-3]} import decode

HERE = Path(__file__).resolve().parent
labels = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))
tokenizer = AutoTokenizer.from_pretrained(str(HERE))
session = ort.InferenceSession(str(HERE / "{ONNX_FP32_FILENAME}"))


def extract(text):
    encoded = tokenizer(
        [text], return_offsets_mapping=True, truncation=True, max_length=8192
    )
    offsets = encoded["offset_mapping"][0]
    logits = session.run(
        None,
        {{
            "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
        }},
    )[0]
    return decode(text, logits[0][: len(offsets)], offsets, labels["id2label"])
'''


def published_merged_spans(loader_module: object, model_root: Path, texts: Sequence[str]) -> list[list[CharSpan]]:
    """Decode through the published loader's merged branch.

    Nothing else in the pipeline executes this branch: packaging measures the
    merged delta through its own modules, and the adapter default never reaches
    from_config or the prefixed state dict. dtype is left to the published config
    so the arithmetic matches the packaging-time measurement.

    Returns:
        The spans each text decodes to through the published merged loader, one list per text
        in input order. This is the only caller of that branch, so these lists are the sole
        evidence the published merged weights decode as packaging measured them.

    Raises:
        TypeError: If the dynamically loaded release module does not expose the extractor
            construction and extraction interface that the published loader promises.

    """
    extractor_factory = getattr(loader_module, "MeddiesPiiExtractor", None)
    if not isinstance(extractor_factory, _PublishedExtractorFactory):
        msg = "published loader does not expose a compatible MeddiesPiiExtractor"
        raise TypeError(msg)
    merged = extractor_factory.from_pretrained(str(model_root), device="cuda", weights="merged")
    return [list(merged.extract(text)) for text in texts]


def reconcile_merged_parity(config: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the published merged weights reproduce the recorded span delta.

    Both counts compare merged against adapter form at the same shape, dtype, and
    device, so the arithmetic is deterministic. An unequal count means the
    published merged bytes are not the ones packaging measured, which fails the
    release closed rather than shipping an unmeasured artifact.

    Returns:
        The row count with the expected and observed mismatch counts, as the evidence behind
        the pass rather than a bare verdict.

    Raises:
        RuntimeError: If the observed mismatch count differs from the one the published
            config records, reporting both. It compares against the config's own recorded
            number, so a release cannot pass by measuring itself afresh.

    """
    expected = int(config["meddies_release"]["parity"]["merged_mismatched_rows"])
    observed = int(comparison["mismatched_rows"])
    if expected != observed:
        msg = (
            "published merged weights do not reproduce the recorded span delta: "
            f"config records {expected} mismatched rows, the published merged "
            f"loader decodes {observed} of {comparison['rows']}"
        )
        raise RuntimeError(msg)
    return {
        "rows": int(comparison["rows"]),
        "expected_mismatched_rows": expected,
        "observed_mismatched_rows": observed,
    }


def verify_manifest_tree(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Check every manifest entry against the downloaded bytes.

    Returns:
        Every verified path, in manifest order.

    Raises:
        RuntimeError: If the manifest's recorded digest does not match a digest recomputed
            over its own body with that field removed -- so a tampered manifest is caught
            before any file it lists is trusted; if a listed file is absent; or if an
            entry's recorded hash is not a well-formed SHA-256, or the file's bytes do not
            hash to it. Note the scope: this verifies every file the manifest LISTS, and
            does not reject an unlisted extra file in the directory.

    """
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if manifest.get("manifest_digest") != canonical_sha256(body):
        msg = "release manifest digest does not match its body"
        raise RuntimeError(msg)
    verified: list[str] = []
    for entry in manifest["files"]:
        relative = str(entry["path"])
        path = root / relative
        if not path.is_file():
            msg = f"release file is absent: {relative}"
            raise RuntimeError(msg)
        if not is_sha256(entry.get("sha256")) or file_sha256(str(path)) != entry["sha256"]:
            msg = f"release file failed hash verification: {relative}"
            raise RuntimeError(msg)
        verified.append(relative)
    return verified
