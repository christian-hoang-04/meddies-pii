"""Eval adapter for the LFM2.5-350M BIOES spike checkpoint.

This is the first init of the own-architecture PII tagger (150 steps, checkpoint
``bioes/20260604_h100_8192_r128a256_pack_bs128_150step_ckpt10/unsloth``).

This is an ENCODER token-classifier: a frozen LFM2.5-350M backbone + a LoRA ``backbone_adapter``
+ a ``classifier.pt`` BIOES head, all trained together. The adapter reassembles the
three pieces (mirroring the proven load in ``scripts/ops/run_baseline_eval.py``),
then ``predict()`` tokenizes raw text with char offsets, forwards once per document,
Viterbi-decodes the BIOES tags under transition constraints, and maps them back to
char spans via ``decode_bioes_from_offsets``.

No Triton or custom kernels are required. The native ``transformers`` loader restores
the exact base-model revision, then PEFT restores the saved LoRA adapter from the
mounted Volume. The classifier head is loaded separately from ``classifier.pt``.
The checkpoint directory is a constructor argument, not a hardcoded path.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: adapters load their model stack inside `load()`, so listing an adapter costs nothing.
# ruff: file-ignore[print]
# reason: per-shard telemetry goes to the run log so a degraded cell is visible rather than silent.
from typing import TYPE_CHECKING

from meddies_pii.annotations.bioes import (
    decode_bioes_from_offsets,
    tokenize_and_align,
    viterbi_decode_logits,
)
from meddies_pii.taxonomy import PII_LABEL_SET, PiiLabel

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from meddies_pii.spans import CharSpan
    from meddies_pii.training.bioes.data.artifacts import ProbeArtifacts

LFM_BIOES_SUPPORTED_LABELS: frozenset[PiiLabel] = PII_LABEL_SET
"""The spike classifier was trained on the full 9-label taxonomy.

Its ``ENTITY_LABELS`` vocab == ``PII_LABEL_SET``), so it can emit every label — the only baseline besides openmed that
covers ``private_url`` and ``secret``. The runtime vocab-size assert in ``_load_checkpoint_artifacts`` re-checks this on
load.

"""

MODEL_ID = "LiquidAI/LFM2.5-350M-Base"
MODEL_REVISION = "9960764e30892e01f29a6dc23df2533fcd8bd5ae"
MAX_SEQ_LENGTH = 8192
"""The checkpoint was trained at 8192-token packing.

Tokenize at the same length so the model sees documents exactly as it did in training. Docs longer than this are
head-truncated to the first window (see ``tokenize_and_align``); predict() logs the per-shard truncation rate so a
low-recall cell on long docs is visible, not silent.

"""


def spans_from_logits(
    text: str,
    offset_mapping: Sequence[tuple[int, int]],
    logits_row: torch.Tensor,
    id_to_label: dict[int, str],
) -> list[CharSpan]:
    """Viterbi-decode one document's ``[tokens, labels]`` logits into char spans.

    Factored out of ``predict()`` so the BIOES decode path is unit-testable on CPU
    with hand-built logits — no GPU, no checkpoint. Every returned span slices
    ``text[start:end]`` by construction (``decode_bioes_from_offsets``), so the
    ``surface == text[start:end]`` alignment invariant holds without a separate check.

    Returns:
        The decoded character spans for that document. Every span slices ``text[start:end]`` by
        construction, so the surface-alignment invariant holds without a separate assertion.
        An empty list means the Viterbi path assigned no entity, not that decoding failed.

    """
    pred_ids = viterbi_decode_logits(logits_row, id_to_label, offset_mapping)
    return list(decode_bioes_from_offsets(text, offset_mapping, pred_ids, id_to_label))


def _load_checkpoint_artifacts(
    checkpoint_dir: str,
    *,
    model_id: str,
) -> ProbeArtifacts:
    """Load the historical adapter with native Transformers + PEFT.

    Returns:
        The probe artifacts -- model, tokenizer and label mapping -- built from the checkpoint's
        own backbone adapter and classifier, loaded through native Transformers and PEFT rather
        than the training-time stack.

    Raises:
        FileNotFoundError: If the checkpoint holds no ``backbone_adapter`` directory or no
            ``classifier.pt``. Both are checked before anything loads, so a partial checkpoint
            names the missing path instead of failing later inside the loader.
        RuntimeError: If the artifacts cannot be assembled into a usable probe once loaded.

    """
    from pathlib import Path

    import torch

    from meddies_pii.training.bioes.data.artifacts import (
        build_native_checkpoint_artifacts,
    )

    checkpoint = Path(checkpoint_dir)
    adapter_dir = checkpoint / "backbone_adapter"
    classifier_path = checkpoint / "classifier.pt"
    if not adapter_dir.is_dir():
        msg = f"Missing adapter dir: {adapter_dir}"
        raise FileNotFoundError(msg)
    if not classifier_path.is_file():
        msg = f"Missing classifier: {classifier_path}"
        raise FileNotFoundError(msg)

    artifacts = build_native_checkpoint_artifacts(model_id, adapter_dir, model_revision=MODEL_REVISION)

    payload = torch.load(classifier_path, map_location="cpu", weights_only=True)
    saved = int(payload["num_labels"])
    head = int(artifacts.tagger.classifier.out_features)
    vocab = len(artifacts.label_vocab)
    if saved != head or saved != vocab:
        msg = f"num_labels mismatch: classifier.pt={saved} head={head} vocab={vocab}"
        raise RuntimeError(msg)
    classifier_state = {
        key: value.to(
            dtype=artifacts.tagger.classifier.weight.dtype,
            device=artifacts.tagger.classifier.weight.device,
        )
        for key, value in payload["classifier"].items()
    }
    artifacts.tagger.classifier.load_state_dict(classifier_state, strict=True)
    return artifacts


class LfmBioesAdapter:
    name = "lfm-bioes-spike"
    supported_labels = LFM_BIOES_SUPPORTED_LABELS

    def __init__(self, checkpoint_dir: str, *, max_seq_length: int = MAX_SEQ_LENGTH) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.max_seq_length = max_seq_length
        self._artifacts: ProbeArtifacts | None = None
        self._id_to_label: dict[int, str] | None = None

    def load(self) -> None:
        if self._artifacts is not None:
            return
        artifacts = _load_checkpoint_artifacts(
            self.checkpoint_dir,
            model_id=MODEL_ID,
        )
        artifacts.tagger.eval()
        self._artifacts = artifacts
        self._id_to_label = {index: label for label, index in artifacts.label_to_id.items()}

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        artifacts = self._artifacts
        id_to_label = self._id_to_label
        if artifacts is None or id_to_label is None:
            msg = "LfmBioesAdapter.load() must be called before predict()"
            raise RuntimeError(msg)

        import torch

        tokenizer = artifacts.tokenizer
        tagger = artifacts.tagger
        device = str(next(tagger.parameters()).device)

        predictions: list[list[CharSpan]] = []
        truncated = 0
        with torch.no_grad():
            for text in texts:
                example = tokenize_and_align(tokenizer, text, [], max_length=self.max_seq_length)
                if example.truncated:
                    truncated += 1
                input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
                attention_mask = torch.tensor([example.attention_mask], dtype=torch.long, device=device)
                logits = tagger(input_ids=input_ids, attention_mask=attention_mask)["logits"]
                predictions.append(spans_from_logits(text, example.offset_mapping, logits[0], id_to_label))

        if truncated:
            print(
                f"LFM_BIOES_TRUNCATED::docs={len(texts)} truncated={truncated} rate={truncated / len(texts):.3f}",
                flush=True,
            )
        return predictions
