from __future__ import annotations

# ruff: file-ignore[no-self-use]
# reason: the stateless load double retains the bound detector method shape required by the paired runner.
import json
from typing import TYPE_CHECKING

import pytest

from meddies_pii.eval_baseline.baseline.datasets import EvalRow
from meddies_pii.eval_baseline.baseline.run import (
    ShardSpec,
    shard_output_path,
)
from meddies_pii.eval_baseline.baseline.views import (
    MODEL_CORE_VIEW,
    MODEL_PLUS_REGEX_VIEW,
)
from meddies_pii.eval_baseline.regex_release.paired_evaluation import (
    paired_regex_document_counts,
    run_regex_view_shards,
)
from meddies_pii.eval_baseline.regex_release.regex_bootstrap import source_row_sha256
from meddies_pii.eval_baseline.regex_release.regex_ignore_list import ignore_list_from_rows
from meddies_pii.eval_baseline.regex_release.regex_report import RegexCorpusManifest
from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    EvaluationContract,
    dataset_shard_identity,
    file_sha256,
)
from meddies_pii.spans import CharSpan, char_span_to_dict
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from pathlib import Path


class _CountingAdapter:
    name = "counting"
    supported_labels = PII_LABEL_SET

    def __init__(self) -> None:
        self.predict_calls: list[list[str]] = []

    def load(self) -> None:
        return None

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        self.predict_calls.append(texts)
        return [[CharSpan(0, 7, text[:7], "human_name")] for text in texts]


class _Volume:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


def _contract(view: str) -> EvaluationContract:
    def artifact(name: str, digit: str) -> ArtifactIdentity:
        return ArtifactIdentity(name, "f" * 40, digit * 64)

    return EvaluationContract(
        model=artifact("model", "a"),
        vendor_inference_source=artifact("vendor", "b"),
        local_adapter_source=artifact("adapter", "c"),
        applied_label_prediction_contract=artifact(f"labels-{view}", "d"),
        decoder_contract=artifact(f"decoder-{view}", "e"),
        resolved_runtime_environment=artifact("runtime", "f"),
        scorer_contract=artifact("scorer", "0"),
        supported_labels=tuple(sorted(PII_LABEL_SET)),
        result_schema=(
            "id",
            "doc_id",
            "pred_spans",
            "gold_spans",
            "language",
            "slice",
            "length_bucket",
            "document_chars",
            "evaluation_view",
            "shared_inference_identity",
            "regex_manifest_sha256",
            "comparison_id",
            "source_row_sha256",
            "corpus_manifest_sha256",
        ),
    )


# reason: one vertical test keeps setup, execution, and both persisted-view assertions in one readable proof.
def test_regex_views_are_persisted_from_one_model_forward(  # ruff: ignore[too-many-locals]
    tmp_path: Path,
) -> None:
    adapter = _CountingAdapter()
    volume = _Volume()
    rows = [
        EvalRow(
            doc_id="synthetic-1",
            dataset="smoke",
            shard="full",
            text="Example alpha@example.invalid",
            gold_spans=(),
            language="en",
            slices=frozenset({"synthetic"}),
        ),
    ]
    specs = {
        MODEL_CORE_VIEW: ShardSpec("model", "smoke", "full", MODEL_CORE_VIEW),
        MODEL_PLUS_REGEX_VIEW: ShardSpec("model", "smoke", "full", MODEL_PLUS_REGEX_VIEW),
    }
    row_source_sha256 = source_row_sha256(
        stable_id=rows[0].stable_id,
        text_sha256=rows[0].text_sha256,
        gold_spans=rows[0].gold_spans,
        language=rows[0].language,
        slices=tuple(sorted(rows[0].slices)),
    )
    fixture_sha256 = dataset_shard_identity(
        _contract(MODEL_CORE_VIEW),
        dataset="smoke",
        shard="full",
        rows=rows,
    ).fixture_identity.digest
    corpus_manifest = RegexCorpusManifest.create(
        corpus_id="synthetic-smoke",
        dataset="smoke",
        dataset_revision="synthetic-revision",
        shard="full",
        fixture_sha256=fixture_sha256,
        source_row_sha256s=(row_source_sha256,),
        historically_exposed=True,
        blinded=False,
        preregistered_strata=(("en", "short"),),
    )

    result = run_regex_view_shards(
        adapter=adapter,
        rows=rows,
        output_root=tmp_path,
        specs=specs,
        evaluation_contracts={view: _contract(view) for view in specs},
        volume=volume,
        comparison_id="regex-candidate",
        corpus_manifest=corpus_manifest,
    )

    assert adapter.predict_calls == [["Example alpha@example.invalid"]]
    assert volume.commits == 1
    assert set(result.by_view) == set(specs)
    persisted = {view: json.loads(shard.output_path.read_text(encoding="utf-8")) for view, shard in result.by_view.items()}
    assert (
        persisted[MODEL_CORE_VIEW]["shared_inference_identity"]
        == (persisted[MODEL_PLUS_REGEX_VIEW]["shared_inference_identity"])
    )
    assert len(persisted[MODEL_CORE_VIEW]["pred_spans"]) == 1
    assert len(persisted[MODEL_PLUS_REGEX_VIEW]["pred_spans"]) == 2
    assert persisted[MODEL_PLUS_REGEX_VIEW]["length_bucket"] == "short"
    assert persisted[MODEL_PLUS_REGEX_VIEW]["document_chars"] == len(rows[0].text)
    assert len(persisted[MODEL_PLUS_REGEX_VIEW]["regex_manifest_sha256"]) == 64
    assert persisted[MODEL_PLUS_REGEX_VIEW]["source_row_sha256"] == row_source_sha256
    assert persisted[MODEL_PLUS_REGEX_VIEW]["corpus_manifest_sha256"] == (corpus_manifest.sha256)
    regex_meta = json.loads(
        result.by_view[MODEL_PLUS_REGEX_VIEW].output_path.with_suffix(".meta.json").read_text(encoding="utf-8"),
    )
    assert (
        RegexCorpusManifest.from_payload(
            regex_meta["corpus_manifest"],
            expected_sha256=regex_meta["corpus_manifest_sha256"],
        )
        == corpus_manifest
    )
    tampered_payload = dict(regex_meta["corpus_manifest"])
    tampered_payload["historically_exposed"] = False
    with pytest.raises(ValueError, match="does not match its SHA-256"):
        RegexCorpusManifest.from_payload(
            tampered_payload,
            expected_sha256=regex_meta["corpus_manifest_sha256"],
        )
    for view, shard in result.by_view.items():
        assert shard.output_path != shard_output_path(tmp_path, specs[view])
        assert (f"/regex_ablation/regex-candidate/{persisted[view]['regex_manifest_sha256']}/{view}/") in str(
            shard.output_path,
        )
    counts = paired_regex_document_counts(
        (persisted[MODEL_CORE_VIEW],),
        (persisted[MODEL_PLUS_REGEX_VIEW],),
    )
    assert (counts[0].model_tp, counts[0].model_fp, counts[0].model_fn) == (0, 1, 0)
    assert (counts[0].regex_tp, counts[0].regex_fp, counts[0].regex_fn) == (0, 2, 0)
    assert counts[0].source_row_sha256 == row_source_sha256
    assert counts[0].corpus_manifest_sha256 == corpus_manifest.sha256

    for shard in result.by_view.values():
        row = json.loads(shard.output_path.read_text(encoding="utf-8"))
        row["regex_manifest_sha256"] = "f" * 64
        shard.output_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        meta_path = shard.output_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["regex_manifest_sha256"] = "f" * 64
        meta["result_sha256"] = file_sha256(str(shard.output_path))
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    rerun_adapter = _CountingAdapter()
    rerun = run_regex_view_shards(
        adapter=rerun_adapter,
        rows=rows,
        output_root=tmp_path,
        specs=specs,
        evaluation_contracts={view: _contract(view) for view in specs},
        volume=_Volume(),
        comparison_id="regex-candidate",
        corpus_manifest=corpus_manifest,
    )

    assert rerun_adapter.predict_calls == [["Example alpha@example.invalid"]]
    assert all(not shard.skipped for shard in rerun.by_view.values())


def test_paired_counting_rejects_a_stale_same_forward_identity() -> None:
    base = {
        "id": "a" * 64,
        "doc_id": "synthetic",
        "language": "en",
        "slice": [],
        "gold_spans": [],
        "pred_spans": [],
        "length_bucket": "short",
        "regex_manifest_sha256": "b" * 64,
        "comparison_id": "regex-candidate",
        "source_row_sha256": "e" * 64,
        "corpus_manifest_sha256": "f" * 64,
    }
    model_row = {
        **base,
        "evaluation_view": MODEL_CORE_VIEW,
        "shared_inference_identity": "c" * 64,
    }
    regex_row = {
        **base,
        "evaluation_view": MODEL_PLUS_REGEX_VIEW,
        "shared_inference_identity": "d" * 64,
    }

    with pytest.raises(ValueError, match="same-forward provenance"):
        paired_regex_document_counts((model_row,), (regex_row,))


def _paired_rows(
    *,
    doc_id: str,
    gold_spans: tuple[CharSpan, ...],
    model_pred_spans: tuple[CharSpan, ...],
    regex_pred_spans: tuple[CharSpan, ...],
    document_chars: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    text_sha256 = "a" * 64
    row_sha256 = source_row_sha256(
        stable_id=doc_id,
        text_sha256=text_sha256,
        gold_spans=gold_spans,
        language="vi",
        slices=(),
    )
    base: dict[str, object] = {
        "id": text_sha256,
        "doc_id": doc_id,
        "language": "vi",
        "slice": [],
        "gold_spans": [char_span_to_dict(span) for span in gold_spans],
        "length_bucket": "short",
        "regex_manifest_sha256": "b" * 64,
        "comparison_id": "regex-candidate",
        "source_row_sha256": row_sha256,
        "corpus_manifest_sha256": "f" * 64,
    }
    if document_chars is not None:
        base["document_chars"] = document_chars
    model_row = {
        **base,
        "pred_spans": [char_span_to_dict(span) for span in model_pred_spans],
        "evaluation_view": MODEL_CORE_VIEW,
        "shared_inference_identity": "c" * 64,
    }
    regex_row = {
        **base,
        "pred_spans": [char_span_to_dict(span) for span in regex_pred_spans],
        "evaluation_view": MODEL_PLUS_REGEX_VIEW,
        "shared_inference_identity": "c" * 64,
    }
    return model_row, regex_row


def test_ignore_list_nets_the_audited_false_positive_out_of_both_views() -> None:
    audited = CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name")
    model_row, regex_row = _paired_rows(
        doc_id="audit-doc-vi-0c33d83fd43447dd",
        gold_spans=(),
        model_pred_spans=(audited,),
        regex_pred_spans=(audited,),
    )
    ignore_list = ignore_list_from_rows([
        {
            "split": "eval",
            "uid": "audit-doc-vi-0c33d83fd43447dd",
            "start": 23,
            "end": 48,
            "text": "Bệnh Viện Đa Khoa Sài Gòn",
            "category": "company_name",
        },
    ])

    counts = paired_regex_document_counts((model_row,), (regex_row,), ignore_list=ignore_list, split="eval")

    assert (counts[0].model_tp, counts[0].model_fp, counts[0].model_fn) == (0, 0, 0)
    assert (counts[0].regex_tp, counts[0].regex_fp, counts[0].regex_fn) == (0, 0, 0)
    assert counts[0].model_fp_ignored == 1
    assert counts[0].regex_fp_ignored == 1


def test_character_safety_truth_includes_audited_ignore_list_spans() -> None:
    """Include audited ignore-list spans in character-safety truth.

    Safety truth is gold [0, 2) plus audited [4, 7): 2 + 3 = 5 chars. Model masks all 5 safety chars; regex leaves
    audited [4, 7) unmasked: leak 3. Both also mask unaudited [8, 10): 2 over-redacted chars each.

    Exact scoring still uses gold alone. Only the predicted audited span is ignored.
    """
    gold = CharSpan(0, 2, "AB", "human_name")
    audited = CharSpan(4, 7, "EFG", "company_name")
    unaudited = CharSpan(8, 10, "IJ", "address")
    model_row, regex_row = _paired_rows(
        doc_id="audited-character-safety",
        gold_spans=(gold,),
        model_pred_spans=(gold, audited, unaudited),
        regex_pred_spans=(gold, unaudited),
        document_chars=12,
    )
    ignore_list = ignore_list_from_rows([
        {
            "split": "eval",
            "uid": "audited-character-safety",
            "start": audited.start,
            "end": audited.end,
            "text": audited.text,
            "category": audited.label,
        },
    ])

    (counts,) = paired_regex_document_counts((model_row,), (regex_row,), ignore_list=ignore_list, split="eval")

    assert counts.gold_pii_chars == 5
    assert (counts.model_leaked_chars, counts.regex_leaked_chars) == (0, 3)
    assert (counts.model_overredacted_chars, counts.regex_overredacted_chars) == (2, 2)
    assert (counts.model_tp, counts.model_fp, counts.model_fn) == (1, 1, 0)
    assert (counts.regex_tp, counts.regex_fp, counts.regex_fn) == (1, 1, 0)
    assert (counts.model_fp_ignored, counts.regex_fp_ignored) == (1, 0)


def test_ignore_list_does_not_exempt_the_same_text_at_a_different_coordinate() -> None:
    elsewhere = CharSpan(100, 125, "Bệnh Viện Đa Khoa Sài Gòn", "company_name")
    model_row, regex_row = _paired_rows(
        doc_id="audit-doc-vi-0c33d83fd43447dd",
        gold_spans=(),
        model_pred_spans=(elsewhere,),
        regex_pred_spans=(elsewhere,),
    )
    ignore_list = ignore_list_from_rows([
        {
            "split": "eval",
            "uid": "audit-doc-vi-0c33d83fd43447dd",
            "start": 23,
            "end": 48,
            "text": "Bệnh Viện Đa Khoa Sài Gòn",
            "category": "company_name",
        },
    ])

    counts = paired_regex_document_counts((model_row,), (regex_row,), ignore_list=ignore_list, split="eval")

    assert (counts[0].model_tp, counts[0].model_fp, counts[0].model_fn) == (0, 1, 0)
    assert (counts[0].regex_tp, counts[0].regex_fp, counts[0].regex_fn) == (0, 1, 0)
    assert counts[0].model_fp_ignored == 0
    assert counts[0].regex_fp_ignored == 0


def test_ignore_list_never_pulls_an_already_gold_span_out_of_true_positives() -> None:
    audited = CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name")
    model_row, regex_row = _paired_rows(
        doc_id="audit-doc-vi-0c33d83fd43447dd",
        gold_spans=(audited,),
        model_pred_spans=(audited,),
        regex_pred_spans=(audited,),
    )
    ignore_list = ignore_list_from_rows([
        {
            "split": "eval",
            "uid": "audit-doc-vi-0c33d83fd43447dd",
            "start": 23,
            "end": 48,
            "text": "Bệnh Viện Đa Khoa Sài Gòn",
            "category": "company_name",
        },
    ])

    counts = paired_regex_document_counts((model_row,), (regex_row,), ignore_list=ignore_list, split="eval")

    assert (counts[0].model_tp, counts[0].model_fp, counts[0].model_fn) == (1, 0, 0)
    assert (counts[0].regex_tp, counts[0].regex_fp, counts[0].regex_fn) == (1, 0, 0)
    assert counts[0].model_fp_ignored == 0
    assert counts[0].regex_fp_ignored == 0


def test_no_ignore_list_contract_is_byte_identical_to_today() -> None:
    audited = CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name")
    model_row, regex_row = _paired_rows(
        doc_id="audit-doc-vi-0c33d83fd43447dd",
        gold_spans=(),
        model_pred_spans=(audited,),
        regex_pred_spans=(audited,),
    )

    counts = paired_regex_document_counts((model_row,), (regex_row,))

    assert (counts[0].model_tp, counts[0].model_fp, counts[0].model_fn) == (0, 1, 0)
    assert (counts[0].regex_tp, counts[0].regex_fp, counts[0].regex_fn) == (0, 1, 0)
    assert counts[0].model_fp_ignored == 0
    assert counts[0].regex_fp_ignored == 0


def test_ignore_list_without_split_is_refused() -> None:
    ignore_list = ignore_list_from_rows([
        {
            "split": "eval",
            "uid": "doc",
            "start": 0,
            "end": 3,
            "text": "abc",
            "category": "company_name",
        },
    ])
    model_row, regex_row = _paired_rows(doc_id="doc", gold_spans=(), model_pred_spans=(), regex_pred_spans=())

    with pytest.raises(ValueError, match="needs split when ignore_list is set"):
        paired_regex_document_counts((model_row,), (regex_row,), ignore_list=ignore_list)


def test_paired_counts_measure_label_agnostic_gold_character_leak() -> None:
    """Measure gold-character leaks independently of predicted labels.

    model_core covers 2 of the 3 gold characters. model_plus_regex covers all three under a DIFFERENT label, which
    still redacts the text, so it leaks nothing even though it scores zero true positives on exact-span matching.
    """
    gold = CharSpan(0, 3, "ABC", "human_name")
    model_row, regex_row = _paired_rows(
        doc_id="leak-doc",
        gold_spans=(gold,),
        model_pred_spans=(CharSpan(0, 2, "AB", "human_name"),),
        regex_pred_spans=(CharSpan(0, 3, "ABC", "address"),),
    )

    (counts,) = paired_regex_document_counts((model_row,), (regex_row,))

    assert counts.gold_pii_chars == 3
    assert counts.model_leaked_chars == 1
    assert counts.regex_leaked_chars == 0


def test_paired_counts_measure_label_agnostic_gold_character_overredaction() -> None:
    """Measure non-gold over-redaction independently of predicted labels.

    20-char document. Gold covers chars [0, 3) -> 3 gold chars, 17 non-gold. model_core predicts [0, 5): overlaps gold
    on [0, 3) and over-masks the 2 non-gold chars [3, 5) -> model_overredacted_chars == 2. model_plus_regex predicts
    [10, 15): no overlap with gold at all, so every one of its 5 masked chars is non-gold -> regex_overredacted_chars
    == 5.
    """
    gold = CharSpan(0, 3, "ABC", "human_name")
    model_row, regex_row = _paired_rows(
        doc_id="overredaction-doc",
        gold_spans=(gold,),
        model_pred_spans=(CharSpan(0, 5, "ABCDE", "human_name"),),
        regex_pred_spans=(CharSpan(10, 15, "KLMNO", "address"),),
        document_chars=20,
    )

    (counts,) = paired_regex_document_counts((model_row,), (regex_row,))

    assert counts.document_chars == 20
    assert counts.gold_pii_chars == 3
    assert counts.model_overredacted_chars == 2
    assert counts.regex_overredacted_chars == 5


def test_paired_counts_leave_overredaction_unknown_without_document_chars() -> None:
    gold = CharSpan(0, 3, "ABC", "human_name")
    model_row, regex_row = _paired_rows(
        doc_id="no-length-doc",
        gold_spans=(gold,),
        model_pred_spans=(CharSpan(0, 5, "ABCDE", "human_name"),),
        regex_pred_spans=(CharSpan(10, 15, "KLMNO", "address"),),
    )

    (counts,) = paired_regex_document_counts((model_row,), (regex_row,))

    assert counts.document_chars is None
    assert counts.model_overredacted_chars == 0
    assert counts.regex_overredacted_chars == 0


def test_paired_counts_refuse_mismatched_document_chars_between_views() -> None:
    gold = CharSpan(0, 3, "ABC", "human_name")
    model_row, regex_row = _paired_rows(
        doc_id="mismatched-doc",
        gold_spans=(gold,),
        model_pred_spans=(),
        regex_pred_spans=(),
        document_chars=20,
    )
    regex_row["document_chars"] = 21

    with pytest.raises(ValueError, match="mismatched document_chars"):
        paired_regex_document_counts((model_row,), (regex_row,))


def test_paired_counts_refuse_present_non_integer_document_chars() -> None:
    model_row, regex_row = _paired_rows(
        doc_id="malformed-length-doc",
        gold_spans=(),
        model_pred_spans=(),
        regex_pred_spans=(),
    )
    model_row["document_chars"] = None
    regex_row["document_chars"] = None

    with pytest.raises(ValueError, match="document_chars"):
        paired_regex_document_counts((model_row,), (regex_row,))


def test_paired_counts_zero_predictions_have_zero_overredaction() -> None:
    gold = CharSpan(0, 2, "AB", "human_name")
    model_row, regex_row = _paired_rows(
        doc_id="zero-predictions",
        gold_spans=(gold,),
        model_pred_spans=(),
        regex_pred_spans=(),
        document_chars=10,
    )

    (counts,) = paired_regex_document_counts((model_row,), (regex_row,))

    assert counts.model_overredacted_chars == 0
    assert counts.regex_overredacted_chars == 0


def test_paired_counts_fully_masked_document_counts_only_non_gold_chars() -> None:
    """Count only non-gold characters when the whole document is masked.

    10 total chars - 2 gold chars = 8 non-gold chars. A full-document mask therefore over-redacts exactly 8/8 non-gold
    chars, not all 10 chars.
    """
    gold = CharSpan(0, 2, "AB", "human_name")
    full_mask = CharSpan(0, 10, "ABCDEFGHIJ", "address")
    model_row, regex_row = _paired_rows(
        doc_id="fully-masked",
        gold_spans=(gold,),
        model_pred_spans=(full_mask,),
        regex_pred_spans=(full_mask,),
        document_chars=10,
    )

    (counts,) = paired_regex_document_counts((model_row,), (regex_row,))

    assert counts.model_overredacted_chars == 8
    assert counts.regex_overredacted_chars == 8
