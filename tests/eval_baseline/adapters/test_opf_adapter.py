from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from meddies_pii.eval_baseline.adapters.opf import (
    OPF_LABEL_FOLD,
    OPF_SUPPORTED_LABELS,
    OpfAdapter,
    map_opf_label,
    spans_from_native_spans,
)
from meddies_pii.eval_baseline.adapters.opf_backend import (
    NativePredictorDoc,
    detected_span_to_char_span,
)
from meddies_pii.eval_baseline.adapters.opf_windowing import (
    TokenizedOpfDoc,
    pack_token_windows,
    reassemble_window_values,
)
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.taxonomy import PiiLabel


def test_adapter_runtime_has_no_benchmark_dependency() -> None:
    adapter_dir = Path(__file__).parents[3] / "src/meddies_pii/eval_baseline/adapters"

    for module_path in adapter_dir.glob("opf*.py"):
        assert "opf_benchmark" not in module_path.read_text(encoding="utf-8")


def test_opf_label_fold_maps_all_native_labels_to_pii_label() -> None:
    expected: dict[str, PiiLabel] = {
        "private_address": "address",
        "private_date": "date",
        "private_email": "email_address",
        "private_person": "human_name",
        "private_phone": "phone_number",
        "private_url": "private_url",
        "secret": "secret",
        "account_number": "id_number",
    }

    assert dict(OPF_LABEL_FOLD) == expected
    for native, meddies in expected.items():
        assert map_opf_label(native) == meddies
        assert map_opf_label(f"B-{native}") == meddies

    assert map_opf_label("company_name") is None
    assert map_opf_label("diagnosis") is None
    assert map_opf_label("O") is None


def test_supported_labels_are_exactly_the_meddies8_without_company_name() -> None:
    assert frozenset(OPF_LABEL_FOLD.values()) == OPF_SUPPORTED_LABELS
    assert len(OPF_SUPPORTED_LABELS) == 8
    assert "company_name" not in OPF_SUPPORTED_LABELS


def test_mock_detected_spans_fold_to_original_text_offsets_and_drop_unknowns() -> None:
    text = "Name: John Smith, email: alice@example.com. Token: sk-live."
    name_start = text.index(" John")
    email_start = text.index("alice@example.com")
    token_start = text.index("sk-live")
    detected = [
        SimpleNamespace(
            start=name_start,
            end=name_start + len(" John Smith"),
            text="decoded-name",
            label="private_person",
        ),
        SimpleNamespace(
            start=email_start,
            end=email_start + len("alice@example.com."),
            text="decoded-email",
            label="private_email",
        ),
        SimpleNamespace(
            start=token_start,
            end=token_start + len("sk-live"),
            text="decoded-secret",
            label="secret",
        ),
        SimpleNamespace(start=0, end=4, text="Name", label="company_name"),
        SimpleNamespace(start=0, end=4, text="Name", label="diagnosis"),
        SimpleNamespace(start=-1, end=4, text="bad", label="private_address"),
    ]

    native_spans = [detected_span_to_char_span(span) for span in detected]
    spans = spans_from_native_spans(text, native_spans)

    assert spans == [
        CharSpan(
            start=name_start + 1,
            end=name_start + len(" John Smith"),
            text="John Smith",
            label="human_name",
        ),
        CharSpan(
            start=email_start,
            end=email_start + len("alice@example.com"),
            text="alice@example.com",
            label="email_address",
        ),
        CharSpan(
            start=token_start,
            end=token_start + len("sk-live"),
            text="sk-live",
            label="secret",
        ),
    ]
    assert all(text[span.start : span.end] == span.text for span in spans)
    assert {span.label for span in spans} <= OPF_SUPPORTED_LABELS


def test_predict_uses_injected_predictor_without_loading_gpu_model() -> None:
    calls: list[list[str]] = []

    def fake_predict(
        docs: Sequence[NativePredictorDoc],
    ) -> dict[str, list[CharSpan]]:
        calls.append([doc.doc_id for doc in docs])
        output: dict[str, list[CharSpan]] = {}
        for doc in docs:
            start = doc.text.index("secret")
            output[doc.doc_id] = [
                CharSpan(
                    start=start,
                    end=start + len("secret"),
                    text="decoded-secret",
                    label="secret",
                ),
            ]
        return output

    adapter = OpfAdapter(predictor=fake_predict, max_docs_per_call=1)
    texts = ["first secret", "second secret"]

    assert adapter.predict(texts) == [
        [CharSpan(start=6, end=12, text="secret", label="secret")],
        [CharSpan(start=7, end=13, text="secret", label="secret")],
    ]
    assert calls == [["0"], ["1"]]


def test_window_packer_pads_and_reassembles_by_example_id() -> None:
    docs = (
        TokenizedOpfDoc(example_id="doc-a", token_ids=(1, 2, 3, 4, 5)),
        TokenizedOpfDoc(example_id="doc-b", token_ids=(9, 8)),
    )

    batches = pack_token_windows(docs, window_size=3, batch_size=2, pad_token_id=0)

    assert batches[0].input_ids == ((1, 2, 3), (4, 5, 0))
    assert batches[0].attention_mask == ((True, True, True), (True, True, False))
    assert batches[1].input_ids == ((9, 8),)

    window_values = [
        (("a0", "a1", "a2"), ("a3", "a4", "pad")),
        (("b0", "b1"),),
    ]
    assert reassemble_window_values(batches, window_values) == {
        "doc-a": ("a0", "a1", "a2", "a3", "a4"),
        "doc-b": ("b0", "b1"),
    }
