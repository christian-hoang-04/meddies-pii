from __future__ import annotations

from meddies_pii.translation import clean_output


def test_clean_output_strips_think_block() -> None:
    assert clean_output("<think>reasoning</think>\nXin chào bạn") == "Xin chào bạn"


def test_clean_output_passes_plain_text_through() -> None:
    assert clean_output("  Bệnh nhân Nguyễn Văn A  ") == "Bệnh nhân Nguyễn Văn A"
