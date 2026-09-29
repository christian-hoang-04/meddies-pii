from __future__ import annotations

from meddies_pii.generation.label_corpus.audit import markdown_table_cell


def test_markdown_table_cell_escapes_newlines_pipes_and_truncates() -> None:
    assert markdown_table_cell("090\n123|456", limit=20) == "090\\n123/456"
    assert markdown_table_cell("x" * 10, limit=6) == "xxxxx…"
