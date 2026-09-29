from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from meddies_pii.tags import strip_code_blocks, strip_pii_tags


class TestPropertyBasedTagHelpers:
    @staticmethod
    @given(st.text())
    def test_strip_code_blocks_properties(content: str) -> None:
        cleaned = strip_code_blocks(content)
        assert "```" not in cleaned, "Cleaned content should not contain code blocks"
        if "[" in content and "]" in content:
            assert "[" in cleaned, "PII tag opener should be preserved"
            assert "]" in cleaned, "PII tag closer should be preserved"

    @staticmethod
    @given(st.text())
    def test_strip_pii_tags_properties(content: str) -> None:
        stripped = strip_pii_tags(content)
        assert not re.search(r"\[[^\]]*\]\s*<[a-zA-Z][^>]*>", stripped), (
            "Stripped content should not contain PII tag patterns"
        )
        assert len(stripped) <= len(content), "Stripped content should be shorter or equal"
