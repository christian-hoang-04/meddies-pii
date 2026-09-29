# ruff: file-ignore[print]
# reason: diagnostic output for a module that also runs standalone under `__main__`.
from meddies_pii.tags import strip_code_blocks, strip_pii_tags


def test_strip_code_blocks() -> None:
    test_cases = [
        ("```markdown\nHello world\n```", "Hello world"),
        ("```xml\n<tag>Value</tag>\n```", "<tag>Value</tag>"),
        ("```\nPlain text block\n```", "Plain text block"),
        ("Just some text", "Just some text"),
        (
            "```\nText with triple backticks inside: ```\n```",
            "Text with triple backticks inside:",
        ),
        ("```markdown\nMulti-line\nContent\n```", "Multi-line\nContent"),
        ("```\nNo start newline", "No start newline"),
        ("No end newline\n```", "No end newline"),
        ("```json\nNo end newline\n```", "No end newline"),
    ]

    for input_str, expected in test_cases:
        result = strip_code_blocks(input_str)
        print(f"Input: {input_str!r}")
        print(f"Result: {result!r}")
        print(f"Expected: {expected!r}")
        assert result == expected, f"Failed for {input_str!r}"
        print("-" * 20)


def test_strip_pii_tags() -> None:
    test_cases = [
        (
            "[John Doe]<human_name> went to [Mayo Clinic]<company_name>",
            "John Doe went to Mayo Clinic",
        ),
        ("Date: [2023-10-10]<date>", "Date: 2023-10-10"),
        ("No tags here", "No tags here"),
        ("[Empty Value]<>", "[Empty Value]<>"),
        ("[Value]<label1> and [Value]<label2>", "Value and Value"),
    ]

    for input_str, expected in test_cases:
        result = strip_pii_tags(input_str)
        print(f"Input: {input_str!r}")
        print(f"Result: {result!r}")
        print(f"Expected: {expected!r}")
        assert result == expected, f"Failed for {input_str!r}"
        print("-" * 20)


if __name__ == "__main__":
    test_strip_code_blocks()
    print("=" * 20)
    test_strip_pii_tags()
    print("All tests passed!")
