from __future__ import annotations

from meddies_pii.regex_runtime import apply
from meddies_pii.spans import CharSpan


def test_runtime_owns_union_provenance_manifest_and_fail_closed_policy() -> None:
    caller_text = "Email: alpha@example.invalid"
    email_start = caller_text.index("alpha")
    model_email = CharSpan(
        email_start,
        email_start + len("alpha"),
        "alpha",
        "email_address",
    )

    caller = apply((caller_text,), ((model_email,),), language="en")

    assert [(span.text, span.label) for span in caller.spans[0]] == [("alpha@example.invalid", "email_address")]
    assert caller.regex_language == "en"
    assert caller.regex_language_source == "caller"
    assert caller.manifest_sha256 is not None
    assert len(caller.manifest_sha256) == 64

    detected_text = "Người bệnh được chuyển đến Bệnh viện Đa khoa Sông Xanh để tiếp tục điều trị."
    detected = apply((detected_text,), ((),))

    assert [(span.text, span.label) for span in detected.spans[0]] == [("Bệnh viện Đa khoa Sông Xanh", "company_name")]
    assert detected.regex_language == "vi"
    assert detected.regex_language_source == "detected"
    assert detected.manifest_sha256 == caller.manifest_sha256

    undetected = apply(("Bệnh viện X.",), ((),))

    assert undetected.spans == ((),)
    assert undetected.regex_language is None
    assert undetected.regex_language_source == "undetected"
    assert undetected.manifest_sha256 == caller.manifest_sha256

    disabled = apply((caller_text,), ((model_email,),), enabled=False)

    assert disabled.spans == ((model_email,),)
    assert disabled.regex_language is None
    assert disabled.regex_language_source is None
    assert disabled.manifest_sha256 is None
