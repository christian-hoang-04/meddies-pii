"""False rows often omit ValueStart/End (empty) — must parse without crashing.

A tagged row without a value column → span the whole line (trimmed), not drop.

The 1,070-credential case: a multi-line PEM key marked at line granularity (LineStart..LineEnd, empty value cols) → span
the whole key block as secret.

A credential embedded in a URL → span the WHOLE URL as private_url.

A "URL Credentials" row whose credential is NOT inside a URL → secret.

"""

from __future__ import annotations

from meddies_pii.training.bioes.data.creddata import (
    CredMetaRow,
    convert_creddata_file,
    creddata_category_to_pii_label,
)


def _resolve(category: str, crypto: str = "", gt: str = "T") -> str | None:
    return creddata_category_to_pii_label(category, crypto, gt)


def test_resolver_maps_categories_to_pii_label() -> None:
    """colon-combo priority: URL Credentials > credential > id.

    public crypto key → untagged even if category looks credential-ish.

    """
    assert _resolve("Password") == "secret"
    assert _resolve("Secret:Auth") == "secret"
    assert _resolve("PEM Private Key", crypto="Private") == "secret"
    assert _resolve("URL Credentials") == "private_url"
    assert _resolve("UUID") == "id_number"
    assert _resolve("AWS Client ID") == "id_number"
    assert _resolve("Tencent WeChat API App ID") == "id_number"
    assert _resolve("Token:UUID") == "secret"
    assert _resolve("Password:URL Credentials") == "private_url"
    assert _resolve("UUID:Token") == "secret"
    for cat in ("Nonce", "Salt", "AWS S3 Bucket", "Firebase Domain", "Public Key"):
        assert _resolve(cat) is None, cat
    assert _resolve("Key", crypto="Public") is None
    assert _resolve("Password", gt="F") is None
    assert _resolve("Password", gt="X") is None


def test_convert_single_line_secret_offset() -> None:
    text = 'import os\nAPI_KEY = "sk_live_SECRET_VALUE_123"\nDEBUG = True\n'
    rows = [CredMetaRow(2, 2, 11, 35, "T", "", "Secret:Auth")]

    records, dropped = convert_creddata_file(text, rows, repo="r", file_id="f")

    assert not dropped
    assert len(records) == 1
    rec = records[0]
    assert len(rec["label"]) == 1
    span = rec["label"][0]
    assert span["category"] == "secret"
    assert span["text"] == "sk_live_SECRET_VALUE_123"
    assert rec["text"][span["start"] : span["end"]] == span["text"]
    assert rec["info"]["domain_bucket"] == "general"


def test_convert_multiline_value_offset() -> None:
    """Value spans line 2 (col 0) ..

    line 5 (col 17 = end of '-----END KEY-----').

    """
    text = "cfg:\n-----BEGIN KEY-----\nABC\nDEF\n-----END KEY-----\ndone\n"
    rows = [CredMetaRow(2, 5, 0, 17, "T", "Private", "PEM Private Key")]

    records, _dropped = convert_creddata_file(text, rows, repo="r", file_id="f")

    span = records[0]["label"][0]
    assert span["category"] == "secret"
    assert span["text"] == "-----BEGIN KEY-----\nABC\nDEF\n-----END KEY-----"
    assert records[0]["text"][span["start"] : span["end"]] == span["text"]


def test_convert_normalizes_crlf_before_offsets() -> None:
    text = 'a\r\nKEY="xyz"\r\nb\r\n'
    rows = [CredMetaRow(2, 2, 5, 8, "T", "", "Password")]

    records, _dropped = convert_creddata_file(text, rows, repo="r", file_id="f")

    span = records[0]["label"][0]
    assert span["text"] == "xyz"
    assert records[0]["text"][span["start"] : span["end"]] == "xyz"


def test_convert_keeps_false_positive_untagged_in_text() -> None:
    """Real → secret false positive → untagged.

    the false-positive value stays in the snippet text, just untagged (a negative).

    """
    text = 'token_a = "secret_TRUE_val"\ntoken_b = "EXAMPLE_PLACEHOLDER"\n'
    rows = [
        CredMetaRow(1, 1, 11, 26, "T", "", "Password"),
        CredMetaRow(2, 2, 11, 30, "F", "", "Password"),
    ]

    records, _dropped = convert_creddata_file(text, rows, repo="r", file_id="f")

    assert len(records) == 1
    rec = records[0]
    assert [s["text"] for s in rec["label"]] == ["secret_TRUE_val"]
    assert "EXAMPLE_PLACEHOLDER" in rec["text"]
    assert all(s["text"] != "EXAMPLE_PLACEHOLDER" for s in rec["label"])


def test_from_csv_tolerates_empty_value_positions() -> None:
    row = CredMetaRow.from_csv({
        "LineStart": "5",
        "LineEnd": "5",
        "ValueStart": "",
        "ValueEnd": "",
        "GroundTruth": "F",
        "CryptographyKey": "",
        "Category": "Password",
    })
    assert (row.value_start, row.value_end) == (-1, -1)


def test_converter_spans_whole_line_when_no_value_pos() -> None:
    text = "x\n  KEY = secret_value  \ny\n"
    row = CredMetaRow(2, 2, -1, -1, "T", "", "Password")

    records, _dropped = convert_creddata_file(text, [row], repo="r", file_id="f")

    span = records[0]["label"][0]
    assert span["category"] == "secret"
    assert span["text"] == "KEY = secret_value"
    assert records[0]["text"][span["start"] : span["end"]] == span["text"]


def test_converter_spans_multiline_pem_when_no_value_pos() -> None:
    text = "cfg:\n-----BEGIN KEY-----\nABCDEF\n-----END KEY-----\ndone\n"
    row = CredMetaRow(2, 4, -1, -1, "T", "Private", "PEM Private Key")

    records, _dropped = convert_creddata_file(text, [row], repo="r", file_id="f")

    span = records[0]["label"][0]
    assert span["category"] == "secret"
    assert span["text"] == "-----BEGIN KEY-----\nABCDEF\n-----END KEY-----"
    assert records[0]["text"][span["start"] : span["end"]] == span["text"]


def test_convert_url_credentials_expands_to_private_url() -> None:
    text = 'proxy = "http://john:zavvfuco@proxy.com"\n'
    row = CredMetaRow(1, 1, 21, 29, "T", "", "URL Credentials")

    records, _dropped = convert_creddata_file(text, [row], repo="r", file_id="f")

    span = records[0]["label"][0]
    assert span["category"] == "private_url"
    assert span["text"] == "http://john:zavvfuco@proxy.com"
    assert records[0]["text"][span["start"] : span["end"]] == span["text"]


def test_convert_url_credentials_without_url_falls_back_to_secret() -> None:
    text = "password = zavvfuco\n"
    row = CredMetaRow(1, 1, 11, 19, "T", "", "URL Credentials")

    records, _dropped = convert_creddata_file(text, [row], repo="r", file_id="f")

    span = records[0]["label"][0]
    assert span["category"] == "secret"
    assert span["text"] == "zavvfuco"


def test_convert_drops_out_of_range_offsets() -> None:
    text = 'KEY = "v"\n'
    rows = [CredMetaRow(99, 99, 0, 5, "T", "", "Password")]

    records, dropped = convert_creddata_file(text, rows, repo="r", file_id="f")

    assert records == []
    assert sum(dropped.values()) == 1
