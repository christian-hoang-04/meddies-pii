from __future__ import annotations

REMOTE_CODE_TOKENIZER_ALLOWLIST = frozenset({"LiquidAI/LFM2.5-350M-Base"})


def validate_remote_code_tokenizer_policy(
    *,
    tokenizer_model_id: str,
    tokenizer_revision: str | None,
    trust_remote_code: bool,
) -> None:
    if not trust_remote_code:
        return
    if tokenizer_model_id not in REMOTE_CODE_TOKENIZER_ALLOWLIST:
        msg = f"tokenizer model {tokenizer_model_id!r} is not allowlisted for trust_remote_code"
        raise ValueError(msg)
    if tokenizer_revision is None or not tokenizer_revision.strip():
        msg = f"trust_remote_code requires a pinned tokenizer revision for {tokenizer_model_id!r}"
        raise ValueError(msg)
