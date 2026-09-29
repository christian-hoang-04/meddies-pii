from __future__ import annotations

from html import escape


def metric_card(label: str, value: object) -> str:
    """Render the shared summary metric card used by BIOES reports.

    Returns:
        One card as an HTML fragment, not a whole document. Both the label and the value are
        HTML-escaped, so a value carrying markup renders as text rather than injecting into
        the report -- callers must not pre-escape or the entities will show up doubled.

    """
    return (
        f'<div class="metric"><div class="label">{escape(label)}</div><div class="value">{escape(str(value))}</div></div>'
    )
