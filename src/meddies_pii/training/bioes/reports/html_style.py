"""Static CSS for the training-readiness report."""

from __future__ import annotations


# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: the multiplication sign is typography meaning "by" in an operator-facing report label
# reason: (source x language x bucket, 2xA100); the ASCII letter x would misrender the heading.
def _css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f7f5;
  --panel: #ffffff;
  --ink: #171717;
  --muted: #626262;
  --line: #deded8;
  --accent: #1d4ed8;
  --warn: #b45309;
  --fail: #b91c1c;
  --pass: #047857;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header, main { max-width: 1240px; margin: 0 auto; padding: 28px; }
header { padding-top: 40px; }
h1 { margin: 0; font-size: 34px; letter-spacing: -0.03em; }
h2 { margin-top: 42px; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
.eyebrow { text-transform: uppercase; letter-spacing: .14em; color: var(--muted); font-size: 12px; }
.sub, .hint, .note, .empty { color: var(--muted); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre {
  white-space: pre-wrap;
  overflow: auto;
  padding: 16px;
  background: #111827;
  color: #f9fafb;
  border-radius: 10px;
}
.grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.card, section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,.03);
}
section .card { margin: 0; }
.metric-title { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.metric { font-size: 28px; font-weight: 750; letter-spacing: -0.03em; }
.stack { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.stack > div { border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: #fbfbfa; }
.checks { list-style: none; padding: 0; display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; }
.checks li { border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
.checks li::before { display: inline-block; width: 1.4em; font-weight: 700; }
.checks li.pass::before { content: "✓"; color: var(--pass); }
.checks li.warn::before { content: "!"; color: var(--warn); }
.checks li.fail::before { content: "×"; color: var(--fail); }
.checks span { display: block; color: var(--muted); margin-left: 1.4em; }
table { width: 100%; border-collapse: collapse; }
th, td { border-top: 1px solid var(--line); padding: 8px; vertical-align: top; }
th { text-align: left; background: #fafaf9; position: sticky; top: 0; z-index: 1; }
.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.dense td, .dense th { padding: 6px 8px; }
.examples { font-size: 13px; }
.snippet { min-width: 340px; white-space: pre-wrap; overflow-wrap: anywhere; }
.bar { height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
.bar span { display: block; height: 100%; background: var(--accent); }
details { margin: 10px 0; }
summary { cursor: pointer; font-weight: 650; }
mark.pii {
  color: inherit;
  padding: 1px 3px;
  border-radius: 5px;
  border: 1px solid rgba(0,0,0,.12);
}
.tag {
  margin-left: 4px;
  color: #111;
  font-size: 10px;
  opacity: .7;
}
.pill {
  display: inline-block;
  margin: 1px;
  padding: 2px 6px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,.12);
  font-size: 11px;
}
.label-address { background: #dbeafe; }
.label-company-name { background: #e0e7ff; }
.label-date { background: #fef3c7; }
.label-email-address { background: #dcfce7; }
.label-human-name { background: #fae8ff; }
.label-id-number { background: #fee2e2; }
.label-phone-number { background: #ccfbf1; }
.label-private-url { background: #ffedd5; }
.label-secret { background: #f3e8ff; }
.good { color: var(--pass); }
a { color: var(--accent); }
@media (max-width: 900px) {
  .grid, .stack, .checks { grid-template-columns: 1fr; }
  header, main { padding: 16px; }
}
"""
