"""Static presentation assets for the training-setup preview."""

from __future__ import annotations

STYLE = """
:root {
  color-scheme: dark;
  --bg: #070a12;
  --panel: #0d1220;
  --line: #223049;
  --text: #e5edf8;
  --muted: #92a0b8;
  --accent: #7dd3fc;
  --ok: #86efac;
  --warn: #fbbf24;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: radial-gradient(circle at top left, #13213d 0, var(--bg) 36rem);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
main { max-width: 1800px; margin: 0 auto; padding: 32px; }
header {
  position: sticky;
  top: 0;
  z-index: 5;
  margin: -32px -32px 24px;
  padding: 28px 32px 22px;
  background: rgba(7, 10, 18, 0.93);
  backdrop-filter: blur(18px);
  border-bottom: 1px solid var(--line);
}
h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: -0.04em; }
.subtitle { margin: 0; color: var(--muted); font-size: 14px; }
.metrics { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }
.metric { border: 1px solid var(--line); border-radius: 16px; padding: 12px 14px; background: rgba(13, 18, 32, 0.78); }
.metric .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.11em; }
.metric .value { margin-top: 4px; font-size: 18px; font-weight: 750; }
section.setup { border: 1px solid var(--line); border-radius: 22px; background: rgba(13, 18, 32, \
0.78); margin: 20px 0; overflow: hidden; }
.setup-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
.setup-panel { padding: 18px; border-right: 1px solid var(--line); min-width: 0; }
.setup-panel:last-child { border-right: 0; }
h2 { margin: 0 0 12px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; }
h3 { margin: 24px 0 10px; color: #c7d2fe; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; overflow: auto; max-height: 620px; \
color: #e8eef9; font: 12.5px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #dbeafe; }
.command, .textbox { border: 1px solid #1d2b44; border-radius: 16px; padding: 14px; background: #050914; overflow: auto; }
.card { border: 1px solid var(--line); border-radius: 22px; background: linear-gradient(180deg, \
rgba(17, 24, 39, 0.96), rgba(8, 13, 25, 0.96)); box-shadow: 0 16px 70px rgba(0, 0, 0, 0.26); \
margin: 20px 0; overflow: hidden; }
.card-header { display: flex; justify-content: space-between; gap: 18px; padding: 18px 20px; \
border-bottom: 1px solid var(--line); background: rgba(255, 255, 255, 0.015); }
.title-line { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.rank { display: inline-flex; align-items: center; justify-content: center; min-width: 36px; \
height: 32px; border-radius: 999px; background: #172554; color: #bfdbfe; font-weight: 800; }
.path { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; color: #dbeafe; }
.badge { border: 1px solid #24405f; color: #c7d2fe; background: #111c33; border-radius: 999px; \
padding: 5px 9px; font-size: 12px; font-weight: 700; }
.badge.ok { color: var(--ok); border-color: #1f5f3f; background: #0f2419; }
.badge.warn { color: var(--warn); border-color: #7c5f1d; background: #281f0d; }
.meta { display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 12px; justify-content: flex-end; }
.card-body { padding: 18px; }
.snippet { color: #dbeafe; font-size: 14px; white-space: pre-wrap; overflow-wrap: anywhere; }
.table-wrap { overflow: auto; border: 1px solid #1d2b44; border-radius: 16px; background: #050914; }
table { width: 100%; border-collapse: collapse; }
th, td { border-top: 1px solid #1d2b44; padding: 8px 10px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .09em; }
.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
mark.pii { color: inherit; padding: 1px 3px; border-radius: 5px; border: 1px solid rgba(0,0,0,.18); }
.tag { margin-left: 4px; color: #111; font-size: 10px; opacity: .75; }
.label-address { background: #dbeafe; color: #0f172a; }
.label-company-name { background: #e0e7ff; color: #0f172a; }
.label-date { background: #fef3c7; color: #0f172a; }
.label-email-address { background: #dcfce7; color: #0f172a; }
.label-human-name { background: #fae8ff; color: #0f172a; }
.label-id-number { background: #fee2e2; color: #0f172a; }
.label-phone-number { background: #ccfbf1; color: #0f172a; }
.label-private-url { background: #ffedd5; color: #0f172a; }
.label-secret { background: #f3e8ff; color: #0f172a; }
footer { color: var(--muted); padding: 32px 0 12px; }
@media (max-width: 1320px) {
  .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .setup-grid { grid-template-columns: 1fr; }
  .setup-panel { border-right: 0; border-bottom: 1px solid var(--line); }
  .setup-panel:last-child { border-bottom: 0; }
}
"""

SCRIPT = """
const input = document.querySelector('#search');
const cards = Array.from(document.querySelectorAll('.card[data-search]'));
const visibleCount = document.querySelector('#visible-count');
function update() {
  const q = input.value.trim().toLowerCase();
  let count = 0;
  for (const card of cards) {
    const show = !q || card.dataset.search.includes(q);
    card.style.display = show ? '' : 'none';
    if (show) count += 1;
  }
  visibleCount.textContent = `${count} / ${cards.length} samples shown`;
}
input.addEventListener('input', update);
update();
"""
