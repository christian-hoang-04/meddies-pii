"""Static presentation assets for the inference preview report."""

from __future__ import annotations

STYLE = """
:root {
  color-scheme: dark;
  --bg: #070a12;
  --panel: #0d1220;
  --line: #213047;
  --line-soft: #172136;
  --text: #e7edf8;
  --muted: #98a6bd;
  --blue: #7dd3fc;
  --green: #86efac;
  --red: #fca5a5;
  --amber: #fbbf24;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 12% -10%, rgba(56, 189, 248, 0.18), transparent 34rem),
    radial-gradient(circle at 90% 4%, rgba(167, 139, 250, 0.13), transparent 30rem),
    var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
main { max-width: 1680px; margin: 0 auto; padding: 32px; }
header {
  position: sticky;
  top: 0;
  z-index: 5;
  margin: -32px -32px 24px;
  padding: 28px 32px 22px;
  border-bottom: 1px solid var(--line);
  background: rgba(7, 10, 18, 0.92);
  backdrop-filter: blur(18px);
}
h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: -0.045em; }
h2 {
  margin: 0 0 12px;
  color: var(--muted);
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.subtitle { margin: 0; color: var(--muted); font-size: 14px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 12px;
  margin-top: 20px;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px 14px;
  background: rgba(13, 18, 32, 0.8);
}
.metric .label {
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.metric .value { margin-top: 4px; font-size: 18px; font-weight: 780; }
.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin: 20px 0;
}
input[type="search"] {
  width: min(520px, 100%);
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #070b14;
  color: var(--text);
  padding: 11px 15px;
  outline: none;
}
.panel {
  border: 1px solid var(--line);
  border-radius: 22px;
  background: rgba(13, 18, 32, 0.8);
  margin: 20px 0;
  overflow: hidden;
}
.panel-body { padding: 18px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.textbox {
  border: 1px solid var(--line-soft);
  border-radius: 16px;
  background: #050914;
  padding: 14px;
  min-height: 180px;
  overflow: auto;
}
.snippet {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: #dce8f8;
  font: 13px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.card {
  border: 1px solid var(--line);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(8, 13, 25, 0.96));
  box-shadow: 0 16px 70px rgba(0, 0, 0, 0.28);
  margin: 20px 0;
  overflow: hidden;
}
.card-header {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 18px 20px;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.018);
}
.title-line { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 38px;
  height: 32px;
  border-radius: 999px;
  background: #172554;
  color: #bfdbfe;
  font-weight: 850;
}
.path {
  color: #dbeafe;
  font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid #24405f;
  border-radius: 999px;
  background: #111c33;
  color: #c7d2fe;
  padding: 5px 9px;
  font-size: 12px;
  font-weight: 740;
}
.badge.ok { color: var(--green); border-color: #1f5f3f; background: #0f2419; }
.badge.warn { color: var(--amber); border-color: #7c5f1d; background: #281f0d; }
.badge.bad { color: var(--red); border-color: #6f2a2a; background: #271111; }
.meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 12px;
  color: var(--muted);
  font-size: 12px;
}
.card-body { padding: 18px; }
mark.pii {
  color: #0f172a;
  padding: 1px 4px;
  border-radius: 6px;
  border: 1px solid rgba(15, 23, 42, 0.22);
}
mark.pred { box-shadow: 0 0 0 2px rgba(125, 211, 252, 0.32); }
mark.gold { box-shadow: 0 0 0 2px rgba(134, 239, 172, 0.3); }
.tag {
  margin-left: 5px;
  color: #0f172a;
  font-size: 10px;
  font-weight: 820;
  opacity: 0.72;
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
.table-wrap {
  overflow: auto;
  border: 1px solid var(--line-soft);
  border-radius: 16px;
  background: #050914;
}
table { width: 100%; border-collapse: collapse; }
th, td { border-top: 1px solid var(--line-soft); padding: 8px 10px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 11px; letter-spacing: .09em; text-transform: uppercase; }
td { font-size: 13px; }
.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.muted { color: var(--muted); }
pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 520px;
  overflow: auto;
  color: #dbeafe;
  font: 12.5px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
footer { color: var(--muted); padding: 30px 0 12px; }
@media (max-width: 1280px) {
  .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .grid-2 { grid-template-columns: 1fr; }
  .toolbar { align-items: stretch; flex-direction: column; }
}
"""

SCRIPT = """
const input = document.querySelector('#search');
const cards = Array.from(document.querySelectorAll('.card[data-search]'));
const visibleCount = document.querySelector('#visible-count');
function update() {
  const query = input.value.trim().toLowerCase();
  let visible = 0;
  for (const card of cards) {
    const show = !query || card.dataset.search.includes(query);
    card.style.display = show ? '' : 'none';
    if (show) visible += 1;
  }
  visibleCount.textContent = `${visible} / ${cards.length} rows shown`;
}
input.addEventListener('input', update);
update();
"""
