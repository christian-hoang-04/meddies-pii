"""Meddies BIOES training mode.

Token-classification training over the PII-label-label vocabulary in
BIOES form (37 output classes: 9 entity labels x {B,I,E,S} + 1
background). The trained model emits per-token tags decoded into spans
via constrained Viterbi; the SFT sibling mode emits JSON entity dicts.
"""
# ruff: file-ignore[implicit-namespace-package]
# reason: `training/` is a namespace package BY DECISION — `pyproject.toml:439` records it and sets
# reason: `include_namespace_packages = true` so coverage inventories its unexecuted files. Ruff's
# reason: suggested `training/__init__.py` would reverse that documented choice, not fix a defect.
