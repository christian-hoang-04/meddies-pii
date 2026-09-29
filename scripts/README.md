# Maintainer scripts

The Python package under `src/meddies_pii/` owns reusable behavior. The files
under `scripts/` are thin adapters for dataset construction, generation,
evaluation, reporting, and operational jobs; they are not part of the public
`meddies-pii` CLI surface.

## Directory guide

| Directory | Purpose | Runtime boundary |
| --- | --- | --- |
| `generation/` | Provider-backed synthetic-data generation | Local or provider credentials |
| `migrations/` | One-off and repeatable dataset migrations | Local files and hosted datasets |
| `ops/` | Modal jobs, release gates, baselines, and private benchmarks | Modal, GPU, secrets, or private data |
| `quality/` | Repository and evaluation quality checks | Local or Modal CPU |
| `reports/` | Reproducible report and preview renderers | Local or hosted artifacts |
| `archive/` | Historical one-off scripts kept for provenance | Not a supported release path |

## How to choose a home for new code

1. Put reusable parsing, validation, policy, scoring, and data transformation
   in `src/meddies_pii/` first.
2. Keep the script as a small argument/configuration adapter around that package
   code.
3. Put a workflow in `ops/` only when it requires Modal, a GPU, a secret, a
   private dataset, or a release/evaluation gate.
4. Add a focused test under `tests/` for any package behavior or script wiring
   that can run without external credentials.

The commands embedded in individual script docstrings are authoritative for
that workflow. Operational scripts may require credentials, private data,
large model artifacts, or a hosted runtime and are not part of the offline
quick start.
