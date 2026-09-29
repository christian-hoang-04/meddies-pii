# Contributing to Meddies PII

Meddies PII should be easy to run without credentials. Start with the offline CLI before touching provider-backed generation or training jobs.

## Setup

```bash
git clone https://github.com/meddies-ai/meddies-pii.git
cd meddies-pii
uv sync
uv run meddies-pii demo
```

## Development loop

```bash
uv run pytest
uv run ruff check --config ruff-strict.toml .
uv run meddies-pii labels
uv run meddies-pii validate examples/sample.inline.jsonl
```

Use test-first changes for behavior. Add the smallest failing test that proves the bug or feature, implement the package logic, then keep CLI/scripts as thin adapters.

The source type gate is `uv run mypy src/meddies_pii` plus `uv run basedpyright src/meddies_pii`. Strict MyPy checks every module under `src/meddies_pii` with no MyPy exclusions. BasedPyright runs in standard mode across the same source and excludes only `.venv`.

## Public vocabulary

Use **Meddies Labels** in public docs. Use `PiiLabel`, `PII_LABELS`, and `PII_LABEL_SET` in code. The current labels are:

```text
address, company_name, date, email_address, human_name, id_number, phone_number, private_url, secret
```

Earlier internal versions had 7 labels; do not describe that old set as current.

## Architecture expectations

- Read the [architecture guide](docs/ARCHITECTURE.md) before choosing a module or adding an advanced workflow.
- Public contributor workflows go through `uv run meddies-pii ...`.
- `scripts/` is for maintainer adapters, not reusable implementation.
- Reusable behavior belongs under `src/meddies_pii/...` with tests.
- Offline commands must not read provider credentials or call network services.
- Provider, HuggingFace, Gemini, and Modal workflows must make data movement explicit.

## Security reports

Report suspected vulnerabilities through the private process in [SECURITY.md](SECURITY.md). Do not open a public issue for a suspected vulnerability.

## Pull requests

1. Create a branch.
2. Keep the diff focused on one behavior or refactor seam.
3. Run the relevant targeted tests while developing.
4. Before review, run:

```bash
uv run ruff check --config ruff-strict.toml .
uvx --from 'git+https://github.com/hahuyhoang411/lintmax-py.git@4cac0c5b770b8f9cdb5a1d84186554ffbd0eb2a7' lintmax-py check .
uv run pytest
uv run mypy src/meddies_pii
uv run basedpyright src/meddies_pii
```

5. Explain any data/privacy implications in the PR body.
