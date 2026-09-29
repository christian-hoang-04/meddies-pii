# Meddies-PII

Code and paper release for Meddies-PII, a toolkit for generating, validating,
converting, and evaluating synthetic clinical PII examples across 17 languages.

## Repository contents

- `src/meddies_pii/` — package implementation and command-line interface.
- `tests/` — automated tests for parsing, validation, conversion, generation,
  evaluation, and PDF workflows.
- `scripts/` — reproducible maintainer and evaluation entry points.
- `examples/` — small offline examples for a first run.
- `naacl submission/` — paper source, compiled PDF, bibliography, ACL style
  files, and figure assets.
- `docs/ARCHITECTURE.md` — codebase architecture and navigation guide.

Review-workspace artifacts, internal planning notes, temporary data, and local
environment files are intentionally excluded from this publication repository.

## Paper

The paper is available at:

- [`naacl submission/meddies-pii-naacl-final.pdf`](naacl%20submission/meddies-pii-naacl-final.pdf)
- [`naacl submission/meddies-pii-naacl-final.tex`](naacl%20submission/meddies-pii-naacl-final.tex)

The paper directory also contains the bibliography, ACL style files, formatting
notes, and the figure used by the manuscript.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run meddies-pii labels
uv run meddies-pii validate examples/sample.inline.jsonl
uv run meddies-pii demo
```

The quick-start commands run offline and do not require API keys, cloud
credentials, or network access after dependencies are installed.

## Development checks

```bash
uv run ruff check --config ruff-strict.toml .
uv run pytest
uv run mypy src/meddies_pii
uv run basedpyright src/meddies_pii
```

## Annotation format

Meddies-PII uses inline annotations in the form `[value]<label>`:

```text
Patient [Nguyen Van A]<human_name> visited [Cho Ray Hospital]<company_name>
on [2024-03-15]<date>.
```

The public label set includes `address`, `company_name`, `date`,
`email_address`, `human_name`, `id_number`, `phone_number`, `private_url`, and
`secret`.

## License

This project is released under the [Apache License 2.0](LICENSE).
