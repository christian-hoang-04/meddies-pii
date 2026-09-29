# Meddies-PII

Official code and paper repository for **Meddies-PII: A Multilingual Framework
for Personally Identifiable Information Extraction in Clinical
De-identification**.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.12544-b31b1b)](https://arxiv.org/abs/2609.12544)

**Authors:** Le Linh Uyen*, Ho Dinh Tri*, Christian Hoang, and Huy Hoang Ha

\* These authors contributed equally.

This repository contains the research code and paper artifacts for a
multilingual synthetic clinical PII generation and evaluation framework. The
paper's local PDF and source are the authoritative publication artifacts; the
linked arXiv record is the associated preprint.

## Paper

| Resource | Link |
| --- | --- |
| Paper PDF | [`naacl submission/meddies-pii-naacl-final.pdf`](naacl%20submission/meddies-pii-naacl-final.pdf) |
| LaTeX source | [`naacl submission/meddies-pii-naacl-final.tex`](naacl%20submission/meddies-pii-naacl-final.tex) |
| arXiv preprint | [arXiv:2609.12544](https://arxiv.org/abs/2609.12544) |
| Paper assets | [`naacl submission/`](naacl%20submission/) |

The paper presents a synthetic clinical PII corpus with one million documents
across seventeen languages and a nine-label ontology. Attribute-conditioned
prompts control document and generation attributes, while thirteen deterministic
gates check structural and annotation validity before samples are accepted.

## What the paper contributes

- **Meddies-PII-Dataset:** one million synthetic clinical documents spanning
  seventeen languages and nine PII labels.
- **Controlled generation:** prompts explicitly control language, document type,
  text format, scenario, ontology constraints, and difficult surface forms.
- **Deterministic verification:** generated samples are repaired or rejected
  through structural, annotation, offset, coverage, and duplicate checks.
- **Meddies-PII-Model:** a BIOES token classifier used to evaluate the utility
  of the generated data.
- **Multilingual evaluation:** exact-match entity-level micro-F1 is reported on
  fifteen external benchmarks and the Meddies-PII Benchmark.

### Reported results

The paper reports an external mean F1 of **0.827** for Meddies-PII-Model,
compared with **0.658** for the strongest baseline. On the in-domain
Meddies-PII Benchmark, the model reaches **0.878** F1; the overall mean across
external and in-domain evaluation is **0.833**.

These are paper-reported benchmark results, not a guarantee of safe deployment
or regulatory compliance.

## Repository structure

```text
src/meddies_pii/             Package implementation and CLI
tests/                       Automated tests
scripts/                     Generation, evaluation, migration, and reporting tools
examples/                    Small offline examples
naacl submission/            Paper PDF, LaTeX, bibliography, ACL styles, and figures
docs/ARCHITECTURE.md         Codebase architecture guide
```

The publication release intentionally excludes review-workspace artifacts,
internal planning notes, temporary data, virtual environments, and local
execution caches. The dataset and trained model are not bundled in this Git
repository; follow the paper's release status and linked project artifacts for
those resources.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run meddies-pii labels
uv run meddies-pii validate examples/sample.inline.jsonl
uv run meddies-pii demo
```

The quick-start workflow is offline after dependencies are installed. It does
not require provider API keys, cloud credentials, or network access to validate
the bundled example.

The demo writes three local artifacts under its output directory:

- `plain.jsonl` - stripped clinical text;
- `spans.jsonl` - text with character-offset spans; and
- `report.json` - record, span, and label summary.

## Development and reproduction checks

Run the core checks with:

```bash
uv run pytest
uv run ruff check --config ruff-strict.toml .
uv run mypy src/meddies_pii
uv run basedpyright src/meddies_pii
```

The broader generation, evaluation, training, and reporting workflows are
implemented under `src/meddies_pii/` and `scripts/`. Their commands may require
large model artifacts, external datasets, GPU runtimes, or provider
credentials; the offline quick start is the minimal reproducibility path.

## Annotation format

Meddies-PII uses inline annotations in the form `[value]<label>`:

```text
Patient [Nguyen Van A]<human_name> visited [Cho Ray Hospital]<company_name>
on [2024-03-15]<date>. Contact: [0901234567]<phone_number>.
```

The nine-label ontology is:

| Label | Scope |
| --- | --- |
| `address` | Street addresses, postal codes, coordinates, and care locations |
| `company_name` | Hospitals, clinics, departments, companies, and organizations |
| `date` | Person- or encounter-linked dates and datetimes |
| `email_address` | Email addresses |
| `human_name` | Patient, clinician, and other person names |
| `id_number` | Medical records, government IDs, passports, IPs, and account numbers |
| `phone_number` | Telephone and fax numbers |
| `private_url` | Access-bearing portal, result, signed, or private-record URLs |
| `secret` | Passwords, API keys, session tokens, cookies, PINs, and OTPs |

## Limitations and responsible use

The corpus is entirely synthetic. Deterministic gates enforce declared
structural and annotation requirements, but they do not directly measure the
semantic naturalness of every generated document. The evaluation focuses on
synthetic PII benchmarks and the Meddies-PII Benchmark; evaluation on
appropriately governed real clinical records remains future work.

This project is a research resource, not a certification that clinical text is
safe to share. Any deployment in a regulated workflow requires local validation,
privacy review, audit controls, and human oversight.

## Citation

If you use this code or paper, please cite:

```bibtex
@misc{meddiespii2026,
  title         = {Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification},
  author        = {Le, Linh Uyen and Ho, Dinh Tri and Hoang, Christian and Ha, Huy Hoang},
  year          = {2026},
  eprint        = {2609.12544},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.12544}
}
```

## License

Code in this repository is released under the
[Apache License 2.0](LICENSE). See [SECURITY.md](SECURITY.md) for responsible
vulnerability reporting.
