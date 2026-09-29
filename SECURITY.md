# Security policy

## Supported code

Security fixes are considered for the current `main` branch. The repository does not publish support windows for historical releases.

## Report a vulnerability privately

Use GitHub private vulnerability reporting for this repository when the **Security** tab shows **Report a vulnerability**. That channel lets reporters and maintainers discuss a vulnerability privately before disclosure. Include the affected commit or release, reproduction steps, impact, and any safe mitigation you know.

Do not open a public issue for a suspected vulnerability. This repository does not publish an email address or another versioned private reporting channel. If the GitHub control is absent, a maintainer must enable private vulnerability reporting before an external reporter has a documented private fallback.

## Disclosure and fixes

Maintainers should reproduce and assess the report, prepare and validate a fix in a private advisory or fork when appropriate, then publish an advisory after a fixed revision is available. A public advisory should identify affected versions and the fixed version when one exists.

## Operational boundaries

Never place provider, HuggingFace, Gemini, or Modal credentials in source, examples, test fixtures, issues, or benchmark output. Keep them in the runtime environment described by [`.env.example`](.env.example). Treat generated clinical text and artifacts as potentially sensitive even when a workflow is intended to use synthetic data.

The PDF private-fixture benchmark path is intentionally separate: it returns the unreviewed result to the caller and records hashes, counts, timings, and reason codes rather than fixture contents. It remains a benchmark path, not a production redaction approval.
