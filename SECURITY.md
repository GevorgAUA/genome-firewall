# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's security-advisory feature. Do not include identifiable clinical records, unpublished pathogen data, credentials, or production model artifacts in a report.

## Supported scope

This repository is a research prototype. The local upload surface enforces file-size and
extension checks, treats uploads strictly as data, uses a temporary directory for the
FASTA, and never executes uploaded content. Local inference caches derived AMRFinder
reports by FASTA hash. The hosted deployment serializes resource-intensive annotation
jobs and deletes both uploaded FASTAs and derived AMRFinder reports immediately after
each prediction.

Serialized model artifacts must come from this trusted training run; Python pickle and
joblib formats are not safe for untrusted files.
