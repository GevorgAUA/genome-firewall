# Contributing

Contributions must preserve the project's defensive research scope and visible non-clinical-use warning. Do not submit organism-design features, resistance-enhancing mutation selection, wet-lab protocols, patient-specific recommendations, real challenge data, credentials, or untrusted serialized model artifacts.

Set up the project with Python 3.11 and `uv`:

```bash
uv sync --all-extras --dev
uv run pre-commit run --all-files
uv run pytest
```

Use focused commits, add tests for behavioral changes, and document assumptions rather than presenting statistical associations as biological causation.
