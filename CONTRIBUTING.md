# Contributing

Please keep changes focused on the domain-independent transformation and avoid
committing private datasets, internal file paths, credentials, or large arrays.

Development setup:

```bash
python -m venv .venv
python -m pip install -e ".[dev,plot]"
pytest
ruff check .
python -m build
```

New numerical behavior should include tests and a changelog entry. Changes to the
transform must be clearly identified and must not silently alter the default
output.
