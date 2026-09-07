# Validation

FDSM is tested as a standalone numerical transformation library. The checks focus
on its public API, numerical behavior, input validation, command-line interface,
and public-data examples.

## Automated tests

The test suite covers:

- agreement of integer and fractional FFT derivatives with analytic periodic
  signals;
- preservation of the input at derivative order zero;
- order-ladder construction and transform output coordinates;
- row-wise normalization, batch transforms, and output data types;
- ascending, descending, invalid, and nonuniform sampling axes;
- Savitzky-Golay parameter validation and optional preprocessing controls;
- both unpaired-Nyquist policies;
- command-line archive output and metadata; and
- parsers and resampling used by the public GC-MS and FT-IR example workflow.

GitHub Actions runs the complete test suite and Ruff static checks on Ubuntu,
Windows, and macOS with Python 3.10, 3.11, 3.12, and 3.13.

Run the same checks locally with:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Public-data example

The reproducible overview example downloads hash-pinned public source files. The
workflow explicitly resamples the irregularly timed GC-MS TIC, preserves the
conventional descending display direction of the FT-IR axis, and produces finite
FDSM arrays for both modalities. Source attribution, reuse terms, hashes, and
processing details are documented in [example-data.md](example-data.md).

The example can be regenerated with:

```bash
python -m pip install -e ".[plot]"
python examples/make_toc_figure.py
```
