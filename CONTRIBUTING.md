# Contributing

Thanks for your interest in improving this project. It is a reproducible
genomics ML prototype, so the bar for contributions is correctness,
reproducibility, and honest claims rather than raw model scores.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

The CNN path is optional. Install it only if you touch `cnn.py` or its tests:

```bash
pip install -e ".[cnn,dev]"
# or CPU-only torch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`pysam` is imported lazily inside the I/O functions, so the pure helpers and
their tests import and run without it. You only need `pysam` (and `samtools`)
for `download-data`, `preprocess`, and the smoke test.

## Before opening a pull request

Run the same checks CI runs:

```bash
ruff check src tests
pytest
```

The `pytest` suite for the pure functions requires no network, no data files,
and no `torch`. The sequence-encoding tests are skipped automatically when
`torch` is not installed.

Optionally, exercise the end-to-end pipeline on synthetic data:

```bash
python scripts/run_smoke_test.py
```

## Guidelines

- Keep the leakage-aware split invariants intact. Any change to `assign_split`
  must keep same-region variants in the same split and stay deterministic.
- Do not commit large artifacts. Raw downloads, processed datasets, and model
  checkpoints stay local (see `.gitignore`). Preserve headline metrics in
  `reports/` instead.
- Match the existing style: type hints, `from __future__ import annotations`,
  and explanatory comments for non-obvious ML/genomics choices.
- Keep claims defensible. The README and results summary deliberately avoid
  clinical-grade language; new documentation should do the same.

## Commit messages

Use conventional-commit style prefixes (`feat:`, `fix:`, `docs:`, `test:`,
`refactor:`, `chore:`, `ci:`), one logical change per commit.
