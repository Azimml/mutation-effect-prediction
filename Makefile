.PHONY: help install install-dev install-cnn lint format test smoke clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in editable mode.
	pip install -e .

install-dev:  ## Install with dev extras (ruff, pytest).
	pip install -e ".[dev]"

install-cnn:  ## Install with CNN + dev extras (pulls in torch).
	pip install -e ".[cnn,dev]"

lint:  ## Run ruff over the source and tests.
	ruff check src tests

format:  ## Auto-fix lint issues where ruff can.
	ruff check --fix src tests

test:  ## Run the test suite.
	pytest

smoke:  ## Run the end-to-end smoke test on synthetic data.
	python scripts/run_smoke_test.py

clean:  ## Remove caches and build artifacts.
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
