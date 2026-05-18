.PHONY: help install install-dev dev start worker lint format fix test clean

help:                 ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:              ## Install dependencies
	uv sync --all-groups

install-dev:          ## Install dependencies + dev tools (pytest, ruff, mypy, etc.)
	uv sync --group dev

dev:                  ## Start dev server with reload
	uv run uvicorn graph_core.main:app --reload --host 0.0.0.0 --port 8000

start:                ## Start production server
	uv run uvicorn graph_core.main:app --host 0.0.0.0 --port 8000 --workers 4

worker:               ## Start Dramatiq worker
	uv run dramatiq graph_core.workers --processes 4 --threads 8

lint:                 ## Run all lint checks
	uv run ruff check src/
	uv run mypy src/

test:                 ## Run tests (requires make install-dev)
	uv run pytest

format:               ## Check formatting (no changes)
	uv run black --check src/
	uv run isort --check src/

fix:                  ## Auto-fix lint + format
	uv run autoflake --remove-all-unused-imports --remove-dummy-unused-variables --in-place --recursive src/
	uv run ruff check --fix src/
	uv run black src/
	uv run isort src/

clean:                ## Remove build artifacts
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/
