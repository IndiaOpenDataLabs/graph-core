.PHONY: help install install-dev lint format fix test clean \
	docker-up docker-down docker-logs docker-clean docker-ps \
	docker-logs-app docker-logs-worker \
	db-migrate db-revision db-current db-stamp db-downgrade \
	infra-check seed smoke-test tui

# ─── Project ──────────────────────────────────────────────────────────────────

help:                 ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:              ## Install dependencies
	uv sync --all-groups

install-dev:          ## Install dependencies + dev tools (pytest, ruff, mypy, etc.)
	uv sync --group dev

# ─── Quality ──────────────────────────────────────────────────────────────────

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

# ─── Docker ───────────────────────────────────────────────────────────────────
## Services: postgres (pgvector), falkordb (graph db), redis (dramatiq + sse)

docker-up:            ## Build + start everything (infra + app + worker)
	docker compose up -d --build

docker-down:          ## Stop all services
	docker compose down

docker-logs:          ## Follow logs for all services
	docker compose logs -f

docker-logs-app:      ## Follow app logs
	docker compose logs -f app

docker-logs-worker:   ## Follow worker logs
	docker compose logs -f worker

docker-ps:            ## List running containers
	docker compose ps

docker-clean:         ## Stop and remove all containers, volumes, networks
	docker compose down -v --remove-orphans

# ─── Database Migrations (Alembic) ───────────────────────────────────────────

db-migrate:           ## Run pending Alembic migrations
	uv run alembic upgrade head

db-revision:          ## Generate a new migration (pass message: make db-revision m="add jobs table")
	uv run alembic revision --autogenerate -m "$(m)"

db-current:           ## Show current migration version
	uv run alembic current

db-stamp:             ## Set current migration version without running
	uv run alembic stamp head

db-downgrade:         ## Downgrade one migration revision
	uv run alembic downgrade -1

# ─── Utilities ────────────────────────────────────────────────────────────────

infra-check:          ## Verify all infrastructure services are reachable
	@echo "Checking PostgreSQL..."
	@docker compose exec -T postgres pg_isready -U graphcore -d graphcore || echo "  PostgreSQL: NOT READY"
	@echo "Checking Redis..."
	@docker compose exec -T redis redis-cli ping || echo "  Redis: NOT READY"
	@echo "Checking FalkorDB..."
	@docker compose exec -T falkordb redis-cli ping || echo "  FalkorDB: NOT READY"
	@echo "All services checked."

seed:                 ## Run seed script (create default embedding/llm profiles)
	uv run python -m graph_core.scripts.seed

smoke-test:           ## Run end-to-end smoke test (requires make docker-up; add --llm-key/--embed-key for all strategies)
	uv run python -m graph_core.scripts.smoke_test $(SMOKE_ARGS)

smoke-test-local:     ## Run smoke test against local LLM/embedding servers
	uv run python -m graph_core.scripts.smoke_test \
		--llm-key test-key --llm-url http://host.docker.internal:8080/v1 \
		--embed-key test-key --embed-url http://host.docker.internal:8002/v1 \
		--embed-dimensions 4096

tui:                  ## Run the terminal UI client
	uv run python -m graph_core.cli

server:               ## Start the FastAPI server with MCP endpoint
	uv run uvicorn graph_core.main:app --host 0.0.0.0 --port 8000 --reload
