FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy project files and install everything in one step
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/
COPY scripts/entrypoint.sh ./
RUN chmod +x ./entrypoint.sh

RUN uv sync --frozen --no-dev --compile-bytecode --all-extras

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001 8002 8003

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "graph_core.main:app", "--host", "0.0.0.0", "--port", "8001"]
