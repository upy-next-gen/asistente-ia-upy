FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --frozen --no-install-project

COPY . .

RUN uv sync --no-dev --frozen

# Usar el venv directamente para evitar que uv run re-sincronice al iniciar
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "-h"]
