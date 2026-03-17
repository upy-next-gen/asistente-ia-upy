FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --frozen --no-install-project

COPY . .

RUN uv sync --no-dev --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
