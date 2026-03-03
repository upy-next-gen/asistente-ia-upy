FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --frozen --no-install-project

# Pre-descargar el modelo de embeddings para evitar descarga en cada cold start
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

# Instalar el proyecto e indexar documentos para RAG
RUN uv sync --no-dev --frozen && \
    uv run python scripts/index_docs.py

# Forzar modo offline para que no contacte HuggingFace Hub al arrancar
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "-h"]
