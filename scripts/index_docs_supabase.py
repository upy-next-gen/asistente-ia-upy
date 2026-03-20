"""
Indexer para Supabase pgvector usando embeddings contextuales de Perplexity.

Uso:
    uv run python scripts/index_docs_supabase.py
"""

import logging
import os
import sys
import time
from collections import defaultdict

import httpx
from llama_index.core import SimpleDirectoryReader
from perplexity import Perplexity

from core.clients import get_supabase_client
from core.config import settings
from core.supabase_vector_db.indexing_utils import build_metadata_rows, decode_embedding

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
logger = logging.getLogger(__name__)
DEFAULT_VERSION_TAG = "current"
DEFAULT_CATEGORY = "general"


def build_metadata_lookup_key(source_file_name: str, version_tag: str) -> str:
    return f"{source_file_name}::{version_tag}"


def validate_embedding(embedding: list[float], expected_dim: int) -> None:
    if len(embedding) != expected_dim:
        raise ValueError(
            f"Embedding con dimension invalida: {len(embedding)}. Esperada: {expected_dim}."
        )
    if not all(isinstance(value, (int, float)) for value in embedding):
        raise TypeError("Todos los valores del embedding deben ser numericos.")


def upsert_metadata_rows(supabase_client, metadata_rows: list[dict]) -> dict[str, str]:
    if not metadata_rows:
        return {}

    normalized_rows = []
    for row in metadata_rows:
        source_file_name = str(row.get("source_file_name", "")).strip()
        version_tag = str(row.get("version_tag", "")).strip()
        if not source_file_name:
            raise ValueError("Cada metadata row debe incluir source_file_name.")
        if not version_tag:
            raise ValueError("Cada metadata row debe incluir version_tag.")
        normalized_rows.append(
            {
                "source_file_name": source_file_name,
                "document_type": row.get("document_type"),
                "version_tag": version_tag,
                "category": row.get("category"),
            }
        )

    supabase_client.table("documents_metadata").upsert(
        normalized_rows,
        on_conflict="source_file_name,version_tag",
    ).execute()

    file_names = [row["source_file_name"] for row in normalized_rows]
    rows = (
        supabase_client.table("documents_metadata")
        .select("id,source_file_name,version_tag")
        .in_("source_file_name", file_names)
        .execute()
        .data
        or []
    )

    mapping = {
        build_metadata_lookup_key(row["source_file_name"], row["version_tag"]): row["id"]
        for row in rows
        if row.get("source_file_name") and row.get("version_tag") and row.get("id")
    }

    missing = [
        build_metadata_lookup_key(row["source_file_name"], row["version_tag"])
        for row in normalized_rows
        if build_metadata_lookup_key(row["source_file_name"], row["version_tag"]) not in mapping
    ]
    if missing:
        raise RuntimeError(f"No se obtuvo metadata_id para: {missing}")

    return mapping


def upsert_document_chunks(supabase_client, chunk_rows: list[dict], batch_size: int = settings.SAFE_BATCH_SIZE) -> None:
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor o igual a 1.")
    if not chunk_rows:
        return

    normalized = []
    for row in chunk_rows:
        metadata_id = str(row.get("metadata_id", "")).strip()
        content = str(row.get("content", "")).strip()
        embedding = row.get("embedding")
        chunk_index = row.get("chunk_index")

        if not metadata_id or not content:
            raise ValueError("Cada chunk row requiere metadata_id y content.")
        if not isinstance(chunk_index, int):
            raise TypeError("chunk_index debe ser int.")
        if chunk_index < 0:
            raise ValueError("chunk_index debe ser mayor o igual a 0.")
        if not isinstance(embedding, list):
            raise TypeError("embedding debe ser list[float].")

        validate_embedding(embedding, settings.EMBEDDING_DIM)
        normalized.append(
            {
                "metadata_id": metadata_id,
                "content": content,
                "chunk_index": chunk_index,
                "embedding": embedding,
            }
        )

    for i in range(0, len(normalized), batch_size):
        supabase_client.table("documents_embeddings").upsert(
            normalized[i : i + batch_size],
            on_conflict="metadata_id,chunk_index",
        ).execute()


def _generate_embeddings_with_retry(
    pplx_client: Perplexity,
    chunks: list[str],
    embedding_model: str,
    max_retries: int = 5,
):
    for attempt in range(max_retries):
        try:
            return pplx_client.contextualized_embeddings.create(
                input=[chunks],
                model=embedding_model,
            )
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            if status_code == 429 and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                logger.warning(
                    "Rate limit recibido, esperando %ss (intento %s/%s).",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
                continue
            raise
        except httpx.RequestError:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                logger.warning(
                    "Error de red, reintentando en %ss (intento %s/%s).",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("No fue posible generar embeddings tras los reintentos configurados.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        logger.error("No se encontraron documentos en docs/. Agrega PDFs y vuelve a ejecutar.")
        sys.exit(1)

    logger.info("Cargando documentos de %s...", DOCS_DIR)
    documents = SimpleDirectoryReader(DOCS_DIR, exclude_hidden=False).load_data()
    file_names = sorted(set(d.metadata.get("file_name", "") for d in documents if d.metadata.get("file_name")))
    logger.info("Cargados %s fragmentos de %s archivos.", len(documents), len(file_names))

    docs_by_file = defaultdict(list)
    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown")
        docs_by_file[file_name].append(doc)

    grouped_chunks = []
    file_order = []
    for file_name, file_docs in docs_by_file.items():
        chunk_texts = []
        for doc in file_docs:
            text = doc.text.strip()
            if text:
                chunk_texts.append(text)
        if chunk_texts:
            grouped_chunks.append(chunk_texts)
            file_order.append(file_name)

    request_groups = []
    for chunks, file_name in zip(grouped_chunks, file_order):
        if len(chunks) <= settings.MAX_CHUNKS_PER_REQUEST:
            request_groups.append((chunks, file_name, 0))
            continue
        for start in range(0, len(chunks), settings.MAX_CHUNKS_PER_REQUEST):
            request_groups.append((chunks[start : start + settings.MAX_CHUNKS_PER_REQUEST], file_name, start))

    total_chunks = sum(len(group[0]) for group in request_groups)
    logger.info(
        "%s documentos -> %s requests, %s chunks totales.",
        len(grouped_chunks),
        len(request_groups),
        total_chunks,
    )

    logger.info("Generando embeddings con %s...", settings.EMBEDDING_MODEL)
    pplx_client = Perplexity(api_key=settings.PERPLEXITY_API_KEY, max_retries=0)
    supabase_client = get_supabase_client()

    metadata_map = upsert_metadata_rows(
        supabase_client=supabase_client,
        metadata_rows=build_metadata_rows(
            file_names=file_order,
            version_tag=DEFAULT_VERSION_TAG,
            category=DEFAULT_CATEGORY,
        ),
    )

    all_rows = []
    total_tokens = 0
    for req_idx, (chunks, file_name, offset) in enumerate(request_groups):
        part_info = f" (parte desde chunk {offset})" if offset > 0 else ""
        logger.info(
            "[%s/%s] %s (%s chunks)%s...",
            req_idx + 1,
            len(request_groups),
            file_name,
            len(chunks),
            part_info,
        )
        response = _generate_embeddings_with_retry(
            pplx_client=pplx_client,
            chunks=chunks,
            embedding_model=settings.EMBEDDING_MODEL,
        )

        total_tokens += response.usage.total_tokens
        metadata_key = build_metadata_lookup_key(file_name, DEFAULT_VERSION_TAG)
        metadata_id = metadata_map[metadata_key]

        for chunk_obj in response.data[0].data:
            global_index = offset + chunk_obj.index
            all_rows.append(
                {
                    "metadata_id": metadata_id,
                    "content": chunks[chunk_obj.index],
                    "chunk_index": global_index,
                    "embedding": decode_embedding(chunk_obj.embedding),
                }
            )

        if req_idx < len(request_groups) - 1:
            time.sleep(5)

    upsert_document_chunks(
        supabase_client=supabase_client,
        chunk_rows=all_rows,
        batch_size=settings.SAFE_BATCH_SIZE,
    )

    logger.info("Indexacion completada.")
    logger.info("%s fragmentos en Supabase.", len(all_rows))
    logger.info("Tokens usados: %s", total_tokens)


if __name__ == "__main__":
    main()
