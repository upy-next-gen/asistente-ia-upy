"""
Indexer independiente para Supabase pgvector usando embeddings contextuales de Perplexity.

Uso:
    uv run python -m vector_supabase.index_docs_pplx_supabase
    uv run upy-index-supabase
"""

import os
import sys
import time
import logging
from collections import defaultdict

from dotenv import load_dotenv
import httpx
from llama_index.core import SimpleDirectoryReader
from perplexity import Perplexity

from vector_supabase.indexing_utils import build_metadata_rows, decode_embedding
from vector_supabase.settings import get_indexer_settings
from vector_supabase.vector_store import (
    SAFE_BATCH_SIZE,
    build_metadata_lookup_key,
    create_supabase_client_from_env,
    upsert_document_chunks,
    upsert_metadata_rows,
)

load_dotenv()

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
logger = logging.getLogger(__name__)
DEFAULT_VERSION_TAG = "current"
DEFAULT_CATEGORY = "general"


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
    settings = get_indexer_settings()

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

    max_chunks_per_request = 20
    request_groups = []
    for chunks, file_name in zip(grouped_chunks, file_order):
        if len(chunks) <= max_chunks_per_request:
            request_groups.append((chunks, file_name, 0))
            continue
        for start in range(0, len(chunks), max_chunks_per_request):
            request_groups.append((chunks[start : start + max_chunks_per_request], file_name, start))

    total_chunks = sum(len(group[0]) for group in request_groups)
    logger.info(
        "%s documentos -> %s requests, %s chunks totales.",
        len(grouped_chunks),
        len(request_groups),
        total_chunks,
    )

    logger.info("Generando embeddings con %s...", settings.EMBEDDING_MODEL)
    pplx_client = Perplexity(api_key=settings.PERPLEXITY_API_KEY, max_retries=0)
    supabase_client = create_supabase_client_from_env()

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
        batch_size=SAFE_BATCH_SIZE,
    )

    logger.info("Indexacion completada.")
    logger.info("%s fragmentos en Supabase.", len(all_rows))
    logger.info("Tokens usados: %s", total_tokens)


if __name__ == "__main__":
    main()

