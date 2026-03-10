"""
Script experimental para indexar documentos PDF usando embeddings contextuales de Perplexity.

Los contextual embeddings procesan cada chunk considerando el contexto completo
del documento al que pertenece (late chunking), mejorando la calidad del retrieval.

Uso:
    uv run python scripts/index_docs_pplx.py

Requiere PERPLEXITY_API_KEY en .env.
"""

import os
import sys
import base64
import time
from collections import defaultdict

import numpy as np
import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from perplexity import Perplexity

load_dotenv()

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
COLLECTION_NAME = "upy_docs_pplx"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def decode_embedding(b64_string: str) -> list[float]:
    """Decodificar un embedding base64 int8 a lista de floats."""
    raw = base64.b64decode(b64_string)
    return np.frombuffer(raw, dtype=np.int8).astype(np.float32).tolist()


def main():
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY no configurada en .env")
        sys.exit(1)

    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print("No se encontraron documentos en docs/. Agrega PDFs y vuelve a ejecutar.")
        sys.exit(1)

    # 1. Cargar documentos con LlamaIndex (solo para parsing de PDFs)
    print(f"Cargando documentos de {DOCS_DIR}...")
    documents = SimpleDirectoryReader(DOCS_DIR, exclude_hidden=False).load_data()
    file_names = set(d.metadata.get("file_name", "") for d in documents)
    print(f"  {len(documents)} fragmentos de {len(file_names)} archivos.")

    # 2. Agrupar chunks por documento fuente, preservando orden
    docs_by_file = defaultdict(list)
    for doc in documents:
        fname = doc.metadata.get("file_name", "unknown")
        docs_by_file[fname].append(doc)

    grouped_chunks = []  # Lista de listas de strings (input para Perplexity)
    chunk_metadata = []  # Metadata plana para mapear respuesta → ChromaDB
    file_order = []  # Orden de archivos para mapear doc_index → file_name

    for fname, file_docs in docs_by_file.items():
        chunk_texts = []
        for i, doc in enumerate(file_docs):
            text = doc.text.strip()
            if text:
                chunk_texts.append(text)
                chunk_metadata.append({
                    "file_name": fname,
                    "chunk_index": i,
                })
        if chunk_texts:
            grouped_chunks.append(chunk_texts)
            file_order.append(fname)

    # Dividir documentos grandes en sub-grupos para evitar límites de la API
    MAX_CHUNKS_PER_REQUEST = 20
    request_groups = []  # Lista de (sub_chunks, fname, offset)

    for chunks, fname in zip(grouped_chunks, file_order):
        if len(chunks) <= MAX_CHUNKS_PER_REQUEST:
            request_groups.append((chunks, fname, 0))
        else:
            for start in range(0, len(chunks), MAX_CHUNKS_PER_REQUEST):
                sub = chunks[start:start + MAX_CHUNKS_PER_REQUEST]
                request_groups.append((sub, fname, start))

    total_chunks = sum(len(g[0]) for g in request_groups)
    print(f"  {len(grouped_chunks)} documentos → {len(request_groups)} requests, {total_chunks} chunks totales.")

    # 3. Generar embeddings con Perplexity API
    print(f"Generando embeddings con {EMBEDDING_MODEL}...")
    pplx_client = Perplexity(api_key=api_key, max_retries=0)

    all_embeddings = []
    all_texts = []
    all_ids = []
    all_metadatas = []
    total_tokens = 0

    for req_idx, (chunks, fname, offset) in enumerate(request_groups):
        part_info = f" (parte desde chunk {offset})" if offset > 0 else ""
        print(f"  [{req_idx + 1}/{len(request_groups)}] {fname} ({len(chunks)} chunks){part_info}...")

        # Retry con backoff exponencial
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = pplx_client.contextualized_embeddings.create(
                    input=[chunks],
                    model=EMBEDDING_MODEL,
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"    Rate limit, esperando {wait}s (intento {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise

        total_tokens += response.usage.total_tokens

        for chunk_obj in response.data[0].data:
            embedding = decode_embedding(chunk_obj.embedding)
            chunk_text = chunks[chunk_obj.index]
            global_index = offset + chunk_obj.index
            chunk_id = f"pplx_{fname}_{global_index}"

            all_embeddings.append(embedding)
            all_texts.append(chunk_text)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "file_name": fname,
                "chunk_index": global_index,
            })

        # Pausa entre requests para evitar rate limit
        if req_idx < len(request_groups) - 1:
            time.sleep(5)

    print(f"  {len(all_embeddings)} embeddings generados (dim={len(all_embeddings[0])}).")

    # 5. Guardar en ChromaDB
    print(f"Guardando en ChromaDB colección '{COLLECTION_NAME}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Eliminar colección existente para re-indexar limpio
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insertar en batches
    BATCH_SIZE = 5000
    for i in range(0, len(all_ids), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(all_ids))
        collection.add(
            ids=all_ids[i:end],
            embeddings=all_embeddings[i:end],
            documents=all_texts[i:end],
            metadatas=all_metadatas[i:end],
        )

    print(f"\nIndexación completada:")
    print(f"  {len(all_ids)} fragmentos en colección '{COLLECTION_NAME}'")
    print(f"  Base vectorial: {CHROMA_DIR}")
    print(f"  Tokens usados: {total_tokens}")


if __name__ == "__main__":
    main()
