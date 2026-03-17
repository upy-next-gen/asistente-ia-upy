from typing import Any

from supabase import create_client

from vector_supabase.settings import get_vector_settings

EMBEDDING_DIM = 1024
SAFE_BATCH_SIZE = 200


def build_metadata_lookup_key(source_file_name: str, version_tag: str) -> str:
    return f"{source_file_name}::{version_tag}"


def create_supabase_client_from_env():
    settings = get_vector_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def validate_embedding(embedding: list[float], expected_dim: int = EMBEDDING_DIM) -> None:
    if len(embedding) != expected_dim:
        raise ValueError(
            f"Embedding con dimension invalida: {len(embedding)}. Esperada: {expected_dim}."
        )
    if not all(isinstance(value, (int, float)) for value in embedding):
        raise TypeError("Todos los valores del embedding deben ser numericos.")


def build_match_documents_payload(
    query_embedding: list[float],
    match_count: int = 5,
    min_similarity: float = 0.3,
) -> dict[str, Any]:
    if match_count < 1:
        raise ValueError("match_count debe ser mayor o igual a 1.")
    if not isinstance(min_similarity, (int, float)):
        raise TypeError("min_similarity debe ser numerico.")
    validate_embedding(query_embedding)
    return {
        "query_embedding": query_embedding,
        "match_count": match_count,
        "min_similarity": float(min_similarity),
    }


def search_documents_by_embedding(
    supabase_client,
    query_embedding: list[float],
    match_count: int = 5,
    min_similarity: float = 0.3,
) -> list[dict[str, Any]]:
    payload = build_match_documents_payload(
        query_embedding=query_embedding,
        match_count=match_count,
        min_similarity=min_similarity,
    )
    response = supabase_client.rpc("match_documents", payload).execute()
    rows = getattr(response, "data", None) or []
    return [row for row in rows if isinstance(row, dict)]


def upsert_metadata_rows(
    supabase_client,
    metadata_rows: list[dict[str, Any]],
) -> dict[str, str]:
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


def upsert_document_chunks(
    supabase_client,
    chunk_rows: list[dict[str, Any]],
    batch_size: int = SAFE_BATCH_SIZE,
) -> None:
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
        validate_embedding(embedding)

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

