from typing import Any

from core.config import EMBEDDING_DIM


class SupabaseVectorStore:
    def __init__(self, supabase_client):
        self._supabase_client = supabase_client

    def validate_embedding(self, embedding: list[float], expected_dim: int = EMBEDDING_DIM) -> None:
        if len(embedding) != expected_dim:
            raise ValueError(
                f"Embedding con dimension invalida: {len(embedding)}. Esperada: {expected_dim}."
            )
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise TypeError("Todos los valores del embedding deben ser numericos.")

    def build_match_documents_payload(
        self,
        query_embedding: list[float],
        match_count: int,
        min_similarity: float,
    ) -> dict[str, Any]:
        if match_count < 1:
            raise ValueError("match_count debe ser mayor o igual a 1.")
        if not isinstance(min_similarity, (int, float)):
            raise TypeError("min_similarity debe ser numerico.")
        self.validate_embedding(query_embedding)
        return {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "min_similarity": float(min_similarity),
        }

    @property
    def client(self):
        return self._supabase_client
