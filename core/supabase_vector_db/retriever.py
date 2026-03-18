from core.clients import get_supabase_client
from core.supabase_vector_db.store import SupabaseVectorStore


class SupabaseVectorRetriever:
    def __init__(self):
        self._store = SupabaseVectorStore(get_supabase_client())

    def search(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        payload = self._store.build_match_documents_payload(
            query_embedding=query_embedding,
            match_count=match_count,
            min_similarity=min_similarity,
        )
        response = self._store.client.rpc("match_documents", payload).execute()
        rows = getattr(response, "data", None) or []
        return [row for row in rows if isinstance(row, dict)]
