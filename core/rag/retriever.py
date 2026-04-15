from langfuse import observe

from core.config import settings
from core.clients import get_supabase_client
from core.supabase_vector_db.store import SupabaseVectorStore


class DocumentRetriever:
    def __init__(self):
        self._store = SupabaseVectorStore(get_supabase_client())

    @observe(name="document_retrieval")
    def retrieve(self, query_embedding: list[float], top_k: int | None = None) -> list[dict]:
        match_count = top_k if top_k is not None else settings.RETRIEVER_TOP_K
        
        payload = self._store.build_match_documents_payload(
            query_embedding=query_embedding,
            match_count=match_count,
            min_similarity=settings.MIN_SIMILARITY,
        )
        response = self._store.client.rpc("match_documents", payload).execute()
        rows = getattr(response, "data", None) or []
        return [row for row in rows if isinstance(row, dict)]
