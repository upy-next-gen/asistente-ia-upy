from langfuse import observe

from core.config import settings
from core.supabase_vector_db.retriever import SupabaseVectorRetriever


class DocumentRetriever:
    def __init__(self):
        self._retriever = SupabaseVectorRetriever()

    @observe(name="document_retrieval")
    def retrieve(self, query_embedding: list[float], top_k: int | None = None) -> list[dict]:
        match_count = top_k if top_k is not None else settings.RETRIEVER_TOP_K
        return self._retriever.search(
            query_embedding=query_embedding,
            match_count=match_count,
            min_similarity=settings.MIN_SIMILARITY,
        )