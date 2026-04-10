from perplexity import Perplexity

from core.config import settings
from core.supabase_vector_db.indexing_utils import decode_embedding


class EmbeddingService:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = Perplexity(api_key=api_key or settings.PERPLEXITY_API_KEY)
        self._model = model or settings.EMBEDDING_MODEL

    def embed_query(self, query_text: str) -> list[float]:
        response = self._client.contextualized_embeddings.create(
            input=[[query_text]],
            model=self._model,
        )
        return decode_embedding(response.data[0].data[0].embedding)
