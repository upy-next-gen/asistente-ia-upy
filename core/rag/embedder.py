"""Query embedder abstraction for Perplexity contextualized embeddings."""

from perplexity import Perplexity

from core.supabase_vector_db.indexing_utils import decode_embedding


class QueryEmbedder:
    """Genera embeddings de consultas usando la API de Perplexity."""

    def __init__(self, client: Perplexity, model: str) -> None:
        """Initialize the query embedder.

        Args:
            client: Configured Perplexity API client.
            model: Embedding model identifier.
        """
        self._client = client
        self._model = model

    def embed(self, query_text: str) -> list[float]:
        """Genera el embedding de una consulta de texto.

        Args:
            query_text: El texto de la consulta del usuario.

        Returns:
            Lista de floats con el embedding (1024 dimensiones).
        """
        response = self._client.contextualized_embeddings.create(
            input=[[query_text]],
            model=self._model,
        )
        return decode_embedding(response.data[0].data[0].embedding)
