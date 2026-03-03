from langfuse import observe
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from core.clients import clients
from core.config import settings


class DocumentRetriever:
    def __init__(self):
        clients.init_embeddings()
        index = VectorStoreIndex.from_vector_store(clients.vector_store)
        self._retriever = index.as_retriever(
            similarity_top_k=settings.retriever_top_k,
        )

    @observe(name="document_retrieval")
    def retrieve(self, query: str) -> list[NodeWithScore]:
        return self._retriever.retrieve(query)
