import base64

import chromadb
import numpy as np
from langfuse import observe
from llama_index.core.schema import NodeWithScore, TextNode
from perplexity import Perplexity

from core.config import settings


class DocumentRetriever:
    def __init__(self):
        self._pplx_client = Perplexity(api_key=settings.PERPLEXITY_API_KEY)
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self._collection = chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_query(self, query: str) -> list[float]:
        response = self._pplx_client.contextualized_embeddings.create(
            input=[[query]],
            model=settings.EMBEDDING_MODEL,
        )
        raw = base64.b64decode(response.data[0].data[0].embedding)
        return np.frombuffer(raw, dtype=np.int8).astype(np.float32).tolist()

    @observe(name="document_retrieval")
    def retrieve(self, query: str) -> list[NodeWithScore]:
        embedding = self._embed_query(query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=settings.RETRIEVER_TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        nodes = []
        if not results or not results["documents"] or not results["documents"][0]:
            return nodes

        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            node = NodeWithScore(
                node=TextNode(text=text, metadata=metadata),
                score=1.0 - distance,
            )
            nodes.append(node)

        return nodes
    

    