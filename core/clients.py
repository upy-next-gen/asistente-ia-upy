import chromadb
from langfuse.openai import AsyncOpenAI
from supabase import create_client, Client
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from core.config import settings


class ClientManager:
    def __init__(self):
        self._llm_client: AsyncOpenAI | None = None
        self._supabase: Client | None = None
        self._vector_store: ChromaVectorStore | None = None

    @property
    def llm(self) -> AsyncOpenAI:
        if self._llm_client is None:
            self._llm_client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
        return self._llm_client

    @property
    def supabase(self) -> Client | None:
        if self._supabase is None and settings.supabase_url and settings.supabase_key:
            self._supabase = create_client(
                settings.supabase_url,
                settings.supabase_key,
            )
        return self._supabase

    @property
    def vector_store(self) -> ChromaVectorStore:
        if self._vector_store is None:
            chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
            collection = chroma_client.get_or_create_collection(
                settings.chroma_collection_name,
            )
            self._vector_store = ChromaVectorStore(chroma_collection=collection)
        return self._vector_store

    def init_embeddings(self) -> None:
        LlamaSettings.embed_model = HuggingFaceEmbedding(
            model_name=settings.embed_model_name,
        )


clients = ClientManager()