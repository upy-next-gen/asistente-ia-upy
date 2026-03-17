import chromadb
from langfuse.openai import AsyncOpenAI
from supabase import create_client, Client
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
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.LLM_BASE_URL,
            )
        return self._llm_client

    @property
    def supabase(self) -> Client | None:
        if self._supabase is None and settings.SUPABASE_URL and settings.SUPABASE_KEY:
            self._supabase = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY,
            )
        return self._supabase

    @property
    def vector_store(self) -> ChromaVectorStore:
        if self._vector_store is None:
            chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
            collection = chroma_client.get_or_create_collection(
                settings.CHROMA_COLLECTION_NAME,
            )
            self._vector_store = ChromaVectorStore(chroma_collection=collection)
        return self._vector_store


clients = ClientManager()


