from langfuse.openai import AsyncOpenAI
from supabase import create_client, Client

from core.config import settings


class ClientManager:
    def __init__(self):
        self._llm_client: AsyncOpenAI | None = None
        self._supabase: Client | None = None

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


clients = ClientManager()


def get_supabase_client() -> Client:
    supabase = clients.supabase
    if supabase is None:
        raise RuntimeError("Supabase no está configurado. Revisa SUPABASE_URL y SUPABASE_KEY.")
    return supabase
