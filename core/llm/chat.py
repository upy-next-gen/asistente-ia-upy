import asyncio
from langfuse import observe

from core.clients import clients
from core.config import settings


class ChatService:
    def __init__(self):
        self._client = clients.llm
        self._model = settings.LLM_MODEL
        self._timeout = settings.LLM_TIMEOUT

    @observe(name="llm_stream")
    async def stream(self, messages: list[dict]) -> object:
        return await asyncio.wait_for(
            self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
            ),
            timeout=self._timeout,
        )