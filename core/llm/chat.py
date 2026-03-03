from langfuse import observe
from core.clients import clients
from core.config import settings


class ChatService:
    def __init__(self):
        self._client = clients.llm
        self._model = settings.deepseek_model

    @observe(name="llm_stream")
    async def stream(self, messages: list[dict]) -> object:
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )
