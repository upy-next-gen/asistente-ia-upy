import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestChatService:
    @pytest.mark.asyncio
    @patch("core.llm.chat.clients")
    @patch("core.llm.chat.settings")
    async def test_stream_calls_api(self, mock_settings, mock_clients):
        mock_settings.LLM_MODEL = "deepseek-chat"
        mock_settings.LLM_TIMEOUT = 30
        mock_llm = MagicMock()
        mock_llm.chat.completions.create = AsyncMock(return_value="stream_obj")
        mock_clients.llm = mock_llm

        from core.llm.chat import ChatService
        service = ChatService()
        messages = [{"role": "user", "content": "test"}]
        result = await service.stream(messages)

        mock_llm.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=messages,
            stream=True,
        )
        assert result == "stream_obj"

    @pytest.mark.asyncio
    @patch("core.llm.chat.clients")
    @patch("core.llm.chat.settings")
    async def test_stream_passes_full_history(self, mock_settings, mock_clients):
        mock_settings.LLM_MODEL = "deepseek-chat"
        mock_settings.LLM_TIMEOUT = 30
        mock_llm = MagicMock()
        mock_llm.chat.completions.create = AsyncMock(return_value="stream_obj")
        mock_clients.llm = mock_llm

        from core.llm.chat import ChatService
        service = ChatService()
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "primera pregunta"},
            {"role": "assistant", "content": "primera respuesta"},
            {"role": "user", "content": "segunda pregunta"},
        ]
        await service.stream(messages)

        mock_llm.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=messages,
            stream=True,
        )

    @pytest.mark.asyncio
    @patch("core.llm.chat.asyncio")
    @patch("core.llm.chat.clients")
    @patch("core.llm.chat.settings")
    async def test_stream_raises_on_timeout(self, mock_settings, mock_clients, mock_asyncio):
        mock_settings.LLM_MODEL = "deepseek-chat"
        mock_settings.LLM_TIMEOUT = 1
        mock_asyncio.wait_for = AsyncMock(side_effect=asyncio.TimeoutError)

        from core.llm.chat import ChatService
        service = ChatService()

        with pytest.raises(asyncio.TimeoutError):
            await service.stream([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    @patch("core.llm.chat.clients")
    @patch("core.llm.chat.settings")
    async def test_stream_uses_correct_model(self, mock_settings, mock_clients):
        mock_settings.LLM_MODEL = "custom-model"
        mock_settings.LLM_TIMEOUT = 30
        mock_llm = MagicMock()
        mock_llm.chat.completions.create = AsyncMock(return_value="stream_obj")
        mock_clients.llm = mock_llm

        from core.llm.chat import ChatService
        service = ChatService()
        await service.stream([{"role": "user", "content": "test"}])

        call_kwargs = mock_llm.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    @patch("core.llm.chat.clients")
    @patch("core.llm.chat.settings")
    async def test_stream_always_sets_stream_true(self, mock_settings, mock_clients):
        mock_settings.LLM_MODEL = "deepseek-chat"
        mock_settings.LLM_TIMEOUT = 30
        mock_llm = MagicMock()
        mock_llm.chat.completions.create = AsyncMock(return_value="stream_obj")
        mock_clients.llm = mock_llm

        from core.llm.chat import ChatService
        service = ChatService()
        await service.stream([{"role": "user", "content": "test"}])

        call_kwargs = mock_llm.chat.completions.create.call_args
        assert call_kwargs.kwargs["stream"] is True

        