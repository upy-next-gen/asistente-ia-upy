import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestChatService:
    @patch("core.llm.chat.clients")
    @patch("core.llm.chat.settings")
    @pytest.mark.asyncio
    async def test_stream_calls_api(self, mock_settings, mock_clients):
        mock_settings.deepseek_model = "deepseek-chat"
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