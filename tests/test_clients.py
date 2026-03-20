import pytest
from unittest.mock import patch, MagicMock
from core.clients import ClientManager, get_supabase_client


class TestClientManager:
    def test_llm_lazy_init(self):
        with patch("core.clients.AsyncOpenAI") as mock_openai, \
             patch("core.clients.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.LLM_BASE_URL = "https://api.deepseek.com"
            mock_openai.return_value = MagicMock()
            manager = ClientManager()
            assert manager._llm_client is None
            client = manager.llm
            assert client is not None
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.deepseek.com",
            )

    def test_llm_returns_same_instance(self):
        with patch("core.clients.AsyncOpenAI") as mock_openai, \
             patch("core.clients.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.LLM_BASE_URL = "https://api.deepseek.com"
            mock_openai.return_value = MagicMock()
            manager = ClientManager()
            first = manager.llm
            second = manager.llm
            assert first is second
            assert mock_openai.call_count == 1

    def test_supabase_returns_none_without_config(self):
        with patch("core.clients.settings") as mock_settings:
            mock_settings.SUPABASE_URL = ""
            mock_settings.SUPABASE_KEY = ""
            manager = ClientManager()
            assert manager.supabase is None

    def test_supabase_returns_none_with_partial_config(self):
        with patch("core.clients.settings") as mock_settings:
            mock_settings.SUPABASE_URL = "https://test.supabase.co"
            mock_settings.SUPABASE_KEY = ""
            manager = ClientManager()
            assert manager.supabase is None

    def test_supabase_initializes_with_config(self):
        with patch("core.clients.settings") as mock_settings, \
             patch("core.clients.create_client") as mock_create:
            mock_settings.SUPABASE_URL = "https://test.supabase.co"
            mock_settings.SUPABASE_KEY = "sb-key"
            mock_create.return_value = MagicMock()
            manager = ClientManager()
            client = manager.supabase
            assert client is not None
            mock_create.assert_called_once_with(
                "https://test.supabase.co",
                "sb-key",
            )

    def test_supabase_returns_same_instance(self):
        with patch("core.clients.settings") as mock_settings, \
             patch("core.clients.create_client") as mock_create:
            mock_settings.SUPABASE_URL = "https://test.supabase.co"
            mock_settings.SUPABASE_KEY = "sb-key"
            mock_create.return_value = MagicMock()
            manager = ClientManager()
            first = manager.supabase
            second = manager.supabase
            assert first is second
            assert mock_create.call_count == 1

    @patch("core.clients.clients")
    def test_get_supabase_client_raises_if_not_configured(self, mock_clients):
        mock_clients.supabase = None
        with pytest.raises(RuntimeError, match="Supabase no está configurado"):
            get_supabase_client()

    @patch("core.clients.clients")
    def test_get_supabase_client_returns_client(self, mock_clients):
        client = MagicMock()
        mock_clients.supabase = client
        assert get_supabase_client() is client
               