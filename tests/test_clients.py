from unittest.mock import patch, MagicMock
from core.clients import ClientManager


class TestClientManager:
    def test_llm_lazy_init(self):
        with patch("core.clients.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            manager = ClientManager()
            assert manager._llm_client is None
            client = manager.llm
            assert client is not None
            mock_openai.assert_called_once()

    def test_llm_returns_same_instance(self):
        with patch("core.clients.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            manager = ClientManager()
            first = manager.llm
            second = manager.llm
            assert first is second
            assert mock_openai.call_count == 1

    def test_supabase_returns_none_without_config(self):
        with patch("core.clients.settings") as mock_settings:
            mock_settings.supabase_url = ""
            mock_settings.supabase_key = ""
            manager = ClientManager()
            assert manager.supabase is None

    def test_supabase_initializes_with_config(self):
        with patch("core.clients.settings") as mock_settings, \
             patch("core.clients.create_client") as mock_create:
            mock_settings.supabase_url = "https://test.supabase.co"
            mock_settings.supabase_key = "key"
            mock_create.return_value = MagicMock()
            manager = ClientManager()
            client = manager.supabase
            assert client is not None
            mock_create.assert_called_once_with("https://test.supabase.co", "key")

    def test_vector_store_lazy_init(self):
        with patch("core.clients.chromadb") as mock_chroma, \
             patch("core.clients.ChromaVectorStore") as mock_vs:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_vs.return_value = MagicMock()
            manager = ClientManager()
            assert manager._vector_store is None
            store = manager.vector_store
            assert store is not None

    def test_init_embeddings(self):
        with patch("core.clients.LlamaSettings") as mock_llama, \
             patch("core.clients.HuggingFaceEmbedding") as mock_hf:
            mock_hf.return_value = MagicMock()
            manager = ClientManager()
            manager.init_embeddings()
            mock_hf.assert_called_once()