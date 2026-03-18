import os
import pytest
from unittest.mock import patch
from core.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.OPENAI_API_KEY == ""
        assert s.LLM_BASE_URL == ""
        assert s.LLM_MODEL == ""
        assert s.SUPABASE_URL == ""
        assert s.SUPABASE_KEY == ""
        assert s.PERPLEXITY_API_KEY == ""
        assert s.EMBEDDING_MODEL == "pplx-embed-context-v1-0.6b"
        assert s.MIN_SIMILARITY == 0.3
        assert s.EMBEDDING_DIM == 1024
        assert s.SAFE_BATCH_SIZE == 200
        assert s.MAX_CHUNKS_PER_REQUEST == 20
        assert s.RETRIEVER_TOP_K == 5
        assert s.MAX_HISTORY_MESSAGES == 20
        assert s.MAX_MESSAGE_LENGTH == 2000
        assert s.MAX_SUGGESTION_LENGTH == 1000
        assert s.LLM_TIMEOUT == 30
        assert s.RATE_LIMIT_WINDOW == 60
        assert s.RATE_LIMIT_MAX_REQUESTS == 10

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "LLM_BASE_URL": "https://api.deepseek.com",
        "LLM_MODEL": "deepseek-chat",
        "PERPLEXITY_API_KEY": "pplx-test",
        "EMBEDDING_MODEL": "pplx-embed-context-v1-0.6b",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "sb-key",
        "MIN_SIMILARITY": "0.4",
        "EMBEDDING_DIM": "1024",
        "SAFE_BATCH_SIZE": "250",
    })
    def test_from_env_full(self):
        s = Settings.from_env()
        assert s.OPENAI_API_KEY == "test-key"
        assert s.LLM_BASE_URL == "https://api.deepseek.com"
        assert s.LLM_MODEL == "deepseek-chat"
        assert s.PERPLEXITY_API_KEY == "pplx-test"
        assert s.EMBEDDING_MODEL == "pplx-embed-context-v1-0.6b"
        assert s.SUPABASE_URL == "https://test.supabase.co"
        assert s.SUPABASE_KEY == "sb-key"
        assert s.MIN_SIMILARITY == 0.4
        assert s.EMBEDDING_DIM == 1024
        assert s.SAFE_BATCH_SIZE == 250

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "LLM_BASE_URL": "https://api.deepseek.com",
        "LLM_MODEL": "deepseek-chat",
        "PERPLEXITY_API_KEY": "pplx-test",
        "EMBEDDING_MODEL": "pplx-embed-context-v1-0.6b",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "sb-key",
        "RETRIEVER_TOP_K": "10",
        "MAX_HISTORY_MESSAGES": "30",
        "MAX_MESSAGE_LENGTH": "5000",
        "MAX_SUGGESTION_LENGTH": "2000",
        "LLM_TIMEOUT": "60",
        "RATE_LIMIT_WINDOW": "120",
        "RATE_LIMIT_MAX_REQUESTS": "20",
    })
    def test_from_env_operational_params(self):
        s = Settings.from_env()
        assert s.RETRIEVER_TOP_K == 10
        assert s.MAX_HISTORY_MESSAGES == 30
        assert s.MAX_MESSAGE_LENGTH == 5000
        assert s.MAX_SUGGESTION_LENGTH == 2000
        assert s.LLM_TIMEOUT == 60
        assert s.RATE_LIMIT_WINDOW == 120
        assert s.RATE_LIMIT_MAX_REQUESTS == 20

    @patch("core.config.load_dotenv")
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_missing_openai_key_raises(self, mock_dotenv):
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            Settings.from_env()

    @patch("core.config.load_dotenv")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_from_env_missing_llm_base_url_raises(self, mock_dotenv):
        with pytest.raises(EnvironmentError, match="LLM_BASE_URL"):
            Settings.from_env()

    @patch("core.config.load_dotenv")
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "LLM_BASE_URL": "https://api.deepseek.com",
        "LLM_MODEL": "deepseek-chat",
    }, clear=True)
    def test_from_env_missing_perplexity_key_raises(self, mock_dotenv):
        with pytest.raises(EnvironmentError, match="PERPLEXITY_API_KEY"):
            Settings.from_env()

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "LLM_BASE_URL": "https://api.deepseek.com",
        "LLM_MODEL": "deepseek-chat",
        "PERPLEXITY_API_KEY": "pplx-test",
        "EMBEDDING_MODEL": "pplx-embed-context-v1-0.6b",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "sb-key",
    })
    def test_from_env_requires_supabase(self):
        s = Settings.from_env()
        assert s.SUPABASE_URL == "https://test.supabase.co"
        assert s.SUPABASE_KEY == "sb-key"
        