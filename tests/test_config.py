import os
from unittest.mock import patch
from core.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.deepseek_api_key == ""
        assert s.deepseek_base_url == "https://api.deepseek.com"
        assert s.deepseek_model == "deepseek-chat"
        assert s.supabase_url == ""
        assert s.supabase_key == ""
        assert s.embed_model_name == "BAAI/bge-small-en-v1.5"
        assert s.chroma_collection_name == "upy_docs"
        assert s.retriever_top_k == 5

    @patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "test-key",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "sb-key",
    })
    def test_from_env(self):
        s = Settings.from_env()
        assert s.deepseek_api_key == "test-key"
        assert s.supabase_url == "https://test.supabase.co"
        assert s.supabase_key == "sb-key"
        assert s.chroma_dir != ""

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_missing_keys(self):
        s = Settings.from_env()
        assert s.deepseek_api_key == ""
        assert s.supabase_url == ""
        assert s.supabase_key == ""