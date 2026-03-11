import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    supabase_url: str = ""
    supabase_key: str = ""
    perplexity_api_key: str = ""
    embed_model_name: str = "r-text-embedding-3-small"
    chroma_dir: str = ""
    chroma_collection_name: str = "upy_docs_pplx"
    retriever_top_k: int = 5
    max_history_messages: int = 20
    max_message_length: int = 2000
    max_suggestion_length: int = 1000
    llm_timeout: int = 30
    rate_limit_window: int = 60
    rate_limit_max_requests: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY", ""),
            embed_model_name=os.getenv("EMBEDDING_MODEL", "r-text-embedding-3-small"),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_KEY", ""),
            chroma_dir=os.path.join(os.path.dirname(__file__), "..", "chroma_db"),
            max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "20")),
            max_message_length=int(os.getenv("MAX_MESSAGE_LENGTH", "2000")),
            max_suggestion_length=int(os.getenv("MAX_SUGGESTION_LENGTH", "1000")),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "30")),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            rate_limit_max_requests=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10")),
        )


settings = Settings.from_env()