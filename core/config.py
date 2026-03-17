import os
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class Settings:
    OPENAI_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    PERPLEXITY_API_KEY: str = ""
    EMBEDDING_MODEL: str = ""
    CHROMA_DIR: str = ""
    CHROMA_COLLECTION_NAME: str = ""
    RETRIEVER_TOP_K: int = 0
    MAX_HISTORY_MESSAGES: int = 0
    MAX_MESSAGE_LENGTH: int = 0
    MAX_SUGGESTION_LENGTH: int = 0
    LLM_TIMEOUT: int = 0
    RATE_LIMIT_WINDOW: int = 0
    RATE_LIMIT_MAX_REQUESTS: int = 0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        instance = cls(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            LLM_BASE_URL=os.getenv("LLM_BASE_URL", ""),
            LLM_MODEL=os.getenv("LLM_MODEL", ""),
            SUPABASE_URL=os.getenv("SUPABASE_URL", ""),
            SUPABASE_KEY=os.getenv("SUPABASE_KEY", ""),
            PERPLEXITY_API_KEY=os.getenv("PERPLEXITY_API_KEY", ""),
            EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", ""),
            CHROMA_DIR=os.path.join(os.path.dirname(__file__), "..", "chroma_db"),
            CHROMA_COLLECTION_NAME=os.getenv("CHROMA_COLLECTION_NAME", ""),
            RETRIEVER_TOP_K=int(os.getenv("RETRIEVER_TOP_K", "0")),
            MAX_HISTORY_MESSAGES=int(os.getenv("MAX_HISTORY_MESSAGES", "0")),
            MAX_MESSAGE_LENGTH=int(os.getenv("MAX_MESSAGE_LENGTH", "0")),
            MAX_SUGGESTION_LENGTH=int(os.getenv("MAX_SUGGESTION_LENGTH", "0")),
            LLM_TIMEOUT=int(os.getenv("LLM_TIMEOUT", "0")),
            RATE_LIMIT_WINDOW=int(os.getenv("RATE_LIMIT_WINDOW", "0")),
            RATE_LIMIT_MAX_REQUESTS=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "0")),
        )

        missing_str = [
            key for key in (
                "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                "PERPLEXITY_API_KEY", "EMBEDDING_MODEL", "CHROMA_COLLECTION_NAME",
            )
            if not getattr(instance, key)
        ]
        missing_int = [
            key for key in (
                "RETRIEVER_TOP_K", "MAX_HISTORY_MESSAGES", "MAX_MESSAGE_LENGTH",
                "MAX_SUGGESTION_LENGTH", "LLM_TIMEOUT", "RATE_LIMIT_WINDOW",
                "RATE_LIMIT_MAX_REQUESTS",
            )
            if getattr(instance, key) == 0
        ]
        missing = missing_str + missing_int
        if missing:
            raise EnvironmentError(
                f"Variables de entorno requeridas no configuradas: {missing}"
            )

        return instance


settings = Settings.from_env()