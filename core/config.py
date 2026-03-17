import os
from dataclasses import dataclass
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
    RETRIEVER_TOP_K: int = 5
    MAX_HISTORY_MESSAGES: int = 20
    MAX_MESSAGE_LENGTH: int = 2000
    MAX_SUGGESTION_LENGTH: int = 1000
    LLM_TIMEOUT: int = 30
    RATE_LIMIT_WINDOW: int = 60
    RATE_LIMIT_MAX_REQUESTS: int = 10

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
            RETRIEVER_TOP_K=int(os.getenv("RETRIEVER_TOP_K", "5")),
            MAX_HISTORY_MESSAGES=int(os.getenv("MAX_HISTORY_MESSAGES", "20")),
            MAX_MESSAGE_LENGTH=int(os.getenv("MAX_MESSAGE_LENGTH", "2000")),
            MAX_SUGGESTION_LENGTH=int(os.getenv("MAX_SUGGESTION_LENGTH", "1000")),
            LLM_TIMEOUT=int(os.getenv("LLM_TIMEOUT", "30")),
            RATE_LIMIT_WINDOW=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            RATE_LIMIT_MAX_REQUESTS=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10")),
        )

        missing = [
            key for key in ("OPENAI_API_KEY", "LLM_BASE_URL", "PERPLEXITY_API_KEY")
            if not getattr(instance, key)
        ]
        if missing:
            raise EnvironmentError(
                f"Variables de entorno requeridas no configuradas: {missing}"
            )

        return instance


settings = Settings.from_env()


