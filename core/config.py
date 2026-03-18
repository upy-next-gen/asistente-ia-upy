import os
from dataclasses import dataclass
from dotenv import load_dotenv


EMBEDDING_DIM = 1024
SAFE_BATCH_SIZE = 200
MAX_CHUNKS_PER_REQUEST = 20


@dataclass
class Settings:
    OPENAI_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    PERPLEXITY_API_KEY: str = ""
    EMBEDDING_MODEL: str = "pplx-embed-context-v1-0.6b"
    MIN_SIMILARITY: float = 0.3
    EMBEDDING_DIM: int = EMBEDDING_DIM
    SAFE_BATCH_SIZE: int = SAFE_BATCH_SIZE
    MAX_CHUNKS_PER_REQUEST: int = MAX_CHUNKS_PER_REQUEST
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
            EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "pplx-embed-context-v1-0.6b"),
            MIN_SIMILARITY=float(os.getenv("MIN_SIMILARITY", "0.3")),
            EMBEDDING_DIM=int(os.getenv("EMBEDDING_DIM", str(EMBEDDING_DIM))),
            SAFE_BATCH_SIZE=int(os.getenv("SAFE_BATCH_SIZE", str(SAFE_BATCH_SIZE))),
            MAX_CHUNKS_PER_REQUEST=int(os.getenv("MAX_CHUNKS_PER_REQUEST", str(MAX_CHUNKS_PER_REQUEST))),
            RETRIEVER_TOP_K=int(os.getenv("RETRIEVER_TOP_K", "5")),
            MAX_HISTORY_MESSAGES=int(os.getenv("MAX_HISTORY_MESSAGES", "20")),
            MAX_MESSAGE_LENGTH=int(os.getenv("MAX_MESSAGE_LENGTH", "2000")),
            MAX_SUGGESTION_LENGTH=int(os.getenv("MAX_SUGGESTION_LENGTH", "1000")),
            LLM_TIMEOUT=int(os.getenv("LLM_TIMEOUT", "30")),
            RATE_LIMIT_WINDOW=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            RATE_LIMIT_MAX_REQUESTS=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10")),
        )

        missing_str = [
            key for key in (
                "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                "PERPLEXITY_API_KEY", "EMBEDDING_MODEL", "SUPABASE_URL", "SUPABASE_KEY",
            )
            if not getattr(instance, key)
        ]
        missing = missing_str
        if missing:
            raise EnvironmentError(
                f"Variables de entorno requeridas no configuradas: {missing}"
            )

        return instance


settings = Settings.from_env()