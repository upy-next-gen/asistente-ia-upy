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
    MIN_SIMILARITY: float = 0.3
    EMBEDDING_DIM: int = 1024
    SAFE_BATCH_SIZE: int = 200
    MAX_CHUNKS_PER_REQUEST: int = 20
    RETRIEVER_TOP_K: int = 5
    MAX_HISTORY_MESSAGES: int = 20
    MAX_MESSAGE_LENGTH: int = 2000
    MAX_SUGGESTION_LENGTH: int = 1000
    LLM_TIMEOUT: int = 30
    RATE_LIMIT_WINDOW: int = 60
    RATE_LIMIT_MAX_REQUESTS: int = 10
    ENV: str = ""
    ENTRA_TENANT_ID: str = ""
    ENTRA_CLIENT_ID: str = ""
    ENTRA_CLIENT_SECRET: str = ""
    ENTRA_REDIRECT_URI: str = ""
    
    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        instance = cls(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            LLM_BASE_URL=os.getenv("LLM_BASE_URL"),
            LLM_MODEL=os.getenv("LLM_MODEL"),
            SUPABASE_URL=os.getenv("SUPABASE_URL"),
            SUPABASE_KEY=os.getenv("SUPABASE_KEY"),
            PERPLEXITY_API_KEY=os.getenv("PERPLEXITY_API_KEY"),
            EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL"),
            ENV=os.getenv("ENV"),
            ENTRA_TENANT_ID=os.getenv("ENTRA_TENANT_ID", ""),
            ENTRA_CLIENT_ID=os.getenv("ENTRA_CLIENT_ID", ""),
            ENTRA_CLIENT_SECRET=os.getenv("ENTRA_CLIENT_SECRET", ""),
            ENTRA_REDIRECT_URI=os.getenv("ENTRA_REDIRECT_URI", ""),
        )

        missing_str = [
            key for key in (
                "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                "PERPLEXITY_API_KEY", "EMBEDDING_MODEL", "SUPABASE_URL", "SUPABASE_KEY", "ENV",
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
