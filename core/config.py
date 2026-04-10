from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
   

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()