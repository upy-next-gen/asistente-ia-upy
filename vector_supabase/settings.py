from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class SupabaseVectorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    SUPABASE_URL: str
    SUPABASE_KEY: str


class SupabaseIndexerSettings(SupabaseVectorSettings):
    PERPLEXITY_API_KEY: str
    EMBEDDING_MODEL: str


@lru_cache(maxsize=1)
def get_vector_settings() -> SupabaseVectorSettings:
    return SupabaseVectorSettings()


@lru_cache(maxsize=1)
def get_indexer_settings() -> SupabaseIndexerSettings:
    return SupabaseIndexerSettings()

