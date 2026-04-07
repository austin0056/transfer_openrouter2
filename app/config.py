from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = ""
    gateway_api_key: str = ""

    upstream_model: str = "anthropic/claude-opus-4.6"
    upstream_base_url: str = "https://openrouter.ai/api/v1"

    # SOCKS5 / HTTP proxy for outbound OpenRouter (e.g. socks5://user:pass@host:port)
    https_proxy: str | None = None

    database_url: str | None = None

    cache_enabled: bool = True
    # When True, use 1h TTL for Anthropic automatic caching (long sessions)
    cache_ttl_1h: bool = True

    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    # Defaults to openrouter_api_key if empty
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://openrouter.ai/api/v1"

    request_timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 30.0
    http_max_connections: int = 100
    http_max_keepalive: int = 20

    persist_queue_max: int = 10000
    embed_queue_max: int = 10000
    embed_batch_size: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
