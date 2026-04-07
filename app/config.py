from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Admin UI: set via env ADMIN_KEY only (never read from runtime JSON file)
    admin_key: str = ""

    openrouter_api_key: str = ""
    gateway_api_key: str = ""

    upstream_model: str = "anthropic/claude-opus-4.6"
    upstream_base_url: str = "https://openrouter.ai/api/v1"

    https_proxy: str | None = None

    database_url: str | None = None

    cache_enabled: bool = True
    cache_ttl_1h: bool = True

    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://openrouter.ai/api/v1"

    request_timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 30.0
    http_max_connections: int = 100
    http_max_keepalive: int = 20

    persist_queue_max: int = 10000
    embed_queue_max: int = 10000
    embed_batch_size: int = 8


_settings: Settings | None = None


def config_json_path() -> Path:
    return Path(os.environ.get("CONFIG_FILE", "data/config.json"))


def _load_merged_settings() -> Settings:
    base = Settings()
    path = config_json_path()
    if not path.exists():
        return base
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return base
        raw = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable CONFIG_FILE %s: %s", path, e)
        return base
    if not isinstance(raw, dict):
        logger.warning("Ignoring CONFIG_FILE %s: root must be a JSON object", path)
        return base
    raw.pop("admin_key", None)
    allowed: dict[str, Any] = {}
    for k, v in raw.items():
        if k in Settings.model_fields and k != "admin_key":
            allowed[k] = v
    if not allowed:
        return base
    try:
        return base.model_copy(update=allowed)
    except Exception as e:
        logger.warning("Ignoring invalid values in CONFIG_FILE %s: %s", path, e)
        return base


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = _load_merged_settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = _load_merged_settings()
    return _settings


def save_runtime_config(updates: dict[str, Any]) -> Settings:
    """Merge updates into current settings, write JSON file, reload cache."""
    base = get_settings()
    patch = {k: v for k, v in updates.items() if k in Settings.model_fields and k != "admin_key"}
    new_settings = base.model_copy(update=patch)
    path = config_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = new_settings.model_dump()
    to_write.pop("admin_key", None)
    path.write_text(json.dumps(to_write, indent=2, ensure_ascii=False), encoding="utf-8")
    return reload_settings()
