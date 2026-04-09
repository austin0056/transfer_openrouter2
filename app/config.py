from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Admin UI: set via env ADMIN_KEY only (never read from runtime JSON file)
    admin_key: str = ""

    openrouter_api_key: str = ""
    gateway_api_key: str = ""

    upstream_provider: str = "openrouter"  # "openrouter" | "anthropic"

    upstream_model: str = "anthropic/claude-opus-4.6"
    upstream_base_url: str = "https://openrouter.ai/api/v1"

    openrouter_http_referer: str = ""
    openrouter_app_title: str = ""

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-opus-4-20250514"
    anthropic_version: str = "2023-06-01"

    loose_tools_passthrough: bool = False
    log_chat_metadata: bool = False

    identity_prompt_enabled: bool = False
    identity_prompt: str = (
        "我是 Claude Opus 4.6，来自 Anthropic 的 Claude 4.6 模型家族。"
        "Claude 4.6 家族目前包括 Claude Opus 4.6 和 Claude Sonnet 4.6，"
        "其中 Opus 4.6 是最先进、最智能的版本。"
    )

    efficiency_prompt_enabled: bool = True
    efficiency_prompt: str = ""

    # 历史 tool result 截断：保留最近 N 轮完整，更早的截断到 max_chars
    tool_result_truncate_enabled: bool = True
    tool_result_keep_recent_turns: int = 2
    tool_result_max_chars: int = 800

    https_proxy: str | None = None

    database_url: str | None = None

    cache_enabled: bool = True
    cache_ttl_1h: bool = True

    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int | None = 1536
    embedding_use_pgvector: bool = True
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://openrouter.ai/api/v1"

    request_timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 30.0
    http_max_connections: int = 100
    http_max_keepalive: int = 20

    max_request_body_mb: int = 50

    persist_queue_max: int = 10000
    embed_queue_max: int = 10000
    embed_batch_size: int = 8

    @field_validator("persist_queue_max", "embed_queue_max", mode="before")
    @classmethod
    def _queue_cap(cls, v: Any) -> int:
        """CONFIG_FILE 里若写 null，会变成 None，asyncio.Queue(maxsize=None) 在部分 Python 下会崩。0 表示无上限队列。"""
        if v is None or v == "":
            return 10000
        try:
            i = int(v)
        except (TypeError, ValueError):
            return 10000
        if i < 0:
            return 10000
        return min(i, 1_000_000_000)

    @field_validator("embed_batch_size", mode="before")
    @classmethod
    def _batch_size(cls, v: Any) -> int:
        if v is None or v == "":
            return 8
        try:
            i = int(v)
        except (TypeError, ValueError):
            return 8
        return max(1, min(i, 10_000))


_settings: Settings | None = None
_settings_file_mtime: float | None = None


def _config_file_mtime() -> float:
    path = config_json_path()
    try:
        if path.exists():
            return float(path.stat().st_mtime)
    except OSError:
        pass
    return 0.0


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
    """每次比对 CONFIG_FILE 修改时间；多 worker 时避免只在保存请求所在进程生效、其它进程仍用旧 upstream_model。"""
    global _settings, _settings_file_mtime
    mtime = _config_file_mtime()
    if _settings is None or mtime != _settings_file_mtime:
        _settings = _load_merged_settings()
        _settings_file_mtime = mtime
    return _settings


def reload_settings() -> Settings:
    global _settings, _settings_file_mtime
    _settings = _load_merged_settings()
    _settings_file_mtime = _config_file_mtime()
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
