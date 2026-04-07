"""Merge OpenAI-compatible client body with OpenRouter routing + Anthropic cache policy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.config import Settings


def merge_chat_completion_body(client_body: dict[str, Any], settings: Settings) -> dict[str, Any]:
    body = deepcopy(client_body)
    body["model"] = settings.upstream_model
    body["provider"] = {"only": ["anthropic"], "allow_fallbacks": False}
    if settings.cache_enabled:
        if settings.cache_ttl_1h:
            body["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        else:
            body["cache_control"] = {"type": "ephemeral"}
    else:
        body.pop("cache_control", None)
    return body


def openrouter_headers(settings: Settings) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
