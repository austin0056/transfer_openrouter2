"""Merge OpenAI-compatible client body with OpenRouter routing + Anthropic cache policy."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from app.config import Settings


def _function_name_nonempty(fn: Any) -> bool:
    if not isinstance(fn, dict):
        return False
    name = fn.get("name")
    return isinstance(name, str) and bool(name.strip())


def _normalize_tool_function_schema(fn: dict[str, Any]) -> None:
    """OpenRouter/Anthropic 侧常见要求 parameters 为 object；None 会触发 invalid_request。"""
    p = fn.get("parameters")
    if p is None:
        fn["parameters"] = {}
    elif not isinstance(p, dict):
        fn["parameters"] = {}


def _normalize_tool_call_arguments(fn: dict[str, Any]) -> None:
    """tool_calls.function.arguments 须为 JSON 字符串；None/对象会被规范化。"""
    raw = fn.get("arguments")
    if raw is None:
        fn["arguments"] = "{}"
    elif isinstance(raw, str):
        if raw.strip() == "":
            fn["arguments"] = "{}"
    else:
        fn["arguments"] = json.dumps(raw, ensure_ascii=False)


def _adapt_openai_body_for_upstream(body: dict[str, Any]) -> None:
    """转换层：兼容分发/客户端脏负载，上游无需、分发侧也无需改代码。"""
    allowed: set[str] = set()
    tools = body.get("tools")
    if isinstance(tools, list):
        kept: list[Any] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function")
            if not isinstance(fn, dict):
                continue
            if not _function_name_nonempty(fn):
                continue
            _normalize_tool_function_schema(fn)
            allowed.add(str(fn["name"]).strip())
            kept.append(t)
        if kept:
            body["tools"] = kept
        else:
            body.pop("tools", None)
            allowed.clear()
    else:
        body.pop("tools", None)

    tc = body.get("tool_choice")
    if isinstance(tc, dict) and tc.get("type") == "function":
        fn = tc.get("function")
        n = fn.get("name") if isinstance(fn, dict) else None
        if (
            not isinstance(n, str)
            or not n.strip()
            or (allowed and n.strip() not in allowed)
        ):
            body["tool_choice"] = "auto"
    elif isinstance(tc, str) and tc not in ("none", "auto", "required"):
        if allowed and tc.strip() not in allowed:
            body["tool_choice"] = "auto"

    if not body.get("tools"):
        body.pop("tool_choice", None)
        body.pop("parallel_tool_calls", None)

    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    cleaned: list[Any] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool":
            n = m.get("name")
            if not isinstance(n, str) or not n.strip():
                continue
        cleaned.append(m)
    body["messages"] = cleaned

    for m in cleaned:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        raw_tcs = m.get("tool_calls")
        if not isinstance(raw_tcs, list):
            continue
        good: list[Any] = []
        for tc in raw_tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict) or not _function_name_nonempty(fn):
                continue
            _normalize_tool_call_arguments(fn)
            good.append(tc)
        if good:
            m["tool_calls"] = good
        else:
            m.pop("tool_calls", None)
            if m.get("content") is None:
                m["content"] = ""


def merge_chat_completion_body(client_body: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Preserve tools/tool_choice/parallel_tool_calls etc.; only override model + Anthropic routing."""
    body = deepcopy(client_body)
    _adapt_openai_body_for_upstream(body)
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
