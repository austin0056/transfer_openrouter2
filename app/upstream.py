"""Merge OpenAI-compatible client body with OpenRouter routing + Anthropic cache policy."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from app.config import Settings


def _inject_identity_prompt(body: dict[str, Any], settings: Settings) -> None:
    """可选：注入 system，向用户说明本线路固定为 Opus（减少误称 Sonnet）。"""
    if not settings.identity_prompt_enabled:
        return
    fact = (settings.identity_prompt or "").strip()
    if not fact:
        return
    guidance = (
        "当用户问及你的模型名称、厂商、版本或身份时，请基于下文事实简洁作答，"
        "不要自称 Sonnet、GPT 或其他与下文不符的型号；用户未问及身份时不要主动长篇介绍自己。\n\n"
        f"{fact}"
    )
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        body["messages"] = [{"role": "system", "content": guidance}]
        return
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        c0 = msgs[0].get("content")
        if isinstance(c0, str):
            msgs[0]["content"] = guidance + "\n\n---\n\n" + c0
        elif isinstance(c0, list):
            msgs.insert(0, {"role": "system", "content": guidance})
        else:
            msgs[0]["content"] = guidance
    else:
        msgs.insert(0, {"role": "system", "content": guidance})


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


def _convert_top_level_system(body: dict[str, Any]) -> None:
    """Anthropic 原生格式把 system 放顶层；转换为 OpenAI 格式的 messages[0]。"""
    sys_field = body.pop("system", None)
    if sys_field is None:
        return
    # system 可能是字符串或 content-block 数组
    if isinstance(sys_field, str):
        text = sys_field
    elif isinstance(sys_field, list):
        parts: list[str] = []
        for block in sys_field:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        text = "\n".join(parts)
    else:
        return
    if not text.strip():
        return
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        body["messages"] = [{"role": "system", "content": text}]
        return
    # 如果 messages 已有 system 消息，合并；否则插入开头
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        c0 = msgs[0].get("content")
        if isinstance(c0, str):
            msgs[0]["content"] = text + "\n\n" + c0
        else:
            msgs.insert(0, {"role": "system", "content": text})
    else:
        msgs.insert(0, {"role": "system", "content": text})


def _adapt_openai_body_for_upstream(body: dict[str, Any], settings: Settings) -> None:
    """转换层：兼容分发/客户端脏负载，上游无需、分发侧也无需改代码。"""
    _convert_top_level_system(body)
    allowed: set[str] = set()
    loose = settings.loose_tools_passthrough
    tools = body.get("tools")
    if isinstance(tools, list):
        kept: list[Any] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            # Anthropic 原生工具格式: {"name":"X","description":"...","input_schema":{...}}
            # 转换为 OpenAI 格式: {"type":"function","function":{"name":"X","description":"...","parameters":{...}}}
            if "function" not in t and "name" in t:
                fn_obj: dict[str, Any] = {"name": t["name"]}
                if "description" in t:
                    fn_obj["description"] = t["description"]
                fn_obj["parameters"] = t.get("input_schema") or t.get("parameters") or {}
                t = {"type": "function", "function": fn_obj}
            fn = t.get("function")
            ttype = t.get("type") or "function"
            if ttype == "function" and isinstance(fn, dict) and _function_name_nonempty(fn):
                _normalize_tool_function_schema(fn)
                allowed.add(str(fn["name"]).strip())
                kept.append(t)
            elif loose:
                if isinstance(fn, dict):
                    _normalize_tool_function_schema(fn)
                    if _function_name_nonempty(fn):
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
    if isinstance(tc, dict):
        tc_type = tc.get("type", "")
        # Anthropic 原生格式转 OpenAI 格式
        if tc_type == "auto":
            body["tool_choice"] = "auto"
        elif tc_type == "any":
            body["tool_choice"] = "required"
        elif tc_type == "none":
            body["tool_choice"] = "none"
        elif tc_type == "tool":
            # Anthropic: {"type":"tool","name":"X"} → OpenAI: {"type":"function","function":{"name":"X"}}
            name = tc.get("name", "")
            if isinstance(name, str) and name.strip() and (not allowed or name.strip() in allowed):
                body["tool_choice"] = {"type": "function", "function": {"name": name.strip()}}
            else:
                body["tool_choice"] = "auto"
        elif tc_type == "function":
            fn = tc.get("function")
            n = fn.get("name") if isinstance(fn, dict) else None
            if (
                not isinstance(n, str)
                or not n.strip()
                or (allowed and n.strip() not in allowed)
            ):
                body["tool_choice"] = "auto"
        else:
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
            # tool 结果消息须有 tool_call_id 或 name 之一才有效
            has_tool_call_id = isinstance(m.get("tool_call_id"), str) and m["tool_call_id"].strip()
            has_name = isinstance(m.get("name"), str) and m["name"].strip()
            if not has_tool_call_id and not has_name:
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

    for m in cleaned:
        if isinstance(m, dict) and m.get("role") == "developer":
            m["role"] = "system"

    _bridge_max_completion_tokens(body)
    _ensure_min_max_tokens(body)
    _strip_unknown_fields(body)


def _bridge_max_completion_tokens(body: dict[str, Any]) -> None:
    """Cursor / 新版 OpenAI 客户端常发 max_completion_tokens；OpenRouter 仍认 max_tokens。"""
    if body.get("max_tokens") is not None:
        return
    mct = body.get("max_completion_tokens")
    if mct is None:
        return
    try:
        body["max_tokens"] = int(mct)
    except (TypeError, ValueError):
        pass


def _ensure_min_max_tokens(body: dict[str, Any]) -> None:
    """确保 max_tokens 足够大，不限制模型输出。"""
    body.pop("max_completion_tokens", None)
    body["max_tokens"] = 1000000


# OpenAI / OpenRouter 已知接受的顶层字段白名单
_KNOWN_BODY_KEYS: set[str] = {
    "model", "messages", "stream", "stream_options",
    "temperature", "top_p", "n", "stop",
    "max_tokens", "max_completion_tokens",
    "presence_penalty", "frequency_penalty", "logit_bias",
    "user", "tools", "tool_choice", "parallel_tool_calls",
    "response_format", "seed", "logprobs", "top_logprobs",
    "service_tier",
    # OpenRouter extensions
    "provider", "cache_control", "transforms", "route",
    "reasoning_effort", "metadata",
}


def _strip_unknown_fields(body: dict[str, Any]) -> None:
    """移除 Cursor/客户端发送的非标准字段，防止上游 400。"""
    for k in list(body.keys()):
        if k not in _KNOWN_BODY_KEYS:
            body.pop(k)


def merge_chat_completion_body(client_body: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Preserve tools/tool_choice/parallel_tool_calls etc.; only override model + Anthropic routing."""
    body = deepcopy(client_body)
    _adapt_openai_body_for_upstream(body, settings)
    _inject_identity_prompt(body, settings)
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
    h: dict[str, str] = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    ref = (settings.openrouter_http_referer or "").strip()
    if ref:
        h["HTTP-Referer"] = ref
    title = (settings.openrouter_app_title or "").strip()
    if title:
        h["X-Title"] = title
    return h
