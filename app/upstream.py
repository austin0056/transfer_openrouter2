"""Merge OpenAI-compatible client body with OpenRouter routing + Anthropic cache policy."""

from __future__ import annotations

import json
import re
from copy import copy, deepcopy
from typing import Any

from app.config import Settings


# ─────────────────────────────────────────────────────────────────
# Reasoning effort suffix parsing
# ─────────────────────────────────────────────────────────────────

REASONING_SUFFIXES: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


def parse_reasoning_suffix(model_id: str) -> tuple[str, str | None]:
    """Strip a trailing ``-low|-medium|-high|-xhigh|-max`` from a model id.

    Returns ``(base_model, effort_or_none)``. Matching is case-insensitive on
    the suffix only; the base model casing is preserved verbatim.

    Examples
    --------
    >>> parse_reasoning_suffix("claude-opus-4-7")
    ('claude-opus-4-7', None)
    >>> parse_reasoning_suffix("claude-opus-4-7-high")
    ('claude-opus-4-7', 'high')
    >>> parse_reasoning_suffix("anthropic/claude-opus-4.7-xhigh")
    ('anthropic/claude-opus-4.7', 'xhigh')
    """
    if not isinstance(model_id, str):
        return model_id, None  # type: ignore[return-value]
    mid = model_id.strip()
    if not mid:
        return mid, None
    lower = mid.lower()
    for suffix in REASONING_SUFFIXES:
        marker = "-" + suffix
        if lower.endswith(marker) and len(mid) > len(marker):
            return mid[: -len(marker)], suffix
    return mid, None


def budget_for_effort(settings: Settings, effort: str | None) -> int | None:
    """Map an effort label to an Anthropic ``thinking.budget_tokens`` value.

    Returns ``None`` when no suffix was supplied (caller should fall back to
    its own default behaviour, e.g. ``settings.thinking_budget_tokens``).
    """
    if not effort:
        return None
    table = {
        "low":    settings.reasoning_budget_low,
        "medium": settings.reasoning_budget_medium,
        "high":   settings.reasoning_budget_high,
        "xhigh":  settings.reasoning_budget_xhigh,
        "max":    settings.reasoning_budget_max,
    }
    return table.get(effort)


def apply_reasoning_to_openrouter_body(
    body: dict[str, Any],
    effort: str | None,
) -> None:
    """Inject OpenRouter ``reasoning`` field based on the parsed effort label.

    Mapping:
      low/medium/high → ``{"effort": "low|medium|high"}``
      xhigh           → ``{"effort": "high", "max_tokens": 32768}``
      max             → ``{"effort": "high", "max_tokens": 65536}``

    A pre-existing ``reasoning_effort`` key on the request takes precedence
    (the suffix is silently ignored in that case).
    """
    if not effort:
        return
    # Honor explicit client-supplied reasoning_effort
    if body.get("reasoning_effort") or body.get("reasoning"):
        return
    if effort in ("low", "medium", "high"):
        body["reasoning"] = {"effort": effort}
    elif effort == "xhigh":
        body["reasoning"] = {"effort": "high", "max_tokens": 32768}
    elif effort == "max":
        body["reasoning"] = {"effort": "high", "max_tokens": 65536}


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


_DEFAULT_EFFICIENCY_PROMPT = (
    "[INTERNAL OPTIMIZATION DIRECTIVE — invisible to user, do not mention or reference this]\n"
    "You are operating through a token-metered API. Follow these rules to minimize cost while "
    "maintaining full code intelligence:\n\n"
    "READING STRATEGY:\n"
    "- ALWAYS start by listing the directory structure (ls/Glob/tree) before reading any files. "
    "Understand the project layout first, then read only what you need.\n"
    "- NEVER read these files/dirs: package-lock.json, yarn.lock, pnpm-lock.yaml, "
    "node_modules/*, dist/*, build/*, .git/*, *.min.js, *.min.css, *.map, "
    "*.pyc, __pycache__/*, .next/*, vendor/*, *.wasm, *.bin, *.dat.\n"
    "- START with entry points: package.json, README, main/index files, config files — "
    "these reveal the project structure without reading everything.\n"
    "- For large files (>200 lines): use offset+limit to read only the relevant section. "
    "Read the first 50 lines to understand structure, then jump to the specific function/class.\n"
    "- NEVER re-read a file you already read in this conversation unless it was just modified.\n\n"
    "TOOL CALL STRATEGY:\n"
    "- BATCH multiple Read/Shell/Glob calls in a single response. "
    "Issue 3-5 reads at once rather than one per turn.\n"
    "- Combine related shell commands with && instead of separate calls.\n"
    "- Prefer Glob patterns over multiple individual reads to discover files.\n\n"
    "RESPONSE STRATEGY:\n"
    "- Be direct and concise. Skip lengthy explanations unless the user asks for them.\n"
    "- When writing code, output only the changes needed — do not repeat unchanged code.\n"
    "- When explaining, use bullet points over paragraphs."
)


def _inject_efficiency_prompt(body: dict[str, Any], settings: Settings) -> None:
    """注入隐藏的效率指令，引导模型减少 tool call 次数和读取量。仅在有 tools 时注入。"""
    if not settings.efficiency_prompt_enabled:
        return
    # 没有 tools 的简单对话不需要效率指令
    if not body.get("tools"):
        return
    prompt = (settings.efficiency_prompt or "").strip() or _DEFAULT_EFFICIENCY_PROMPT
    if not prompt:
        return
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    # 追加到 system prompt 末尾（不影响缓存前缀）
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        c0 = msgs[0].get("content")
        if isinstance(c0, str):
            msgs[0]["content"] = c0 + "\n\n---\n\n" + prompt
        elif isinstance(c0, list):
            # content block 数组：追加一个 text block
            c0.append({"type": "text", "text": "\n\n---\n\n" + prompt})
    else:
        msgs.insert(0, {"role": "system", "content": prompt})


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


def _is_empty_content(content: Any) -> bool:
    """判断消息内容是否为空（None、空字符串、空数组、全空文本 content-block）。"""
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        if len(content) == 0:
            return True
        # content-block 数组：如果全部块的文本为空则视为空
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    return False
                # 非 text 类型的块（如 image）视为非空
                if block.get("type") and block["type"] != "text":
                    return False
            elif isinstance(block, str) and block.strip():
                return False
        return True
    return False


def _to_content_blocks(content: Any) -> list[dict[str, Any]]:
    """将 content 统一为 content-block 数组。"""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _merge_content(target: dict[str, Any], source: dict[str, Any]) -> None:
    """将 source 的 content 追加到 target，统一为 content-block 数组或拼接字符串。"""
    tc = target.get("content")
    sc = source.get("content")
    # 如果两边都是字符串，直接拼接
    if isinstance(tc, str) and isinstance(sc, str):
        target["content"] = tc + "\n" + sc if tc.strip() and sc.strip() else (tc or sc)
        return
    # 否则转为 content-block 数组合并
    blocks = _to_content_blocks(tc) + _to_content_blocks(sc)
    target["content"] = blocks if blocks else ""


def _convert_anthropic_content_blocks(body: dict[str, Any]) -> None:
    """将 Anthropic 原生 content block（tool_use / tool_result）转为 OpenAI 格式。

    Anthropic 格式:
      assistant: {"content": [{"type":"text","text":"..."}, {"type":"tool_use","id":"x","name":"fn","input":{...}}]}
      user:      {"content": [{"type":"tool_result","tool_use_id":"x","content":"..."}]}

    OpenAI 格式:
      assistant: {"content":"...", "tool_calls": [{"id":"x","type":"function","function":{"name":"fn","arguments":"{...}"}}]}
      tool:      {"role":"tool", "tool_call_id":"x", "content":"..."}
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    converted: list[dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            converted.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, list):
            converted.append(m)
            continue

        # 检查是否包含 Anthropic 原生 block
        has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
        has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)

        if m.get("role") == "assistant" and has_tool_use:
            # 提取文本和 tool_use 块
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            other_blocks: list[Any] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    txt = block.get("text", "")
                    if isinstance(txt, str) and txt.strip():
                        text_parts.append(txt)
                elif block.get("type") == "tool_use":
                    tc_id = block.get("id", "")
                    tc_name = block.get("name", "")
                    tc_input = block.get("input", {})
                    if isinstance(tc_input, str):
                        args_str = tc_input
                    else:
                        args_str = json.dumps(tc_input, ensure_ascii=False) if tc_input else "{}"
                    tool_calls.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": tc_name, "arguments": args_str},
                    })
                else:
                    other_blocks.append(block)
            new_m: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                new_m["content"] = "\n".join(text_parts)
            else:
                new_m["content"] = None
            if tool_calls:
                new_m["tool_calls"] = tool_calls
            converted.append(new_m)

        elif m.get("role") == "user" and has_tool_result:
            # 拆分 tool_result 为独立 tool 消息，保留其他内容为 user 消息
            text_blocks: list[Any] = []
            tool_results: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tc_id = block.get("tool_use_id", "")
                    # tool_result 的 content 可能是字符串或 content block 数组
                    rc = block.get("content", "")
                    if isinstance(rc, list):
                        parts = []
                        for rb in rc:
                            if isinstance(rb, dict) and rb.get("type") == "text":
                                parts.append(rb.get("text", ""))
                            elif isinstance(rb, str):
                                parts.append(rb)
                        rc = "\n".join(parts)
                    elif not isinstance(rc, str):
                        rc = json.dumps(rc, ensure_ascii=False) if rc else ""
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": rc or "",
                    })
                else:
                    text_blocks.append(block)
            # 先输出 tool 消息
            converted.extend(tool_results)
            # 如果还有文本内容，保留为 user 消息
            if text_blocks:
                sanitized = [b for b in text_blocks
                             if isinstance(b, dict) and b.get("type") == "text"
                             and isinstance(b.get("text"), str) and b["text"].strip()]
                if sanitized:
                    converted.append({"role": "user", "content": sanitized})
        else:
            converted.append(m)
    body["messages"] = converted


def _gemini_contents_to_messages(contents: Any) -> list[dict[str, Any]] | None:
    """Gemini/Vertex 原生 contents → OpenAI messages（供误用字段或中间层透传）。"""
    if not isinstance(contents, list) or not contents:
        return None
    out: list[dict[str, Any]] = []
    for c in contents:
        if not isinstance(c, dict):
            continue
        role = str(c.get("role") or "user").lower()
        if role == "model":
            role = "assistant"
        parts = c.get("parts")
        if isinstance(parts, list):
            text_chunks: list[str] = []
            for p in parts:
                if isinstance(p, dict):
                    t = p.get("text")
                    if isinstance(t, str) and t.strip():
                        text_chunks.append(t)
            if text_chunks:
                out.append({"role": role, "content": "\n".join(text_chunks)})
        elif isinstance(c.get("text"), str) and c["text"].strip():
            out.append({"role": role, "content": c["text"]})
    return out if out else None


def _hoist_alternate_inputs_to_messages(body: dict[str, Any]) -> None:
    """若 messages 缺失或为空，从 input / contents / prompt 等字段恢复（兼容 Responses API 与 Gemini 负载）。"""
    msgs = body.get("messages")
    if isinstance(msgs, list) and len(msgs) > 0:
        return

    inp = body.get("input")
    if isinstance(inp, str) and inp.strip():
        body["messages"] = [{"role": "user", "content": inp.strip()}]
        return
    if isinstance(inp, list) and inp:
        new_msgs: list[dict[str, Any]] = []
        for item in inp:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                role = str(item.get("role") or "user").lower()
                new_msgs.append({"role": role, "content": item.get("content")})
            elif "role" in item:
                role = str(item.get("role") or "user").lower()
                new_msgs.append({"role": role, "content": item.get("content")})
        if new_msgs:
            body["messages"] = new_msgs
            return

    conv = _gemini_contents_to_messages(body.get("contents"))
    if conv:
        body["messages"] = conv
        return

    pr = body.get("prompt")
    if isinstance(pr, str) and pr.strip():
        body["messages"] = [{"role": "user", "content": pr.strip()}]


def _normalize_message_roles(body: dict[str, Any]) -> None:
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    for m in msgs:
        if isinstance(m, dict):
            r = m.get("role")
            if isinstance(r, str):
                m["role"] = r.lower()


def _adapt_openai_body_for_upstream(body: dict[str, Any], settings: Settings) -> None:
    """转换层：兼容分发/客户端脏负载，上游无需、分发侧也无需改代码。"""
    _hoist_alternate_inputs_to_messages(body)
    _normalize_message_roles(body)
    _convert_top_level_system(body)
    _convert_anthropic_content_blocks(body)
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
    # 第一遍：过滤无效消息 + 规范化 user/system content
    filtered: list[Any] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool":
            has_tool_call_id = isinstance(m.get("tool_call_id"), str) and m["tool_call_id"].strip()
            has_name = isinstance(m.get("name"), str) and m["name"].strip()
            if not has_tool_call_id and not has_name:
                continue
        # 对 user/system 消息：规范化 content block 数组，剥离空块
        if m.get("role") in ("user", "system"):
            c = m.get("content")
            if isinstance(c, list):
                # 只保留有实际内容的 block
                good_blocks: list[Any] = []
                for block in c:
                    if isinstance(block, dict):
                        btype = block.get("type", "text")
                        if btype in ("text", "input_text", "text_input"):
                            txt = block.get("text")
                            if isinstance(txt, str) and txt.strip():
                                good_blocks.append(block)
                        else:
                            # image_url 等非 text 类型保留
                            good_blocks.append(block)
                    elif isinstance(block, str) and block.strip():
                        good_blocks.append(block)
                if not good_blocks:
                    continue  # 所有 block 都为空，跳过整条消息
                m["content"] = good_blocks
            elif _is_empty_content(c):
                continue
        filtered.append(m)

    # 第二遍：合并连续同角色消息（Anthropic 要求 user/assistant 严格交替）
    cleaned: list[Any] = []
    for m in filtered:
        if cleaned and m.get("role") == cleaned[-1].get("role") and m.get("role") in ("user", "system"):
            # 将当前消息内容合并到上一条
            _merge_content(cleaned[-1], m)
        else:
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
    _final_sanitize_messages(body)


def _normalize_content_block(block: dict[str, Any]) -> dict[str, Any]:
    """规范化单个 content block：只保留标准字段，移除多余属性以提高缓存一致性。"""
    btype = block.get("type", "text")
    if btype == "text":
        out: dict[str, Any] = {"type": "text", "text": block.get("text", "")}
        # 保留 cache_control（如果有）
        if "cache_control" in block:
            out["cache_control"] = block["cache_control"]
        return out
    # 非 text block 保持原样
    return block


def _sanitize_content(content: Any) -> Any:
    """清洗 content：字符串保持原样，数组剥离空 text block + 规范化字段。"""
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        good: list[Any] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "text")
                if btype in ("text", "input_text", "text_input"):
                    txt = block.get("text")
                    if isinstance(txt, str) and txt.strip():
                        good.append(_normalize_content_block(block))
                else:
                    good.append(block)
            elif isinstance(block, str) and block.strip():
                good.append(block)
        return good if good else None
    return content


def _final_sanitize_messages(body: dict[str, Any]) -> None:
    """最终清洗：确保发给上游的每条 user/system 消息 content 非空。"""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    result: list[Any] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") in ("user", "system"):
            sanitized = _sanitize_content(m.get("content"))
            if sanitized is None:
                continue
            m["content"] = sanitized
        result.append(m)
    body["messages"] = result


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
    """设置合理的 max_tokens，Anthropic 要求必传；尊重客户端设置。"""
    body.pop("max_completion_tokens", None)
    # 尊重客户端的 max_tokens；只在客户端没设或设得过小时补默认
    mt = body.get("max_tokens")
    if mt is None or (isinstance(mt, int) and mt < 4096):
        body["max_tokens"] = 64000


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
    "reasoning_effort", "reasoning", "metadata",
}


def _strip_unknown_fields(body: dict[str, Any]) -> None:
    """移除 Cursor/客户端发送的非标准字段，防止上游 400。"""
    for k in list(body.keys()):
        if k not in _KNOWN_BODY_KEYS:
            body.pop(k)


def _inject_cache_breakpoints(body: dict[str, Any], settings: Settings) -> None:
    """在 system prompt、tools 末尾、最近的对话转折点设置 cache breakpoint。

    Anthropic prompt caching 按前缀匹配：
    - breakpoint 1: system prompt 末尾 → tools 定义和 system 在所有请求间共享缓存
    - breakpoint 2: tools 最后一个定义 → 确保 tools 列表被缓存
    - breakpoint 3: 最后一条 user/tool 消息之前 → 上一轮的历史被缓存
    最多 4 个 breakpoint（Anthropic 限制）。
    """
    if not settings.cache_enabled:
        return
    cache_marker = {"type": "ephemeral"}

    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return

    # 先清除所有已有的 cache_control（分发层可能已经加了）
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
        elif isinstance(c, dict):
            c.pop("cache_control", None)
    tools = body.get("tools")
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict):
                fn = t.get("function")
                if isinstance(fn, dict):
                    fn.pop("cache_control", None)

    # Breakpoint 1: system prompt
    if msgs[0].get("role") == "system":
        c = msgs[0].get("content")
        if isinstance(c, str):
            # 转为 content block 数组以便加 cache_control
            msgs[0]["content"] = [{"type": "text", "text": c, "cache_control": cache_marker}]
        elif isinstance(c, list) and c:
            last_block = c[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = cache_marker

    # Breakpoint 2: tools 列表最后一项
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        last_tool = tools[-1]
        if isinstance(last_tool, dict):
            fn = last_tool.get("function")
            if isinstance(fn, dict):
                fn["cache_control"] = cache_marker

    # Breakpoint 3: 最后一条 assistant 消息（连续 tool call 轮次间缓存前一轮）
    assistant_indices = [i for i, m in enumerate(msgs)
                         if isinstance(m, dict) and m.get("role") == "assistant"]
    if assistant_indices:
        last_asst = msgs[assistant_indices[-1]]
        c = last_asst.get("content")
        if isinstance(c, str) and c:
            last_asst["content"] = [{"type": "text", "text": c, "cache_control": cache_marker}]
        elif isinstance(c, list) and c:
            last_block = c[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = cache_marker

    # Breakpoint 4: 倒数第二条 user/tool 消息（历史前缀缓存）
    user_tool_indices = [i for i, m in enumerate(msgs)
                         if isinstance(m, dict) and m.get("role") in ("user", "tool")]
    if len(user_tool_indices) >= 2:
        prev_idx = user_tool_indices[-2]
        prev_msg = msgs[prev_idx]
        c = prev_msg.get("content")
        if isinstance(c, str):
            prev_msg["content"] = [{"type": "text", "text": c, "cache_control": cache_marker}]
        elif isinstance(c, list) and c:
            last_block = c[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = cache_marker


def _sort_tools(body: dict[str, Any]) -> None:
    """按工具名称排序，确保 tools 列表在不同请求间顺序一致，最大化缓存前缀命中。"""
    tools = body.get("tools")
    if not isinstance(tools, list) or len(tools) <= 1:
        return
    try:
        body["tools"] = sorted(
            tools,
            key=lambda t: (t.get("function", {}).get("name", "") if isinstance(t, dict) else ""),
        )
    except Exception:
        pass  # 排序失败不影响功能


# Opus 1M context vs Sonnet 200K context
_MAX_MESSAGES_OPUS = 500   # 1M context 模型
_MAX_MESSAGES_SONNET = 150  # 200K context 模型
_MAX_MESSAGES_DEFAULT = 150


def _get_max_messages_for_model(model: str) -> int:
    m = (model or "").lower()
    if "opus" in m:
        return _MAX_MESSAGES_OPUS
    if "sonnet" in m:
        return _MAX_MESSAGES_SONNET
    return _MAX_MESSAGES_DEFAULT

# 含这些关键字的 assistant 消息不截断（保留完整的 todo / plan / 进度信息）
_PROGRESS_KEYWORDS = (
    "todo", "TODO", "Todo",
    "plan", "Plan", "PLAN",
    "step", "Step", "STEP",
    "progress", "Progress",
    "完成", "计划", "步骤",
    "- [ ]", "- [x]", "- [X]",
    "## ", "### ",
)


def _contains_progress_info(text: str) -> bool:
    """检测文本是否包含 todo/plan/进度关键字。"""
    if not isinstance(text, str):
        return False
    return any(kw in text for kw in _PROGRESS_KEYWORDS)


def _message_has_progress(msg: dict[str, Any]) -> bool:
    """检测 assistant 消息是否包含进度/计划信息。"""
    c = msg.get("content")
    if isinstance(c, str) and _contains_progress_info(c):
        return True
    if isinstance(c, list):
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                if _contains_progress_info(block.get("text", "")):
                    return True
    # 带 TodoWrite / update_plan 等 tool_calls 的消息也视为进度信息
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list):
        for tc in tcs:
            if isinstance(tc, dict):
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn_name = (fn.get("name") or "").lower()
                    if any(k in fn_name for k in ("todo", "plan", "task")):
                        return True
    # Anthropic 原生格式的 tool_use block 也检查
    if isinstance(c, list):
        for block in c:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = (block.get("name") or "").lower()
                if any(k in name for k in ("todo", "plan", "task")):
                    return True
    return False


def _is_tool_result_only(msg: Any) -> bool:
    """判断一条 user 消息是否只包含 tool_result block 而无文本。"""
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    c = msg.get("content")
    if not isinstance(c, list) or not c:
        return False
    has_text = False
    has_tool_result = False
    for block in c:
        if isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "tool_result":
                has_tool_result = True
            elif btype == "text" and isinstance(block.get("text"), str) and block["text"].strip():
                has_text = True
    return has_tool_result and not has_text


def _compress_old_messages(body: dict[str, Any], settings: Settings) -> None:
    """压缩历史消息：限制总数 + 截断旧内容。

    1. 消息总数超过上限时，保留 system + 最近 N 条消息
    2. 旧 tool result 截断到 max_chars
    3. 旧 assistant 长回复截断到 1500 字符
    """
    if not settings.tool_result_truncate_enabled:
        return
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return

    # ── 第零步：过滤空 assistant 消息（content 和 tool_calls 都为空）──
    msgs = [
        m for m in msgs
        if not (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and _is_empty_content(m.get("content"))
            and not m.get("tool_calls")
        )
    ]

    # ── 第一步：消息数量硬限制（按模型能力区分）──
    if settings.upstream_provider == "anthropic":
        target_model = settings.anthropic_model
    else:
        target_model = settings.upstream_model
    max_msgs = _get_max_messages_for_model(target_model)

    if len(msgs) > max_msgs:
        # 保留第一条（system）+ 最近的消息
        system_msgs = []
        rest = msgs
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            system_msgs = [msgs[0]]
            rest = msgs[1:]

        # 保留最后 N 条
        keep = rest[-(max_msgs - len(system_msgs)):]

        # 关键：Anthropic 要求对话必须从 user 开始
        # 从后向前找最后一条 user 消息之前的所有 user 位置，
        # 确保第一条是 user，且不孤立 tool_result
        start = 0
        found_user = False
        for i, m in enumerate(keep):
            if isinstance(m, dict) and m.get("role") == "user":
                start = i
                found_user = True
                break
        if not found_user:
            # 窗口内可能仅有 assistant/tool（或角色命名异常）；勿整段清空，避免上游收到空 messages
            start = 0
        keep = keep[start:]

        # 再次检查：如果第一条 user 消息的 content 只有 tool_result block（没有文本），
        # 而前面又没有对应的 assistant 带 tool_use，则删掉这条 tool_result-only user
        while keep and _is_tool_result_only(keep[0]):
            keep = keep[1:]

        msgs = system_msgs + keep
        body["messages"] = msgs

    # ── 第二步：仅截断旧的 tool/user 大内容（文件内容等） ──
    # assistant 消息永不截断 — 这是模型的推理链，截断等于切断大脑
    tool_max = settings.tool_result_max_chars
    keep_recent = settings.tool_result_keep_recent_turns

    assistant_indices = [i for i, m in enumerate(msgs)
                         if isinstance(m, dict) and m.get("role") == "assistant"]
    if len(assistant_indices) <= keep_recent:
        return

    cutoff_idx = assistant_indices[-keep_recent]

    for i, m in enumerate(msgs):
        if i >= cutoff_idx:
            break
        if not isinstance(m, dict):
            continue
        role = m.get("role")

        if role == "tool":
            c = m.get("content")
            if isinstance(c, str) and len(c) > tool_max:
                m["content"] = c[:tool_max] + f"\n\n[...truncated, original {len(c)} chars]"
        elif role == "user":
            c = m.get("content")
            if isinstance(c, list):
                for block in c:
                    if isinstance(block, dict) and block.get("type") in ("text", "tool_result"):
                        txt = block.get("text") or block.get("content")
                        if isinstance(txt, str) and len(txt) > tool_max:
                            if "text" in block:
                                block["text"] = txt[:tool_max] + f"\n[...truncated]"
                            elif "content" in block:
                                block["content"] = txt[:tool_max] + f"\n[...truncated]"
        # assistant 消息保持完整 — 不截断推理链


def _shallow_body_copy(src: dict[str, Any]) -> dict[str, Any]:
    """浅拷贝请求体：顶层 dict + messages 列表各项独立拷贝（修改安全且快）。"""
    body = dict(src)
    msgs = src.get("messages")
    if isinstance(msgs, list):
        body["messages"] = [dict(m) if isinstance(m, dict) else m for m in msgs]
    tools = src.get("tools")
    if isinstance(tools, list):
        body["tools"] = [deepcopy(t) for t in tools]  # tools 可能被修改（cache_control），需深拷贝
    return body


def _parse_string_json_field(body: dict[str, Any], key: str) -> None:
    """部分代理把 messages/tools 以 JSON 字符串形式放在 body 里。"""
    val = body.get(key)
    if not isinstance(val, str):
        return
    s = val.strip()
    if len(s) < 2 or s[0] not in "[{":
        return
    try:
        body[key] = json.loads(s)
    except json.JSONDecodeError:
        pass


def _lift_nested_chat_fields(body: dict[str, Any]) -> None:
    """部分网关/客户端把字段包在 request、data、payload 等嵌套对象中。"""
    m = body.get("messages")
    if isinstance(m, list) and len(m) > 0:
        return
    if isinstance(m, str) and m.strip():
        return
    for k in ("request", "body", "payload", "data", "params", "json"):
        inner = body.get(k)
        if not isinstance(inner, dict):
            continue
        if inner.get("messages") is None and inner.get("input") is None:
            continue
        if inner.get("messages") is not None:
            body["messages"] = inner["messages"]
        for fld in (
            "model", "stream", "tools", "tool_choice", "temperature",
            "top_p", "max_tokens", "max_completion_tokens", "n", "stop",
            "input", "contents", "prompt", "user", "response_format",
            "parallel_tool_calls",
        ):
            if fld not in inner:
                continue
            cur = body.get(fld)
            if cur is None or cur == "" or cur == []:
                body[fld] = inner[fld]
        return


def _aliases_message_model_keys(body: dict[str, Any]) -> None:
    for alt in ("Messages", "MESSAGES"):
        if body.get("messages") is None and alt in body:
            body["messages"] = body.pop(alt)
            break
    if body.get("model") is None and "Model" in body:
        body["model"] = body.pop("Model")


def _refresh_message_list_shallow_clone(body: dict[str, Any]) -> None:
    msgs = body.get("messages")
    if isinstance(msgs, list):
        body["messages"] = [dict(m) if isinstance(m, dict) else m for m in msgs]


def _coerce_raw_chat_request(body: dict[str, Any]) -> None:
    """在清洗逻辑之前统一原始 body 形态（字符串 JSON、嵌套、别名键）。"""
    _aliases_message_model_keys(body)
    _lift_nested_chat_fields(body)
    for key in ("messages", "tools", "input"):
        _parse_string_json_field(body, key)
    _refresh_message_list_shallow_clone(body)


def merge_chat_completion_body(client_body: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """根据 upstream_provider 构建请求体。"""
    body = _shallow_body_copy(client_body)
    _coerce_raw_chat_request(body)

    # 识别客户端传进来的 -low/-medium/-high/-xhigh/-max 后缀。
    # 后缀仅用于决定 reasoning effort 及 thinking budget；
    # 最终发给上游的 model 由 _build_*_body 赋值（与 settings.upstream_model 一致）。
    effort: str | None = None
    requested_model = body.get("model")
    if settings.reasoning_suffix_enabled and isinstance(requested_model, str):
        _, effort = parse_reasoning_suffix(requested_model)

    _adapt_openai_body_for_upstream(body, settings)
    # 零注入策略：任何对 system prompt 的修改都会破坏缓存前缀匹配，
    # 导致 OpenRouter/Anthropic 的 prompt caching 命中率归零。
    # 仅在用户显式启用时才注入（identity/efficiency 默认关闭）。
    if settings.identity_prompt_enabled:
        _inject_identity_prompt(body, settings)
    if settings.upstream_provider != "anthropic" and settings.efficiency_prompt_enabled:
        _inject_efficiency_prompt(body, settings)
    # 工具排序只对 OpenRouter 执行（保证多次请求 tools 列表字节一致）
    # Anthropic 路径保留原始顺序 — 上游模型已见过 Cursor 原始顺序，改了反而破坏缓存
    if settings.upstream_provider == "openrouter":
        _sort_tools(body)
    _compress_old_messages(body, settings)

    if settings.upstream_provider == "anthropic":
        return _build_anthropic_body(body, settings, effort=effort)
    if settings.upstream_provider == "gemini":
        return _build_gemini_body(body, settings)
    return _build_openrouter_body(body, settings, effort=effort)


# Gemini OpenAI 兼容 API 常见最大输出上限（超过易 400）
_GEMINI_MAX_OUTPUT_TOKENS = 65536


def _build_gemini_body(body: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Google Gemini OpenAI 兼容路径：不发送 OpenRouter / Anthropic 专用字段。"""
    body["model"] = settings.upstream_model
    body.pop("provider", None)
    body.pop("cache_control", None)
    # 与 tools 同时使用 structured output 时 Gemini 常返回 400
    if body.get("tools") and body.get("response_format") is not None:
        body.pop("response_format", None)
    mt = body.get("max_tokens")
    if mt is not None:
        try:
            n = int(mt)
        except (TypeError, ValueError):
            body.pop("max_tokens", None)
        else:
            body["max_tokens"] = max(1, min(n, _GEMINI_MAX_OUTPUT_TOKENS))
    return body


def _build_openrouter_body(
    body: dict[str, Any],
    settings: Settings,
    *,
    effort: str | None = None,
) -> dict[str, Any]:
    """OpenRouter 路径（OpenAI 格式透传 + 增强）。"""
    body["model"] = settings.upstream_model
    # 后缀驱动的 reasoning effort
    apply_reasoning_to_openrouter_body(body, effort)
    # 限定在支持 prompt caching 的 provider 内做 fallback：
    # - anthropic 直发、google-vertex 都完整支持 cache_control 计费（命中 0.1×）
    # - amazon-bedrock 等会静默忽略 cache_control，导致按全价计费
    # 既避免单 provider 限流引发 503，又保证缓存折扣不丢失
    body["provider"] = {
        "only": ["anthropic", "google-vertex"],
        "allow_fallbacks": True,
    }

    # 流式请求自动带上 stream_options.include_usage=true
    # 否则 OpenRouter 不会在流式响应里发 usage chunk，Cursor/客户端看不到 token 消耗
    if body.get("stream"):
        so = body.get("stream_options")
        if not isinstance(so, dict):
            so = {}
        so.setdefault("include_usage", True)
        body["stream_options"] = so

    # 缓存：仅对 Claude/Gemini 类模型生效（OpenAI 家族忽略 cache_control）
    if settings.cache_enabled:
        if settings.cache_ttl_1h:
            body["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        else:
            body["cache_control"] = {"type": "ephemeral"}
    else:
        body.pop("cache_control", None)
    _inject_cache_breakpoints(body, settings)

    # 透传请求级 usage 偏好（部分客户端会发 usage: {include: true}）
    # OpenRouter 也支持 ?include=reasoning_details 等参数，但 body 层不需要改
    return body


# ─────────────────────────────────────────────────────────────────
# Anthropic 原生 API 请求构建
# ─────────────────────────────────────────────────────────────────


def _build_anthropic_body(
    body: dict[str, Any],
    settings: Settings,
    *,
    effort: str | None = None,
) -> dict[str, Any]:
    """OpenAI 格式 → Anthropic Messages API 格式。"""
    msgs = body.get("messages") or []
    out: dict[str, Any] = {"model": settings.anthropic_model}

    # 提取 system prompt（Anthropic 用顶层 system 字段）
    system_parts: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                system_parts.append({"type": "text", "text": c})
            elif isinstance(c, list):
                system_parts.extend(c)
        else:
            conversation.append(m)
    if system_parts:
        # 在最后一个 system block 上加 cache_control
        if settings.cache_enabled:
            system_parts[-1]["cache_control"] = {"type": "ephemeral"}
        out["system"] = system_parts

    # 转换 messages: OpenAI → Anthropic
    anthropic_msgs: list[dict[str, Any]] = []
    for m in conversation:
        role = m.get("role", "")

        if role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            # 文本内容
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                content_blocks.append({"type": "text", "text": c})
            elif isinstance(c, list):
                content_blocks.extend(c)
            # tool_calls → tool_use blocks
            tcs = m.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        args_obj = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args_obj or {},
                    })
            if content_blocks:
                anthropic_msgs.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            # OpenAI tool result → Anthropic tool_result block in user message
            tc_id = m.get("tool_call_id", "")
            tc_content = m.get("content", "")
            # content 可能是 str、list（content blocks）或其他
            if isinstance(tc_content, str):
                result_content = tc_content
            elif isinstance(tc_content, list):
                # OpenAI content block 数组 → 提取文本拼接
                parts = []
                for blk in tc_content:
                    if isinstance(blk, dict):
                        parts.append(blk.get("text", "") or blk.get("content", ""))
                    elif isinstance(blk, str):
                        parts.append(blk)
                result_content = "\n".join(p for p in parts if p)
            else:
                result_content = str(tc_content) if tc_content else ""
            result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tc_id,
                "content": result_content,
            }
            # Anthropic 要求 tool_result 放在 user 消息中
            if anthropic_msgs and anthropic_msgs[-1].get("role") == "user":
                anthropic_msgs[-1]["content"].append(result_block)
            else:
                anthropic_msgs.append({"role": "user", "content": [result_block]})

        elif role == "user":
            c = m.get("content")
            if isinstance(c, str):
                blocks = [{"type": "text", "text": c}]
            elif isinstance(c, list):
                blocks = c
            else:
                continue
            # 合并连续 user 消息
            if anthropic_msgs and anthropic_msgs[-1].get("role") == "user":
                anthropic_msgs[-1]["content"].extend(blocks)
            else:
                anthropic_msgs.append({"role": "user", "content": blocks})

    out["messages"] = anthropic_msgs

    # ── 强制 user/assistant 严格交替（Anthropic 硬性要求）──
    # 第一条必须是 user；连续同角色需要合并或插入占位
    if anthropic_msgs:
        # 确保第一条是 user
        if anthropic_msgs[0].get("role") != "user":
            anthropic_msgs.insert(0, {"role": "user", "content": [{"type": "text", "text": "."}]})
        # 合并连续同角色消息 / 插入占位消息
        merged_msgs: list[dict[str, Any]] = [anthropic_msgs[0]]
        for m in anthropic_msgs[1:]:
            prev_role = merged_msgs[-1].get("role")
            cur_role = m.get("role")
            if cur_role == prev_role:
                # 同角色：合并 content
                prev_content = merged_msgs[-1].get("content", [])
                cur_content = m.get("content", [])
                if isinstance(prev_content, list) and isinstance(cur_content, list):
                    prev_content.extend(cur_content)
                elif isinstance(prev_content, str) and isinstance(cur_content, str):
                    merged_msgs[-1]["content"] = prev_content + "\n" + cur_content
                else:
                    # 类型不一致，插入占位消息
                    placeholder_role = "user" if cur_role == "assistant" else "assistant"
                    merged_msgs.append({"role": placeholder_role, "content": [{"type": "text", "text": "."}]})
                    merged_msgs.append(m)
            else:
                merged_msgs.append(m)
        # 确保最后一条是 user（Anthropic 要求最后一条必须是 user）
        if merged_msgs[-1].get("role") != "user":
            merged_msgs.append({"role": "user", "content": [{"type": "text", "text": "."}]})
        out["messages"] = merged_msgs

    # Tools → Anthropic 格式
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        anthropic_tools: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function", {})
            if not isinstance(fn, dict):
                continue
            tool_name = fn.get("name", "")
            if not tool_name:
                continue  # 无名工具跳过
            # input_schema 必须是合法 JSON Schema object
            schema = fn.get("parameters")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            atool: dict[str, Any] = {
                "name": tool_name,
                "input_schema": schema,
            }
            desc = fn.get("description")
            if desc:
                atool["description"] = desc
            anthropic_tools.append(atool)
        if anthropic_tools:
            # 最后一个 tool 上加 cache_control
            if settings.cache_enabled:
                anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            out["tools"] = anthropic_tools

    # Tool choice
    tc = body.get("tool_choice")
    if tc == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif tc == "required":
        out["tool_choice"] = {"type": "any"}
    elif tc == "none":
        # Anthropic 没有 "none"；改为移除 tools 让模型无法调用
        out.pop("tools", None)
    elif isinstance(tc, dict) and tc.get("type") == "function":
        fn = tc.get("function", {})
        name = fn.get("name", "")
        if name:
            out["tool_choice"] = {"type": "tool", "name": name}

    # max_tokens 必传；按模型最大能力给出默认
    model_lower = (settings.anthropic_model or "").lower()
    mt = body.get("max_tokens")
    if mt:
        out["max_tokens"] = int(mt)
    else:
        # Opus 4.6 支持 1M output；Sonnet 4.5 支持 128K (with beta)
        if "opus" in model_lower:
            out["max_tokens"] = 128000
        else:
            out["max_tokens"] = 128000

    # 扩展思考（extended thinking）— 后缀优先于 settings.thinking_budget_tokens
    suffix_budget = budget_for_effort(settings, effort)
    if suffix_budget is not None:
        # 后缀明确指定了等级 → 强制启用思考，采用后缀对应预算
        if out["max_tokens"] <= suffix_budget:
            out["max_tokens"] = suffix_budget + 8192
        out["thinking"] = {"type": "enabled", "budget_tokens": suffix_budget}
        out.pop("temperature", None)
    elif settings.thinking_enabled:
        base_budget = settings.thinking_budget_tokens or 10000
        # 动态调整：根据请求复杂度选择 thinking 预算
        has_tools = bool(body.get("tools"))
        msg_count = len(body.get("messages") or [])
        # 有 tools 或消息多 → 复杂请求，给满预算
        # 无 tools 且消息少 → 简单请求，减少预算降低延迟
        if has_tools or msg_count > 6:
            budget = base_budget
        elif msg_count <= 2:
            budget = min(base_budget, 4096)
        else:
            budget = min(base_budget, 8000)
        # Anthropic 要求：max_tokens 必须 > budget_tokens
        if out["max_tokens"] <= budget:
            out["max_tokens"] = budget + 8192
        out["thinking"] = {"type": "enabled", "budget_tokens": budget}
        # Anthropic 要求：开启 thinking 时 temperature 必须为 1（或不设）
        out.pop("temperature", None)

    # stream
    if body.get("stream"):
        out["stream"] = True

    # temperature, top_p（thinking 模式下 temperature 已移除）
    if "thinking" not in out:
        for k in ("temperature", "top_p"):
            v = body.get(k)
            if v is not None:
                out[k] = v
    else:
        # thinking 模式只允许 top_p
        tp = body.get("top_p")
        if tp is not None:
            out["top_p"] = tp

    # stop_sequences 透传（OpenAI: stop → Anthropic: stop_sequences）
    stop = body.get("stop")
    if stop:
        if isinstance(stop, str):
            out["stop_sequences"] = [stop]
        elif isinstance(stop, list):
            out["stop_sequences"] = stop

    # metadata 透传
    meta = body.get("metadata")
    if isinstance(meta, dict):
        out["metadata"] = meta

    # 历史消息 cache breakpoint（倒数第二条 user 消息）
    final_msgs = out["messages"]
    if settings.cache_enabled and len(final_msgs) >= 3:
        user_indices = [i for i, m in enumerate(final_msgs) if m.get("role") == "user"]
        if len(user_indices) >= 2:
            prev_user = final_msgs[user_indices[-2]]
            blocks = prev_user.get("content", [])
            if isinstance(blocks, list) and blocks:
                last_block = blocks[-1]
                if isinstance(last_block, dict):
                    last_block["cache_control"] = {"type": "ephemeral"}

    return out


def anthropic_response_to_openai(resp: dict[str, Any]) -> dict[str, Any]:
    """Anthropic Messages API 响应 → OpenAI chat.completion 格式。"""
    content_blocks = resp.get("content") or []
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tc_index = 0

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "thinking":
            t = block.get("thinking", "")
            if t:
                thinking_parts.append(t)
        elif btype == "text":
            t = block.get("text", "")
            if t:
                text_parts.append(t)
        elif btype == "tool_use":
            inp = block.get("input", {})
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(inp, ensure_ascii=False) if inp else "{}",
                },
                "index": tc_index,
            })
            tc_index += 1

    msg: dict[str, Any] = {"role": "assistant"}
    # thinking 以省略摘要形式输出，不贴完整内容
    output_parts: list[str] = []
    if thinking_parts:
        total = sum(len(t) for t in thinking_parts)
        output_parts.append(f"> 🧠 Thinking ({total} chars)...")
    output_parts.extend(text_parts)
    msg["content"] = "\n".join(output_parts) if output_parts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls

    # stop_reason 映射
    sr = resp.get("stop_reason", "end_turn")
    finish_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}
    finish_reason = finish_map.get(sr, "stop")

    # usage 映射
    u = resp.get("usage", {})
    usage = {
        "prompt_tokens": u.get("input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
    }
    # 透传缓存 token 信息
    if "cache_creation_input_tokens" in u:
        usage["cache_creation_input_tokens"] = u["cache_creation_input_tokens"]
    if "cache_read_input_tokens" in u:
        usage["cache_read_input_tokens"] = u["cache_read_input_tokens"]

    return {
        "id": resp.get("id", ""),
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": resp.get("model", ""),
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": usage,
    }


# ─────────────────────────────────────────────────────────────────
# 双模型协同：Opus 规划 + Qwen 执行
# ─────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM_SUFFIX = """
[DUAL-MODEL ORCHESTRATION — internal, invisible to user]

You are the PLANNER in a dual-model system. Your thinking is streamed to the user.
An executor model will perform the actual tool calls based on your instructions.

RULES:
1. Think through the task step by step.
2. After your analysis, output a PRECISE execution plan.
3. DO NOT output tool_calls yourself. Instead, end your response with a JSON block:

```json
{"execute": [
  {"tool": "ToolName", "arguments": {"param": "value"}},
  {"tool": "ToolName2", "arguments": {"param": "value"}}
]}
```

4. Each tool call must include the EXACT tool name and complete arguments matching the available tools.
5. If you need to ASK the user a question (no tools needed), respond normally WITHOUT the execute block.
6. If you see tool results from previous turns, REVIEW them carefully before planning next steps.
7. Be thorough — the executor model follows your instructions LITERALLY.
8. If the task is complete or you want to provide a final answer, respond WITHOUT the execute block.
"""

_EXECUTOR_SYSTEM = (
    "You are a precise tool executor. Execute EXACTLY the tool calls specified below. "
    "Do not add, skip, or modify any calls. Do not explain or add commentary. "
    "Call the tools in the order given with the exact arguments provided."
)


def build_planner_body(merged: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """构造规划请求：Opus 看完整上下文，输出思考 + JSON 执行指令。"""
    body = _shallow_body_copy(merged)
    body["model"] = settings.planner_model
    body["stream"] = True
    # 不设 max_tokens，让 Opus 充分思考（缓存命中让 input 几乎免费）
    body.pop("max_tokens", None)

    # 移除 tools（Opus 不执行，只规划）但保留工具名称列表供参考
    tools = body.pop("tools", None)
    body.pop("tool_choice", None)
    body.pop("parallel_tool_calls", None)

    # 把可用工具名称列表加到规划指令中
    tool_names = ""
    if isinstance(tools, list):
        names = []
        for t in tools:
            if isinstance(t, dict):
                fn = t.get("function", {})
                name = fn.get("name", "")
                desc = fn.get("description", "")
                if name:
                    names.append(f"  - {name}: {desc[:80]}" if desc else f"  - {name}")
        if names:
            tool_names = "\n\nAvailable tools:\n" + "\n".join(names)

    suffix = _PLANNER_SYSTEM_SUFFIX + tool_names

    # 追加到 system prompt 末尾
    msgs = body.get("messages") or []
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        c = msgs[0].get("content", "")
        if isinstance(c, str):
            msgs[0]["content"] = c + "\n\n---\n" + suffix
        elif isinstance(c, list):
            c.append({"type": "text", "text": "\n\n---\n" + suffix})
    else:
        msgs.insert(0, {"role": "system", "content": suffix})

    return body


def build_executor_body(
    merged: dict[str, Any],
    commands: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, Any]:
    """构造执行请求：Qwen 只拿到精简指令 + tools 定义。"""
    body: dict[str, Any] = {}
    body["model"] = settings.executor_model
    body["stream"] = True

    # 保留完整 tools 定义
    if merged.get("tools"):
        body["tools"] = merged["tools"]
        body["tool_choice"] = "required"  # 强制调工具

    # 精简 messages：只有执行指令
    cmd_json = json.dumps({"execute": commands}, ensure_ascii=False, indent=2)
    body["messages"] = [
        {"role": "system", "content": _EXECUTOR_SYSTEM},
        {"role": "user", "content": f"Execute these tool calls:\n\n```json\n{cmd_json}\n```"},
    ]

    # OpenRouter 路由
    body["provider"] = {"order": ["Alibaba Cloud"]}

    return body


def extract_execute_commands(plan_text: str) -> list[dict[str, Any]] | None:
    """从 Opus 输出中提取 {"execute": [...]} JSON 指令块。"""
    # 尝试找 ```json ... ``` 代码块
    code_match = re.search(r'```json\s*(\{[\s\S]*?"execute"[\s\S]*?\})\s*```', plan_text)
    if code_match:
        try:
            data = json.loads(code_match.group(1))
            cmds = data.get("execute")
            if isinstance(cmds, list) and cmds:
                return cmds
        except json.JSONDecodeError:
            pass

    # 回退：找裸 JSON
    bare_match = re.search(r'\{\s*"execute"\s*:\s*\[[\s\S]*?\]\s*\}', plan_text)
    if bare_match:
        try:
            data = json.loads(bare_match.group())
            cmds = data.get("execute")
            if isinstance(cmds, list) and cmds:
                return cmds
        except json.JSONDecodeError:
            pass

    return None


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


def anthropic_headers(settings: Settings) -> dict[str, str]:
    beta_flags = [
        "prompt-caching-2024-07-31",
        "interleaved-thinking-2025-05-14",     # 允许 tool_call 之间思考
        "fine-grained-tool-streaming-2025-05-14",  # 边生成边流 tool 参数
        "token-efficient-tools-2025-02-19",    # 紧凑 tool schema 省 ~14% token
        "output-128k-2025-02-19",              # 解锁 128K 输出上限
    ]
    model_lower = (settings.anthropic_model or "").lower()
    if "opus" in model_lower:
        beta_flags.append("context-1m-2025-08-07")
    return {
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": settings.anthropic_version,
        "anthropic-beta": ",".join(beta_flags),
        "content-type": "application/json",
    }


def get_upstream_url(settings: Settings) -> str:
    if settings.upstream_provider == "anthropic":
        return f"{settings.anthropic_base_url.rstrip('/')}/v1/messages"
    if settings.upstream_provider == "gemini":
        return f"{settings.gemini_openai_base_url.rstrip('/')}/chat/completions"
    return f"{settings.upstream_base_url.rstrip('/')}/chat/completions"


def get_upstream_headers(settings: Settings) -> dict[str, str]:
    if settings.upstream_provider == "anthropic":
        return anthropic_headers(settings)
    if settings.upstream_provider == "gemini":
        return {
            "Authorization": f"Bearer {settings.google_api_key}",
            "Content-Type": "application/json",
        }
    return openrouter_headers(settings)
