"""Parse OpenAI-compatible SSE stream; accumulate assistant text, tool_calls, usage."""

from __future__ import annotations

import json
import time as _time
from typing import Any


def _coerce_tool_index(idx: Any) -> int:
    """统一为 int，避免 bucket.keys() 混用 str/int 导致 sorted 抛错或顺序错乱。"""
    if idx is None:
        return 0
    if isinstance(idx, bool):
        return 0
    if isinstance(idx, int):
        return idx
    try:
        return int(idx)
    except (TypeError, ValueError):
        return 0


def parse_sse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def accumulate_delta(chunk: dict[str, Any], parts: list[str]) -> None:
    """Only assistant text tokens; do not mix tool argument fragments into content."""
    choices = chunk.get("choices") or []
    for ch in choices:
        delta = ch.get("delta") or {}
        c = delta.get("content")
        if isinstance(c, str) and c:
            parts.append(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)


def apply_tool_call_delta(
    bucket: dict[int, dict[str, Any]],
    tc: dict[str, Any],
    *,
    id_to_index: dict[str, int] | None = None,
) -> None:
    """Merge one streaming tool_call fragment (OpenAI format, keyed by index)."""
    idx_raw = tc.get("index")
    if idx_raw is None and id_to_index is not None:
        tid = tc.get("id")
        if isinstance(tid, str) and tid.strip():
            if tid not in id_to_index:
                id_to_index[tid] = len(id_to_index)
            idx = id_to_index[tid]
        else:
            idx = _coerce_tool_index(idx_raw)
    else:
        idx = _coerce_tool_index(idx_raw)
    cur = bucket.setdefault(
        idx,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if tc.get("id"):
        cur["id"] = tc["id"]
    if tc.get("type"):
        cur["type"] = tc["type"]
    fn = tc.get("function")
    if isinstance(fn, dict):
        if fn.get("name"):
            cur["function"]["name"] = fn["name"]
        if "arguments" in fn and fn["arguments"] is not None:
            cur["function"]["arguments"] = cur["function"].get("arguments", "") + str(
                fn["arguments"]
            )


def accumulate_tool_calls(
    chunk: dict[str, Any],
    bucket: dict[int, dict[str, Any]],
    id_to_index: dict[str, int] | None = None,
) -> None:
    choices = chunk.get("choices") or []
    for ch in choices:
        delta = ch.get("delta") or {}
        tcs = delta.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if isinstance(tc, dict):
                apply_tool_call_delta(bucket, tc, id_to_index=id_to_index)


def tool_calls_list(bucket: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    if not bucket:
        return []
    return [bucket[i] for i in sorted(bucket.keys())]


def extract_usage(chunk: dict[str, Any]) -> dict[str, Any] | None:
    u = chunk.get("usage")
    if isinstance(u, dict):
        return u
    return None


def convert_usage_to_additive(
    usage: dict[str, Any],
    cache_write_multiplier: float = 1.25,
    mode: str = "native",
    cache_creation_scale: float = 1.0,
    cache_read_scale: float = 1.0,
) -> bool:
    """Normalize OpenRouter-style usage so the downstream billing layer computes correctly.

    Upstream (OpenRouter) returns:
        usage.prompt_tokens = fresh + cache_read + cache_write
        usage.prompt_tokens_details.cached_tokens      = cache_read
        usage.prompt_tokens_details.cache_write_tokens = cache_write

    Two output modes:

    1. ``native`` (default, recommended):
        Emit the four canonical fields the dispatch layer expects:
            prompt_tokens               = fresh
            cached_tokens (in details)  = cache_read
            cache_creation_input_tokens = cache_write   ← LiteLLM/Anthropic native key
            cache_read_input_tokens     = cache_read    ← LiteLLM/Anthropic native key

        The dispatch layer applies its own per-field price
        (input_cost / cache_creation_cost / cache_read_cost) and the total
        matches the upstream `cost` exactly. The billing UI also shows
        "cache write tokens" line item correctly.

    2. ``additive`` (legacy fallback):
        For dispatch layers that ONLY know `input` + `cache_read`, fold
        cache_write tokens into prompt_tokens at the premium rate:
            new_prompt_tokens = fresh + round(cache_write × multiplier)
        (1.25× for 5min TTL, 2× for 1h TTL)

    Returns True if the usage was modified.
    """
    details = usage.get("prompt_tokens_details")
    cached = 0
    cache_write = 0
    if isinstance(details, dict):
        cached = details.get("cached_tokens", 0) or 0
        cache_write = details.get("cache_write_tokens", 0) or 0
    if not cached and not cache_write:
        return False
    prompt = usage.get("prompt_tokens", 0) or 0
    if not isinstance(prompt, int) or prompt < cached + cache_write:
        return False
    fresh = prompt - cached - cache_write
    completion = usage.get("completion_tokens", 0) or 0

    if mode == "native":
        # Strip cached + cache_write from prompt_tokens so the dispatch layer
        # bills them via cache_creation_input_tokens / cache_read_input_tokens.
        usage["prompt_tokens"] = fresh
        scaled_cw = 0
        scaled_cr = 0
        if cache_write:
            scaled_cw = round(cache_write * cache_creation_scale)
            usage["cache_creation_input_tokens"] = scaled_cw
            # 保持 details 字段与输出一致，避免下游重复计费
            if isinstance(usage.get("prompt_tokens_details"), dict):
                usage["prompt_tokens_details"]["cache_write_tokens"] = scaled_cw
        if cached:
            scaled_cr = round(cached * cache_read_scale)
            usage["cache_read_input_tokens"] = scaled_cr
            if isinstance(usage.get("prompt_tokens_details"), dict):
                usage["prompt_tokens_details"]["cached_tokens"] = scaled_cr
        # 重算 total_tokens：分发层（New-API / One-API 等）常常要求
        # total_tokens == prompt_tokens + completion_tokens (+ cache 字段）。
        # 只保留上游原始 total 会跟改写后的 prompt 不一致、被分发层以
        # "非法 usage" 为由丢弃 → 表现为 token 消耗为 0 / 不显示。
        usage["total_tokens"] = fresh + scaled_cw + scaled_cr + completion
        return True

    # legacy additive mode
    inflated_cache_write = round(cache_write * cache_write_multiplier)
    usage["prompt_tokens"] = fresh + inflated_cache_write
    usage["total_tokens"] = usage["prompt_tokens"] + completion
    return True


_FINISH_REASON_ALIASES = {
    "tool_use": "tool_calls",
    "function_call": "tool_calls",
}


def finish_reason_from_chunk(chunk: dict[str, Any]) -> str | None:
    choices = chunk.get("choices") or []
    for ch in choices:
        fr = ch.get("finish_reason")
        if fr is not None:
            s = str(fr)
            return _FINISH_REASON_ALIASES.get(s, s)
    return None


# ─────────────────────────────────────────────────────────────────
# Anthropic 原生 SSE → OpenAI chunk 实时转换
# ─────────────────────────────────────────────────────────────────

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


class AnthropicStreamState:
    """跟踪 Anthropic SSE 流状态，用于实时转换为 OpenAI chunk。"""

    def __init__(self) -> None:
        self.message_id: str = ""
        self.model: str = ""
        self.current_block_index: int = -1
        self.current_block_type: str = ""
        self.tool_call_index: int = -1  # OpenAI tool_calls 的索引
        self.text_parts: list[str] = []
        self.tool_bucket: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None
        self.finish_reason: str | None = None
        self.error_emitted: bool = False  # 是否已发过 error chunk
        self.role_sent: bool = False  # 是否已发过 role chunk（只发一次）
        self.in_thinking: bool = False  # 当前是否在 thinking block 中
        self.thinking_chars: int = 0  # thinking 累积字符数


def parse_anthropic_sse(raw_lines: list[str]) -> tuple[str, dict[str, Any] | None]:
    """从 Anthropic SSE 行中提取 event type 和 data。

    Anthropic SSE 格式：
      event: content_block_delta
      data: {"type":"content_block_delta",...}
    """
    event_type = ""
    data_str = ""
    for line in raw_lines:
        line = line.strip()
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
    if not data_str:
        return event_type, None
    try:
        return event_type, json.loads(data_str)
    except json.JSONDecodeError:
        return event_type, None


def anthropic_event_to_openai_chunk(
    event_type: str,
    data: dict[str, Any],
    state: AnthropicStreamState,
) -> str | None:
    """将单个 Anthropic SSE 事件转为 OpenAI SSE chunk 字符串。返回 None 表示跳过。"""
    msg_id = state.message_id
    model = state.model
    dtype = data.get("type", "")

    # 错误事件：将错误转为 OpenAI 格式的错误 chunk + [DONE]
    if dtype == "error" or event_type == "error":
        if state.error_emitted:
            return None
        state.error_emitted = True
        err_info = data.get("error", {}) if isinstance(data.get("error"), dict) else {}
        err_msg = err_info.get("message", "upstream error")
        err_type = err_info.get("type", "upstream_error")
        err_chunk = {
            "id": f"chatcmpl-{state.message_id or 'error'}",
            "object": "chat.completion.chunk",
            "model": state.model or "",
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n[Upstream error: {err_msg}]"},
                "finish_reason": "error",
            }],
            # 结构化错误字段（Cursor 能识别为失败响应，不会把内容当正常结果）
            "error": {
                "message": err_msg,
                "type": err_type,
                "code": err_info.get("code", "upstream_error"),
            },
        }
        return (
            f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
            "data: [DONE]\n\n"
        )

    # Ping 事件：返回 SSE 注释行作为心跳透传给客户端
    if dtype == "ping" or event_type == "ping":
        return ": ping\n\n"

    if dtype == "message_start":
        msg = data.get("message", {})
        state.message_id = msg.get("id", "")
        # 只在 state.model 还未被调用方预设时才写入上游 model
        # （允许网关提前设置 client_model 后缀名）
        if not state.model:
            state.model = msg.get("model", "")
        # 提取 input usage
        u = msg.get("usage", {})
        state.usage = {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": 0,
            "total_tokens": u.get("input_tokens", 0),
        }
        if "cache_creation_input_tokens" in u:
            state.usage["cache_creation_input_tokens"] = u["cache_creation_input_tokens"]
        if "cache_read_input_tokens" in u:
            state.usage["cache_read_input_tokens"] = u["cache_read_input_tokens"]
        return None  # 不输出 chunk

    if dtype == "content_block_start":
        state.current_block_index = data.get("index", 0)
        cb = data.get("content_block", {})
        state.current_block_type = cb.get("type", "text")

        if state.current_block_type == "tool_use":
            state.in_thinking = False
            state.tool_call_index += 1
            idx = state.tool_call_index
            tc_id = cb.get("id", "")
            tc_name = cb.get("name", "")
            state.tool_bucket[idx] = {
                "id": tc_id, "type": "function",
                "function": {"name": tc_name, "arguments": ""},
            }
            delta: dict[str, Any] = {
                "tool_calls": [{"index": idx, "id": tc_id, "type": "function",
                                "function": {"name": tc_name, "arguments": ""}}]
            }
            if not state.role_sent:
                delta["role"] = "assistant"
                state.role_sent = True
            chunk = _make_openai_chunk(state, delta=delta)
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        if state.current_block_type == "thinking":
            # thinking block 开始 — 静默跳过，不发 chunk 给客户端
            state.in_thinking = True
            return None

        if state.current_block_type == "text":
            state.in_thinking = False
            # 只在第一次发 role chunk
            if not state.role_sent:
                state.role_sent = True
                chunk = _make_openai_chunk(state, delta={"role": "assistant", "content": ""})
                return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            return None

        return None

    if dtype == "content_block_delta":
        delta = data.get("delta", {})
        delta_type = delta.get("type", "")

        if delta_type == "text_delta":
            text = delta.get("text", "")
            if text:
                state.text_parts.append(text)
                oai_delta: dict[str, Any] = {"content": text}
                if not state.role_sent:
                    oai_delta["role"] = "assistant"
                    state.role_sent = True
                chunk = _make_openai_chunk(state, delta=oai_delta)
                return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        elif delta_type == "input_json_delta":
            partial = delta.get("partial_json", "")
            if partial and state.tool_call_index >= 0:
                idx = state.tool_call_index
                # 防御：tool_bucket[idx] 可能未初始化（如 content_block_start 丢失）
                if idx not in state.tool_bucket:
                    state.tool_bucket[idx] = {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                state.tool_bucket[idx]["function"]["arguments"] += partial
                chunk = _make_openai_chunk(state, delta={
                    "tool_calls": [{"index": idx, "function": {"arguments": partial}}]
                })
                return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        elif delta_type == "thinking_delta":
            # 扩展思考：只累积字符数，不发内容给客户端
            thinking_text = delta.get("thinking", "")
            if thinking_text:
                state.thinking_chars += len(thinking_text)
            return None

        elif delta_type == "signature_delta":
            # 思考签名，跳过
            return None

        return None

    if dtype == "content_block_stop":
        if state.in_thinking and state.thinking_chars > 0:
            # thinking block 结束 — 输出一行省略摘要
            state.in_thinking = False
            summary = f"> 🧠 Thinking ({state.thinking_chars} chars)...\n\n"
            state.thinking_chars = 0
            oai_delta: dict[str, Any] = {"content": summary}
            if not state.role_sent:
                oai_delta["role"] = "assistant"
                state.role_sent = True
            chunk = _make_openai_chunk(state, delta=oai_delta)
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        # tool_use block 结束：修正空 arguments，Cursor 执行工具时需要合法 JSON
        if state.current_block_type == "tool_use" and state.tool_call_index >= 0:
            idx = state.tool_call_index
            tc = state.tool_bucket.get(idx)
            if tc:
                args = tc["function"].get("arguments", "")
                if not args.strip():
                    # 空参数 → 补 "{}"，同时补发一个 chunk 让客户端看到
                    tc["function"]["arguments"] = "{}"
                    chunk = _make_openai_chunk(state, delta={
                        "tool_calls": [{"index": idx, "function": {"arguments": "{}"}}]
                    })
                    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                # 校验累积的 JSON 是否合法；不合法就记警告（不修改，让客户端看到真实问题）
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    import logging
                    logging.getLogger(__name__).warning(
                        "tool_use arguments invalid JSON: idx=%d len=%d head=%r",
                        idx, len(args), args[:200],
                    )
        return None

    if dtype == "message_delta":
        d = data.get("delta", {})
        sr = d.get("stop_reason")
        if sr:
            state.finish_reason = _STOP_REASON_MAP.get(sr, "stop")
        u = data.get("usage", {})
        if u and state.usage:
            state.usage["completion_tokens"] = u.get("output_tokens", 0)
            state.usage["total_tokens"] = (
                state.usage.get("prompt_tokens", 0) + u.get("output_tokens", 0)
            )
        # 不在这里发 finish chunk — 等 message_stop 时一起发
        # 这样避免 Cursor 收到 finish_reason 后提前关闭连接
        return None

    if dtype == "message_stop":
        # 在 [DONE] 之前发 finish chunk（带 usage），然后发 [DONE]
        parts: list[str] = []
        finish = state.finish_reason or "stop"
        chunk = _make_openai_chunk(state, delta={}, finish=finish)
        parts.append(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")
        parts.append("data: [DONE]\n\n")
        return "".join(parts)

    return None


def _make_openai_chunk(
    state: AnthropicStreamState,
    delta: dict[str, Any],
    finish: str | None = None,
) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish:
        choice["finish_reason"] = finish
    else:
        choice["finish_reason"] = None
    chunk: dict[str, Any] = {
        "id": f"chatcmpl-{state.message_id}",
        "object": "chat.completion.chunk",
        "created": int(_time.time()),
        "model": state.model,
        "choices": [choice],
    }
    if finish and state.usage:
        chunk["usage"] = state.usage
    return chunk
