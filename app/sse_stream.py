"""Parse OpenAI-compatible SSE stream; accumulate assistant text, tool_calls, usage."""

from __future__ import annotations

import json
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


def apply_tool_call_delta(bucket: dict[int, dict[str, Any]], tc: dict[str, Any]) -> None:
    """Merge one streaming tool_call fragment (OpenAI format, keyed by index)."""
    idx = _coerce_tool_index(tc.get("index"))
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


def accumulate_tool_calls(chunk: dict[str, Any], bucket: dict[int, dict[str, Any]]) -> None:
    choices = chunk.get("choices") or []
    for ch in choices:
        delta = ch.get("delta") or {}
        tcs = delta.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if isinstance(tc, dict):
                apply_tool_call_delta(bucket, tc)


def tool_calls_list(bucket: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    if not bucket:
        return []
    return [bucket[i] for i in sorted(bucket.keys())]


def extract_usage(chunk: dict[str, Any]) -> dict[str, Any] | None:
    u = chunk.get("usage")
    if isinstance(u, dict):
        return u
    return None


def finish_reason_from_chunk(chunk: dict[str, Any]) -> str | None:
    choices = chunk.get("choices") or []
    for ch in choices:
        fr = ch.get("finish_reason")
        if fr is not None:
            return str(fr)
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
        err_chunk = {
            "id": f"chatcmpl-{state.message_id or 'error'}",
            "object": "chat.completion.chunk",
            "model": state.model or "",
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n[Upstream error: {err_msg}]"},
                "finish_reason": "stop",
            }],
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
            state.tool_call_index += 1
            idx = state.tool_call_index
            tc_id = cb.get("id", "")
            tc_name = cb.get("name", "")
            state.tool_bucket[idx] = {
                "id": tc_id, "type": "function",
                "function": {"name": tc_name, "arguments": ""},
            }
            chunk = _make_openai_chunk(state, delta={
                "tool_calls": [{"index": idx, "id": tc_id, "type": "function",
                                "function": {"name": tc_name, "arguments": ""}}]
            })
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        if state.current_block_type == "text":
            # 第一个 text block 发送 role
            chunk = _make_openai_chunk(state, delta={"role": "assistant", "content": ""})
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        return None

    if dtype == "content_block_delta":
        delta = data.get("delta", {})
        delta_type = delta.get("type", "")

        if delta_type == "text_delta":
            text = delta.get("text", "")
            if text:
                state.text_parts.append(text)
                chunk = _make_openai_chunk(state, delta={"content": text})
                return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        elif delta_type == "input_json_delta":
            partial = delta.get("partial_json", "")
            if partial and state.tool_call_index >= 0:
                idx = state.tool_call_index
                state.tool_bucket[idx]["function"]["arguments"] += partial
                chunk = _make_openai_chunk(state, delta={
                    "tool_calls": [{"index": idx, "function": {"arguments": partial}}]
                })
                return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        elif delta_type in ("thinking_delta", "signature_delta"):
            # 扩展思考模式：不输出给客户端但也不报错，保持流继续
            return None

        return None

    if dtype == "content_block_stop":
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
        # 输出 finish chunk
        chunk = _make_openai_chunk(state, delta={}, finish=state.finish_reason)
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    if dtype == "message_stop":
        return "data: [DONE]\n\n"

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
        "model": state.model,
        "choices": [choice],
    }
    if finish and state.usage:
        chunk["usage"] = state.usage
    return chunk
