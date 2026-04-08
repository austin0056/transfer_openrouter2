"""Parse OpenAI-compatible SSE stream; accumulate assistant text, tool_calls, usage."""

from __future__ import annotations

import json
from typing import Any


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


def apply_tool_call_delta(bucket: dict[int, dict[str, Any]], tc: dict[str, Any]) -> None:
    """Merge one streaming tool_call fragment (OpenAI format, keyed by index)."""
    idx = tc.get("index")
    if idx is None:
        return
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
