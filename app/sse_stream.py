"""Parse OpenAI-compatible SSE stream; accumulate assistant text and usage."""

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
    choices = chunk.get("choices") or []
    for ch in choices:
        delta = ch.get("delta") or {}
        c = delta.get("content")
        if isinstance(c, str) and c:
            parts.append(c)
        # tool_calls fragments (optional)
        tcs = delta.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                fn = (tc or {}).get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    parts.append(fn["arguments"])


def extract_usage(chunk: dict[str, Any]) -> dict[str, Any] | None:
    u = chunk.get("usage")
    if isinstance(u, dict):
        return u
    return None
