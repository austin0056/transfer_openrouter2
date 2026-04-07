"""Async PostgreSQL persistence + queues for request logs and embedding jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class PersistJob:
    external_session_id: str
    request_body: dict[str, Any]
    response_body: dict[str, Any] | None
    usage_json: dict[str, Any] | None
    streamed: bool
    latency_ms: int
    error_text: str | None


@dataclass
class EmbedJob:
    turn_id: UUID
    text: str


def extract_last_user_text(messages: list[Any] | None) -> str:
    if not messages:
        return ""
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts: list[str] = []
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
            return "\n".join(parts)
    return ""


def extract_assistant_text(response_body: dict[str, Any] | None) -> str:
    if not response_body:
        return ""
    choices = response_body.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return ""


async def persist_worker(
    pool: asyncpg.Pool,
    queue: asyncio.Queue[PersistJob | None],
    embed_queue: asyncio.Queue[EmbedJob | None],
) -> None:
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            break
        try:
            await _persist_one(pool, embed_queue, job)
        except Exception:
            logger.exception("persist_worker failed for session %s", job.external_session_id)
        finally:
            queue.task_done()


async def _persist_one(
    pool: asyncpg.Pool,
    embed_queue: asyncio.Queue[EmbedJob | None],
    job: PersistJob,
) -> None:
    user_text = extract_last_user_text(job.request_body.get("messages"))
    assistant_text = extract_assistant_text(job.response_body)

    async with pool.acquire() as conn:
        async with conn.transaction():
            sid = await conn.fetchval(
                "SELECT id FROM sessions WHERE external_session_id = $1",
                job.external_session_id,
            )
            if sid is None:
                sid = await conn.fetchval(
                    "INSERT INTO sessions (external_session_id) VALUES ($1) RETURNING id",
                    job.external_session_id,
                )
            rid = await conn.fetchval(
                """
                INSERT INTO request_logs (
                    session_id, request_json, response_json, usage_json,
                    streamed, latency_ms, error_text
                )
                VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb, $5, $6, $7)
                RETURNING id
                """,
                sid,
                json.dumps(job.request_body),
                json.dumps(job.response_body) if job.response_body is not None else None,
                json.dumps(job.usage_json) if job.usage_json is not None else None,
                job.streamed,
                job.latency_ms,
                job.error_text,
            )
            if assistant_text or user_text:
                tid = await conn.fetchval(
                    """
                    INSERT INTO conversation_turns (
                        session_id, request_log_id, user_text, assistant_text
                    )
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    sid,
                    rid,
                    user_text or None,
                    assistant_text or None,
                )
                if tid and (user_text or assistant_text):
                    combined = f"User:\n{user_text}\n\nAssistant:\n{assistant_text}".strip()
                    if combined:
                        try:
                            embed_queue.put_nowait(EmbedJob(turn_id=tid, text=combined))
                        except asyncio.QueueFull:
                            logger.warning("embed queue full, dropping turn %s", tid)


async def create_pool(database_url: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(database_url, min_size=1, max_size=20)
