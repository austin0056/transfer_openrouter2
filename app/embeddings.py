"""Async embedding worker: OpenRouter/OpenAI-compatible embeddings API + pgvector insert."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import asyncpg
import httpx

from app.config import Settings
from app.storage import EmbedJob

logger = logging.getLogger(__name__)


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" if x == int(x) else str(x) for x in vec) + "]"


async def embed_worker(
    pool: asyncpg.Pool,
    queue: asyncio.Queue[EmbedJob | None],
    client: httpx.AsyncClient,
    settings: Settings,
) -> None:
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            break
        try:
            await _flush_batch(pool, client, settings, [job])
        except Exception:
            logger.exception("embedding failed for turn %s", job.turn_id)
        finally:
            queue.task_done()


async def _flush_batch(
    pool: asyncpg.Pool,
    client: httpx.AsyncClient,
    settings: Settings,
    batch: list[EmbedJob],
) -> None:
    if not batch:
        return
    texts = [j.text for j in batch]
    key = settings.embedding_api_key or settings.openrouter_api_key
    url = f"{settings.embedding_base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": settings.embedding_model,
        "input": texts,
    }
    r = await client.post(url, headers=headers, json=body)
    r.raise_for_status()
    data = r.json()

    embeds_data = data.get("data") or []
    indexed: list[tuple[int, list[float]]] = []
    for item in embeds_data:
        idx = item.get("index", 0)
        emb = item.get("embedding")
        if isinstance(emb, list):
            indexed.append((idx, [float(x) for x in emb]))
    indexed.sort(key=lambda x: x[0])
    vectors = [v for _, v in indexed]
    if len(vectors) != len(batch):
        logger.error("embedding count mismatch: %s vs %s", len(vectors), len(batch))
        return

    emb_key = settings.embedding_model
    async with pool.acquire() as conn:
        async with conn.transaction():
            for job, vec in zip(batch, vectors, strict=True):
                if len(vec) != settings.embedding_dim:
                    logger.error(
                        "vector dim %s != expected %s",
                        len(vec),
                        settings.embedding_dim,
                    )
                    continue
                await conn.execute(
                    """
                    INSERT INTO embeddings (turn_id, embedding, embedding_model)
                    VALUES ($1, $2::vector, $3)
                    """,
                    job.turn_id,
                    _vector_literal(vec),
                    emb_key,
                )
