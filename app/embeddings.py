"""Async embedding worker: OpenRouter/OpenAI-compatible embeddings API + pgvector insert."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import asyncpg
import httpx
from fastapi import FastAPI

from app.config import Settings, get_settings
from app.storage import EmbedJob

logger = logging.getLogger(__name__)


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" if x == int(x) else str(x) for x in vec) + "]"


async def _check_pgvector(pool: asyncpg.Pool) -> bool:
    """检测数据库是否安装了 pgvector 扩展。"""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchval(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )
            return row is not None
    except Exception:
        return False


async def embed_worker(
    app: FastAPI,
    pool: asyncpg.Pool,
    queue: asyncio.Queue[EmbedJob | None],
) -> None:
    """每次任务使用 app.state.http 客户端与 get_settings()，避免 /admin 保存时 aclose 旧 client 后仍引用已关闭实例。"""
    # 启动时检测 pgvector，若不存在则覆盖设置
    has_pgvector = await _check_pgvector(pool)
    if not has_pgvector:
        logger.warning("pgvector extension not found; falling back to double precision[] for embeddings")

    batch: list[EmbedJob] = []

    while True:
        # 尝试累积批次：先阻塞等第一条，然后非阻塞收集更多
        if not batch:
            job = await queue.get()
            if job is None:
                queue.task_done()
                break
            # 跳过极短文本（<50 字符），节省 embedding 费用
            if len(job.text.strip()) < 50:
                queue.task_done()
                continue
            batch.append(job)

        # 非阻塞收集更多 job 到批次上限
        settings = get_settings()
        batch_size = settings.embed_batch_size or 8
        while len(batch) < batch_size:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if job is None:
                queue.task_done()
                # 收到终止信号，先处理当前批次再退出
                await _try_flush(app, pool, has_pgvector, batch)
                batch.clear()
                return
            if len(job.text.strip()) < 50:
                queue.task_done()
                continue
            batch.append(job)

        # 如果批次未满，等一小段时间看有没有更多
        if len(batch) < batch_size:
            try:
                job = await asyncio.wait_for(queue.get(), timeout=2.0)
                if job is None:
                    queue.task_done()
                    await _try_flush(app, pool, has_pgvector, batch)
                    batch.clear()
                    return
                if len(job.text.strip()) >= 50:
                    batch.append(job)
                else:
                    queue.task_done()
            except asyncio.TimeoutError:
                pass  # 超时，处理当前批次

        # 处理批次
        await _try_flush(app, pool, has_pgvector, batch)
        for _ in batch:
            queue.task_done()
        batch.clear()


# 连续 401/403 超过阈值后，熔断嵌入功能避免刷屏
_EMBED_AUTH_FAIL_STREAK = 0
_EMBED_AUTH_FAIL_THRESHOLD = 3
_EMBED_DISABLED_UNTIL_RESTART = False


async def _try_flush(
    app: FastAPI,
    pool: asyncpg.Pool,
    has_pgvector: bool,
    batch: list[EmbedJob],
) -> None:
    """安全地处理一个批次。"""
    global _EMBED_AUTH_FAIL_STREAK, _EMBED_DISABLED_UNTIL_RESTART
    if not batch:
        return
    if _EMBED_DISABLED_UNTIL_RESTART:
        return  # 熔断后直接丢弃，不再打扰上游
    try:
        client: httpx.AsyncClient = app.state.http_client
        settings = get_settings()
        use_pgvector = settings.embedding_use_pgvector and has_pgvector
        await _flush_batch(pool, client, settings, batch, use_pgvector=use_pgvector)
        _EMBED_AUTH_FAIL_STREAK = 0  # 成功后清零
    except httpx.HTTPStatusError as e:
        status_code = getattr(e.response, "status_code", 0)
        if status_code in (401, 403):
            _EMBED_AUTH_FAIL_STREAK += 1
            if _EMBED_AUTH_FAIL_STREAK >= _EMBED_AUTH_FAIL_THRESHOLD:
                _EMBED_DISABLED_UNTIL_RESTART = True
                logger.error(
                    "embedding auth failed %d times in a row (status=%d url=%s). "
                    "DISABLING embedding worker until restart. "
                    "Fix: set embedding_api_key in Admin, or set a valid embedding_base_url, "
                    "or clear DATABASE_URL to disable persistence+embeddings.",
                    _EMBED_AUTH_FAIL_STREAK, status_code,
                    getattr(e.request, "url", "?"),
                )
            else:
                logger.warning(
                    "embedding auth failed (status=%d, streak=%d/%d)",
                    status_code, _EMBED_AUTH_FAIL_STREAK, _EMBED_AUTH_FAIL_THRESHOLD,
                )
        else:
            logger.exception("embedding batch failed for %d turns (status=%d)", len(batch), status_code)
    except Exception:
        logger.exception("embedding batch failed for %d turns", len(batch))


async def _flush_batch(
    pool: asyncpg.Pool,
    client: httpx.AsyncClient,
    settings: Settings,
    batch: list[EmbedJob],
    *,
    use_pgvector: bool = True,
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
                if settings.embedding_dim is not None and len(vec) != settings.embedding_dim:
                    logger.error(
                        "vector dim %s != expected %s",
                        len(vec),
                        settings.embedding_dim,
                    )
                    continue
                if use_pgvector:
                    await conn.execute(
                        """
                        INSERT INTO embeddings (turn_id, embedding, embedding_model)
                        VALUES ($1, $2::vector, $3)
                        """,
                        job.turn_id,
                        _vector_literal(vec),
                        emb_key,
                    )
                else:
                    await conn.execute(
                        """
                        INSERT INTO embeddings (turn_id, embedding, embedding_model)
                        VALUES ($1, $2, $3)
                        """,
                        job.turn_id,
                        vec,
                        emb_key,
                    )
