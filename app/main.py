from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.admin import router as admin_router
from app.config import Settings, get_settings
from app.deps import verify_gateway_key
from app.embeddings import embed_worker
from app.http_client import build_http_client
from app.sse_stream import accumulate_delta, extract_usage, parse_sse_line
from app.storage import EmbedJob, PersistJob, create_pool, persist_worker
from app.upstream import merge_chat_completion_body, openrouter_headers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.openrouter_api_key.strip():
        logger.warning("OPENROUTER_API_KEY is empty — configure via env or /admin")
    if not settings.gateway_api_key.strip():
        logger.warning("GATEWAY_API_KEY is empty — configure via env or /admin")

    app.state.http_client = build_http_client(settings)
    app.state.persist_queue = None
    app.state.embed_queue = None
    app.state.persist_task = None
    app.state.embed_task = None
    app.state.db_pool = None

    if settings.database_url and settings.database_url.strip():
        pool = await create_pool(settings.database_url)
        app.state.db_pool = pool
        persist_q: asyncio.Queue[PersistJob | None] = asyncio.Queue(
            maxsize=settings.persist_queue_max
        )
        embed_q: asyncio.Queue[EmbedJob | None] = asyncio.Queue(
            maxsize=settings.embed_queue_max
        )
        app.state.persist_queue = persist_q
        app.state.embed_queue = embed_q
        app.state.persist_task = asyncio.create_task(
            persist_worker(pool, persist_q, embed_q)
        )
        app.state.embed_task = asyncio.create_task(
            embed_worker(pool, embed_q, app.state.http_client, settings)
        )
        logger.info("PostgreSQL persistence and embedding workers started")
    else:
        logger.warning("DATABASE_URL not set: persistence and embeddings disabled")

    yield

    if app.state.persist_queue is not None:
        await app.state.persist_queue.put(None)
    if app.state.persist_task:
        await app.state.persist_task
    if app.state.embed_queue is not None:
        await app.state.embed_queue.put(None)
    if app.state.embed_task:
        await app.state.embed_task
    if app.state.db_pool:
        await app.state.db_pool.close()
    await app.state.http_client.aclose()


app = FastAPI(title="OpenRouter OpenAI Gateway", lifespan=lifespan)
app.include_router(admin_router, prefix="/admin")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(
    _: None = Depends(verify_gateway_key),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    mid = settings.upstream_model
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": 0,
                "owned_by": "openrouter",
            }
        ],
    }


async def _enqueue_persist(
    app: FastAPI,
    job: PersistJob,
) -> None:
    q = getattr(app.state, "persist_queue", None)
    if q is None:
        return
    try:
        q.put_nowait(job)
    except asyncio.QueueFull:
        logger.warning("persist queue full; dropping log for session %s", job.external_session_id)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: dict[str, Any] = Body(...),
    _: None = Depends(verify_gateway_key),
    settings: Settings = Depends(get_settings),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
) -> Response:
    session_external = (x_session_id or "").strip() or str(uuid4())
    if not settings.openrouter_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENROUTER_API_KEY is not configured",
        )
    merged = merge_chat_completion_body(body, settings)
    url = f"{settings.upstream_base_url.rstrip('/')}/chat/completions"
    headers = openrouter_headers(settings)
    client: httpx.AsyncClient = request.app.state.http_client

    stream = bool(merged.get("stream"))
    t0 = time.perf_counter()

    if stream:
        return await _stream_chat(
            request,
            client,
            url,
            headers,
            merged,
            session_external,
            t0,
        )

    try:
        r = await client.post(url, headers=headers, json=merged)
    except httpx.RequestError as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        await _enqueue_persist(
            request.app,
            PersistJob(
                external_session_id=session_external,
                request_body=merged,
                response_body=None,
                usage_json=None,
                streamed=False,
                latency_ms=latency_ms,
                error_text=str(e),
            ),
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    latency_ms = int((time.perf_counter() - t0) * 1000)
    try:
        payload = r.json()
    except Exception:
        payload = None

    usage = None
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            pass
        else:
            usage = None

    await _enqueue_persist(
        request.app,
        PersistJob(
            external_session_id=session_external,
            request_body=merged,
            response_body=payload if isinstance(payload, dict) else None,
            usage_json=usage,
            streamed=False,
            latency_ms=latency_ms,
            error_text=None if r.is_success else r.text,
        ),
    )

    out = JSONResponse(
        content=payload if payload is not None else {"error": r.text},
        status_code=r.status_code,
        headers={"X-Session-Id": session_external},
    )
    return out


async def _stream_chat(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    merged: dict[str, Any],
    session_external: str,
    t0: float,
) -> StreamingResponse:
    parts: list[str] = []
    last_usage: dict[str, Any] | None = None

    async def gen():
        nonlocal last_usage
        err: str | None = None
        try:
            async with client.stream("POST", url, headers=headers, json=merged) as r:
                if r.status_code >= 400:
                    err_body = await r.aread()
                    err = err_body.decode("utf-8", errors="replace")
                    yield (
                        "data: "
                        + json.dumps(
                            {"error": {"message": err, "type": "upstream_error"}},
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )
                    return
                async for line in r.aiter_lines():
                    yield line + "\n\n"
                    parsed = parse_sse_line(line)
                    if not parsed or parsed.get("__done__"):
                        continue
                    accumulate_delta(parsed, parts)
                    u = extract_usage(parsed)
                    if u is not None:
                        last_usage = u
        except httpx.RequestError as e:
            err = str(e)
            yield (
                "data: "
                + json.dumps(
                    {"error": {"message": err, "type": "proxy_error"}},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        finally:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            full_text = "".join(parts)
            synthetic: dict[str, Any] = {
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }
                ],
            }
            if last_usage is not None:
                synthetic["usage"] = last_usage
            await _enqueue_persist(
                request.app,
                PersistJob(
                    external_session_id=session_external,
                    request_body=merged,
                    response_body=synthetic,
                    usage_json=last_usage,
                    streamed=True,
                    latency_ms=latency_ms,
                    error_text=err,
                ),
            )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "X-Session-Id": session_external,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
