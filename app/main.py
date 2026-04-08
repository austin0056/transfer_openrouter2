from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.admin import router as admin_router
from app.config import Settings, get_settings
from app.deps import verify_gateway_key
from app.embeddings import embed_worker
from app.http_client import build_http_client
from app.sse_stream import (
    accumulate_delta,
    accumulate_tool_calls,
    extract_usage,
    finish_reason_from_chunk,
    parse_sse_line,
    tool_calls_list,
)
from app.storage import EmbedJob, PersistJob, create_pool, persistence_schema_ready, persist_worker
from app.upstream import merge_chat_completion_body, openrouter_headers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Boot: PORT=%s PYTHON=%s",
        os.environ.get("PORT", ""),
        os.environ.get("PYTHON_VERSION", ""),
    )
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
        try:
            pool = await create_pool(settings.database_url)
        except Exception:
            logger.exception(
                "Failed to connect to PostgreSQL (DATABASE_URL). "
                "Continuing without persistence; fix URL or wait for DB to be ready."
            )
        else:
            if not await persistence_schema_ready(pool):
                logger.error(
                    "DATABASE_URL 能连通，但缺少表 public.sessions（未执行 migrations/001_init.sql "
                    "或连到了空库）。已关闭落库与向量任务；请在「当前连接串指向的库」执行 SQL 后重启。"
                )
                await pool.close()
            else:
                app.state.db_pool = pool
                pm = settings.persist_queue_max
                em = settings.embed_queue_max
                if pm is None:
                    pm = 10000
                if em is None:
                    em = 10000
                persist_q: asyncio.Queue[PersistJob | None] = asyncio.Queue(maxsize=pm)
                embed_q: asyncio.Queue[EmbedJob | None] = asyncio.Queue(maxsize=em)
                app.state.persist_queue = persist_q
                app.state.embed_queue = embed_q
                app.state.persist_task = asyncio.create_task(
                    persist_worker(pool, persist_q, embed_q)
                )
                app.state.embed_task = asyncio.create_task(
                    embed_worker(app, pool, embed_q)
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    settings = get_settings()
    max_bytes = settings.max_request_body_mb * 1024 * 1024
    cl = request.headers.get("content-length")
    if cl and int(cl) > max_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "message": f"Request body too large (max {settings.max_request_body_mb}MB)",
                    "type": "invalid_request_error",
                    "code": 413,
                }
            },
        )
    return await call_next(request)


app.include_router(admin_router, prefix="/admin")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.head("/v1")
async def head_v1_root() -> Response:
    """部分探活/负载均衡会 HEAD /v1，避免无意义 404。"""
    return Response(status_code=204)


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
                "owned_by": "anthropic",
                "permission": [],
                "root": mid,
                "parent": None,
                "context_length": 200000,
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                },
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


def _log_chat_upstream_meta(
    settings: Settings,
    merged: dict[str, Any],
    *,
    streamed: bool,
    response_model: str | None,
    ok: bool,
    err_short: str | None = None,
) -> None:
    if not settings.log_chat_metadata:
        return
    tools = merged.get("tools")
    n_tools = len(tools) if isinstance(tools, list) else 0
    extra = ""
    if err_short:
        extra = " err=" + err_short[:300].replace("\n", " ")
    logger.info(
        "chat_upstream_meta ok=%s stream=%s tools=%d request_model=%s response_model=%s%s",
        ok,
        streamed,
        n_tools,
        merged.get("model"),
        response_model,
        extra,
    )


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
    # 调试：记录清洗前的原始 messages 结构（含 content block 详细信息）
    if settings.log_chat_metadata:
        _raw_msgs = body.get("messages") or []
        raw_summary = []
        for i, _m in enumerate(_raw_msgs):
            if not isinstance(_m, dict):
                continue
            r = _m.get("role", "?")
            c = _m.get("content")
            c_type = type(c).__name__
            c_len = len(c) if isinstance(c, (str, list)) else 0
            # 详细记录 content block 类型和文本长度
            blocks_info = ""
            if isinstance(c, list):
                bi = []
                for b in c:
                    if isinstance(b, dict):
                        bt = b.get("type", "?")
                        txt = b.get("text", "")
                        tlen = len(txt) if isinstance(txt, str) else -1
                        bi.append(f"{bt}:{tlen}")
                    else:
                        bi.append(f"raw:{type(b).__name__}")
                blocks_info = f"[{','.join(bi)}]"
            raw_summary.append(f"{i}:{r}({c_type},{c_len}){blocks_info}")
        logger.info("chat_raw_messages session=%s raw=[%s]", session_external, " ".join(raw_summary))

    merged = merge_chat_completion_body(body, settings)
    url = f"{settings.upstream_base_url.rstrip('/')}/chat/completions"
    headers = openrouter_headers(settings)
    client: httpx.AsyncClient = request.app.state.http_client

    # 调试日志：记录发给上游的最终 messages 结构（含 content 详情）
    if settings.log_chat_metadata:
        _msgs = merged.get("messages") or []
        msg_summary = []
        for i, _m in enumerate(_msgs):
            if not isinstance(_m, dict):
                continue
            r = _m.get("role", "?")
            c = _m.get("content")
            has_tc = bool(_m.get("tool_calls"))
            tcid = _m.get("tool_call_id", "")
            # content 概要
            if isinstance(c, str):
                c_info = f"str:{len(c)}"
            elif isinstance(c, list):
                block_infos = []
                for b in c:
                    if isinstance(b, dict):
                        bt = b.get("type", "?")
                        txt = b.get("text", "")
                        tl = len(txt) if isinstance(txt, str) else -1
                        block_infos.append(f"{bt}:{tl}")
                    else:
                        block_infos.append(f"raw")
                c_info = f"[{','.join(block_infos)}]"
            elif c is None:
                c_info = "null"
            else:
                c_info = f"other:{type(c).__name__}"
            info = f"{i}:{r}({c_info})"
            if has_tc:
                tc_names = [tc.get("function", {}).get("name", "?") for tc in _m.get("tool_calls", []) if isinstance(tc, dict)]
                info += f"[tc:{','.join(tc_names)}]"
            if tcid:
                info += f"[tcid:{tcid[:12]}]"
            msg_summary.append(info)
        logger.info(
            "chat_merged_debug session=%s msgs=[%s] max_tokens=%s tools=%d",
            session_external,
            " → ".join(msg_summary),
            merged.get("max_tokens"),
            len(merged.get("tools") or []),
        )

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
            settings,
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
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": {
                    "message": str(e),
                    "type": "proxy_error",
                    "code": 502,
                }
            },
            headers={"X-Session-Id": session_external},
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    try:
        payload = r.json()
    except Exception:
        payload = None

    usage = None
    resp_model: str | None = None
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            pass
        else:
            usage = None
        rm = payload.get("model")
        if isinstance(rm, str) and rm:
            resp_model = rm

    _log_chat_upstream_meta(
        settings,
        merged,
        streamed=False,
        response_model=resp_model,
        ok=r.is_success,
        err_short=r.text if not r.is_success else None,
    )

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

    if payload is None or (not r.is_success and not isinstance(payload, dict)):
        payload = {
            "error": {
                "message": r.text,
                "type": "upstream_error",
                "code": r.status_code,
            }
        }
    elif not r.is_success and isinstance(payload, dict) and "error" not in payload:
        payload = {
            "error": {
                "message": json.dumps(payload, ensure_ascii=False),
                "type": "upstream_error",
                "code": r.status_code,
            }
        }

    out = JSONResponse(
        content=payload,
        status_code=r.status_code,
        headers={"X-Session-Id": session_external},
    )
    return out


@app.post("/v1/embeddings")
async def embeddings_proxy(
    request: Request,
    body: dict[str, Any] = Body(...),
    _: None = Depends(verify_gateway_key),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Proxy embeddings requests to upstream (OpenRouter / OpenAI compatible)."""
    key = settings.embedding_api_key or settings.openrouter_api_key
    if not key or not key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding API key is not configured",
        )
    url = f"{settings.embedding_base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if "model" not in body or not body["model"]:
        body["model"] = settings.embedding_model
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        r = await client.post(url, headers=headers, json=body)
    except httpx.RequestError as e:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(e), "type": "proxy_error", "code": 502}},
        )
    try:
        payload = r.json()
    except Exception:
        payload = None
    if payload is None:
        payload = {"error": {"message": r.text, "type": "upstream_error", "code": r.status_code}}
    return JSONResponse(content=payload, status_code=r.status_code)


async def _stream_chat(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    merged: dict[str, Any],
    session_external: str,
    t0: float,
    settings: Settings,
) -> StreamingResponse:
    parts: list[str] = []
    last_usage: dict[str, Any] | None = None

    async def gen():
        nonlocal last_usage
        err: str | None = None
        stream_resp_model: str | None = None
        upstream_ok = False
        got_done = False
        tool_bucket: dict[int, dict[str, Any]] = {}
        last_finish: str | None = None
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
                    upstream_ok = False
                    return
                upstream_ok = True
                got_done = False
                async for line in r.aiter_lines():
                    if await request.is_disconnected():
                        break
                    yield line + "\n\n"
                    parsed = parse_sse_line(line)
                    if not parsed:
                        continue
                    if parsed.get("__done__"):
                        got_done = True
                        continue
                    if isinstance(parsed, dict):
                        m = parsed.get("model")
                        if isinstance(m, str) and m:
                            stream_resp_model = m
                    accumulate_delta(parsed, parts)
                    accumulate_tool_calls(parsed, tool_bucket)
                    fr = finish_reason_from_chunk(parsed)
                    if fr:
                        last_finish = fr
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
            if not got_done:
                yield "data: [DONE]\n\n"
            _log_chat_upstream_meta(
                settings,
                merged,
                streamed=True,
                response_model=stream_resp_model,
                ok=upstream_ok and err is None,
                err_short=err,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            full_text = "".join(parts)
            tclist = tool_calls_list(tool_bucket)
            msg: dict[str, Any] = {"role": "assistant"}
            if full_text:
                msg["content"] = full_text
            else:
                msg["content"] = None
            if tclist:
                msg["tool_calls"] = tclist
            fin = last_finish or ("tool_calls" if tclist else "stop")
            synthetic: dict[str, Any] = {
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": msg,
                        "finish_reason": fin,
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
            "X-Accel-Buffering": "no",
        },
    )
