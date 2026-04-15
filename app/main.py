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
    AnthropicStreamState,
    accumulate_delta,
    accumulate_tool_calls,
    anthropic_event_to_openai_chunk,
    extract_usage,
    finish_reason_from_chunk,
    parse_sse_line,
    tool_calls_list,
)
from app.storage import EmbedJob, PersistJob, create_pool, persistence_schema_ready, persist_worker
from app.upstream import (
    _coerce_raw_chat_request,
    anthropic_response_to_openai,
    build_executor_body,
    build_planner_body,
    extract_execute_commands,
    get_upstream_headers,
    get_upstream_url,
    merge_chat_completion_body,
    openrouter_headers,
)

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
    if settings.upstream_provider == "anthropic":
        if not settings.anthropic_api_key.strip():
            logger.warning("ANTHROPIC_API_KEY is empty — configure via env or /admin")
    elif settings.upstream_provider == "gemini":
        if not settings.google_api_key.strip():
            logger.warning("GOOGLE_API_KEY is empty — configure via env or /admin")
    else:
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


@app.get("/")
async def root() -> dict[str, Any]:
    """根路径说明（探活/人工访问）；API 仍以 /v1、/health、/admin 为准。"""
    return {
        "service": "openrouter-openai-gateway",
        "health": "/health",
        "admin": "/admin/",
        "openai_compatible": "/v1/models",
    }


@app.get("/favicon.ico")
async def favicon() -> Response:
    """避免浏览器默认请求产生无意义 404 日志。"""
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.head("/v1")
async def head_v1_root() -> Response:
    """部分探活/负载均衡会 HEAD /v1，避免无意义 404。"""
    return Response(status_code=204)


def _openai_models_payload(settings: Settings) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    mid = settings.upstream_model
    owned_by = "openrouter"
    if settings.upstream_provider == "anthropic":
        mid = settings.anthropic_model
        owned_by = "anthropic"
    elif settings.upstream_provider == "gemini":
        owned_by = "google"
    models.append({
        "id": mid,
        "object": "model",
        "created": 0,
        "owned_by": owned_by,
        "permission": [],
        "root": mid,
        "parent": None,
        "context_length": 200000,
        "capabilities": {"vision": True, "function_calling": True},
    })
    if settings.dual_model_enabled and settings.dual_model_name:
        models.append({
            "id": settings.dual_model_name,
            "object": "model",
            "created": 0,
            "owned_by": "gateway",
            "permission": [],
            "root": settings.dual_model_name,
            "parent": None,
            "context_length": 200000,
            "capabilities": {"vision": True, "function_calling": True},
        })
    return {
        "object": "list",
        "data": models,
    }


@app.get("/v1/models")
async def list_models(
    _: None = Depends(verify_gateway_key),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return _openai_models_payload(settings)


@app.get("/v1/v1/models")
async def list_models_double_v1_prefix(
    _: None = Depends(verify_gateway_key),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """客户端把 Base URL 配成 `.../v1` 时会请求 `/v1/v1/models`；与此兼容。"""
    return _openai_models_payload(settings)


# 疑似"丢失上下文"的通用开场白响应模式
_SUSPICIOUS_GENERIC_OPENINGS = (
    "I'm ready to help",
    "I'll assist you",
    "How can I help",
    "What would you like",
    "I don't have any context",
    "I don't have the context",
    "could you please provide more information",
    "could you please clarify",
    "Please let me know what you'd like",
)


def _is_suspicious_generic_response(text: str) -> bool:
    """检测响应是否是'新会话风格'的通用开场白（说明上下文丢失）。"""
    if not text:
        return False
    head = text[:400].lower()
    return any(s.lower() in head for s in _SUSPICIOUS_GENERIC_OPENINGS)


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
@app.post("/v1/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: dict[str, Any] = Body(...),
    _: None = Depends(verify_gateway_key),
    settings: Settings = Depends(get_settings),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
) -> Response:
    session_external = (x_session_id or "").strip() or str(uuid4())
    if settings.upstream_provider == "anthropic":
        if not settings.anthropic_api_key.strip():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ANTHROPIC_API_KEY is not configured",
            )
    elif settings.upstream_provider == "gemini":
        if not settings.google_api_key.strip():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GOOGLE_API_KEY is not configured",
            )
    else:
        if not settings.openrouter_api_key.strip():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENROUTER_API_KEY is not configured",
            )
    # 尽早统一 body 形态，便于日志与后续清洗一致（嵌套 request/data、JSON 字符串 messages 等）
    _coerce_raw_chat_request(body)
    # 调试：记录清洗前的原始 messages 结构（含 content block 详细信息）
    if settings.log_chat_metadata:
        _rm = body.get("messages")
        if not isinstance(_rm, list) or len(_rm) == 0:
            logger.info(
                "chat_body_diag session=%s top_keys=%s messages_type=%s",
                session_external,
                sorted(body.keys()),
                type(_rm).__name__,
            )
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

        # 记录 system prompt 头部 + 工具名列表，帮助诊断"模型不知道能做什么"问题
        _sys_head = ""
        for _m in _raw_msgs:
            if isinstance(_m, dict) and _m.get("role") == "system":
                c = _m.get("content", "")
                _sys_head = (c[:500] if isinstance(c, str) else str(c)[:500])
                break
        _tool_names = []
        _raw_tools = body.get("tools") or []
        for _t in _raw_tools[:30]:
            if isinstance(_t, dict):
                fn = _t.get("function", {})
                _tool_names.append(fn.get("name", "?"))
        logger.info(
            "chat_context session=%s system_head=%r tools=[%s]",
            session_external, _sys_head[:300], ",".join(_tool_names),
        )

    merged = merge_chat_completion_body(body, settings)
    _merged_msgs = merged.get("messages")
    if not isinstance(_merged_msgs, list) or len(_merged_msgs) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "规范化后 messages 为空，无法转发上游。请检查请求体是否包含非空的 messages，"
                "或兼容字段 input / contents / prompt。"
                "若使用 Cursor：OpenAI Base URL 应为主机根路径 + /v1（只出现一次 /v1），"
                "例如 https://你的网关/v1 而非 https://你的网关/v1/v1。"
            ),
        )
    url = get_upstream_url(settings)
    headers = get_upstream_headers(settings)
    client: httpx.AsyncClient = request.app.state.http_client
    is_anthropic = settings.upstream_provider == "anthropic"

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
        # 双模型协同：Opus 规划 + Qwen 执行（仅 OpenRouter 上游）
        if (
            settings.dual_model_enabled
            and merged.get("tools")
            and settings.upstream_provider == "openrouter"
        ):
            return await _dual_model_stream(
                request, client, url, headers, merged,
                session_external, t0, settings,
            )
        return await _stream_chat(
            request,
            client,
            url,
            headers,
            merged,
            session_external,
            t0,
            settings,
            is_anthropic=is_anthropic,
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

    # Anthropic 非流式响应 → 转为 OpenAI 格式
    if is_anthropic and r.is_success and isinstance(payload, dict) and payload.get("type") == "message":
        payload = anthropic_response_to_openai(payload)

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


async def _dual_model_stream(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    merged: dict[str, Any],
    session_external: str,
    t0: float,
    settings: Settings,
) -> StreamingResponse:
    """双模型串联：Opus 流式思考 → Qwen 流式执行 tool_calls。"""

    async def gen():
        chunk_id = f"chatcmpl-{uuid4().hex[:24]}"
        plan_parts: list[str] = []
        disconnected = False
        err: str | None = None

        # ═══ 阶段 1: Opus 规划 ═══
        planner_body = build_planner_body(merged, settings)
        if settings.log_chat_metadata:
            logger.info(
                "dual_planner session=%s model=%s",
                session_external, settings.planner_model,
            )

        try:
            async with client.stream("POST", url, headers=headers, json=planner_body) as r:
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
                    yield "data: [DONE]\n\n"
                    return

                async for line in r.aiter_lines():
                    if await request.is_disconnected():
                        disconnected = True
                        break
                    parsed = parse_sse_line(line)
                    if not parsed:
                        continue
                    if parsed.get("__done__"):
                        break
                    # 提取文字并累积
                    choices = parsed.get("choices") or []
                    for ch in choices:
                        delta = ch.get("delta") or {}
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            plan_parts.append(text)
                    # 重写 chunk id 后输出思考过程
                    parsed["id"] = chunk_id
                    yield f"data: {json.dumps(parsed, ensure_ascii=False)}\n\n"
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
            disconnected = True

        if disconnected:
            yield "data: [DONE]\n\n"
            return

        plan_text = "".join(plan_parts)
        commands = extract_execute_commands(plan_text)

        if settings.log_chat_metadata:
            logger.info(
                "dual_plan session=%s plan_len=%d commands=%s",
                session_external, len(plan_text),
                len(commands) if commands else "none",
            )

        if not commands:
            # Opus 没输出执行指令 → 纯文字回复，正常结束
            finish_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ═══ 阶段 2: Qwen 执行 ═══
        executor_body = build_executor_body(merged, commands, settings)
        if settings.log_chat_metadata:
            logger.info(
                "dual_executor session=%s model=%s tools=%d commands=%d",
                session_external, settings.executor_model,
                len(executor_body.get("tools") or []), len(commands),
            )

        try:
            async with client.stream("POST", url, headers=headers, json=executor_body) as r:
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
                    yield "data: [DONE]\n\n"
                    return

                async for line in r.aiter_lines():
                    if await request.is_disconnected():
                        break
                    parsed = parse_sse_line(line)
                    if not parsed:
                        continue
                    if parsed.get("__done__"):
                        break
                    # 重写 chunk id，只透传 tool_calls 相关 chunks
                    parsed["id"] = chunk_id
                    yield f"data: {json.dumps(parsed, ensure_ascii=False)}\n\n"
        except httpx.RequestError as e:
            yield (
                "data: "
                + json.dumps(
                    {"error": {"message": str(e), "type": "proxy_error"}},
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        yield "data: [DONE]\n\n"

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


@asynccontextmanager
async def _stream_with_retry(client, url, headers, json_body, settings, session_id, is_anthropic):
    """对 5xx 错误做一次透明重试（仅 Anthropic 路径）。"""
    async with client.stream("POST", url, headers=headers, json=json_body) as r:
        if r.status_code >= 500 and is_anthropic:
            await r.aread()  # 读完 body 释放连接
            if settings.log_chat_metadata:
                logger.warning(
                    "upstream_5xx session=%s status=%d, retrying once...",
                    session_id, r.status_code,
                )
            await asyncio.sleep(0.5)
        else:
            yield r
            return
    # 重试一次
    async with client.stream("POST", url, headers=headers, json=json_body) as r2:
        yield r2


async def _stream_chat(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    merged: dict[str, Any],
    session_external: str,
    t0: float,
    settings: Settings,
    *,
    is_anthropic: bool = False,
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
        tool_call_id_to_index: dict[str, int] = {}
        last_finish: str | None = None
        try:
            async with _stream_with_retry(client, url, headers, merged, settings, session_external, is_anthropic) as r:
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

                if is_anthropic:
                    # Anthropic SSE → 实时转为 OpenAI SSE chunk
                    anth_state = AnthropicStreamState()
                    event_type = ""
                    last_data_at = time.monotonic()
                    last_yield_at = time.monotonic()
                    idle_timeout = float(getattr(settings, "stream_idle_timeout_seconds", 300.0))
                    chunk_count = 0
                    first_chunk_at: float | None = None
                    abnormal_exit_reason: str | None = None  # 异常退出原因（用于发错误 chunk）
                    HEARTBEAT_INTERVAL = 2.0  # 每 2 秒发心跳（即使上游 stall）
                    if settings.log_chat_metadata:
                        logger.info("stream_start session=%s upstream=anthropic", session_external)

                    # 使用带超时的迭代，避免 aiter_lines 长时间 block
                    line_iter = r.aiter_lines().__aiter__()
                    stop_reading = False
                    while not stop_reading:
                        if await request.is_disconnected():
                            if settings.log_chat_metadata:
                                logger.info(
                                    "stream_client_disconnect session=%s chunks=%d",
                                    session_external, chunk_count,
                                )
                            abnormal_exit_reason = "client_disconnected"
                            break
                        try:
                            # 最多等 HEARTBEAT_INTERVAL 秒拿一行，拿不到就发心跳
                            line = await asyncio.wait_for(
                                line_iter.__anext__(),
                                timeout=HEARTBEAT_INTERVAL,
                            )
                        except asyncio.TimeoutError:
                            # 上游 stall — 发心跳保持连接
                            now = time.monotonic()
                            if now - last_data_at > idle_timeout:
                                logger.warning(
                                    "stream idle > %.1fs session=%s chunks=%d, forcing close",
                                    idle_timeout, session_external, chunk_count,
                                )
                                abnormal_exit_reason = (
                                    f"upstream stalled for {idle_timeout}s — no data received"
                                )
                                break
                            yield ": heartbeat\n\n"
                            last_yield_at = now
                            continue
                        except StopAsyncIteration:
                            # 上游流正常结束
                            if not got_done:
                                # 上游流结束但没发 message_stop，这是异常
                                abnormal_exit_reason = (
                                    "upstream stream ended without message_stop"
                                )
                            break

                        now = time.monotonic()
                        last_data_at = now
                        if first_chunk_at is None:
                            first_chunk_at = now
                            if settings.log_chat_metadata:
                                logger.info(
                                    "stream_first_byte session=%s delay=%.2fs",
                                    session_external, now - t0,
                                )

                        line_s = line.strip()
                        # 透传 SSE 注释行作为心跳（: comment）
                        if line_s.startswith(":"):
                            yield line_s + "\n\n"
                            continue
                        if line_s.startswith("event:"):
                            event_type = line_s[6:].strip()
                            continue
                        if not line_s.startswith("data:"):
                            continue
                        data_str = line_s[5:].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError as je:
                            logger.warning(
                                "malformed anthropic SSE data session=%s: %s (err=%s)",
                                session_external, data_str[:200], je,
                            )
                            continue
                        chunk_str = anthropic_event_to_openai_chunk(event_type, data, anth_state)
                        if chunk_str:
                            yield chunk_str
                            chunk_count += 1
                            last_yield_at = time.monotonic()
                            if "data: [DONE]" in chunk_str:
                                got_done = True
                                if settings.log_chat_metadata:
                                    logger.info(
                                        "stream_done session=%s chunks=%d total=%.2fs",
                                        session_external, chunk_count, time.monotonic() - t0,
                                    )
                        else:
                            # 处理了事件但没 yield 到客户端（如 thinking_delta、message_start）
                            # 超过心跳阈值就发 SSE 注释行，防止客户端认为连接死
                            if now - last_yield_at >= HEARTBEAT_INTERVAL:
                                yield ": heartbeat\n\n"
                                last_yield_at = now
                        # 同步内部状态
                        if anth_state.model:
                            stream_resp_model = anth_state.model
                        parts.clear()
                        parts.extend(anth_state.text_parts)
                        tool_bucket.update(anth_state.tool_bucket)
                        if anth_state.finish_reason:
                            last_finish = anth_state.finish_reason
                        if anth_state.usage:
                            last_usage = anth_state.usage
                        event_type = ""
                        if got_done:
                            stop_reading = True
                    # 循环结束后如果是异常退出，明确告知客户端 — 不要发假 DONE
                    if not got_done:
                        reason = abnormal_exit_reason or "stream ended unexpectedly"
                        if settings.log_chat_metadata:
                            logger.warning(
                                "stream_exit_without_done session=%s chunks=%d total=%.2fs "
                                "reason=%s last_finish=%s text_len=%d tool_calls=%d",
                                session_external, chunk_count, time.monotonic() - t0,
                                reason, last_finish,
                                len("".join(anth_state.text_parts)),
                                len(anth_state.tool_bucket),
                            )
                        # 发送错误 chunk，让 Cursor 知道这是失败响应而不是正常结束
                        # 关键：这样 Cursor 不会把半截内容存入历史作为"已完成"
                        err_chunk = {
                            "id": f"chatcmpl-{anth_state.message_id or 'err'}",
                            "object": "chat.completion.chunk",
                            "model": stream_resp_model or "",
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "error",
                            }],
                            "error": {
                                "message": f"Upstream stream interrupted: {reason}",
                                "type": "stream_interrupted",
                                "code": 502,
                            },
                        }
                        yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                        # 让 finally 自然发 [DONE] 终止 SSE 流
                        # Cursor 会看到 finish_reason=error + error 字段，知道是失败响应
                else:
                    # OpenRouter/OpenAI SSE 透传
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
                        accumulate_tool_calls(parsed, tool_bucket, tool_call_id_to_index)
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
            # 检测疑似"上下文丢失"的通用开场白响应
            if full_text and _is_suspicious_generic_response(full_text):
                _msgs_sent = merged.get("messages") or []
                logger.warning(
                    "suspicious_generic_response session=%s msgs_count=%d "
                    "text_head=%r last_user_msg_head=%r",
                    session_external,
                    len(_msgs_sent),
                    full_text[:200],
                    (str(_msgs_sent[-1].get("content", ""))[:200]
                     if _msgs_sent and isinstance(_msgs_sent[-1], dict) else ""),
                )
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
