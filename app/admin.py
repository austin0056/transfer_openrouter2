from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from app.config import Settings, config_json_path, get_settings, save_runtime_config
from app.http_client import build_http_client

router = APIRouter(tags=["admin"])


def _admin_secret(settings: Settings) -> str:
    return (settings.admin_key or "").strip()


async def verify_admin(
    request: Request,
    settings: Settings = Depends(get_settings),
    authorization: str | None = Header(None),
    x_admin_key: str | None = Header(None, alias="X-Admin-Key"),
) -> None:
    if not _admin_secret(settings):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin disabled")
    token = None
    if x_admin_key:
        token = x_admin_key.strip()
    elif authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    if not token or token != _admin_secret(settings):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key")


def _public_config(settings: Settings) -> dict[str, Any]:
    d = settings.model_dump()
    d.pop("admin_key", None)
    return d


@router.get("/api/config", dependencies=[Depends(verify_admin)])
async def admin_get_config(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {"config_path": str(config_json_path()), "settings": _public_config(settings)}


@router.post("/api/generate-gateway-key", dependencies=[Depends(verify_admin)])
async def generate_gateway_key() -> dict[str, str]:
    """Cryptographically strong key for Cursor / OpenAI-compatible clients."""
    return {"gateway_api_key": secrets.token_urlsafe(48)}


@router.post("/api/config", dependencies=[Depends(verify_admin)])
async def admin_save_config(
    request: Request,
    body: dict[str, Any],
) -> dict[str, Any]:
    save_runtime_config(body)
    new_client = build_http_client(get_settings())
    old = request.app.state.http_client
    request.app.state.http_client = new_client
    await old.aclose()
    return {
        "ok": True,
        "message": "已保存并刷新 HTTP 客户端。嵌入任务会自动使用新客户端。若修改了 DATABASE_URL，仍需重启以重建数据库连接池与 worker。",
        "config_path": str(config_json_path()),
    }


@router.get("/", response_class=HTMLResponse)
async def admin_page() -> str:
    return _ADMIN_HTML


_ADMIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>网关管理 · OpenRouter</title>
  <style>
    :root {
      --bg: #0d1117;
      --surface: #161b22;
      --surface2: #21262d;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #238636;
      --accent-hover: #2ea043;
      --danger: #f85149;
      --ok: #3fb950;
      --code: #79c0ff;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "PingFang SC", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 0 16px 48px; max-width: 920px; margin-left: auto; margin-right: auto; }
    .topbar {
      position: sticky; top: 0; z-index: 10;
      background: linear-gradient(180deg, rgba(13,17,23,0.98), rgba(13,17,23,0.92));
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      padding: 16px 0 12px; margin: 0 -16px 20px; padding-left: 16px; padding-right: 16px;
    }
    .topbar h1 { margin: 0 0 4px; font-size: 1.35rem; font-weight: 600; letter-spacing: -0.02em; }
    .topbar .sub { font-size: 0.85rem; color: var(--muted); margin: 0; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 14px; }
    .intro {
      background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
      padding: 14px 16px; margin-bottom: 20px; font-size: 0.9rem; color: var(--muted);
    }
    .intro strong { color: var(--text); }
    .intro code { color: var(--code); font-size: 0.85em; padding: 1px 6px; background: var(--surface2); border-radius: 4px; }
    .card {
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
      padding: 18px 20px 22px; margin-bottom: 18px;
    }
    .card h2 {
      margin: 0 0 6px; font-size: 1rem; font-weight: 600; color: var(--text);
      display: flex; align-items: center; gap: 8px;
    }
    .card .section-desc { margin: 0 0 16px; font-size: 0.8rem; color: var(--muted); line-height: 1.45; }
    .field { margin-bottom: 18px; }
    .field:last-child { margin-bottom: 0; }
    .field-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 12px; margin-bottom: 6px; }
    .field-title { font-size: 0.9rem; font-weight: 600; color: var(--text); }
    .field-key {
      font-family: ui-monospace, "Cascadia Code", monospace; font-size: 0.72rem;
      color: var(--muted); background: var(--bg); padding: 2px 8px; border-radius: 4px; border: 1px solid var(--border);
    }
    .field-desc {
      font-size: 0.8rem; color: var(--muted); line-height: 1.5; margin: 0 0 8px;
    }
    input[type="text"], input[type="password"], input[type="number"], textarea {
      width: 100%; padding: 10px 12px; border-radius: 8px;
      border: 1px solid var(--border); background: var(--bg); color: var(--text);
      font-size: 0.9rem;
    }
    input:focus, textarea:focus { outline: none; border-color: #388bfd; box-shadow: 0 0 0 3px rgba(56,139,253,0.15); }
    textarea { min-height: 72px; font-family: ui-monospace, monospace; font-size: 0.82rem; resize: vertical; }
    .checks { display: flex; flex-direction: column; gap: 12px; }
    @media (min-width: 560px) { .checks { flex-direction: row; flex-wrap: wrap; } }
    .check-item {
      flex: 1; min-width: 200px;
      border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; background: var(--bg);
    }
    .check-item label { display: flex; align-items: flex-start; gap: 10px; cursor: pointer; margin: 0; font-size: 0.88rem; }
    .check-item input { width: auto; margin-top: 3px; accent-color: var(--accent); }
    .check-item .t { font-weight: 600; color: var(--text); display: block; }
    .check-item .d { font-size: 0.78rem; color: var(--muted); margin-top: 4px; line-height: 1.4; }
    button {
      padding: 10px 18px; border-radius: 8px; border: 0; cursor: pointer; font-weight: 600; font-size: 0.9rem;
      transition: background 0.15s, transform 0.05s;
    }
    button:active { transform: scale(0.98); }
    .primary { background: var(--accent); color: #fff; }
    .primary:hover { background: var(--accent-hover); }
    .btn-secondary {
      background: var(--surface2); color: var(--text); border: 1px solid var(--border);
      white-space: nowrap; flex-shrink: 0;
    }
    .btn-secondary:hover { background: #30363d; }
    .input-row { display: flex; gap: 10px; align-items: stretch; flex-wrap: wrap; }
    .input-row input { flex: 1; min-width: 0; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .mono-readonly {
      flex: 1; min-width: 200px; font-family: ui-monospace, monospace; font-size: 0.85rem;
      padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border);
      background: var(--bg); color: var(--ok); word-break: break-all;
    }
    .card-highlight { border-color: rgba(56, 139, 253, 0.35); }
    .foot-note { font-size: 0.8rem; color: var(--muted); margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border); }
    .foot-note code { color: var(--code); font-size: 0.85em; }
    #msg { margin-top: 12px; min-height: 1.2em; font-size: 0.88rem; }
    .err { color: var(--danger); white-space: pre-wrap; }
    .ok { color: var(--ok); }
    .admin-login label { display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 6px; }
  </style>
</head>
<body>
  <header class="topbar">
    <h1>网关配置</h1>
    <p class="sub">OpenRouter 兼容网关 · 保存至 CONFIG_FILE（与 ADMIN_KEY 分离）</p>
    <div class="toolbar admin-login">
      <div style="flex:1; min-width:200px; max-width:420px">
        <label for="adminKey">管理密钥（环境变量 ADMIN_KEY，仅用于登录本页）</label>
        <input type="password" id="adminKey" placeholder="与服务器 ADMIN_KEY 一致" autocomplete="off"/>
      </div>
      <button type="button" class="primary" id="btnLoad">加载配置</button>
      <button type="button" class="primary" id="btnSave">保存全部</button>
    </div>
    <div id="msg"></div>
  </header>

  <p class="intro">
    <strong>推荐流程：</strong>在 Zeabur 等平台只设置 <code>ADMIN_KEY</code>；本页填写 OpenRouter 密钥、对外 Gateway 密钥、代理与数据库等，保存后写入 <code>CONFIG_FILE</code>（默认 <code>data/config.json</code>，可用数据盘路径如 <code>/data/config.json</code>）。
    <code>ADMIN_KEY</code> 不会写入配置文件。修改 <code>DATABASE_URL</code> 或嵌入相关项后<strong>建议重启服务</strong>以应用后台任务。
  </p>

  <div class="card card-highlight">
    <h2>Cursor / OpenAI 兼容客户端</h2>
    <p class="section-desc">网关对外提供与 OpenAI 相同的 <code>/v1/chat/completions</code> 等路径。下面 Base URL 由<strong>当前访问域名</strong>自动算出（你部署在 Zeabur 后打开本页即可得到正确地址）。</p>
    <div class="field">
      <div class="field-head"><span class="field-title">OpenAI API Base URL</span><span class="field-key">填到 Cursor 的 Override OpenAI Base URL</span></div>
      <p class="field-desc">须以 <code>/v1</code> 结尾，例如 <code>https://你的域名/v1</code>。下面框内为当前站点对应值，点「复制 URL」粘贴到 Cursor。</p>
      <div class="input-row">
        <div class="mono-readonly" id="openaiBaseUrl" title="OpenAI-compatible base URL"></div>
        <button type="button" class="btn-secondary" id="btnCopyBaseUrl">复制 URL</button>
      </div>
    </div>
    <p class="field-desc" style="margin:0">API Key 填下方 <strong>Gateway 密钥</strong>；Model 填与 <code>upstream_model</code> 相同的模型 ID（如 <code>anthropic/claude-opus-4.6</code>）。</p>
  </div>

  <form id="cfg" onsubmit="return false;">

    <div class="card">
      <h2>上游与模型</h2>
      <p class="section-desc">网关代表客户端向 OpenRouter 发起请求时使用的密钥与模型标识。</p>

      <div class="field">
        <div class="field-head"><span class="field-title">OpenRouter 上游 API 密钥</span><span class="field-key">openrouter_api_key</span></div>
        <p class="field-desc">在 <a href="https://openrouter.ai/" target="_blank" rel="noopener" style="color:#58a6ff">openrouter.ai</a> 控制台创建。网关用此密钥调用上游 <code>chat/completions</code>，请妥善保管、勿提交到公开仓库。</p>
        <input type="password" name="openrouter_api_key" autocomplete="off" placeholder="sk-or-v1-..."/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">对外 Gateway 密钥（给 Cursor 用）</span><span class="field-key">gateway_api_key</span></div>
        <p class="field-desc">客户端（如 Cursor）在 <code>Authorization: Bearer</code> 里填的密钥，<strong>与 OpenRouter Key 独立</strong>。可点<strong>随机生成</strong>；生成或保存后点<strong>复制密钥</strong>粘贴到 Cursor。若留空后点「保存全部」，也会<strong>自动生成</strong>并写入配置。</p>
        <div class="input-row">
          <input type="password" name="gateway_api_key" id="gatewayApiKey" autocomplete="off" placeholder="点击随机生成，或保存时留空则自动生成"/>
          <div class="btn-row">
            <button type="button" class="btn-secondary" id="btnGenGateway">随机生成</button>
            <button type="button" class="btn-secondary" id="btnCopyGateway">复制密钥</button>
          </div>
        </div>
      </div>

      <div class="field" id="cursorSnippetField">
        <div class="field-head"><span class="field-title">Cursor 快速配置</span></div>
        <p class="field-desc">将以下 JSON 片段粘贴到 Cursor 的 <code>Settings &gt; Models &gt; OpenAI API Key</code> 和 <code>Override OpenAI Base URL</code>，或直接复制到 <code>.cursor/settings.json</code>。</p>
        <pre id="cursorSnippet" style="background:#1a1a2e;color:#e0e0e0;padding:12px;border-radius:6px;font-size:13px;overflow-x:auto;white-space:pre-wrap;word-break:break-all;cursor:pointer" title="点击复制">正在生成…</pre>
        <button type="button" class="btn-secondary" id="btnCopyCursorSnippet" style="margin-top:6px">复制 Cursor 配置</button>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">上游供应商协议</span><span class="field-key">upstream_provider</span></div>
        <p class="field-desc">选择上游 API 协议。<strong>OpenRouter</strong> 走 OpenAI 兼容协议，<strong>Anthropic</strong> 走原生 Messages API 直连。</p>
        <select name="upstream_provider" id="upstreamProvider">
          <option value="openrouter">OpenRouter (OpenAI 兼容)</option>
          <option value="anthropic">Anthropic 原生直连</option>
        </select>
      </div>

      <div id="openrouterFields">
        <div class="field">
          <div class="field-head"><span class="field-title">上游模型 ID</span><span class="field-key">upstream_model</span></div>
          <p class="field-desc">OpenRouter 模型 slug，例如 <code>anthropic/claude-opus-4.6</code>。</p>
          <input type="text" name="upstream_model" placeholder="anthropic/claude-opus-4.6"/>
        </div>
        <div class="field">
          <div class="field-head"><span class="field-title">上游 API 根地址</span><span class="field-key">upstream_base_url</span></div>
          <p class="field-desc">一般为 <code>https://openrouter.ai/api/v1</code>。</p>
          <input type="text" name="upstream_base_url" placeholder="https://openrouter.ai/api/v1"/>
        </div>
        <div class="field">
          <div class="field-head"><span class="field-title">OpenRouter HTTP-Referer（可选）</span><span class="field-key">openrouter_http_referer</span></div>
          <p class="field-desc">OpenRouter 统计用 URL，例如公开仓库链接。</p>
          <input type="text" name="openrouter_http_referer" placeholder="https://github.com/you/repo"/>
        </div>
        <div class="field">
          <div class="field-head"><span class="field-title">OpenRouter X-Title（可选）</span><span class="field-key">openrouter_app_title</span></div>
          <p class="field-desc">在 OpenRouter 侧展示的应用标题。</p>
          <input type="text" name="openrouter_app_title" placeholder="OpenRouter Gateway"/>
        </div>
      </div>

      <div id="anthropicFields" style="display:none">
        <div class="field">
          <div class="field-head"><span class="field-title">Anthropic API Key</span><span class="field-key">anthropic_api_key</span></div>
          <p class="field-desc">Anthropic 原生 API 密钥（sk-ant-...）。</p>
          <input type="password" name="anthropic_api_key" autocomplete="off" placeholder="sk-ant-api03-..."/>
        </div>
        <div class="field">
          <div class="field-head"><span class="field-title">Anthropic 模型 ID</span><span class="field-key">anthropic_model</span></div>
          <p class="field-desc">原生模型 ID，例如 <code>claude-opus-4-20250514</code>。</p>
          <input type="text" name="anthropic_model" placeholder="claude-opus-4-20250514"/>
        </div>
        <div class="field">
          <div class="field-head"><span class="field-title">Anthropic Base URL</span><span class="field-key">anthropic_base_url</span></div>
          <p class="field-desc">默认 <code>https://api.anthropic.com</code>。使用反代时修改。</p>
          <input type="text" name="anthropic_base_url" placeholder="https://api.anthropic.com"/>
        </div>
        <div class="field">
          <div class="field-head"><span class="field-title">Anthropic API Version</span><span class="field-key">anthropic_version</span></div>
          <p class="field-desc">一般不需要修改。</p>
          <input type="text" name="anthropic_version" placeholder="2023-06-01"/>
        </div>
      </div>

      <div class="checks">
        <div class="check-item">
          <label><input type="checkbox" name="loose_tools_passthrough"/>
            <span><span class="t">宽松工具透传</span><span class="field-key" style="display:inline;margin-left:6px">loose_tools_passthrough</span>
            <span class="d">开启后保留非 <code>function</code> 标准形态的 <code>tools</code> 项；关闭时仅保留带合法 <code>function.name</code> 的项。</span></span>
          </label>
        </div>
        <div class="check-item">
          <label><input type="checkbox" name="log_chat_metadata"/>
            <span><span class="t">记录对话元数据日志</span><span class="field-key" style="display:inline;margin-left:6px">log_chat_metadata</span>
            <span class="d">每条 chat 一行 INFO：流式与否、tools 数量、请求/响应 model，<strong>不包含</strong>消息正文。</span></span>
          </label>
        </div>
      </div>

      <div class="checks">
        <div class="check-item">
          <label><input type="checkbox" name="identity_prompt_enabled"/>
            <span><span class="t">注入身份说明（system）</span><span class="field-key" style="display:inline;margin-left:6px">identity_prompt_enabled</span>
            <span class="d">开启后，每次对话会在消息前合并一条 system，便于用户问「你是什么模型」时按下方文案回答（减少误称 Sonnet）。</span></span>
          </label>
        </div>
      </div>
      <div class="field">
        <div class="field-head"><span class="field-title">身份说明正文</span><span class="field-key">identity_prompt</span></div>
        <p class="field-desc">仅当上一项勾选时生效；会包在一小段「仅在问身份时作答」的指令后发给上游。</p>
        <textarea name="identity_prompt" rows="4" placeholder="我是 Claude Opus 4.6…"></textarea>
      </div>
    </div>

    <div class="card">
      <h2>网络代理</h2>
      <p class="section-desc">出站访问 OpenRouter 时使用的 HTTP/SOCKS 代理（如住宅 SOCKS5）。保存后会自动重建 HTTP 客户端。</p>
      <div class="field">
        <div class="field-head"><span class="field-title">HTTPS / SOCKS 代理 URL</span><span class="field-key">https_proxy</span></div>
        <p class="field-desc">示例：<code>socks5://用户:密码@主机:端口</code>。留空表示直连。若代理格式错误，服务会记录日志并尝试无代理启动。</p>
        <input type="text" name="https_proxy" placeholder="socks5://user:pass@host:port"/>
      </div>
    </div>

    <div class="card">
      <h2>PostgreSQL</h2>
      <p class="section-desc">用于异步保存会话请求记录与向量（需已执行仓库内 <code>migrations</code>）。支持 pgvector 或普通 PostgreSQL（见嵌入卡片说明）。</p>
      <div class="field">
        <div class="field-head"><span class="field-title">数据库连接串</span><span class="field-key">database_url</span></div>
        <p class="field-desc">标准 PostgreSQL URI，例如 <code>postgresql://用户:密码@主机:端口/库名</code>。留空则<strong>不落库、不写向量</strong>，仅转发对话。修改后建议<strong>重启进程</strong>以让连接池与后台任务生效。</p>
        <textarea name="database_url" rows="3" placeholder="postgresql://..."></textarea>
      </div>
    </div>

    <div class="card">
      <h2>提示词缓存（Anthropic / OpenRouter）</h2>
      <p class="section-desc">通过顶层 <code>cache_control</code> 启用自动缓存，配合 OpenRouter 路由；短对话可能因最小 token 门槛收益不明显。</p>
      <div class="checks">
        <div class="check-item">
          <label><input type="checkbox" name="cache_enabled"/>
            <span><span class="t">启用缓存注入</span><span class="field-key" style="display:inline;margin-left:6px">cache_enabled</span>
            <span class="d">关闭则不在请求中附加 <code>cache_control</code>，完全依赖上游默认行为。</span></span>
          </label>
        </div>
        <div class="check-item">
          <label><input type="checkbox" name="cache_ttl_1h"/>
            <span><span class="t">使用 1 小时 TTL</span><span class="field-key" style="display:inline;margin-left:6px">cache_ttl_1h</span>
            <span class="d">开启：约 1 小时缓存窗口；关闭：约 5 分钟 ephemeral。长会话可开 1 小时，注意计费与文档说明。</span></span>
          </label>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>嵌入（向量）</h2>
      <p class="section-desc">对话落库后异步调用嵌入 API。若托管库<strong>无 pgvector</strong>，请执行 <code>migrations/001_init_no_pgvector.sql</code>，并<strong>取消勾选</strong>下方「使用 pgvector 列类型」。无扩展时维度仍须与模型一致。</p>

      <div class="checks">
        <div class="check-item">
          <label><input type="checkbox" name="embedding_use_pgvector" checked/>
            <span><span class="t">使用 pgvector 列类型写入</span><span class="field-key" style="display:inline;margin-left:6px">embedding_use_pgvector</span>
            <span class="d">开启：对应 <code>001_init.sql</code> 的 <code>vector(N)</code>。关闭：对应 <code>001_init_no_pgvector.sql</code> 的 <code>double precision[]</code>。</span></span>
          </label>
        </div>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">嵌入模型</span><span class="field-key">embedding_model</span></div>
        <p class="field-desc">OpenRouter 兼容的 <code>/embeddings</code> 模型名，需与下方维度匹配。</p>
        <input type="text" name="embedding_model" placeholder="openai/text-embedding-3-small"/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">向量维度</span><span class="field-key">embedding_dim</span></div>
        <p class="field-desc">例如 <code>1536</code>（text-embedding-3-small）。若改模型，请同步修改数据库迁移中的 <code>vector(维度)</code> 并重建或迁移表。</p>
        <input name="embedding_dim" type="number" placeholder="1536"/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">嵌入专用 API Key（可选）</span><span class="field-key">embedding_api_key</span></div>
        <p class="field-desc">留空则使用上方的 <strong>OpenRouter 上游密钥</strong> 调用嵌入接口。</p>
        <input type="password" name="embedding_api_key" autocomplete="off"/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">嵌入 API 根地址</span><span class="field-key">embedding_base_url</span></div>
        <p class="field-desc">一般为 <code>https://openrouter.ai/api/v1</code>，与上游一致即可。</p>
        <input type="text" name="embedding_base_url" placeholder="https://openrouter.ai/api/v1"/>
      </div>
    </div>

    <div class="card">
      <h2>HTTP 客户端与队列</h2>
      <p class="section-desc">控制 httpx 超时、连接池与内存中队列长度；高并发或慢代理时可按需调大。</p>

      <div class="field">
        <div class="field-head"><span class="field-title">请求总超时（秒）</span><span class="field-key">request_timeout_seconds</span></div>
        <p class="field-desc">单次上游请求（含流式拉流）的最长等待时间，长对话或慢网络可适当增大。</p>
        <input name="request_timeout_seconds" type="number" step="any" placeholder="600"/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">连接超时（秒）</span><span class="field-key">connect_timeout_seconds</span></div>
        <p class="field-desc">建立 TCP/TLS 连接的最长时间。</p>
        <input name="connect_timeout_seconds" type="number" step="any" placeholder="30"/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">最大连接数</span><span class="field-key">http_max_connections</span></div>
        <p class="field-desc">httpx 连接池上限，并发高时可略增。</p>
        <input name="http_max_connections" type="number" placeholder="100"/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">最大 Keep-Alive 连接</span><span class="field-key">http_max_keepalive</span></div>
        <p class="field-desc">池中保持空闲复用的连接数。</p>
        <input name="http_max_keepalive" type="number" placeholder="20"/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">持久化队列容量</span><span class="field-key">persist_queue_max</span></div>
        <p class="field-desc">异步写库队列；满时会丢弃新日志并打警告。</p>
        <input name="persist_queue_max" type="number" placeholder="10000"/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">嵌入队列容量</span><span class="field-key">embed_queue_max</span></div>
        <p class="field-desc">嵌入任务队列；满时丢弃嵌入任务。</p>
        <input name="embed_queue_max" type="number" placeholder="10000"/>
      </div>

      <div class="field">
        <div class="field-head"><span class="field-title">嵌入批大小（预留）</span><span class="field-key">embed_batch_size</span></div>
        <p class="field-desc">当前实现多为单条嵌入，本字段预留给后续批量优化。</p>
        <input name="embed_batch_size" type="number" placeholder="8"/>
      </div>
    </div>
  </form>

  <p class="foot-note">
    配置文件路径由环境变量 <code>CONFIG_FILE</code> 指定（例如数据盘 <code>/data/config.json</code>）；未设置时默认为项目下 <code>data/config.json</code>。
  </p>

  <script>
    const $ = (sel) => document.querySelector(sel);
    const msg = $("#msg");

    (function initOpenAiBaseUrl() {
      const origin = window.location.origin || (window.location.protocol + "//" + window.location.host);
      const root = origin.endsWith("/") ? origin.slice(0, -1) : origin;
      const base = root + "/v1";
      const el = document.getElementById("openaiBaseUrl");
      if (el) el.textContent = base;
    })();

    async function copyText(text, okMsg) {
      try {
        await navigator.clipboard.writeText(text);
        setMsg(okMsg || "已复制到剪贴板", "ok");
      } catch (e) {
        setMsg("复制失败，请手动选中复制。(" + (e && e.message) + ")", "err");
      }
    }

    $("#btnCopyBaseUrl").onclick = () => {
      const t = $("#openaiBaseUrl").textContent.trim();
      if (!t) return;
      copyText(t, "已复制 OpenAI API Base URL");
    };

    $("#btnCopyGateway").onclick = () => {
      const v = $("#gatewayApiKey").value.trim();
      if (!v) {
        setMsg("请先生成或保存 Gateway 密钥", "err");
        return;
      }
      copyText(v, "已复制 Gateway 密钥，可粘贴到 Cursor API Key");
    };

    function updateCursorSnippet() {
      const base = location.origin + "/v1";
      const key = $("#gatewayApiKey").value.trim() || "<your-gateway-api-key>";
      const model = (document.querySelector('[name="upstream_model"]')?.value || "").trim() || "anthropic/claude-opus-4.6";
      const snippet = JSON.stringify({
        "openai.com/v1": {
          "Override OpenAI Base URL": base,
          "OpenAI API Key": key,
          "Model": model
        }
      }, null, 2);
      $("#cursorSnippet").textContent = "Base URL: " + base + "\\nAPI Key:  " + key + "\\nModel:    " + model;
    }
    $("#gatewayApiKey").addEventListener("input", updateCursorSnippet);
    if (document.querySelector('[name="upstream_model"]')) {
      document.querySelector('[name="upstream_model"]').addEventListener("input", updateCursorSnippet);
    }
    setTimeout(updateCursorSnippet, 500);

    function toggleProviderFields() {
      const v = $("#upstreamProvider").value;
      document.getElementById("openrouterFields").style.display = v === "openrouter" ? "" : "none";
      document.getElementById("anthropicFields").style.display = v === "anthropic" ? "" : "none";
    }
    $("#upstreamProvider").addEventListener("change", toggleProviderFields);
    setTimeout(toggleProviderFields, 100);

    $("#btnCopyCursorSnippet").onclick = () => {
      const t = $("#cursorSnippet").textContent;
      copyText(t, "已复制 Cursor 配置信息");
    };

    function authHeaders() {
      const k = $("#adminKey").value.trim();
      if (!k) throw new Error("请先填写 Admin 密钥（ADMIN_KEY）");
      return { "Authorization": "Bearer " + k, "Content-Type": "application/json" };
    }

    function setMsg(text, cls) {
      msg.className = cls || "";
      msg.textContent = text || "";
    }

    function fillForm(s) {
      const form = $("#cfg");
      for (const el of form.elements) {
        if (!el.name) continue;
        const v = s[el.name];
        if (v === undefined || v === null) continue;
        if (el.type === "checkbox") el.checked = !!v;
        else el.value = v;
      }
    }

    function readForm() {
      const form = $("#cfg");
      const out = {};
      for (const el of form.elements) {
        if (!el.name) continue;
        if (el.type === "checkbox") {
          out[el.name] = el.checked;
        } else if (el.type === "number") {
          const t = el.value.trim();
          if (t === "") continue;
          out[el.name] = el.value.includes(".") ? parseFloat(el) : parseInt(el, 10);
        } else {
          const t = el.value.trim();
          if (t === "") continue;
          out[el.name] = t;
        }
      }
      return out;
    }

    async function fetchNewGatewayKey() {
      const r = await fetch("/admin/api/generate-gateway-key", {
        method: "POST",
        headers: authHeaders(),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText || "生成失败");
      return j.gateway_api_key;
    }

    $("#btnGenGateway").onclick = async () => {
      setMsg("生成中…", "");
      try {
        const key = await fetchNewGatewayKey();
        $("#gatewayApiKey").value = key;
        setMsg("已生成 Gateway 密钥，请点「保存全部」写入配置，再用「复制密钥」粘贴到 Cursor。", "ok");
      } catch (e) {
        setMsg(String(e.message || e), "err");
      }
    };

    $("#btnLoad").onclick = async () => {
      setMsg("加载中…", "");
      try {
        const r = await fetch("/admin/api/config", { headers: authHeaders() });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.detail || r.statusText || "加载失败");
        fillForm(j.settings || {});
        updateCursorSnippet();
        toggleProviderFields();
        setMsg("已加载。路径: " + (j.config_path || ""), "ok");
      } catch (e) {
        setMsg(String(e.message || e), "err");
      }
    };

    $("#btnSave").onclick = async () => {
      setMsg("保存中…", "");
      try {
        const body = readForm();
        const gw = $("#gatewayApiKey").value.trim();
        if (!gw) {
          const key = await fetchNewGatewayKey();
          body.gateway_api_key = key;
          $("#gatewayApiKey").value = key;
        }
        const r = await fetch("/admin/api/config", {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify(body),
        });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(
          typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail) || "保存失败"
        );
        setMsg(j.message || "已保存", "ok");
      } catch (e) {
        setMsg(String(e.message || e), "err");
      }
    };
  </script>
</body>
</html>
"""
