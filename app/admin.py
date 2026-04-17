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
        "message": "已保存并刷新 HTTP 客户端。若修改了 DATABASE_URL 或嵌入配置，建议重启进程。",
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
  <title>网关管理</title>
  <style>
    :root {
      /* 浅色卡其主题 */
      --bg: #f0e8d6;              /* 卡其米色背景 */
      --bg-soft: #f7f1e3;          /* 更浅的米色 */
      --surface: #ffffff;          /* 卡片白色 */
      --surface-alt: #faf6ea;      /* 次级面板 */
      --border: #d9ccae;           /* 卡其边框 */
      --border-strong: #b8a47d;
      --text: #2d2519;             /* 深棕 */
      --text-muted: #7a6e58;       /* 暖灰 */
      --accent: #6b5a3a;           /* 深卡其 (主色) */
      --accent-hover: #544629;
      --accent-soft: #e8dcc0;      /* 浅卡其背景 */
      --danger: #b04a3c;
      --ok: #5a7c4a;
      --code: #8a6a2e;
      --shadow: 0 1px 3px rgba(107, 90, 58, 0.08);
      --shadow-lg: 0 4px 16px rgba(107, 90, 58, 0.12);
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "PingFang SC", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 0; background: var(--bg); min-height: 100vh; }

    /* ========== 顶部 header ========== */
    .header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      box-shadow: var(--shadow);
      position: sticky; top: 0; z-index: 10;
    }
    .header-inner {
      max-width: 1100px; margin: 0 auto;
      padding: 14px 24px;
      display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
    }
    .brand {
      font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em;
      color: var(--accent);
      display: flex; align-items: center; gap: 8px;
    }
    .brand-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); }
    .header-sub { font-size: 0.82rem; color: var(--text-muted); margin-left: 4px; }

    /* tab 导航 */
    .tabs {
      display: flex; gap: 2px; margin-left: auto;
      background: var(--accent-soft);
      border-radius: 10px; padding: 4px;
    }
    .tab-btn {
      padding: 8px 16px; border: 0; background: transparent;
      color: var(--text-muted); font-weight: 600; font-size: 0.88rem;
      cursor: pointer; border-radius: 7px;
      transition: background 0.15s, color 0.15s;
    }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active {
      background: var(--surface); color: var(--accent);
      box-shadow: var(--shadow);
    }

    /* 工具栏 */
    .toolbar {
      max-width: 1100px; margin: 0 auto;
      padding: 14px 24px 8px;
      display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end;
    }
    .admin-key-field { flex: 1; min-width: 220px; max-width: 380px; }
    .admin-key-field label {
      display: block; font-size: 0.78rem; color: var(--text-muted); margin-bottom: 5px;
    }

    /* 容器 */
    .container {
      max-width: 1100px; margin: 0 auto;
      padding: 8px 24px 60px;
    }

    /* 卡片 */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px 24px;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
    }
    .card h2 {
      margin: 0 0 6px; font-size: 1rem; font-weight: 700;
      color: var(--accent);
      display: flex; align-items: center; gap: 8px;
    }
    .card .section-desc {
      margin: 0 0 18px; font-size: 0.82rem;
      color: var(--text-muted); line-height: 1.5;
    }
    .card-highlight {
      border-color: var(--accent);
      background: linear-gradient(180deg, var(--surface) 0%, var(--bg-soft) 100%);
    }

    /* 字段 */
    .field { margin-bottom: 18px; }
    .field:last-child { margin-bottom: 0; }
    .field-head {
      display: flex; flex-wrap: wrap; align-items: baseline;
      gap: 8px 12px; margin-bottom: 6px;
    }
    .field-title { font-size: 0.9rem; font-weight: 600; color: var(--text); }
    .field-key {
      font-family: ui-monospace, "Cascadia Code", Menlo, monospace;
      font-size: 0.72rem; color: var(--text-muted);
      background: var(--bg-soft); padding: 2px 8px;
      border-radius: 4px; border: 1px solid var(--border);
    }
    .field-desc {
      font-size: 0.8rem; color: var(--text-muted);
      line-height: 1.55; margin: 0 0 8px;
    }
    .field-desc code { background: var(--accent-soft); color: var(--code); padding: 1px 6px; border-radius: 4px; font-size: 0.85em; }

    /* 输入控件 */
    input[type="text"], input[type="password"], input[type="number"], textarea, select {
      width: 100%; padding: 10px 12px; border-radius: 8px;
      border: 1px solid var(--border); background: var(--surface); color: var(--text);
      font-size: 0.9rem; appearance: none; -webkit-appearance: none;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    select {
      cursor: pointer;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%237a6e58' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 12px center;
      padding-right: 36px;
    }
    input:focus, textarea:focus, select:focus {
      outline: none; border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(107, 90, 58, 0.15);
    }
    input::placeholder, textarea::placeholder { color: #b8a47d; }
    textarea {
      min-height: 72px; resize: vertical;
      font-family: ui-monospace, "Cascadia Code", Menlo, monospace;
      font-size: 0.82rem;
    }

    /* 勾选项 */
    .checks {
      display: flex; flex-direction: column; gap: 12px;
    }
    @media (min-width: 600px) {
      .checks { flex-direction: row; flex-wrap: wrap; }
    }
    .check-item {
      flex: 1; min-width: 220px;
      border: 1px solid var(--border); border-radius: 8px;
      padding: 12px 14px; background: var(--bg-soft);
      transition: border-color 0.15s, background 0.15s;
    }
    .check-item:hover { border-color: var(--border-strong); }
    .check-item label {
      display: flex; align-items: flex-start; gap: 10px;
      cursor: pointer; margin: 0; font-size: 0.88rem;
    }
    .check-item input { width: auto; margin-top: 3px; accent-color: var(--accent); cursor: pointer; }
    .check-item .t { font-weight: 600; color: var(--text); display: block; }
    .check-item .d { font-size: 0.78rem; color: var(--text-muted); margin-top: 4px; line-height: 1.45; }

    /* 按钮 */
    button {
      padding: 10px 18px; border-radius: 8px; border: 0; cursor: pointer;
      font-weight: 600; font-size: 0.88rem;
      transition: background 0.15s, transform 0.05s;
    }
    button:active { transform: scale(0.98); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .primary {
      background: var(--accent); color: #fff;
      box-shadow: var(--shadow);
    }
    .primary:hover { background: var(--accent-hover); }
    .btn-secondary {
      background: var(--bg-soft); color: var(--text);
      border: 1px solid var(--border);
      white-space: nowrap; flex-shrink: 0;
    }
    .btn-secondary:hover { background: var(--accent-soft); border-color: var(--border-strong); }

    /* 输入行 */
    .input-row { display: flex; gap: 10px; align-items: stretch; flex-wrap: wrap; }
    .input-row input { flex: 1; min-width: 0; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }

    /* 只读显示框 */
    .mono-readonly {
      flex: 1; min-width: 220px;
      font-family: ui-monospace, Menlo, monospace; font-size: 0.85rem;
      padding: 10px 12px; border-radius: 8px;
      border: 1px solid var(--border); background: var(--bg-soft);
      color: var(--accent); word-break: break-all;
    }

    /* Cursor 快速配置预览 */
    .snippet-box {
      background: var(--surface-alt);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 12px 14px; border-radius: 8px;
      font-size: 0.82rem; font-family: ui-monospace, Menlo, monospace;
      overflow-x: auto; white-space: pre-wrap; word-break: break-all;
      cursor: pointer;
    }

    /* Tab 内容 */
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    /* 状态消息 */
    #msg {
      margin-left: auto; min-height: 1.2em;
      font-size: 0.85rem; font-weight: 500;
    }
    .err { color: var(--danger); white-space: pre-wrap; }
    .ok { color: var(--ok); }

    /* 底注 */
    .foot-note {
      font-size: 0.78rem; color: var(--text-muted);
      margin-top: 24px; padding-top: 14px;
      border-top: 1px dashed var(--border);
      max-width: 1100px; margin-left: auto; margin-right: auto;
      padding-left: 24px; padding-right: 24px;
    }
    .foot-note code {
      color: var(--code); background: var(--accent-soft);
      padding: 1px 6px; border-radius: 4px; font-size: 0.85em;
    }

    /* 响应式 */
    @media (max-width: 640px) {
      .header-inner { padding: 12px 16px; gap: 10px; }
      .tabs { width: 100%; overflow-x: auto; }
      .toolbar, .container { padding-left: 16px; padding-right: 16px; }
      .card { padding: 18px; }
    }
  </style>
</head>
<body>
  <header class="header">
    <div class="header-inner">
      <div class="brand"><span class="brand-dot"></span>网关配置</div>
      <span class="header-sub">OpenAI 兼容 · 支持 Anthropic / OpenRouter / Gemini</span>
      <nav class="tabs" id="tabs">
        <button type="button" class="tab-btn active" data-tab="quick">快速开始</button>
        <button type="button" class="tab-btn" data-tab="upstream">上游供应商</button>
        <button type="button" class="tab-btn" data-tab="models">模型列表</button>
        <button type="button" class="tab-btn" data-tab="behavior">行为</button>
        <button type="button" class="tab-btn" data-tab="advanced">高级</button>
      </nav>
    </div>

    <div class="toolbar">
      <div class="admin-key-field">
        <label for="adminKey">管理密钥（ADMIN_KEY，仅登录本页）</label>
        <input type="password" id="adminKey" placeholder="与服务器 ADMIN_KEY 一致" autocomplete="off"/>
      </div>
      <button type="button" class="primary" id="btnLoad">加载</button>
      <button type="button" class="primary" id="btnSave">保存</button>
      <div id="msg"></div>
    </div>
  </header>

  <main class="container">
    <form id="cfg" onsubmit="return false;">

      <!-- ============ 快速开始 ============ -->
      <section class="tab-panel active" data-panel="quick">
        <div class="card card-highlight">
          <h2>Cursor / OpenAI 兼容客户端</h2>
          <p class="section-desc">网关对外提供 <code>/v1/chat/completions</code>。Base URL 自动从当前域名算出。</p>

          <div class="field">
            <div class="field-head">
              <span class="field-title">OpenAI API Base URL</span>
              <span class="field-key">填到 Cursor 的 Override OpenAI Base URL</span>
            </div>
            <p class="field-desc">须以 <code>/v1</code> 结尾。</p>
            <div class="input-row">
              <div class="mono-readonly" id="openaiBaseUrl"></div>
              <button type="button" class="btn-secondary" id="btnCopyBaseUrl">复制 URL</button>
            </div>
          </div>

          <div class="field">
            <div class="field-head">
              <span class="field-title">Gateway 密钥</span>
              <span class="field-key">gateway_api_key</span>
            </div>
            <p class="field-desc">填到 Cursor 的 <code>OpenAI API Key</code>。留空保存时自动生成。</p>
            <div class="input-row">
              <input type="password" name="gateway_api_key" id="gatewayApiKey" autocomplete="off" placeholder="点击随机生成"/>
              <div class="btn-row">
                <button type="button" class="btn-secondary" id="btnGenGateway">随机生成</button>
                <button type="button" class="btn-secondary" id="btnCopyGateway">复制密钥</button>
              </div>
            </div>
          </div>

          <div class="field">
            <div class="field-head"><span class="field-title">Cursor 快速配置</span></div>
            <div class="snippet-box" id="cursorSnippet">正在生成…</div>
            <div class="btn-row" style="margin-top:8px">
              <button type="button" class="btn-secondary" id="btnCopyCursorSnippet">复制配置</button>
            </div>
          </div>
        </div>
      </section>

      <!-- ============ 上游供应商 ============ -->
      <section class="tab-panel" data-panel="upstream">
        <div class="card">
          <h2>🔄 上游供应商</h2>
          <p class="section-desc">切换 API 协议。配置互相独立，切换后保留原值。</p>

          <div class="field">
            <div class="field-head"><span class="field-title">供应商协议</span><span class="field-key">upstream_provider</span></div>
            <select name="upstream_provider" id="upstreamProvider">
              <option value="openrouter">OpenRouter (OpenAI 兼容)</option>
              <option value="gemini">Gemini (OpenAI 兼容)</option>
              <option value="anthropic">Anthropic 原生直连</option>
            </select>
          </div>

          <div id="sharedUpstreamModelField">
            <div class="field">
              <div class="field-head"><span class="field-title">主模型 ID</span><span class="field-key">upstream_model</span></div>
              <p class="field-desc">OpenRouter / Gemini 线路共用；Anthropic 请用下方专用字段。</p>
              <input type="text" name="upstream_model" id="upstreamModelInput" placeholder="anthropic/claude-opus-4.6"/>
            </div>
          </div>

          <div id="openrouterFields">
            <div class="field">
              <div class="field-head"><span class="field-title">OpenRouter API Key</span><span class="field-key">openrouter_api_key</span></div>
              <input type="password" name="openrouter_api_key" autocomplete="off" placeholder="sk-or-v1-..."/>
            </div>
            <div class="field">
              <div class="field-head"><span class="field-title">API 根地址</span><span class="field-key">upstream_base_url</span></div>
              <input type="text" name="upstream_base_url" placeholder="https://openrouter.ai/api/v1"/>
            </div>
            <div class="field">
              <div class="field-head"><span class="field-title">HTTP-Referer（可选）</span><span class="field-key">openrouter_http_referer</span></div>
              <input type="text" name="openrouter_http_referer" placeholder="https://github.com/you/repo"/>
            </div>
            <div class="field">
              <div class="field-head"><span class="field-title">X-Title（可选）</span><span class="field-key">openrouter_app_title</span></div>
              <input type="text" name="openrouter_app_title" placeholder="OpenRouter Gateway"/>
            </div>
          </div>

          <div id="geminiFields" style="display:none">
            <div class="field">
              <div class="field-head"><span class="field-title">Google AI API Key</span><span class="field-key">google_api_key</span></div>
              <input type="password" name="google_api_key" autocomplete="off" placeholder="AIza…"/>
            </div>
            <div class="field">
              <div class="field-head"><span class="field-title">OpenAI 兼容 Base URL</span><span class="field-key">gemini_openai_base_url</span></div>
              <input type="text" name="gemini_openai_base_url" placeholder="https://generativelanguage.googleapis.com/v1beta/openai"/>
            </div>
          </div>

          <div id="anthropicFields" style="display:none">
            <div class="field">
              <div class="field-head"><span class="field-title">Anthropic API Key</span><span class="field-key">anthropic_api_key</span></div>
              <input type="password" name="anthropic_api_key" autocomplete="off" placeholder="sk-ant-api03-..."/>
            </div>
            <div class="field">
              <div class="field-head"><span class="field-title">模型 ID</span><span class="field-key">anthropic_model</span></div>
              <p class="field-desc">上游实际请求的模型名，例如 <code>claude-opus-4-20250514</code> 或 AWS Bedrock 的 inference profile。</p>
              <input type="text" name="anthropic_model" placeholder="claude-opus-4-20250514"/>
            </div>
            <div class="field">
              <div class="field-head"><span class="field-title">Base URL</span><span class="field-key">anthropic_base_url</span></div>
              <input type="text" name="anthropic_base_url" placeholder="https://api.anthropic.com"/>
            </div>
            <div class="field">
              <div class="field-head"><span class="field-title">API Version</span><span class="field-key">anthropic_version</span></div>
              <input type="text" name="anthropic_version" placeholder="2023-06-01"/>
            </div>
            <div class="checks" style="margin-top:12px">
              <div class="check-item">
                <label><input type="checkbox" name="thinking_enabled" checked/>
                  <span><span class="t">扩展思考</span><span class="field-key" style="display:inline;margin-left:6px">thinking_enabled</span>
                  <span class="d">模型回答前深度推理，显著提升代码理解和架构决策质量。</span></span>
                </label>
              </div>
            </div>
            <div class="field" style="margin-top:10px">
              <div class="field-head"><span class="field-title">思考预算 (tokens)</span><span class="field-key">thinking_budget_tokens</span></div>
              <input type="number" name="thinking_budget_tokens" placeholder="10000"/>
            </div>
          </div>
        </div>
      </section>

      <!-- ============ 模型列表 ============ -->
      <section class="tab-panel" data-panel="models">
        <div class="card">
          <h2>📋 对外广告模型名</h2>
          <p class="section-desc">在 <code>/v1/models</code> 里对 Cursor/分发层暴露的模型名列表。Cursor 会根据这个列表显示可选模型。留空则只显示上游主模型。</p>

          <div class="field">
            <div class="field-head">
              <span class="field-title">广告模型名（多个，逗号分隔）</span>
              <span class="field-key">advertised_models</span>
            </div>
            <p class="field-desc">
              例如 <code>claude-opus-4-7,claude-sonnet-4-5,claude-opus-4-6</code>。
              Cursor 发来任何一个名字的请求都会路由到上游配置的实际模型。
              <strong>含 "opus" 的名字自动标记 1M 上下文</strong>。
            </p>
            <textarea name="advertised_models" rows="2" placeholder="claude-opus-4-7, claude-sonnet-4-5, claude-opus-4-6"></textarea>
          </div>

          <div class="field">
            <div class="field-head"><span class="field-title">当前 /v1/models 返回</span></div>
            <p class="field-desc">下列是保存配置后 <code>GET /v1/models</code> 实际返回的模型 ID：</p>
            <div class="snippet-box" id="modelsPreview">保存后刷新查看</div>
          </div>
        </div>

        <div class="card">
          <h2>🧠 双模型协同</h2>
          <p class="section-desc">Opus 规划 + Qwen 执行，节省约 50% 成本。仅对有 tools 的流式请求生效。</p>
          <div class="checks">
            <div class="check-item">
              <label><input type="checkbox" name="dual_model_enabled"/>
                <span><span class="t">启用双模型协同</span><span class="field-key" style="display:inline;margin-left:6px">dual_model_enabled</span>
                <span class="d">开启后 Opus 规划 + Qwen 执行；关闭则全程使用上游模型。</span></span>
              </label>
            </div>
          </div>
          <div class="field" style="margin-top:14px">
            <div class="field-head"><span class="field-title">对外模型名称</span><span class="field-key">dual_model_name</span></div>
            <input type="text" name="dual_model_name" placeholder="opus-qwen-hybrid"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">规划模型</span><span class="field-key">planner_model</span></div>
            <input type="text" name="planner_model" placeholder="anthropic/claude-opus-4.6"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">执行模型</span><span class="field-key">executor_model</span></div>
            <input type="text" name="executor_model" placeholder="qwen/qwen3.6-plus"/>
          </div>
        </div>
      </section>

      <!-- ============ 行为 ============ -->
      <section class="tab-panel" data-panel="behavior">
        <div class="card">
          <h2>⚙️ 请求行为</h2>
          <p class="section-desc">请求清洗、日志、prompt 注入等开关。</p>

          <div class="checks">
            <div class="check-item">
              <label><input type="checkbox" name="loose_tools_passthrough"/>
                <span><span class="t">宽松工具透传</span><span class="field-key" style="display:inline;margin-left:6px">loose_tools_passthrough</span>
                <span class="d">保留非标准 tools 项；上游报错时可关闭排查。</span></span>
              </label>
            </div>
            <div class="check-item">
              <label><input type="checkbox" name="log_chat_metadata"/>
                <span><span class="t">记录对话元数据</span><span class="field-key" style="display:inline;margin-left:6px">log_chat_metadata</span>
                <span class="d">每条 chat 一行 INFO，不含消息正文。</span></span>
              </label>
            </div>
          </div>

          <div class="checks" style="margin-top:12px">
            <div class="check-item">
              <label><input type="checkbox" name="identity_prompt_enabled"/>
                <span><span class="t">注入身份说明</span><span class="field-key" style="display:inline;margin-left:6px">identity_prompt_enabled</span>
                <span class="d">仅 OpenRouter 路径生效。Anthropic 路径从不注入。</span></span>
              </label>
            </div>
            <div class="check-item">
              <label><input type="checkbox" name="efficiency_prompt_enabled"/>
                <span><span class="t">注入效率指令</span><span class="field-key" style="display:inline;margin-left:6px">efficiency_prompt_enabled</span>
                <span class="d">仅 OpenRouter 路径生效。Anthropic 路径零注入。</span></span>
              </label>
            </div>
          </div>

          <div class="field" style="margin-top:16px">
            <div class="field-head"><span class="field-title">身份说明正文</span><span class="field-key">identity_prompt</span></div>
            <textarea name="identity_prompt" rows="3" placeholder="我是 Claude Opus 4.6…"></textarea>
          </div>
        </div>

        <div class="card">
          <h2>🗃️ 历史压缩 / 缓存</h2>
          <p class="section-desc">长对话的历史裁剪、Anthropic prompt caching 开关。</p>

          <div class="checks">
            <div class="check-item">
              <label><input type="checkbox" name="tool_result_truncate_enabled" checked/>
                <span><span class="t">截断旧 tool result</span><span class="field-key" style="display:inline;margin-left:6px">tool_result_truncate_enabled</span>
                <span class="d">超过保留轮数的 tool 输出按下方字符数截断。</span></span>
              </label>
            </div>
            <div class="check-item">
              <label><input type="checkbox" name="cache_enabled"/>
                <span><span class="t">启用缓存注入</span><span class="field-key" style="display:inline;margin-left:6px">cache_enabled</span>
                <span class="d">请求中附加 <code>cache_control</code>。</span></span>
              </label>
            </div>
            <div class="check-item">
              <label><input type="checkbox" name="cache_ttl_1h"/>
                <span><span class="t">1h 缓存 TTL</span><span class="field-key" style="display:inline;margin-left:6px">cache_ttl_1h</span>
                <span class="d">开启约 1 小时；关闭约 5 分钟。</span></span>
              </label>
            </div>
          </div>

          <div class="field" style="margin-top:14px">
            <div class="field-head"><span class="field-title">保留最近 N 轮完整</span><span class="field-key">tool_result_keep_recent_turns</span></div>
            <input name="tool_result_keep_recent_turns" type="number" placeholder="2"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">旧 tool result 截断字符</span><span class="field-key">tool_result_max_chars</span></div>
            <input name="tool_result_max_chars" type="number" placeholder="4000"/>
          </div>
        </div>
      </section>

      <!-- ============ 高级 ============ -->
      <section class="tab-panel" data-panel="advanced">
        <div class="card">
          <h2>🌐 网络代理</h2>
          <div class="field">
            <div class="field-head"><span class="field-title">HTTPS / SOCKS 代理</span><span class="field-key">https_proxy</span></div>
            <p class="field-desc">示例：<code>socks5://user:pass@host:port</code>。留空直连。</p>
            <input type="text" name="https_proxy" placeholder="socks5://user:pass@host:port"/>
          </div>
        </div>

        <div class="card">
          <h2>🐘 PostgreSQL</h2>
          <div class="field">
            <div class="field-head"><span class="field-title">数据库连接串</span><span class="field-key">database_url</span></div>
            <p class="field-desc">留空则不落库。修改后<strong>重启进程</strong>生效。</p>
            <textarea name="database_url" rows="2" placeholder="postgresql://..."></textarea>
          </div>
        </div>

        <div class="card">
          <h2>📦 嵌入（向量）</h2>
          <div class="checks">
            <div class="check-item">
              <label><input type="checkbox" name="embedding_use_pgvector" checked/>
                <span><span class="t">使用 pgvector 列</span><span class="field-key" style="display:inline;margin-left:6px">embedding_use_pgvector</span>
                <span class="d">关闭时对应 <code>double precision[]</code> 列。</span></span>
              </label>
            </div>
          </div>
          <div class="field" style="margin-top:14px">
            <div class="field-head"><span class="field-title">嵌入模型</span><span class="field-key">embedding_model</span></div>
            <input type="text" name="embedding_model" placeholder="openai/text-embedding-3-small"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">向量维度</span><span class="field-key">embedding_dim</span></div>
            <input name="embedding_dim" type="number" placeholder="1536"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">嵌入专用 API Key</span><span class="field-key">embedding_api_key</span></div>
            <input type="password" name="embedding_api_key" autocomplete="off"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">嵌入 API 根地址</span><span class="field-key">embedding_base_url</span></div>
            <input type="text" name="embedding_base_url" placeholder="https://openrouter.ai/api/v1"/>
          </div>
        </div>

        <div class="card">
          <h2>⏱️ HTTP 客户端</h2>
          <div class="field">
            <div class="field-head"><span class="field-title">请求总超时（秒）</span><span class="field-key">request_timeout_seconds</span></div>
            <input name="request_timeout_seconds" type="number" step="any" placeholder="600"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">连接超时（秒）</span><span class="field-key">connect_timeout_seconds</span></div>
            <input name="connect_timeout_seconds" type="number" step="any" placeholder="30"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">流 idle 超时（秒）</span><span class="field-key">stream_idle_timeout_seconds</span></div>
            <input name="stream_idle_timeout_seconds" type="number" step="any" placeholder="300"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">最大连接数</span><span class="field-key">http_max_connections</span></div>
            <input name="http_max_connections" type="number" placeholder="200"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">最大 Keep-Alive 连接</span><span class="field-key">http_max_keepalive</span></div>
            <input name="http_max_keepalive" type="number" placeholder="50"/>
          </div>
        </div>

        <div class="card">
          <h2>📊 队列</h2>
          <div class="field">
            <div class="field-head"><span class="field-title">持久化队列容量</span><span class="field-key">persist_queue_max</span></div>
            <input name="persist_queue_max" type="number" placeholder="10000"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">嵌入队列容量</span><span class="field-key">embed_queue_max</span></div>
            <input name="embed_queue_max" type="number" placeholder="10000"/>
          </div>
          <div class="field">
            <div class="field-head"><span class="field-title">嵌入批大小</span><span class="field-key">embed_batch_size</span></div>
            <input name="embed_batch_size" type="number" placeholder="8"/>
          </div>
        </div>
      </section>

    </form>

    <p class="foot-note">
      配置由环境变量 <code>CONFIG_FILE</code> 指定（默认 <code>data/config.json</code>）；
      <code>ADMIN_KEY</code> 仅存于环境变量，不写入配置文件。
    </p>
  </main>

  <script>
    const $ = (sel) => document.querySelector(sel);
    const msg = $("#msg");

    // ========== 初始化 Base URL ==========
    (function initOpenAiBaseUrl() {
      const origin = window.location.origin;
      const base = (origin.replace(/\\/$/, "")) + "/v1";
      $("#openaiBaseUrl").textContent = base;
    })();

    // ========== Tab 切换 ==========
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        const name = btn.dataset.tab;
        const panel = document.querySelector(`.tab-panel[data-panel="${name}"]`);
        if (panel) panel.classList.add("active");
      };
    });

    // ========== 复制工具 ==========
    async function copyText(text, okMsg) {
      try {
        await navigator.clipboard.writeText(text);
        setMsg(okMsg || "已复制", "ok");
      } catch (e) {
        setMsg("复制失败：" + (e && e.message), "err");
      }
    }

    $("#btnCopyBaseUrl").onclick = () => copyText($("#openaiBaseUrl").textContent.trim(), "已复制 Base URL");
    $("#btnCopyGateway").onclick = () => {
      const v = $("#gatewayApiKey").value.trim();
      if (!v) { setMsg("请先生成或保存 Gateway 密钥", "err"); return; }
      copyText(v, "已复制 Gateway 密钥");
    };

    // ========== Cursor 配置预览 ==========
    function updateCursorSnippet() {
      const base = location.origin + "/v1";
      const key = $("#gatewayApiKey").value.trim() || "<your-gateway-api-key>";
      const prov = $("#upstreamProvider")?.value || "openrouter";
      let model = "";
      if (prov === "anthropic") {
        model = (document.querySelector('[name="anthropic_model"]')?.value || "").trim();
      } else {
        model = ($("#upstreamModelInput")?.value || "").trim();
      }
      // advertised_models 优先显示
      const adv = (document.querySelector('[name="advertised_models"]')?.value || "").trim();
      if (adv) {
        const first = adv.split(/[,\\n;]/)[0].trim();
        if (first) model = first;
      }
      if (!model) model = prov === "anthropic" ? "claude-opus-4-20250514" : "anthropic/claude-opus-4.6";
      $("#cursorSnippet").textContent =
        "Base URL: " + base + "\\n" +
        "API Key:  " + key + "\\n" +
        "Model:    " + model;
    }

    $("#gatewayApiKey").addEventListener("input", updateCursorSnippet);
    document.querySelector('[name="advertised_models"]').addEventListener("input", updateCursorSnippet);
    const _um = $("#upstreamModelInput");
    if (_um) _um.addEventListener("input", updateCursorSnippet);
    const _am = document.querySelector('[name="anthropic_model"]');
    if (_am) _am.addEventListener("input", updateCursorSnippet);
    $("#upstreamProvider").addEventListener("change", () => { updateCursorSnippet(); toggleProviderFields(); });
    setTimeout(updateCursorSnippet, 300);

    $("#btnCopyCursorSnippet").onclick = () => copyText($("#cursorSnippet").textContent, "已复制 Cursor 配置");

    // ========== 供应商字段切换 ==========
    function setSectionDisabled(el, disabled) {
      if (!el) return;
      for (const x of el.querySelectorAll("input,select,textarea,button")) {
        x.disabled = !!disabled;
      }
    }
    function toggleProviderFields() {
      const v = $("#upstreamProvider").value;
      const openrouter = document.getElementById("openrouterFields");
      const gemini = document.getElementById("geminiFields");
      const anthropic = document.getElementById("anthropicFields");
      const shared = document.getElementById("sharedUpstreamModelField");
      openrouter.style.display = v === "openrouter" ? "" : "none";
      gemini.style.display = v === "gemini" ? "" : "none";
      anthropic.style.display = v === "anthropic" ? "" : "none";
      if (shared) {
        shared.style.display = (v === "openrouter" || v === "gemini") ? "" : "none";
        setSectionDisabled(shared, v !== "openrouter" && v !== "gemini");
      }
      setSectionDisabled(openrouter, v !== "openrouter");
      setSectionDisabled(gemini, v !== "gemini");
      setSectionDisabled(anthropic, v !== "anthropic");
    }
    setTimeout(toggleProviderFields, 100);

    // ========== 认证 ==========
    function authHeaders() {
      const k = $("#adminKey").value.trim();
      if (!k) throw new Error("请先填写 Admin 密钥（ADMIN_KEY）");
      return { "Authorization": "Bearer " + k, "Content-Type": "application/json" };
    }

    function setMsg(text, cls) {
      msg.className = cls || "";
      msg.textContent = text || "";
    }

    // ========== 表单读写 ==========
    function fillForm(s) {
      const form = $("#cfg");
      for (const el of form.elements) {
        if (el.name) el.disabled = false;
      }
      for (const el of form.elements) {
        if (!el.name) continue;
        const v = s[el.name];
        if (v === undefined || v === null) continue;
        if (el.type === "checkbox") el.checked = !!v;
        else el.value = v;
      }
      toggleProviderFields();
    }

    function readForm() {
      const form = $("#cfg");
      const out = {};
      for (const el of form.elements) {
        if (!el.name) continue;
        if (el.disabled) continue;
        if (el.type === "checkbox") {
          out[el.name] = el.checked;
        } else if (el.type === "number") {
          const t = el.value.trim();
          if (t === "") continue;
          out[el.name] = t.includes(".") ? parseFloat(t) : parseInt(t, 10);
        } else {
          const t = el.value.trim();
          if (t === "") continue;
          out[el.name] = t;
        }
      }
      return out;
    }

    async function fetchNewGatewayKey() {
      const r = await fetch("/admin/api/generate-gateway-key", { method: "POST", headers: authHeaders() });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText || "生成失败");
      return j.gateway_api_key;
    }

    $("#btnGenGateway").onclick = async () => {
      setMsg("生成中…", "");
      try {
        const key = await fetchNewGatewayKey();
        $("#gatewayApiKey").value = key;
        updateCursorSnippet();
        setMsg("已生成。请保存后粘贴到 Cursor。", "ok");
      } catch (e) {
        setMsg(String(e.message || e), "err");
      }
    };

    async function fetchModelsPreview() {
      try {
        const gw = $("#gatewayApiKey").value.trim();
        if (!gw) { $("#modelsPreview").textContent = "（先配置 gateway_api_key 并保存）"; return; }
        const r = await fetch("/v1/models", { headers: { "Authorization": "Bearer " + gw } });
        const j = await r.json().catch(() => ({}));
        const ids = (j.data || []).map(m => m.id);
        $("#modelsPreview").textContent = ids.length ? ids.join("\\n") : "（暂无）";
      } catch (e) {
        $("#modelsPreview").textContent = "加载失败：" + (e && e.message || e);
      }
    }

    $("#btnLoad").onclick = async () => {
      setMsg("加载中…", "");
      try {
        const r = await fetch("/admin/api/config", { headers: authHeaders() });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.detail || r.statusText || "加载失败");
        fillForm(j.settings || {});
        updateCursorSnippet();
        fetchModelsPreview();
        setMsg("已加载：" + (j.config_path || ""), "ok");
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
        fetchModelsPreview();
      } catch (e) {
        setMsg(String(e.message || e), "err");
      }
    };
  </script>
</body>
</html>
"""
