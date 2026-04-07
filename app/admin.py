from __future__ import annotations

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
        "message": "已保存并刷新 HTTP 客户端。若修改了 DATABASE_URL 或嵌入相关项，建议重启进程以应用后台任务。",
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
    :root { font-family: system-ui, sans-serif; background:#0f1419; color:#e6edf3; }
    body { max-width: 720px; margin: 24px auto; padding: 0 16px; }
    h1 { font-size: 1.25rem; font-weight: 600; }
    label { display:block; margin-top:12px; font-size:0.8rem; color:#8b949e; }
    input, textarea { width:100%; box-sizing:border-box; padding:8px 10px; border-radius:6px;
      border:1px solid #30363d; background:#161b22; color:#e6edf3; }
    textarea { min-height: 64px; font-family: ui-monospace, monospace; font-size: 0.85rem; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:8px; }
    button { padding:8px 16px; border-radius:6px; border:0; cursor:pointer; font-weight:500; }
    .primary { background:#238636; color:#fff; }
    .ghost { background:#21262d; color:#e6edf3; border:1px solid #30363d; }
    .hint { font-size:0.8rem; color:#8b949e; margin-top:8px; line-height:1.4; }
    .err { color:#f85149; margin-top:8px; white-space:pre-wrap; }
    .ok { color:#3fb950; margin-top:8px; }
    hr { border:0; border-top:1px solid #30363d; margin:20px 0; }
  </style>
</head>
<body>
  <h1>OpenRouter 网关 · 配置</h1>
  <p class="hint">在环境变量中设置 <code>ADMIN_KEY</code> 后，在下方输入同一密钥进行加载与保存。密钥不会写入配置文件。</p>

  <label>Admin 密钥（ADMIN_KEY）</label>
  <input type="password" id="adminKey" placeholder="与服务器环境变量 ADMIN_KEY 相同" autocomplete="off"/>

  <div class="row" style="margin-top:12px">
    <button type="button" class="primary" id="btnLoad">加载配置</button>
    <button type="button" class="primary" id="btnSave">保存</button>
  </div>
  <div id="msg"></div>

  <hr/>
  <form id="cfg" onsubmit="return false;">
    <label>OPENROUTER_API_KEY</label>
    <input type="password" name="openrouter_api_key" autocomplete="off"/>

    <label>GATEWAY_API_KEY（Cursor 使用）</label>
    <input type="password" name="gateway_api_key" autocomplete="off"/>

    <label>UPSTREAM_MODEL</label>
    <input name="upstream_model" />

    <label>UPSTREAM_BASE_URL</label>
    <input name="upstream_base_url" />

    <label>HTTPS_PROXY（SOCKS5 等）</label>
    <input name="https_proxy" placeholder="socks5://user:pass@host:port" />

    <label>DATABASE_URL</label>
    <textarea name="database_url" rows="2" placeholder="postgresql://..."></textarea>

    <div class="row">
      <label><input type="checkbox" name="cache_enabled"/> CACHE_ENABLED</label>
      <label><input type="checkbox" name="cache_ttl_1h"/> CACHE_TTL_1H（1 小时缓存）</label>
    </div>

    <label>EMBEDDING_MODEL</label>
    <input name="embedding_model" />

    <label>EMBEDDING_DIM</label>
    <input name="embedding_dim" type="number" />

    <label>EMBEDDING_API_KEY（留空则沿用 OpenRouter Key）</label>
    <input type="password" name="embedding_api_key" autocomplete="off"/>

    <label>EMBEDDING_BASE_URL</label>
    <input name="embedding_base_url" />

    <label>REQUEST_TIMEOUT_SECONDS</label>
    <input name="request_timeout_seconds" type="number" step="any"/>

    <label>CONNECT_TIMEOUT_SECONDS</label>
    <input name="connect_timeout_seconds" type="number" step="any"/>

    <label>HTTP_MAX_CONNECTIONS</label>
    <input name="http_max_connections" type="number"/>

    <label>HTTP_MAX_KEEPALIVE</label>
    <input name="http_max_keepalive" type="number"/>

    <label>PERSIST_QUEUE_MAX</label>
    <input name="persist_queue_max" type="number"/>

    <label>EMBED_QUEUE_MAX</label>
    <input name="embed_queue_max" type="number"/>

    <label>EMBED_BATCH_SIZE</label>
    <input name="embed_batch_size" type="number"/>
  </form>

  <p class="hint">配置文件路径由环境变量 <code>CONFIG_FILE</code> 决定，默认 <code>data/config.json</code>（已加入 .gitignore）。</p>

  <script>
    const $ = (sel) => document.querySelector(sel);
    const msg = $("#msg");

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

    $("#btnLoad").onclick = async () => {
      setMsg("加载中…", "");
      try {
        const r = await fetch("/admin/api/config", { headers: authHeaders() });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.detail || r.statusText || "加载失败");
        fillForm(j.settings || {});
        setMsg("已加载。路径: " + (j.config_path || ""), "ok");
      } catch (e) {
        setMsg(String(e.message || e), "err");
      }
    };

    $("#btnSave").onclick = async () => {
      setMsg("保存中…", "");
      try {
        const body = readForm();
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
