# 部署说明

## 环境变量

复制 [.env.example](.env.example) 为 `.env` 并按环境填写。正常转发请求前需要有效的 `OPENROUTER_API_KEY` 与 `GATEWAY_API_KEY`；若尚未写入环境变量，可在服务器上设置 `ADMIN_KEY` 后打开 `/admin/` 将密钥与连接信息保存到 `CONFIG_FILE`（默认 `data/config.json`）。

### 管理面板

- 设置环境变量 **`ADMIN_KEY`** 后，浏览器打开 **`/admin/`**（注意尾部斜杠）。
- 在页面输入与 `ADMIN_KEY` 相同的密钥，点击「加载配置」「保存」。
- 配置写入 **`CONFIG_FILE`**（默认 `data/config.json`），与 `.env` 合并；**`ADMIN_KEY` 仅来自环境变量**，不会写入文件。
- 保存后会自动重建出站 **HTTP 客户端**（例如更新 `HTTPS_PROXY` 立即生效）。若修改 **`DATABASE_URL`** 或嵌入相关项，**需要重启进程** 才能让后台持久化/嵌入任务使用新库与新参数。

## 数据库

1. 安装 PostgreSQL，并安装扩展 **pgvector**。
2. 执行迁移：

```bash
psql "$DATABASE_URL" -f migrations/001_init.sql
```

若更换嵌入模型维度，请同步修改 `migrations/001_init.sql` 中的 `vector(1536)` 与 `.env` 中的 `EMBEDDING_DIM`。

## 运行

单机开发：

```bash
pip install -r requirements.txt
set OPENROUTER_API_KEY=...
set GATEWAY_API_KEY=...
set DATABASE_URL=postgresql://...
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

生产建议多 worker（按 CPU 调整）：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

前置 **Nginx** 终止 TLS，将 `proxy_read_timeout` 设得足够大（长对话、流式响应）。

## Cursor 接入

- **Base URL**：`https://你的域名/v1`（注意末尾 `/v1`）。
- **API Key**：填 `GATEWAY_API_KEY`。
- **Model**：选网关列出的模型 ID（默认与 `UPSTREAM_MODEL` 一致，如 `anthropic/claude-opus-4.6`）。
- **Session**：建议客户端在请求头携带 `X-Session-Id`（网关也会在响应头回写）；未携带时会自动生成 UUID。

## SOCKS5 住宅代理

将 `HTTPS_PROXY`（或系统级 `ALL_PROXY`）设为 SOCKS5 URI，例如：

`socks5://user:pass@proxy.example.com:1080`

出站 `httpx` 会使用该代理访问 OpenRouter。住宅带宽常是瓶颈，请监控延迟与超时，必要时降低并发或在 Nginx 侧限流。

## 训练数据导出

```bash
set DATABASE_URL=postgresql://...
python scripts/export_training_data.py --out data/train.jsonl
python scripts/export_training_data.py --out data/train.parquet --format parquet
```

## 运维注意

- 上游固定 `provider.only: ["anthropic"]` 且注入 `cache_control`；不要用 `provider.order`，否则会干扰 OpenRouter 的 sticky routing 行为。
- 短对话可能因最小可缓存 token 门槛而几乎无缓存收益，属预期。
- 若 `DATABASE_URL` 未设置，网关仍可转发对话，但不会落库与写向量。
