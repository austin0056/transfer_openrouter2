# 部署说明

## 环境变量

复制 [.env.example](.env.example) 为 `.env` 并按环境填写。网关进程**必须**配置 `OPENROUTER_API_KEY` 与 `GATEWAY_API_KEY`。

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
