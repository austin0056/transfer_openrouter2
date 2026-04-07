# 部署说明

## 环境变量

**推荐**：在 Zeabur / 平台里**只配置 `ADMIN_KEY`**（强随机字符串），其它 **OpenRouter Key、Gateway Key、代理、数据库、模型等**一律在 **[ENV.md](ENV.md)** 所述的 **`/admin/` 管理面板**中填写并保存，无需在控制台再配一长串变量。

复制 [.env.example](.env.example) 可得到最小模板；详细说明见 [ENV.md](ENV.md)。

### 管理面板

- 设置环境变量 **`ADMIN_KEY`** 后，浏览器打开 **`/admin/`**（注意尾部斜杠）。
- 在页面输入与 `ADMIN_KEY` 相同的密钥，在表单中填写 **上游 Key、对外 Key、代理、数据库** 等，点击「保存」。
- 配置写入 **`CONFIG_FILE`**（默认 `data/config.json`），与 `.env` / 平台环境变量合并；**`ADMIN_KEY` 仅来自环境变量**，不会写入文件。
- 保存后会自动重建出站 **HTTP 客户端**（例如更新 `HTTPS_PROXY` 立即生效）。若修改 **`DATABASE_URL`** 或嵌入相关项，**需要重启进程** 才能让后台持久化/嵌入任务使用新库与新参数。

## 数据库

1. 安装 PostgreSQL，并安装扩展 **pgvector**。
2. 执行迁移：

```bash
psql "$DATABASE_URL" -f migrations/001_init.sql
```

若更换嵌入模型维度，请同步修改 `migrations/001_init.sql` 中的 `vector(1536)` 与 `.env` 中的 `EMBEDDING_DIM`。

## Zeabur（容器反复 BackOff / Restarting）

常见原因：

1. **监听端口与平台不一致**  
   Zeabur 会注入环境变量 **`PORT`**（常见为 `8080`）。若启动命令写死 `--port 8000`，健康检查访问容器端口会失败，Pod 会一直重启。  
   **请使用** 启动命令：  
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`（Linux）  
   或本地/兼容方式：`python run.py`（[run.py](run.py) 仅作便捷入口，**未推送 run.py 时勿用**）。  
   仓库内 [Dockerfile](Dockerfile) 使用 **uvicorn + `$PORT`**，不依赖 `run.py`。[Procfile](Procfile) 同理。

2. **数据库连不上**  
   若配置了 **`DATABASE_URL`** 但连接串错误、库未就绪或网络未打通，旧版本会在启动时直接退出。当前版本会在连接失败时**记录错误并继续启动**（不落库），修复 `DATABASE_URL` 后**重启服务**即可恢复持久化。

在 Zeabur 服务设置里确认：**Start Command** 为空（使用 Dockerfile 默认）或显式为：  
`uvicorn app.main:app --host 0.0.0.0 --port $PORT`  
**不要**再写 `python run.py`（除非仓库里已包含 [run.py](run.py) 且路径正确）。

若仍 **BackOff**，请到 Zeabur **Runtime Logs / 容器日志** 查看第一段 **Python Traceback**（常见：`ModuleNotFoundError` 说明构建/根目录不对；`Address already in use` 说明端口冲突）。仓库根目录已提供 **[Procfile](Procfile)**，部分平台会自动识别。

## 运行

单机开发：

```bash
pip install -r requirements.txt
set ADMIN_KEY=你的管理密钥
# 可选：也可在此设置 OPENROUTER_API_KEY / GATEWAY_API_KEY；否则启动后在 /admin/ 配置
python run.py
# 或: uvicorn app.main:app --host 0.0.0.0 --port 8000
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

在 **`/admin/`** 的 **HTTPS_PROXY** 字段填写 SOCKS5 URI（或写在 `CONFIG_FILE` / 环境变量），例如：

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
