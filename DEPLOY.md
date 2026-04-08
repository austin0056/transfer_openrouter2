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

1. 安装 PostgreSQL；若需向量相似检索，安装扩展 **pgvector** 并执行 `migrations/001_init.sql`。
2. **托管库若不支持 `CREATE EXTENSION vector`**（报 extension "vector" is not available）：在 SQL 控制台执行 **`migrations/001_init_no_pgvector.sql`**（向量列为 `double precision[]`），并在 **`/admin/` 取消勾选**「使用 pgvector 列类型」，或设置环境变量 **`EMBEDDING_USE_PGVECTOR=false`**，然后重启服务。

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

### OpenAI 兼容面（当前网关实现）

| 路径 | 说明 |
|------|------|
| `POST /v1/chat/completions` | 支持流式/非流式、`tools`、`tool_choice`、`stream_options`、`messages` 多模态 list 等；合并后转发 OpenRouter。 |
| `GET /v1/models` | 返回配置中的 `upstream_model` 一项。 |
| `GET /health`、`HEAD /v1` | 探活。 |
| `POST /v1/embeddings` | **未实现**；若 Cursor 将同一 Base URL 用于嵌入会 404（本仓库嵌入仅服务端后台任务直连 OpenRouter）。 |

### Cursor 客户端设置

- **Base URL**：`https://你的域名/v1`（注意末尾 `/v1`，避免与反代叠加成 `/v1/v1/...`）。
- **API Key**：填 `GATEWAY_API_KEY`（管理面板生成的对外密钥）。
- **Model**：与 `GET /v1/models` 中 `id` 一致（默认与 `UPSTREAM_MODEL` 相同，如 `anthropic/claude-opus-4.6`）。
- **工具与 MCP**：浏览器侧 **MCP** 在 Cursor 本机执行；模型通过 **`tools` / `tool_calls`** 与 Agent 交互。网关默认只保留合法 **`function`** 工具；若客户端使用扩展形态，可在 **`/admin/`** 开启 **「宽松工具透传」**（可能增加上游 400 风险，按需开关对比）。
- **Session**：可携带请求头 **`X-Session-Id`**（响应头会回写）。

### Agent / Plan / Debug 排错建议

1. 在 **`/admin/`** 开启 **「记录对话元数据日志」**，在容器日志中查看 `chat_upstream_meta`：流式与否、`tools` 数量、请求与响应中的 **`model`**（不含正文），用于区分「Cursor 显示问题」与「上游真实 model」。  
2. Agent 异常时可 **暂时关闭「提示词缓存」**，排除 `cache_control` 与长 system（含身份注入）的干扰。  
3. **身份说明（system）** 会增加上下文长度；Agent 场景可关闭或缩短文案。  
4. 读写出错时先看 **Cursor Output / MCP 日志**（本机工具失败）；再看网关与 **OpenRouter** 返回是否含完整 **`tool_calls`**。  
5. 可选配置 **OpenRouter HTTP-Referer / X-Title**（见管理面板），符合 OpenRouter 统计习惯。

### 已做的请求侧适配摘要

- `role: developer` → `system`；`max_completion_tokens` → `max_tokens`（在未提供 `max_tokens` 时）。  
- SSE 增加 **`X-Accel-Buffering: no`**；全局 **CORS** 宽松策略便于内嵌视图。  
- 对 **tools / tool_calls / tool 消息** 做清洗与参数补全，减少 `Tool ''` 等上游 400。

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
