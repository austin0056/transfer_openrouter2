# 环境变量说明

## 推荐做法：环境变量尽量少，其余用管理面板

除 **管理入口** 与 **配置文件路径** 外，**OpenRouter 密钥、对外 Gateway 密钥、代理、数据库、模型、缓存、嵌入、HTTP 参数等**均建议在 **`/admin/` 管理面板** 中填写并保存到 **`CONFIG_FILE`**（默认 `data/config.json`），无需在云平台重复配置一长串环境变量。

加载顺序：**先读环境变量与 `.env`，再读 `CONFIG_FILE`，后者覆盖同名项**。  
**`ADMIN_KEY` 只从环境变量读取**，不会写入 JSON。

---

## 环境变量（建议只保留这些）

| 变量 | 是否必填 | 说明 |
|------|----------|------|
| `ADMIN_KEY` | **强烈建议** | 非空时启用 `/admin/` 与 `/admin/api/config`。用于首次在浏览器里填写其余配置。**不要写进 `CONFIG_FILE`。** |
| `CONFIG_FILE` | 否 | 运行时 JSON 路径，默认 `data/config.json`。仅当需要改路径时在环境中设置。 |

未设置 `ADMIN_KEY` 时：管理 API 为 **404**；若此时也未通过其它方式提供有效的 `CONFIG_FILE`（例如挂载卷），则需在环境变量或 `.env` 中自行提供 `OPENROUTER_API_KEY`、`GATEWAY_API_KEY` 等，否则聊天接口为 **503**。

---

## 在管理面板中配置（不推荐再配一份环境变量）

以下字段与 [应用配置](app/config.py) 一致，**在 `/admin/` 中编辑并保存即可**（与 JSON 键名为 snake_case 对应关系见面板表单）：

| 配置项 | 说明 |
|--------|------|
| `openrouter_api_key` | OpenRouter 上游 Key。 |
| `gateway_api_key` | Cursor / 客户端使用的 `Bearer` 密钥。 |
| `upstream_model` / `upstream_base_url` | 模型与上游根 URL。 |
| `https_proxy` | SOCKS5 等出站代理。 |
| `database_url` | PostgreSQL（需 pgvector）；空则不落库、不写向量。 |
| `cache_enabled` / `cache_ttl_1h` | 提示词缓存策略。 |
| `embedding_*` | 嵌入模型、维度、独立 Key、Base URL。 |
| `request_timeout_seconds` 等 | HTTP 与队列参数。 |

完整键名见保存后的 `CONFIG_FILE` 或 **「高级：全部键名参考」** 一节。

---

## 高级：本地 `.env`（可选）

本地开发或 CI 可继续使用根目录 `.env`；与 `CONFIG_FILE` 并存时，**文件内覆盖**同名环境变量。适合临时调试，**生产仍推荐以 `ADMIN_KEY` + 面板为主**。

---

## 高级：全部键名参考（与面板 / JSON 一致）

仅当**不使用管理面板**、改为纯环境变量或纯 JSON 时查阅。名称与 Pydantic 字段一致，环境变量一般为 **大写 + 下划线**（如 `UPSTREAM_MODEL`）。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENROUTER_API_KEY` | 空 | 上游 Key。 |
| `GATEWAY_API_KEY` | 空 | 对外 Key。 |
| `UPSTREAM_MODEL` | `anthropic/claude-opus-4.6` | 模型 slug。 |
| `UPSTREAM_BASE_URL` | `https://openrouter.ai/api/v1` | 上游根 URL。 |
| `HTTPS_PROXY` | 空 | 出站代理。 |
| `DATABASE_URL` | 空 | PostgreSQL。 |
| `CACHE_ENABLED` | `true` | 是否注入 `cache_control`。 |
| `CACHE_TTL_1H` | `true` | 长 TTL / 短 TTL。 |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | 嵌入模型。 |
| `EMBEDDING_DIM` | `1536` | 与 `migrations` 中 `vector(N)` 一致。 |
| `EMBEDDING_API_KEY` | 空 | 空则沿用 OpenRouter Key。 |
| `EMBEDDING_BASE_URL` | `https://openrouter.ai/api/v1` | 嵌入 API 根。 |
| `REQUEST_TIMEOUT_SECONDS` | `600` | 请求总超时（秒）。 |
| `CONNECT_TIMEOUT_SECONDS` | `30` | 连接超时（秒）。 |
| `HTTP_MAX_CONNECTIONS` | `100` | 最大连接数。 |
| `HTTP_MAX_KEEPALIVE` | `20` | keep-alive 数。 |
| `PERSIST_QUEUE_MAX` | `10000` | 持久化队列。 |
| `EMBED_QUEUE_MAX` | `10000` | 嵌入队列。 |
| `EMBED_BATCH_SIZE` | `8` | 嵌入批大小（预留）。 |

---

## 最小 `.env` 模板（推荐）

```dotenv
# 仅保留管理入口；其余在 /admin/ 中配置并保存到 CONFIG_FILE
ADMIN_KEY=请改为强随机字符串

# 可选：配置文件路径（默认 data/config.json，一般不必设）
# CONFIG_FILE=data/config.json
```

本地若需脱离面板调试，可再增加 `OPENROUTER_API_KEY`、`GATEWAY_API_KEY` 等，详见上一节。
