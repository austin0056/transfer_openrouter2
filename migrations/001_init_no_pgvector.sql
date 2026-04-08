-- 托管 PostgreSQL 无 pgvector 扩展时使用本脚本（Zeabur 等若报 extension "vector" is not available）。
-- 向量列使用 double precision[]；应用内需关闭「使用 pgvector 类型写入」或设置 embedding_use_pgvector=false。
-- 若托管库日后支持 pgvector，可改用 001_init.sql 并迁移数据（需重建 embeddings 列类型）。

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_session_id TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS request_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    request_json JSONB NOT NULL,
    response_json JSONB,
    usage_json JSONB,
    streamed BOOLEAN NOT NULL DEFAULT false,
    latency_ms INTEGER,
    error_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    request_log_id UUID NOT NULL REFERENCES request_logs (id) ON DELETE CASCADE,
    user_text TEXT,
    assistant_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id UUID NOT NULL REFERENCES conversation_turns (id) ON DELETE CASCADE,
    embedding double precision[],
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_request_logs_session ON request_logs (session_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_created ON request_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_conv_turns_session ON conversation_turns (session_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_turn ON embeddings (turn_id);
