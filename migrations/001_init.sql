-- Requires PostgreSQL with pgvector. Adjust vector(1536) if EMBEDDING_DIM differs.
CREATE EXTENSION IF NOT EXISTS vector;

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
    embedding vector(1536),
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_request_logs_session ON request_logs (session_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_created ON request_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_conv_turns_session ON conversation_turns (session_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_turn ON embeddings (turn_id);
