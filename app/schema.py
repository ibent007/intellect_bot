import asyncpg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bot_users (
    chat_id BIGINT PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'study'
        CHECK (mode IN ('study', 'translate', 'review')),
    temperature DOUBLE PRECISION NOT NULL DEFAULT 0.3
        CHECK (temperature IN (0.0, 0.3, 0.7, 1.0)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dialogue_messages (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chat_id BIGINT NOT NULL REFERENCES bot_users(chat_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL CHECK (content <> ''),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS dialogue_messages_chat_id_id_idx
    ON dialogue_messages (chat_id, id);
"""


async def initialize_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(SCHEMA_SQL)
