import asyncpg


class DialogueStorage:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_or_create_user(self, chat_id: int) -> dict:
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO bot_users (chat_id)
                VALUES ($1)
                ON CONFLICT (chat_id) DO NOTHING
                """,
                chat_id,
            )
            row = await connection.fetchrow(
                """
                SELECT mode, temperature
                FROM bot_users
                WHERE chat_id = $1
                """,
                chat_id,
            )

        if row is None:
            raise RuntimeError("Не удалось получить настройки пользователя.")

        return dict(row)

    async def get_history(self, chat_id: int, max_messages: int) -> list[dict[str, str]]:
        # Берём только целые пары: вопрос и ответ.
        limit = max_messages - max_messages % 2

        rows = await self.pool.fetch(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM dialogue_messages
                WHERE chat_id = $1
                ORDER BY id DESC
                LIMIT $2
            ) AS recent
            ORDER BY id
            """,
            chat_id,
            limit,
        )

        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def save_turn(self, chat_id: int, user_text: str, assistant_text: str) -> None:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO dialogue_messages (chat_id, role, content)
                    VALUES ($1, 'user', $2)
                    """,
                    chat_id,
                    user_text,
                )
                await connection.execute(
                    """
                    INSERT INTO dialogue_messages (chat_id, role, content)
                    VALUES ($1, 'assistant', $2)
                    """,
                    chat_id,
                    assistant_text,
                )

    async def clear_history(self, chat_id: int) -> None:
        await self.pool.execute(
            "DELETE FROM dialogue_messages WHERE chat_id = $1",
            chat_id,
        )

    async def set_mode(self, chat_id: int, mode: str) -> None:
        if mode not in ("study", "translate", "review"):
            raise ValueError("Неизвестный режим.")

        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO bot_users (chat_id, mode)
                    VALUES ($1, $2)
                    ON CONFLICT (chat_id)
                    DO UPDATE SET mode = EXCLUDED.mode
                    """,
                    chat_id,
                    mode,
                )
                await connection.execute(
                    "DELETE FROM dialogue_messages WHERE chat_id = $1",
                    chat_id,
                )

    async def set_temperature(self, chat_id: int, temperature: float) -> None:
        if isinstance(temperature, bool) or temperature not in (0.0, 0.3, 0.7, 1.0):
            raise ValueError("Недопустимая temperature.")

        await self.pool.execute(
            """
            INSERT INTO bot_users (chat_id, temperature)
            VALUES ($1, $2)
            ON CONFLICT (chat_id)
            DO UPDATE SET temperature = EXCLUDED.temperature
            """,
            chat_id,
            temperature,
        )
