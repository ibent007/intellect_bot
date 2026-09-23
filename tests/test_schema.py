import os
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import asyncpg
import pytest

from app.config import LLMSettings, Settings
from app.dialogue import DialogueService
from app.llm import LLMResponse
from app.schema import initialize_schema
from app.storage import DialogueStorage

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_LAB1_DB_TESTS") != "1",
        reason="Включите RUN_LAB1_DB_TESTS=1 для проверки учебной схемы.",
    ),
]


@pytest.fixture
async def database_pool():
    settings = Settings.load(
        environ={
            **os.environ,
            "BOT_TOKEN": "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
        }
    )

    parameters = {
        "host": settings.postgres_host,
        "port": settings.postgres_port,
        "database": settings.postgres_db,
        "user": settings.postgres_user,
        "password": settings.postgres_password,
        "timeout": 5,
        "command_timeout": 5,
    }

    schema = f"lab1_test_{uuid4().hex}"
    admin = await asyncpg.connect(**parameters)
    pool = None

    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        pool = await asyncpg.create_pool(
            **parameters,
            min_size=1,
            max_size=2,
            server_settings={"search_path": schema},
        )
        await initialize_schema(pool)
        yield pool
    finally:
        try:
            if pool is not None:
                await pool.close()
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


async def test_new_user_has_default_settings(database_pool):
    # Arrange
    chat_id = 101

    # Act
    await database_pool.execute(
        "INSERT INTO bot_users (chat_id) VALUES ($1)",
        chat_id,
    )
    row = await database_pool.fetchrow(
        "SELECT mode, temperature FROM bot_users WHERE chat_id = $1",
        chat_id,
    )

    # Assert
    assert row["mode"] == "study"
    assert row["temperature"] == 0.3


async def test_repeated_initialization_preserves_messages(database_pool):
    # Arrange
    await database_pool.execute(
        "INSERT INTO bot_users (chat_id) VALUES ($1)",
        101,
    )
    await database_pool.execute(
        """
        INSERT INTO dialogue_messages (chat_id, role, content)
        VALUES ($1, $2, $3)
        """,
        101,
        "user",
        "Что такое цикл?",
    )

    # Act
    await initialize_schema(database_pool)
    rows = await database_pool.fetch(
        """
        SELECT role, content
        FROM dialogue_messages
        WHERE chat_id = $1
        ORDER BY id
        """,
        101,
    )

    # Assert
    assert [(row["role"], row["content"]) for row in rows] == [("user", "Что такое цикл?")]


async def test_message_without_user_is_rejected(database_pool):
    # Arrange
    chat_id = 999

    # Act
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await database_pool.execute(
            """
            INSERT INTO dialogue_messages (chat_id, role, content)
            VALUES ($1, $2, $3)
            """,
            chat_id,
            "user",
            "Вопрос",
        )

    # Assert
    count = await database_pool.fetchval("SELECT COUNT(*) FROM dialogue_messages")
    assert count == 0


@pytest.mark.parametrize("temperature", [-1.0, 0.5, 2.0])
async def test_invalid_temperature_is_rejected(database_pool, temperature):
    # Arrange
    chat_id = 101

    # Act
    with pytest.raises(asyncpg.CheckViolationError):
        await database_pool.execute(
            """
            INSERT INTO bot_users (chat_id, temperature)
            VALUES ($1, $2)
            """,
            chat_id,
            temperature,
        )

    # Assert
    count = await database_pool.fetchval("SELECT COUNT(*) FROM bot_users")
    assert count == 0


async def test_failed_transaction_does_not_save_partial_dialogue(database_pool):
    # Arrange
    await database_pool.execute(
        "INSERT INTO bot_users (chat_id) VALUES ($1)",
        101,
    )

    # Act
    with pytest.raises(asyncpg.CheckViolationError):
        async with database_pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO dialogue_messages (chat_id, role, content)
                    VALUES ($1, $2, $3)
                    """,
                    101,
                    "user",
                    "Вопрос",
                )
                await connection.execute(
                    """
                    INSERT INTO dialogue_messages (chat_id, role, content)
                    VALUES ($1, $2, $3)
                    """,
                    101,
                    "invalid-role",
                    "Ответ",
                )

    # Assert
    count = await database_pool.fetchval("SELECT COUNT(*) FROM dialogue_messages")
    assert count == 0


async def test_storage_separates_users_and_preserves_roles(database_pool):
    # Arrange
    storage = DialogueStorage(database_pool)
    await storage.get_or_create_user(101)
    await storage.get_or_create_user(202)

    # Act
    await storage.save_turn(101, "Вопрос первого", "Ответ первого")
    await storage.save_turn(202, "Вопрос второго", "Ответ второго")
    first = await storage.get_history(101, 12)
    second = await storage.get_history(202, 12)

    # Assert
    assert first == [
        {"role": "user", "content": "Вопрос первого"},
        {"role": "assistant", "content": "Ответ первого"},
    ]
    assert second == [
        {"role": "user", "content": "Вопрос второго"},
        {"role": "assistant", "content": "Ответ второго"},
    ]


async def test_history_survives_connection_replacement(database_pool):
    # Arrange
    storage = DialogueStorage(database_pool)
    await storage.get_or_create_user(101)
    await storage.save_turn(101, "Первый вопрос", "Первый ответ")
    await storage.save_turn(101, "Второй вопрос", "Второй ответ")

    # Act
    await database_pool.expire_connections()
    restored = DialogueStorage(database_pool)
    history = await restored.get_history(101, 2)

    # Assert
    assert history == [
        {"role": "user", "content": "Второй вопрос"},
        {"role": "assistant", "content": "Второй ответ"},
    ]


async def test_reset_preserves_settings_and_other_users_history(database_pool):
    # Arrange
    storage = DialogueStorage(database_pool)
    await storage.get_or_create_user(101)
    await storage.get_or_create_user(202)
    await database_pool.execute(
        """
        UPDATE bot_users
        SET mode = $2, temperature = $3
        WHERE chat_id = $1
        """,
        101,
        "review",
        0.7,
    )
    await storage.save_turn(101, "Первый вопрос", "Первый ответ")
    await storage.save_turn(202, "Чужой вопрос", "Чужой ответ")

    # Act
    await storage.clear_history(101)
    await database_pool.expire_connections()
    restored = DialogueStorage(database_pool)

    # Assert
    assert await restored.get_history(101, 12) == []
    assert await restored.get_history(202, 12) == [
        {"role": "user", "content": "Чужой вопрос"},
        {"role": "assistant", "content": "Чужой ответ"},
    ]
    user = await database_pool.fetchrow(
        "SELECT mode, temperature FROM bot_users WHERE chat_id = $1",
        101,
    )
    assert user is not None
    assert user["mode"] == "review"
    assert user["temperature"] == 0.7


async def test_reset_is_safe_for_missing_and_empty_history(database_pool):
    # Arrange
    storage = DialogueStorage(database_pool)

    # Act
    await storage.clear_history(999)
    await storage.get_or_create_user(101)
    await storage.clear_history(101)
    await storage.clear_history(101)

    # Assert
    assert await storage.get_history(101, 12) == []
    assert (
        await database_pool.fetchval(
            "SELECT COUNT(*) FROM bot_users WHERE chat_id = $1",
            101,
        )
        == 1
    )


@pytest.mark.parametrize("mode", ["study", "translate", "review"])
async def test_mode_change_preserves_temperature_and_other_user(database_pool, mode):
    # Arrange
    storage = DialogueStorage(database_pool)
    await storage.get_or_create_user(101)
    await storage.get_or_create_user(202)
    await database_pool.execute(
        "UPDATE bot_users SET temperature = $2 WHERE chat_id = $1",
        101,
        0.7,
    )
    await storage.save_turn(101, "Старый вопрос", "Старый ответ")
    await storage.save_turn(202, "Чужой вопрос", "Чужой ответ")

    # Act
    await storage.set_mode(101, mode)
    await database_pool.expire_connections()
    restored = DialogueStorage(database_pool)
    user = await restored.get_or_create_user(101)
    other = await restored.get_or_create_user(202)

    # Assert
    assert user["mode"] == mode
    assert user["temperature"] == 0.7
    assert await restored.get_history(101, 12) == []
    assert other["mode"] == "study"
    assert other["temperature"] == 0.3
    assert await restored.get_history(202, 12) == [
        {"role": "user", "content": "Чужой вопрос"},
        {"role": "assistant", "content": "Чужой ответ"},
    ]


async def test_mode_can_be_selected_before_first_question(database_pool):
    # Arrange
    storage = DialogueStorage(database_pool)

    # Act
    await storage.set_mode(101, "translate")
    user = await storage.get_or_create_user(101)

    # Assert
    assert user["mode"] == "translate"
    assert user["temperature"] == 0.3
    assert await storage.get_history(101, 12) == []


async def test_invalid_mode_preserves_existing_dialogue(database_pool):
    # Arrange
    storage = DialogueStorage(database_pool)
    await storage.get_or_create_user(101)
    await storage.save_turn(101, "Вопрос", "Ответ")

    # Act
    with pytest.raises(ValueError):
        await storage.set_mode(101, "unknown")

    # Assert
    user = await storage.get_or_create_user(101)
    assert user["mode"] == "study"
    assert await storage.get_history(101, 12) == [
        {"role": "user", "content": "Вопрос"},
        {"role": "assistant", "content": "Ответ"},
    ]


@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.7, 1.0])
async def test_temperature_persists_without_changing_dialogue(database_pool, temperature):
    # Arrange
    storage = DialogueStorage(database_pool)
    await storage.set_mode(101, "review")
    await storage.get_or_create_user(202)
    await storage.save_turn(101, "Мой вопрос", "Мой ответ")
    await storage.save_turn(202, "Чужой вопрос", "Чужой ответ")

    # Act
    await storage.set_temperature(101, temperature)
    await database_pool.expire_connections()
    restored = DialogueStorage(database_pool)
    user = await restored.get_or_create_user(101)
    other = await restored.get_or_create_user(202)

    # Assert
    assert user["temperature"] == temperature
    assert user["mode"] == "review"
    assert other["temperature"] == 0.3
    assert await restored.get_history(101, 12) == [
        {"role": "user", "content": "Мой вопрос"},
        {"role": "assistant", "content": "Мой ответ"},
    ]
    assert await restored.get_history(202, 12) == [
        {"role": "user", "content": "Чужой вопрос"},
        {"role": "assistant", "content": "Чужой ответ"},
    ]


async def test_temperature_can_be_set_before_first_question(database_pool):
    # Arrange
    storage = DialogueStorage(database_pool)

    # Act
    await storage.set_temperature(101, 1.0)
    user = await storage.get_or_create_user(101)

    # Assert
    assert user["temperature"] == 1.0
    assert user["mode"] == "study"
    assert await storage.get_history(101, 12) == []


async def test_next_request_uses_changed_temperature(database_pool):
    # Arrange
    storage = DialogueStorage(database_pool)
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=LLMResponse(text="Ответ"))
    settings = LLMSettings(
        base_url="https://api.groq.com/openai/v1",
        api_key="fake-key",
        model="qwen/qwen3.8-27b",
    )
    service = DialogueService(storage, llm, settings)

    # Act
    await service.set_temperature(101, 1.0)
    await service.reply(101, "Что такое переменная?")

    # Assert
    llm.generate.assert_awaited_once()
    assert llm.generate.call_args.kwargs["temperature"] == 1.0
