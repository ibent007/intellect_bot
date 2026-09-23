from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.config import LLMSettings
from app.dialogue import DialogueService
from app.handlers import assistant
from app.llm import LLMError, LLMResponse
from app.prompts import STUDY_PROMPT

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


@pytest.fixture
def handler_case(monkeypatch):
    bot = Bot(TOKEN)
    bot.session = AsyncMock()

    llm = MagicMock()
    llm.generate = AsyncMock(return_value=LLMResponse(text="Ответ модели"))
    storage = MagicMock()
    storage.get_or_create_user = AsyncMock(return_value={"mode": "study", "temperature": 0.3})
    storage.get_history = AsyncMock(return_value=[])
    storage.save_turn = AsyncMock()
    storage.clear_history = AsyncMock()
    storage.set_mode = AsyncMock()

    settings = LLMSettings(
        base_url="https://api.groq.com/openai/v1",
        api_key="fake-key-for-tests",
        model="openai/gpt-oss-120b",
    )
    dialogue = DialogueService(storage, llm, settings)

    typing = MagicMock(return_value=AsyncMock())
    monkeypatch.setattr(assistant.ChatActionSender, "typing", typing)

    dispatcher = Dispatcher()
    dispatcher["dialogue"] = dialogue
    dispatcher.include_router(assistant.create_router())

    return dispatcher, bot, llm, typing


async def send_update(handler_case, text, chat_type="private"):
    dispatcher, bot, llm, _ = handler_case
    message = Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=42, type=chat_type),
        from_user=User(id=42, is_bot=False, first_name="Студент"),
        text=text,
    )
    await dispatcher.feed_update(
        bot,
        Update(update_id=1, message=message),
    )


def sent_messages(bot):
    return [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]


async def test_start_returns_help_without_calling_llm(handler_case):
    # Arrange
    _, bot, llm, _ = handler_case

    # Act
    await send_update(handler_case, "/start")

    # Assert
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == (
        "Привет! Я твой AI-ассистент.\n\n"
        "Помогаю разобраться в программировании, перевести текст "
        "и проверить код.\n\n"
        "Я учитываю недавнюю историю нашего диалога "
        "и сохраняю её после перезапуска.\n\n"
        "Команды:\n"
        "/start — показать эту справку.\n"
        "/study — обучение программированию.\n"
        "/translate — перевод между русским и английским.\n"
        "/review — проверка кода.\n"
        "/reset — очистить историю диалога.\n\n"
        "Выбор режима очищает историю. Temperature сохраняется.\n"
        "По умолчанию включён режим обучения."
    )
    llm.generate.assert_not_awaited()


async def test_unknown_command_does_not_call_llm(handler_case):
    # Arrange
    _, bot, llm, _ = handler_case

    # Act
    await send_update(handler_case, "/unknown")

    # Assert
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == (
        "Неизвестная команда. Используйте /start или напишите вопрос по программированию."
    )
    llm.generate.assert_not_awaited()


async def test_question_uses_study_prompt_and_typing(handler_case):
    # Arrange
    _, bot, llm, typing = handler_case

    # Act
    await send_update(handler_case, "Что такое цикл?")

    # Assert
    llm.generate.assert_awaited_once_with(
        [
            {"role": "system", "content": STUDY_PROMPT},
            {"role": "user", "content": "Что такое цикл?"},
        ],
        temperature=0.3,
    )
    typing.assert_called_once_with(bot=bot, chat_id=42)
    typing.return_value.__aenter__.assert_awaited_once()
    typing.return_value.__aexit__.assert_awaited_once()
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == "Ответ модели"
    assert messages[0].parse_mode is None


@pytest.mark.parametrize("chat_type", ["group", "supergroup"])
async def test_group_messages_are_ignored(handler_case, chat_type):
    # Arrange
    _, bot, llm, _ = handler_case

    # Act
    await send_update(handler_case, "Что такое цикл?", chat_type)

    # Assert
    llm.generate.assert_not_awaited()
    assert sent_messages(bot) == []


async def test_non_text_message_is_ignored(handler_case):
    # Arrange
    _, bot, llm, _ = handler_case

    # Act
    await send_update(handler_case, None)

    # Assert
    llm.generate.assert_not_awaited()
    assert sent_messages(bot) == []


async def test_whitespace_does_not_call_llm(handler_case):
    # Arrange
    _, bot, llm, _ = handler_case

    # Act
    await send_update(handler_case, " \n ")

    # Assert
    llm.generate.assert_not_awaited()
    assert sent_messages(bot)[0].text == "Напишите непустой текстовый вопрос."


async def test_llm_error_returns_safe_message(handler_case):
    # Arrange
    _, bot, llm, _ = handler_case
    llm.generate.side_effect = LLMError("timeout")

    # Act
    await send_update(handler_case, "Что такое цикл?")

    # Assert
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == ("Модель не успела ответить. Попробуйте ещё раз чуть позже.")


async def test_long_answer_is_sent_in_order_without_text_loss(handler_case):
    # Arrange
    _, bot, llm, _ = handler_case
    text = "Начало\n" + "Пример 😀\n" * 1000 + "Конец"
    llm.generate.return_value = LLMResponse(text=text)

    # Act
    await send_update(handler_case, "Покажи пример.")

    # Assert
    messages = sent_messages(bot)
    assert len(messages) > 1
    assert "".join(message.text for message in messages) == text
    assert all(0 < len(message.text.encode("utf-16-le")) // 2 <= 4000 for message in messages)
    assert all(message.parse_mode is None for message in messages)


async def test_reset_clears_history_without_calling_llm(handler_case):
    # Arrange
    dispatcher, bot, llm, typing = handler_case
    storage = dispatcher["dialogue"].storage

    # Act
    await send_update(handler_case, "/reset")

    # Assert
    storage.clear_history.assert_awaited_once_with(42)
    llm.generate.assert_not_awaited()
    typing.assert_not_called()
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == ("История диалога очищена. Режим и temperature сохранены.")


async def test_reset_failure_does_not_report_success(handler_case):
    # Arrange
    dispatcher, bot, llm, _ = handler_case
    dispatcher["dialogue"].storage.clear_history.side_effect = OSError("private-db-details")

    # Act
    await send_update(handler_case, "/reset")

    # Assert
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == ("Не удалось очистить историю. Попробуйте позже.")
    llm.generate.assert_not_awaited()


@pytest.mark.parametrize("chat_type", ["group", "supergroup"])
async def test_reset_in_group_does_not_clear_history(handler_case, chat_type):
    # Arrange
    dispatcher, bot, llm, _ = handler_case
    storage = dispatcher["dialogue"].storage

    # Act
    await send_update(handler_case, "/reset", chat_type)

    # Assert
    storage.clear_history.assert_not_awaited()
    llm.generate.assert_not_awaited()
    assert sent_messages(bot) == []


@pytest.mark.parametrize(
    ("mode", "name"),
    [
        ("study", "Обучение"),
        ("translate", "Перевод"),
        ("review", "Проверка кода"),
    ],
)
async def test_mode_command_changes_mode_without_llm(handler_case, mode, name):
    # Arrange
    dispatcher, bot, llm, typing = handler_case
    storage = dispatcher["dialogue"].storage

    # Act
    await send_update(handler_case, f"/{mode}")

    # Assert
    storage.set_mode.assert_awaited_once_with(42, mode)
    llm.generate.assert_not_awaited()
    typing.assert_not_called()
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == (
        f"Режим: {name}.\nИстория очищена. Temperature сохранена.\nОтправьте новое сообщение."
    )


async def test_mode_command_failure_does_not_report_success(handler_case):
    # Arrange
    dispatcher, bot, llm, _ = handler_case
    dispatcher["dialogue"].storage.set_mode.side_effect = OSError("private-db-details")

    # Act
    await send_update(handler_case, "/translate")

    # Assert
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == ("Не удалось изменить режим. Попробуйте позже.")
    llm.generate.assert_not_awaited()
