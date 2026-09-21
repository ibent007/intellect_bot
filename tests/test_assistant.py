from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

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

    typing = MagicMock(return_value=AsyncMock())
    monkeypatch.setattr(assistant.ChatActionSender, "typing", typing)

    dispatcher = Dispatcher()
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
        llm=llm,
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
        "Привет! Я AI-ассистент студента.\n\n"
        "Сейчас я умею объяснять вопросы по программированию. "
        "Напиши вопрос обычным текстом.\n\n"
        "Пока каждый вопрос обрабатывается отдельно, без истории диалога.\n\n"
        "Команды:\n"
        "/start — показать эту справку."
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
