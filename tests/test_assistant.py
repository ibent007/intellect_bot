from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendChatAction, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

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
    storage.set_temperature = AsyncMock()

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
        "/study — обучение программированию.\n"
        "/translate — перевод между русским и английским.\n"
        "/review — проверка кода.\n"
        "/settings — показать настройки.\n"
        "/temperature — изменить уровень креативности.\n"
        "/reset — очистить историю диалога.\n"
        "/start — показать это меню.\n\n"
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
    typing.assert_called_once_with(
        bot=bot,
        chat_id=42,
        initial_sleep=4,
        interval=4,
    )
    actions = [
        call.args[1]
        for call in bot.session.call_args_list
        if isinstance(call.args[1], SendChatAction)
    ]
    assert len(actions) == 1
    assert actions[0].chat_id == 42
    assert actions[0].action == "typing"
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
    assert messages[0].text == ("История диалога очищена. Режим и уровень креативности сохранены.")


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
        f"Режим: {name}\n"
        "История очищена. Уровень креативности сохранён.\n"
        "Отправьте новое сообщение."
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


async def test_settings_command_shows_values_without_llm(handler_case):
    # Arrange
    _, bot, llm, _ = handler_case

    # Act
    await send_update(handler_case, "/settings")

    # Assert
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert "Режим: Обучение\n" in messages[0].text
    assert "Модель: GPT-OSS 120B\n" in messages[0].text
    assert "Уровень креативности: 0.3" in messages[0].text
    llm.generate.assert_not_awaited()


@pytest.mark.parametrize(
    ("argument", "temperature"),
    [("0", 0.0), ("0.3", 0.3), ("0.7", 0.7), ("1", 1.0)],
)
async def test_temperature_command_saves_value(handler_case, argument, temperature):
    # Arrange
    dispatcher, bot, llm, _ = handler_case
    storage = dispatcher["dialogue"].storage

    # Act
    await send_update(handler_case, f"/temperature {argument}")

    # Assert
    storage.set_temperature.assert_awaited_once_with(42, temperature)
    storage.clear_history.assert_not_awaited()
    llm.generate.assert_not_awaited()
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == f"Уровень креативности: {temperature:.1f}."


@pytest.mark.parametrize("argument", ["", "abc", "0.5", "nan", "inf"])
async def test_invalid_temperature_command_does_not_save(handler_case, argument):
    # Arrange
    dispatcher, bot, llm, _ = handler_case
    storage = dispatcher["dialogue"].storage

    # Act
    await send_update(handler_case, f"/temperature {argument}".strip())

    # Assert
    storage.set_temperature.assert_not_awaited()
    llm.generate.assert_not_awaited()
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert "0.3" in messages[0].text


async def test_temperature_storage_failure_returns_error(handler_case):
    # Arrange
    dispatcher, bot, llm, _ = handler_case
    dispatcher["dialogue"].storage.set_temperature.side_effect = OSError("private-db-details")

    # Act
    await send_update(handler_case, "/temperature 0.7")

    # Assert
    messages = sent_messages(bot)
    assert len(messages) == 1
    assert messages[0].text == ("Не удалось изменить уровень креативности. Попробуйте позже.")
    llm.generate.assert_not_awaited()


async def send_callback(handler_case, data, chat_type="private", user_id=42):
    dispatcher, bot, _, _ = handler_case

    message = Message(
        message_id=100,
        date=datetime.now(UTC),
        chat=Chat(id=42, type=chat_type),
        from_user=User(id=bot.id, is_bot=True, first_name="Бот"),
        text="Меню",
    )
    callback = CallbackQuery(
        id="test-callback",
        from_user=User(id=user_id, is_bot=False, first_name="Студент"),
        chat_instance="test-chat",
        message=message,
        data=data,
    )

    await dispatcher.feed_update(
        bot,
        Update(update_id=2, callback_query=callback),
    )


def edited_messages(bot):
    return [
        call.args[1]
        for call in bot.session.call_args_list
        if isinstance(call.args[1], EditMessageText)
    ]


def callback_answers(bot):
    return [
        call.args[1]
        for call in bot.session.call_args_list
        if isinstance(call.args[1], AnswerCallbackQuery)
    ]


async def test_start_has_inline_menu(handler_case):
    # Arrange
    _, bot, _, _ = handler_case

    # Act
    await send_update(handler_case, "/start")

    # Assert
    keyboard = sent_messages(bot)[0].reply_markup
    actions = {button.callback_data for row in keyboard.inline_keyboard for button in row}
    assert actions == {
        "menu:study",
        "menu:translate",
        "menu:review",
        "menu:settings",
        "menu:reset",
        "menu:home",
    }


@pytest.mark.parametrize("mode", ["study", "translate", "review"])
async def test_mode_button_edits_existing_message(handler_case, mode):
    # Arrange
    dispatcher, bot, llm, _ = handler_case
    storage = dispatcher["dialogue"].storage

    # Act
    await send_callback(handler_case, f"menu:{mode}")

    # Assert
    storage.set_mode.assert_awaited_once_with(42, mode)
    llm.generate.assert_not_awaited()
    assert sent_messages(bot) == []
    edits = edited_messages(bot)
    assert len(edits) == 1
    assert edits[0].chat_id == 42
    assert edits[0].message_id == 100
    assert f"Режим: {assistant.MODE_NAMES[mode]}" in edits[0].text
    assert len(callback_answers(bot)) == 1


async def test_reset_button_clears_memory_and_edits_menu(handler_case):
    # Arrange
    dispatcher, bot, llm, _ = handler_case

    # Act
    await send_callback(handler_case, "menu:reset")

    # Assert
    dispatcher["dialogue"].storage.clear_history.assert_awaited_once_with(42)
    llm.generate.assert_not_awaited()
    assert sent_messages(bot) == []
    assert "Память диалога очищена." in edited_messages(bot)[0].text


async def test_settings_button_opens_settings_in_same_message(handler_case):
    # Arrange
    _, bot, llm, _ = handler_case

    # Act
    await send_callback(handler_case, "menu:settings")

    # Assert
    assert sent_messages(bot) == []
    edits = edited_messages(bot)
    assert len(edits) == 1
    assert edits[0].message_id == 100
    assert "Уровень креативности: 0.3" in edits[0].text
    buttons = edits[0].reply_markup.inline_keyboard[0]
    assert [button.text for button in buttons] == ["0", "✓ 0.3", "0.7", "1"]
    llm.generate.assert_not_awaited()


@pytest.mark.parametrize(
    ("label", "temperature"),
    [("0", 0.0), ("0.3", 0.3), ("0.7", 0.7), ("1", 1.0)],
)
async def test_temperature_button_saves_and_refreshes_settings(handler_case, label, temperature):
    # Arrange
    dispatcher, bot, llm, _ = handler_case
    storage = dispatcher["dialogue"].storage

    async def save_temperature(chat_id, value):
        storage.get_or_create_user.return_value = {
            "mode": "study",
            "temperature": value,
        }

    storage.set_temperature.side_effect = save_temperature

    # Act
    await send_callback(handler_case, f"menu:temperature:{label}")

    # Assert
    storage.set_temperature.assert_awaited_once_with(42, temperature)
    storage.clear_history.assert_not_awaited()
    assert sent_messages(bot) == []
    edit = edited_messages(bot)[0]
    assert f"Уровень креативности: {temperature:.1f}" in edit.text
    assert f"✓ {label}" in [button.text for button in edit.reply_markup.inline_keyboard[0]]
    llm.generate.assert_not_awaited()


async def test_home_button_restores_help_without_reset(handler_case):
    # Arrange
    dispatcher, bot, llm, _ = handler_case
    storage = dispatcher["dialogue"].storage

    # Act
    await send_callback(handler_case, "menu:home")

    # Assert
    assert edited_messages(bot)[0].text == assistant.START_TEXT
    assert sent_messages(bot) == []
    storage.clear_history.assert_not_awaited()
    storage.set_mode.assert_not_awaited()
    llm.generate.assert_not_awaited()


@pytest.mark.parametrize(
    ("chat_type", "user_id"),
    [("group", 42), ("supergroup", 42), ("private", 99)],
)
async def test_menu_rejects_invalid_chat_or_user(handler_case, chat_type, user_id):
    # Arrange
    dispatcher, bot, _, _ = handler_case
    storage = dispatcher["dialogue"].storage

    # Act
    await send_callback(handler_case, "menu:reset", chat_type, user_id)

    # Assert
    storage.clear_history.assert_not_awaited()
    assert edited_messages(bot) == []
    assert sent_messages(bot) == []
    assert callback_answers(bot)[0].show_alert is True


async def test_invalid_temperature_button_does_not_save(handler_case):
    # Arrange
    dispatcher, bot, _, _ = handler_case

    # Act
    await send_callback(handler_case, "menu:temperature:0.5")

    # Assert
    dispatcher["dialogue"].storage.set_temperature.assert_not_awaited()
    assert edited_messages(bot) == []
    assert callback_answers(bot)[0].show_alert is True


async def test_menu_storage_failure_is_shown_in_same_message(handler_case):
    # Arrange
    dispatcher, bot, _, _ = handler_case
    dispatcher["dialogue"].storage.set_mode.side_effect = OSError("private-db-details")

    # Act
    await send_callback(handler_case, "menu:translate")

    # Assert
    assert sent_messages(bot) == []
    assert edited_messages(bot)[0].text == ("Не удалось изменить режим. Попробуйте позже.")


async def test_repeated_menu_content_does_not_raise(handler_case):
    # Arrange
    _, bot, _, _ = handler_case

    async def telegram_response(*args, **kwargs):
        method = args[1]
        if isinstance(method, EditMessageText):
            raise TelegramBadRequest(
                method=method,
                message="Bad Request: message is not modified",
            )
        return True

    bot.session.side_effect = telegram_response

    # Act
    await send_callback(handler_case, "menu:home")

    # Assert
    assert len(edited_messages(bot)) == 1
    assert len(callback_answers(bot)) == 1
    assert sent_messages(bot) == []
