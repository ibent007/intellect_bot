import asyncio
from contextlib import suppress

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.dialogue import DialogueError, DialogueService
from app.llm import LLMError
from app.prompts import MODE_NAMES
from app.reliability import handle_error, typing_indicator

START_TEXT = (
    "Привет! Я твой AI-ассистент.\n\n"
    "Помогаю разобраться в программировании, "
    "проверить код и перевести текст.\n\n"
    "Я учитываю недавнюю историю нашего диалога "
    "и сохраняю её после перезапуска.\n\n"
    "Выберите действие кнопкой ниже "
    "или откройте меню команд рядом с полем ввода."
)

UNKNOWN_COMMAND_TEXT = (
    "Неизвестная команда. Используйте /start или напишите вопрос по программированию."
)

EMPTY_TEXT = "Напишите непустой текстовый вопрос."


TEMPERATURE_BUTTONS = {
    "0": 0.0,
    "0.3": 0.3,
    "0.7": 0.7,
    "1": 1.0,
}


def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📚 Обучение", callback_data="menu:study"),
                InlineKeyboardButton(text="🔎 Проверка кода", callback_data="menu:review"),
            ],
            [
                InlineKeyboardButton(text="🌐 Перевод", callback_data="menu:translate"),
                InlineKeyboardButton(text="⚙️ Настройки", callback_data="menu:settings"),
            ],
            [
                InlineKeyboardButton(text="🗑 Сброс памяти", callback_data="menu:reset"),
                InlineKeyboardButton(text="🏠︎ Главное меню", callback_data="menu:home"),
            ],
        ]
    )


def settings_keyboard(temperature: float) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=f"✓ {label}" if value == temperature else label,
            callback_data=f"menu:temperature:{label}",
        )
        for label, value in TEMPERATURE_BUTTONS.items()
    ]

    return InlineKeyboardMarkup(
        inline_keyboard=[
            buttons,
            [InlineKeyboardButton(text="⬅️ Главное меню", callback_data="menu:home")],
        ]
    )


def settings_text(settings: dict) -> str:
    model_name = {
        "qwen/qwen3.8-27b": "Qwen 3.8 27B",
        "openai/gpt-oss-120b": "GPT-OSS 120B",
    }.get(settings["model"], settings["model"])

    return (
        f"Режим: {MODE_NAMES[settings['mode']]}\n"
        f"Модель: {model_name}\n"
        f"Уровень креативности: {settings['temperature']:.1f}\n\n"
        "Выберите уровень креативности кнопкой ниже.\n\n"
        "Меньшие значения обычно дают более сдержанные ответы, "
        "большие — более разнообразные.\n"
    )


async def edit_menu(
    message: Message,
    text: str,
    keyboard: InlineKeyboardMarkup,
) -> None:
    try:
        await message.edit_text(
            text,
            reply_markup=keyboard,
            parse_mode=None,
        )
    except TelegramBadRequest as error:
        # Повторное нажатие может оставить текст и кнопки прежними.
        if "message is not modified" not in error.message.lower():
            raise


def split_answer(text: str) -> list[str]:
    parts = []
    start = 0
    size = 0

    for position, character in enumerate(text):
        character_size = 2 if ord(character) > 0xFFFF else 1

        if size + character_size > 4000:
            parts.append(text[start:position])
            start = position
            size = 0

        size += character_size

    if start < len(text):
        parts.append(text[start:])

    return parts


async def start_command(message: Message) -> None:
    await message.answer(
        START_TEXT,
        reply_markup=main_keyboard(),
        parse_mode=None,
    )


async def unknown_command(message: Message) -> None:
    await message.answer(UNKNOWN_COMMAND_TEXT, parse_mode=None)


async def answer_question(message: Message, bot: Bot, dialogue: DialogueService) -> None:
    text = message.text
    if text is None or not text.strip():
        await message.answer(EMPTY_TEXT, parse_mode=None)
        return

    # Ошибка отправки индикатора не должна мешать ответу.
    with suppress(TelegramAPIError, TimeoutError):
        async with asyncio.timeout(2):
            await bot.send_chat_action(
                chat_id=message.chat.id,
                action="typing",
            )

    try:
        async with typing_indicator(
            bot=bot,
            chat_id=message.chat.id,
            initial_sleep=4,
            interval=4,
        ):
            result = await dialogue.reply(message.chat.id, text)
    except (LLMError, DialogueError) as error:
        await message.answer(error.user_message, parse_mode=None)
        return

    for part in split_answer(result.text):
        await message.answer(part, parse_mode=None)


async def reset_command(message: Message, dialogue: DialogueService) -> None:
    try:
        await dialogue.reset(message.chat.id)
    except DialogueError as error:
        await message.answer(error.user_message, parse_mode=None)
        return

    await message.answer(
        "Память диалога очищена.\n\n"
        "Режим и уровень креативности сохранены.\n"
        "Сообщения в Telegram остались в чате.\n\n"
        "Можно начать новый диалог.",
        parse_mode=None,
    )


async def mode_command(
    message: Message,
    command: CommandObject,
    dialogue: DialogueService,
) -> None:
    mode = command.command

    try:
        await dialogue.set_mode(message.chat.id, mode)
    except DialogueError as error:
        await message.answer(error.user_message, parse_mode=None)
        return

    await message.answer(
        f"Режим: {MODE_NAMES[mode]}\n\n"
        "Память диалога очищена.\n"
        "Уровень креативности сохранён.\n"
        "Отправьте новое сообщение.\n\n"
        "Выбор другого режима также очищает память диалога.",
        parse_mode=None,
    )


async def settings_command(message: Message, dialogue: DialogueService) -> None:
    try:
        settings = await dialogue.get_settings(message.chat.id)
    except DialogueError as error:
        await message.answer(error.user_message, parse_mode=None)
        return

    await message.answer(
        settings_text(settings),
        reply_markup=settings_keyboard(settings["temperature"]),
        parse_mode=None,
    )


async def temperature_command(
    message: Message,
    command: CommandObject,
    dialogue: DialogueService,
) -> None:
    try:
        temperature = float(command.args or "")
    except ValueError:
        await message.answer(
            "Укажите уровень креативности: 0, 0.3, 0.7 или 1.\nНапример: /temperature 0.7",
            parse_mode=None,
        )
        return

    try:
        await dialogue.set_temperature(message.chat.id, temperature)
    except DialogueError as error:
        await message.answer(error.user_message, parse_mode=None)
        return

    await message.answer(
        f"Уровень креативности: {temperature:.1f}.",
        parse_mode=None,
    )


async def menu_callback(
    callback: CallbackQuery,
    dialogue: DialogueService,
) -> None:
    message = callback.message

    if (
        not isinstance(message, Message)
        or message.chat.type != "private"
        or message.chat.id != callback.from_user.id
    ):
        await callback.answer(
            "Откройте меню в личном чате с ботом.",
            show_alert=True,
        )
        return

    action = (callback.data or "").removeprefix("menu:")
    temperature_actions = {
        f"temperature:{label}": value for label, value in TEMPERATURE_BUTTONS.items()
    }
    allowed = {"home", "settings", "reset", *MODE_NAMES, *temperature_actions}

    if action not in allowed:
        await callback.answer(
            "Неизвестная кнопка. Откройте /start.",
            show_alert=True,
        )
        return

    # Сразу убираем индикатор ожидания на кнопке.
    await callback.answer()

    keyboard = main_keyboard()

    try:
        if action == "home":
            text = START_TEXT

        elif action in MODE_NAMES:
            await dialogue.set_mode(message.chat.id, action)
            text = (
                f"Режим: {MODE_NAMES[action]}\n\n"
                "Память диалога очищена.\n"
                "Уровень креативности сохранён.\n"
                "Отправьте новое сообщение.\n\n"
                "Выбор другого режима также очищает память диалога."
            )

        elif action == "reset":
            await dialogue.reset(message.chat.id)
            text = (
                "Память диалога очищена.\n\n"
                "Режим и уровень креативности сохранены.\n"
                "Сообщения в Telegram остались в чате.\n\n"
                "Можно начать новый диалог."
            )

        else:
            if action in temperature_actions:
                await dialogue.set_temperature(
                    message.chat.id,
                    temperature_actions[action],
                )

            settings = await dialogue.get_settings(message.chat.id)
            text = settings_text(settings)
            keyboard = settings_keyboard(settings["temperature"])

    except DialogueError as error:
        text = error.user_message

    await edit_menu(message, text, keyboard)


def create_router() -> Router:
    router = Router(name="assistant")
    router.message.filter(F.chat.type == "private", F.text)

    router.message.register(start_command, CommandStart())
    router.message.register(reset_command, Command("reset"))
    router.message.register(
        mode_command,
        Command("study", "translate", "review"),
    )
    router.message.register(settings_command, Command("settings"))
    router.message.register(temperature_command, Command("temperature"))
    router.message.register(unknown_command, F.text.startswith("/"))
    router.message.register(answer_question)

    router.callback_query.register(
        menu_callback,
        F.data.startswith("menu:"),
    )

    router.errors.register(handle_error)

    return router
