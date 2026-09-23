from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.dialogue import DialogueError, DialogueService
from app.llm import LLMError
from app.prompts import MODE_NAMES

START_TEXT = (
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
    "/reset — очистить историю диалога.\n"
    "/settings — показать настройки.\n\n"
    "По умолчанию включён режим обучения."
)

UNKNOWN_COMMAND_TEXT = (
    "Неизвестная команда. Используйте /start или напишите вопрос по программированию."
)

EMPTY_TEXT = "Напишите непустой текстовый вопрос."


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
    await message.answer(START_TEXT, parse_mode=None)


async def unknown_command(message: Message) -> None:
    await message.answer(UNKNOWN_COMMAND_TEXT, parse_mode=None)


async def answer_question(message: Message, bot: Bot, dialogue: DialogueService) -> None:
    text = message.text
    if text is None or not text.strip():
        await message.answer(EMPTY_TEXT, parse_mode=None)
        return

    try:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
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
        "История диалога очищена. Режим и temperature сохранены.",
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
        f"Режим: {MODE_NAMES[mode]}.\n"
        "История очищена. Temperature сохранена.\n"
        "Отправьте новое сообщение.",
        parse_mode=None,
    )


async def settings_command(message: Message, dialogue: DialogueService) -> None:
    try:
        settings = await dialogue.get_settings(message.chat.id)
    except DialogueError as error:
        await message.answer(error.user_message, parse_mode=None)
        return

    await message.answer(
        f"Режим: {MODE_NAMES[settings['mode']]}.\n"
        f"Модель: {settings['model']}\n"
        f"Temperature: {settings['temperature']:.1f}\n\n"
        "Изменить temperature:\n"
        "/temperature 0\n"
        "/temperature 0.3\n"
        "/temperature 0.7\n"
        "/temperature 1\n\n"
        "Меньшие значения обычно дают более сдержанные ответы, "
        "большие — более разнообразные. "
        "Temperature не гарантирует правильность ответа.",
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
            "Укажите temperature: 0, 0.3, 0.7 или 1.\nНапример: /temperature 0.7",
            parse_mode=None,
        )
        return

    try:
        await dialogue.set_temperature(message.chat.id, temperature)
    except DialogueError as error:
        await message.answer(error.user_message, parse_mode=None)
        return

    await message.answer(
        f"Temperature: {temperature:.1f}.\n"
        "Настройка сохранена и применяется к следующему запросу. "
        "История сохранена.",
        parse_mode=None,
    )


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

    return router
