from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.dialogue import DialogueError, DialogueService
from app.llm import LLMError

START_TEXT = (
    "Привет! Я твой AI-ассистент.\n\n"
    "Сейчас я умею объяснять вопросы по программированию. "
    "Напиши вопрос обычным текстом.\n\n"
    "Я учитываю недавнюю историю нашего диалога и сохраняю её после перезапуска.\n\n"
    "Команды:\n"
    "/start — показать эту справку.\n"
    "/reset — очистить историю диалога."
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


def create_router() -> Router:
    router = Router(name="assistant")
    router.message.filter(F.chat.type == "private", F.text)

    router.message.register(start_command, CommandStart())
    router.message.register(reset_command, Command("reset"))
    router.message.register(unknown_command, F.text.startswith("/"))
    router.message.register(answer_question)

    return router
