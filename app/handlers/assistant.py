from aiogram import Bot, F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.llm import LLMClient, LLMError
from app.prompts import STUDY_PROMPT

START_TEXT = (
    "Привет! Я AI-ассистент студента.\n\n"
    "Сейчас я умею объяснять вопросы по программированию. "
    "Напиши вопрос обычным текстом.\n\n"
    "Пока каждый вопрос обрабатывается отдельно, без истории диалога.\n\n"
    "Команды:\n"
    "/start — показать эту справку."
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


async def answer_question(message: Message, bot: Bot, llm: LLMClient) -> None:
    text = message.text
    if text is None or not text.strip():
        await message.answer(EMPTY_TEXT, parse_mode=None)
        return

    messages = [
        {"role": "system", "content": STUDY_PROMPT},
        {"role": "user", "content": text},
    ]

    try:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            result = await llm.generate(messages, temperature=0.3)
    except LLMError as error:
        await message.answer(error.user_message, parse_mode=None)
        return

    for part in split_answer(result.text):
        await message.answer(part, parse_mode=None)


def create_router() -> Router:
    router = Router(name="assistant")
    router.message.filter(F.chat.type == "private", F.text)

    router.message.register(start_command, CommandStart())
    router.message.register(unknown_command, F.text.startswith("/"))
    router.message.register(answer_question)

    return router
