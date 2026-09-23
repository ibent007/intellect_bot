import asyncio
import logging
from collections import defaultdict

import asyncpg

from app.config import LLMSettings
from app.llm import LLMClient, LLMResponse
from app.prompts import STUDY_PROMPT
from app.storage import DialogueStorage

logger = logging.getLogger("app.dialogue")


class DialogueError(Exception):
    def __init__(self, message: str):
        self.user_message = message
        super().__init__(message)


class DialogueService:
    def __init__(
        self,
        storage: DialogueStorage,
        llm: LLMClient,
        settings: LLMSettings,
    ):
        self.storage = storage
        self.llm = llm
        self.settings = settings
        self.locks = defaultdict(asyncio.Lock)

    def build_messages(self, text: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
        required_size = len(STUDY_PROMPT) + len(text)

        if required_size > self.settings.context_max_chars:
            raise DialogueError("Вопрос слишком длинный. Сократите его и отправьте ещё раз.")

        limit = self.settings.history_max_messages
        limit -= limit % 2
        selected = list(history[-limit:]) if limit else []

        history_size = sum(len(message["content"]) for message in selected)

        while selected and required_size + history_size > self.settings.context_max_chars:
            removed = selected[:2]
            del selected[:2]
            history_size -= sum(len(message["content"]) for message in removed)

        return [
            {"role": "system", "content": STUDY_PROMPT},
            *selected,
            {"role": "user", "content": text},
        ]

    async def reply(self, chat_id: int, text: str) -> LLMResponse:
        # Один пользователь получает ответы последовательно.
        # Другие пользователи могут обращаться к модели одновременно.
        async with self.locks[chat_id]:
            try:
                user = await self.storage.get_or_create_user(chat_id)
                history = await self.storage.get_history(
                    chat_id, self.settings.history_max_messages
                )
                messages = self.build_messages(text, history)

                result = await self.llm.generate(messages, temperature=user["temperature"])

                await self.storage.save_turn(chat_id, text, result.text)
                return result

            except (
                asyncpg.PostgresError,
                asyncpg.InterfaceError,
                OSError,
                TimeoutError,
            ):
                logger.warning("Ошибка обращения к хранилищу диалогов.")
                raise DialogueError(
                    "Не удалось прочитать или сохранить историю. Попробуйте позже."
                ) from None

    async def reset(self, chat_id: int) -> None:
        async with self.locks[chat_id]:
            try:
                await self.storage.clear_history(chat_id)
            except (
                asyncpg.PostgresError,
                asyncpg.InterfaceError,
                OSError,
                TimeoutError,
            ):
                logger.warning("Ошибка очистки истории диалога.")
                raise DialogueError("Не удалось очистить историю. Попробуйте позже.") from None
