import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import ErrorEvent, Message

logger = logging.getLogger("app.reliability")


@asynccontextmanager
async def typing_indicator(
    bot: Bot,
    chat_id: int,
    initial_sleep: float = 4,
    interval: float = 4,
) -> AsyncIterator[None]:
    async def worker() -> None:
        await asyncio.sleep(initial_sleep)
        while True:
            try:
                async with asyncio.timeout(2):
                    await bot.send_chat_action(chat_id=chat_id, action="typing")
            except (TelegramAPIError, TimeoutError):
                logger.warning("Не удалось обновить индикатор набора текста.")
            await asyncio.sleep(interval)

    task = asyncio.create_task(worker())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def handle_error(event: ErrorEvent) -> bool:
    logger.error(
        "Ошибка обработки update: update_id=%s kind=%s",
        event.update.update_id,
        type(event.exception).__name__,
    )

    # При ошибке самого Telegram не пытаемся повторять отправку.
    if isinstance(event.exception, TelegramAPIError):
        return True

    message = event.update.message
    if event.update.callback_query is not None:
        message = event.update.callback_query.message

    if isinstance(message, Message) and message.chat.type == "private":
        with suppress(TelegramAPIError, TimeoutError):
            async with asyncio.timeout(3):
                await message.answer(
                    "Не удалось обработать запрос. Попробуйте позже.",
                    parse_mode=None,
                )
    return True


async def close_resource(
    name: str,
    close: Callable[[], Awaitable[None]],
    terminate: Callable[[], None] | None = None,
) -> None:
    try:
        async with asyncio.timeout(10):
            await close()
    except Exception as error:
        logger.warning(
            "Ошибка закрытия ресурса: resource=%s kind=%s",
            name,
            type(error).__name__,
        )
        if terminate is not None:
            try:
                terminate()
            except Exception as termination_error:
                logger.warning(
                    "Ошибка принудительного закрытия: resource=%s kind=%s",
                    name,
                    type(termination_error).__name__,
                )
