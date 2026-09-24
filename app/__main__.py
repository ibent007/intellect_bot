import argparse
import asyncio
import logging

import aiohttp
from aiogram import Dispatcher

from app.config import ConfigError, LLMSettings, Settings
from app.db import create_pool
from app.dialogue import DialogueService
from app.handlers.assistant import create_router
from app.health import HealthState, start_health_server
from app.llm import LLMClient
from app.logging_setup import configure_logging
from app.reliability import close_resource
from app.schema import initialize_schema
from app.storage import DialogueStorage
from app.telegram import create_bot

logger = logging.getLogger("app")


async def run(settings: Settings, llm_settings: LLMSettings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    runner = None
    llm_session = None
    try:
        state.pool = await create_pool(settings)
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        await initialize_schema(state.pool)
        logger.info("Таблицы настроек и истории готовы.")
        llm_session = aiohttp.ClientSession()
        llm = LLMClient(llm_settings, llm_session)
        storage = DialogueStorage(state.pool)
        dialogue = DialogueService(storage, llm, llm_settings)
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router())
        runner = await start_health_server(state, settings.health_port)
        state.polling_task = asyncio.create_task(
            dispatcher.start_polling(
                bot,
                db=state.pool,
                dialogue=dialogue,
                allowed_updates=dispatcher.resolve_used_update_types(),
                close_bot_session=False,
            )
        )
        state.initialized = True
        await state.polling_task
    finally:
        state.initialized = False
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)

        if runner is not None:
            await close_resource("health", runner.cleanup)

        await close_resource("telegram", bot.session.close)

        if llm_session is not None:
            await close_resource("llm", llm_session.close)

        if state.pool is not None:
            await close_resource(
                "postgres",
                state.pool.close,
                state.pool.terminate,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-ассистент студента")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
        llm_settings = LLMSettings.load(args.env_file)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings, llm_settings)
    try:
        asyncio.run(run(settings, llm_settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception as error:
        logger.error(
            "Ошибка приложения: kind=%s. Проверьте БД, токен и прокси.",
            type(error).__name__,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
