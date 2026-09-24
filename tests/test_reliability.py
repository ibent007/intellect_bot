import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendChatAction

from app.reliability import close_resource, typing_indicator


async def test_typing_recovers_after_telegram_error(caplog):
    # Arrange
    recovered = asyncio.Event()
    calls = 0

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TelegramBadRequest(
                method=SendChatAction(chat_id=42, action="typing"),
                message="private-details",
            )
        recovered.set()

    bot = Mock(send_chat_action=AsyncMock(side_effect=send))

    # Act
    async with typing_indicator(bot, 42, initial_sleep=0, interval=0):
        await asyncio.wait_for(recovered.wait(), timeout=1)

    calls_after_exit = calls
    await asyncio.sleep(0)

    # Assert
    assert calls >= 2
    assert calls == calls_after_exit
    assert "private-details" not in caplog.text


async def test_typing_stops_when_body_fails():
    # Arrange
    bot = Mock(send_chat_action=AsyncMock())

    # Act
    with pytest.raises(ValueError):
        async with typing_indicator(bot, 42, initial_sleep=60):
            raise ValueError("failure")

    # Assert
    bot.send_chat_action.assert_not_awaited()


@pytest.mark.parametrize(
    "failure",
    [OSError("private-details"), TimeoutError()],
)
async def test_failed_close_uses_termination_without_leaking(failure, caplog):
    # Arrange
    close = AsyncMock(side_effect=failure)
    terminate = Mock()

    # Act
    await close_resource("postgres", close, terminate)

    # Assert
    close.assert_awaited_once()
    terminate.assert_called_once()
    assert "resource=postgres" in caplog.text
    assert "private-details" not in caplog.text
