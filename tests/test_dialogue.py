import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import LLMSettings
from app.dialogue import DialogueError, DialogueService
from app.llm import LLMError, LLMResponse
from app.prompts import STUDY_PROMPT, SYSTEM_PROMPTS


@pytest.fixture
def dialogue_case():
    storage = MagicMock()
    storage.get_or_create_user = AsyncMock(return_value={"mode": "study", "temperature": 0.7})
    storage.get_history = AsyncMock(
        return_value=[
            {"role": "user", "content": "Что такое цикл?"},
            {"role": "assistant", "content": "Повторение действий."},
        ]
    )
    storage.save_turn = AsyncMock()
    storage.clear_history = AsyncMock()
    storage.set_mode = AsyncMock()
    storage.set_temperature = AsyncMock()

    llm = MagicMock()
    llm.generate = AsyncMock(return_value=LLMResponse(text="Новый ответ"))

    settings = LLMSettings(
        base_url="https://api.groq.com/openai/v1",
        api_key="fake-key",
        model="openai/gpt-oss-120b",
    )
    service = DialogueService(storage, llm, settings)
    return service, storage, llm


async def test_reply_sends_history_and_saves_successful_turn(dialogue_case):
    # Arrange
    service, storage, llm = dialogue_case

    # Act
    result = await service.reply(101, "Приведи пример.")

    # Assert
    llm.generate.assert_awaited_once_with(
        [
            {"role": "system", "content": STUDY_PROMPT},
            {"role": "user", "content": "Что такое цикл?"},
            {"role": "assistant", "content": "Повторение действий."},
            {"role": "user", "content": "Приведи пример."},
        ],
        temperature=0.7,
    )
    storage.get_history.assert_awaited_once_with(101, 12)
    storage.save_turn.assert_awaited_once_with(101, "Приведи пример.", "Новый ответ")
    assert result.text == "Новый ответ"


async def test_llm_failure_does_not_save_turn(dialogue_case):
    # Arrange
    service, storage, llm = dialogue_case
    llm.generate.side_effect = LLMError("timeout")

    # Act
    with pytest.raises(LLMError):
        await service.reply(101, "Вопрос")

    # Assert
    storage.save_turn.assert_not_awaited()


async def test_oversized_question_is_rejected_before_api_call(dialogue_case):
    # Arrange
    service, storage, llm = dialogue_case
    text = "А" * service.settings.context_max_chars

    # Act
    with pytest.raises(DialogueError) as error:
        await service.reply(101, text)

    # Assert
    assert str(error.value) == ("Вопрос слишком длинный. Сократите его и отправьте ещё раз.")
    llm.generate.assert_not_awaited()
    storage.save_turn.assert_not_awaited()


def test_character_limit_removes_oldest_whole_pair(dialogue_case):
    # Arrange
    service, _, _ = dialogue_case
    history = [
        {"role": "user", "content": "А" * 6000},
        {"role": "assistant", "content": "Старый ответ"},
        {"role": "user", "content": "Новый вопрос"},
        {"role": "assistant", "content": "Новый ответ"},
    ]

    # Act
    messages = service.build_messages("Продолжи.", history)

    # Assert
    assert messages == [
        {"role": "system", "content": STUDY_PROMPT},
        *history[2:],
        {"role": "user", "content": "Продолжи."},
    ]
    assert sum(len(message["content"]) for message in messages) <= 6000
    assert len(history) == 4


def test_message_limit_keeps_only_complete_pairs(dialogue_case):
    # Arrange
    service, _, _ = dialogue_case
    service.settings = replace(service.settings, history_max_messages=3)
    history = [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ"},
        {"role": "user", "content": "Второй вопрос"},
        {"role": "assistant", "content": "Второй ответ"},
    ]

    # Act
    messages = service.build_messages("Третий вопрос", history)

    # Assert
    assert messages == [
        {"role": "system", "content": STUDY_PROMPT},
        *history[2:],
        {"role": "user", "content": "Третий вопрос"},
    ]


async def test_reset_clears_history_without_calling_llm(dialogue_case):
    # Arrange
    service, storage, llm = dialogue_case

    # Act
    await service.reset(101)

    # Assert
    storage.clear_history.assert_awaited_once_with(101)
    storage.get_or_create_user.assert_not_awaited()
    storage.save_turn.assert_not_awaited()
    llm.generate.assert_not_awaited()


async def test_reset_failure_has_safe_message(dialogue_case, caplog):
    # Arrange
    service, storage, llm = dialogue_case
    storage.clear_history.side_effect = OSError("private-db-details")

    # Act
    with pytest.raises(DialogueError) as error:
        await service.reset(101)

    # Assert
    assert error.value.user_message == ("Не удалось очистить историю. Попробуйте позже.")
    assert "private-db-details" not in caplog.text
    llm.generate.assert_not_awaited()


async def test_reset_waits_for_same_user_lock(dialogue_case):
    # Arrange
    service, storage, _ = dialogue_case
    started = asyncio.Event()

    async def reset():
        started.set()
        await service.reset(101)

    # Act
    async with service.locks[101]:
        task = asyncio.create_task(reset())
        await started.wait()
        waiting = not task.done()
        calls_while_locked = storage.clear_history.await_count

    await asyncio.wait_for(task, timeout=1)

    # Assert
    assert waiting
    assert calls_while_locked == 0
    storage.clear_history.assert_awaited_once_with(101)


@pytest.mark.parametrize("mode", ["study", "translate", "review"])
async def test_reply_uses_saved_mode_prompt(dialogue_case, mode):
    # Arrange
    service, storage, llm = dialogue_case
    storage.get_or_create_user.return_value = {
        "mode": mode,
        "temperature": 0.7,
    }
    storage.get_history.return_value = []

    # Act
    await service.reply(101, "Текст пользователя")

    # Assert
    llm.generate.assert_awaited_once_with(
        [
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {"role": "user", "content": "Текст пользователя"},
        ],
        temperature=0.7,
    )


@pytest.mark.parametrize("mode", ["study", "translate", "review"])
async def test_mode_change_does_not_call_llm(dialogue_case, mode):
    # Arrange
    service, storage, llm = dialogue_case

    # Act
    await service.set_mode(101, mode)

    # Assert
    storage.set_mode.assert_awaited_once_with(101, mode)
    llm.generate.assert_not_awaited()


async def test_invalid_mode_does_not_change_storage(dialogue_case):
    # Arrange
    service, storage, llm = dialogue_case

    # Act
    with pytest.raises(DialogueError):
        await service.set_mode(101, "unknown")

    # Assert
    storage.set_mode.assert_not_awaited()
    llm.generate.assert_not_awaited()


async def test_mode_change_failure_has_safe_message(dialogue_case, caplog):
    # Arrange
    service, storage, llm = dialogue_case
    storage.set_mode.side_effect = OSError("private-db-details")

    # Act
    with pytest.raises(DialogueError) as error:
        await service.set_mode(101, "translate")

    # Assert
    assert error.value.user_message == ("Не удалось изменить режим. Попробуйте позже.")
    assert "private-db-details" not in caplog.text
    llm.generate.assert_not_awaited()


async def test_settings_include_saved_values_and_current_model(dialogue_case):
    # Arrange
    service, storage, llm = dialogue_case

    # Act
    settings = await service.get_settings(101)

    # Assert
    assert settings == {
        "mode": "study",
        "temperature": 0.7,
        "model": service.settings.model,
    }
    storage.get_or_create_user.assert_awaited_once_with(101)
    llm.generate.assert_not_awaited()


@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.7, 1.0])
async def test_temperature_change_preserves_history(dialogue_case, temperature):
    # Arrange
    service, storage, llm = dialogue_case

    # Act
    await service.set_temperature(101, temperature)

    # Assert
    storage.set_temperature.assert_awaited_once_with(101, temperature)
    storage.clear_history.assert_not_awaited()
    llm.generate.assert_not_awaited()


@pytest.mark.parametrize("temperature", [-1.0, 0.5, 2.0, True, float("nan")])
async def test_invalid_temperature_does_not_change_storage(dialogue_case, temperature):
    # Arrange
    service, storage, llm = dialogue_case

    # Act
    with pytest.raises(DialogueError):
        await service.set_temperature(101, temperature)

    # Assert
    storage.set_temperature.assert_not_awaited()
    llm.generate.assert_not_awaited()


async def test_temperature_failure_has_safe_message(dialogue_case, caplog):
    # Arrange
    service, storage, _ = dialogue_case
    storage.set_temperature.side_effect = OSError("private-db-details")

    # Act
    with pytest.raises(DialogueError) as error:
        await service.set_temperature(101, 0.3)

    # Assert
    assert error.value.user_message == ("Не удалось изменить temperature. Попробуйте позже.")
    assert "private-db-details" not in caplog.text


async def test_settings_failure_has_safe_message(dialogue_case, caplog):
    # Arrange
    service, storage, _ = dialogue_case
    storage.get_or_create_user.side_effect = OSError("private-db-details")

    # Act
    with pytest.raises(DialogueError) as error:
        await service.get_settings(101)

    # Assert
    assert error.value.user_message == ("Не удалось прочитать настройки. Попробуйте позже.")
    assert "private-db-details" not in caplog.text
