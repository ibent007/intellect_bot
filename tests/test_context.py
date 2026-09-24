from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import LLMSettings
from app.dialogue import DialogueError, DialogueService
from app.prompts import SYSTEM_PROMPTS


@pytest.fixture
def context_case():
    storage = MagicMock()
    storage.get_or_create_user = AsyncMock()
    storage.get_history = AsyncMock(return_value=[])
    storage.save_turn = AsyncMock()

    llm = MagicMock()
    llm.generate = AsyncMock()

    settings = LLMSettings(
        base_url="https://api.groq.com/openai/v1",
        api_key="fake-key",
        model="qwen/qwen3.8-27b",
        history_max_messages=12,
        context_max_chars=6000,
    )

    service = DialogueService(storage, llm, settings)
    return service, storage, llm


@pytest.mark.parametrize("mode", ["study", "translate", "review"])
def test_exact_context_limit_keeps_complete_history(context_case, mode):
    # Arrange
    service, _, _ = context_case
    text = "Продолжи."
    history = [
        {"role": "user", "content": "Вопрос 😀"},
        {"role": "assistant", "content": "Ответ"},
    ]
    limit = (
        len(SYSTEM_PROMPTS[mode]) + len(text) + sum(len(message["content"]) for message in history)
    )
    service.settings = replace(service.settings, context_max_chars=limit)

    # Act
    messages = service.build_messages(text, history, mode=mode)

    # Assert
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPTS[mode]},
        *history,
        {"role": "user", "content": text},
    ]
    assert sum(len(message["content"]) for message in messages) == limit


@pytest.mark.parametrize("mode", ["study", "translate", "review"])
def test_one_character_over_limit_removes_oldest_pair(context_case, mode):
    # Arrange
    service, _, _ = context_case
    text = "Продолжи."
    history = [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ"},
        {"role": "user", "content": "Второй вопрос"},
        {"role": "assistant", "content": "Второй ответ"},
    ]
    original = [message.copy() for message in history]
    full_size = (
        len(SYSTEM_PROMPTS[mode]) + len(text) + sum(len(message["content"]) for message in history)
    )
    service.settings = replace(
        service.settings,
        context_max_chars=full_size - 1,
    )

    # Act
    messages = service.build_messages(text, history, mode=mode)

    # Assert
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPTS[mode]},
        *history[2:],
        {"role": "user", "content": text},
    ]
    assert history == original
    assert sum(len(message["content"]) for message in messages) <= full_size - 1


@pytest.mark.parametrize("mode", ["study", "translate", "review"])
def test_history_can_be_removed_entirely_without_truncating_question(context_case, mode):
    # Arrange
    service, _, _ = context_case
    text = "Новый вопрос"
    history = [
        {"role": "user", "content": "Старый вопрос"},
        {"role": "assistant", "content": "Очень длинный ответ" * 1000},
    ]
    service.settings = replace(
        service.settings,
        context_max_chars=len(SYSTEM_PROMPTS[mode]) + len(text),
    )

    # Act
    messages = service.build_messages(text, history, mode=mode)

    # Assert
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPTS[mode]},
        {"role": "user", "content": text},
    ]


@pytest.mark.parametrize("mode", ["study", "translate", "review"])
async def test_question_over_remaining_budget_does_not_call_model(context_case, mode):
    # Arrange
    service, storage, llm = context_case
    storage.get_or_create_user.return_value = {
        "mode": mode,
        "temperature": 0.3,
    }
    service.settings = replace(
        service.settings,
        context_max_chars=len(SYSTEM_PROMPTS[mode]) + 10,
    )
    text = "А" * 11

    # Act
    with pytest.raises(DialogueError) as error:
        await service.reply(101, text)

    # Assert
    assert error.value.user_message == (
        "Вопрос слишком длинный. Сократите его и отправьте ещё раз."
    )
    llm.generate.assert_not_awaited()
    storage.save_turn.assert_not_awaited()


def test_message_and_character_limits_apply_together(context_case):
    # Arrange
    service, _, _ = context_case
    text = "Продолжи."
    history = [
        {"role": "user", "content": "Самый старый вопрос"},
        {"role": "assistant", "content": "Самый старый ответ"},
        {"role": "user", "content": "Большой вопрос" * 100},
        {"role": "assistant", "content": "Большой ответ" * 100},
        {"role": "user", "content": "Новый вопрос"},
        {"role": "assistant", "content": "Новый ответ"},
    ]
    latest_pair = history[-2:]
    limit = (
        len(SYSTEM_PROMPTS["study"])
        + len(text)
        + sum(len(message["content"]) for message in latest_pair)
    )
    service.settings = replace(
        service.settings,
        history_max_messages=4,
        context_max_chars=limit,
    )

    # Act
    messages = service.build_messages(text, history)

    # Assert
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPTS["study"]},
        *latest_pair,
        {"role": "user", "content": text},
    ]
