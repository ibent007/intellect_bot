import logging
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from app.config import LLMSettings
from app.llm import LLMClient, LLMError


@pytest.fixture
def llm_case():
    settings = LLMSettings(
        base_url="https://api.groq.com/openai/v1",
        api_key="fake-key-for-tests",
        model="openai/gpt-oss-120b",
    )

    data = {
        "choices": [
            {
                "message": {"content": "Ответ модели"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
        },
    }

    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value=data)

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value = context

    client = LLMClient(settings, session)
    return client, session, response, context, data


@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.7, 1.0])
async def test_request_preserves_messages_and_parameters(llm_case, temperature):
    # Arrange
    client, session, _, _, _ = llm_case
    messages = [
        {"role": "system", "content": "Объясняй программирование."},
        {"role": "user", "content": "Что такое переменная?"},
        {"role": "assistant", "content": "Это имя, связанное со значением."},
        {"role": "user", "content": "Приведи пример."},
    ]

    # Act
    await client.generate(messages, temperature)

    # Assert
    session.post.assert_called_once()
    arguments = session.post.call_args
    assert arguments.args[0] == ("https://api.groq.com/openai/v1/chat/completions")
    assert arguments.kwargs["json"] == {
        "model": "openai/gpt-oss-120b",
        "messages": messages,
        "temperature": temperature,
        "reasoning_effort": "low",
        "include_reasoning": False,
        "max_completion_tokens": 2048,
    }
    assert arguments.kwargs["headers"] == {"Authorization": "Bearer fake-key-for-tests"}
    assert arguments.kwargs["timeout"].total == 60
    assert arguments.kwargs["allow_redirects"] is False


async def test_success_returns_text_and_usage(llm_case):
    # Arrange
    client, _, _, _, _ = llm_case

    # Act
    result = await client.generate([{"role": "user", "content": "Вопрос"}], 0.3)

    # Assert
    assert result.text == "Ответ модели"
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 10


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": "20"}])
async def test_missing_or_invalid_usage_does_not_discard_answer(llm_case, usage):
    # Arrange
    client, _, _, _, data = llm_case
    data["usage"] = usage

    # Act
    result = await client.generate([{"role": "user", "content": "Вопрос"}], 0.3)

    # Assert
    assert result.text == "Ответ модели"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None


@pytest.mark.parametrize("temperature", [-1.0, 0.5, 2.0, True, "0.3"])
async def test_invalid_temperature_does_not_call_api(llm_case, temperature):
    # Arrange
    client, session, _, _, _ = llm_case

    # Act
    with pytest.raises(LLMError) as error:
        await client.generate([{"role": "user", "content": "Вопрос"}], temperature)

    # Assert
    assert str(error.value) == ("Допустимые значения temperature: 0.0, 0.3, 0.7, 1.0.")
    session.post.assert_not_called()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "Не удалось подключиться к сервису ИИ. Сообщите администратору."),
        (403, "Не удалось подключиться к сервису ИИ. Сообщите администратору."),
        (429, "Достигнут лимит запросов к ИИ. Попробуйте позже."),
        (500, "Сервис ИИ временно недоступен. Попробуйте позже."),
        (503, "Сервис ИИ временно недоступен. Попробуйте позже."),
    ],
)
async def test_http_error_has_safe_message(llm_case, status, expected):
    # Arrange
    client, _, response, _, _ = llm_case
    response.status = status

    # Act
    with pytest.raises(LLMError) as error:
        await client.generate([{"role": "user", "content": "Вопрос"}], 0.3)

    # Assert
    assert str(error.value) == expected
    response.json.assert_not_awaited()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            TimeoutError("private details"),
            "Модель не успела ответить. Попробуйте ещё раз чуть позже.",
        ),
        (
            aiohttp.ClientConnectionError("private details"),
            "Сервис ИИ временно недоступен. Попробуйте позже.",
        ),
    ],
)
async def test_network_failure_has_safe_message(llm_case, failure, expected):
    # Arrange
    client, _, _, context, _ = llm_case
    context.__aenter__.side_effect = failure

    # Act
    with pytest.raises(LLMError) as error:
        await client.generate([{"role": "user", "content": "Вопрос"}], 0.3)

    # Assert
    assert str(error.value) == expected
    assert "private details" not in str(error.value)


@pytest.mark.parametrize("text", ["", " \n ", None])
async def test_empty_answer_is_rejected(llm_case, text):
    # Arrange
    client, _, _, _, data = llm_case
    data["choices"][0]["message"]["content"] = text

    # Act
    with pytest.raises(LLMError) as error:
        await client.generate([{"role": "user", "content": "Вопрос"}], 0.3)

    # Assert
    assert str(error.value) == ("Модель вернула пустой ответ. Попробуйте переформулировать вопрос.")


async def test_incomplete_answer_is_rejected(llm_case):
    # Arrange
    client, _, _, _, data = llm_case
    data["choices"][0]["finish_reason"] = "length"

    # Act
    with pytest.raises(LLMError) as error:
        await client.generate([{"role": "user", "content": "Вопрос"}], 0.3)

    # Assert
    assert str(error.value) == ("Ответ модели не завершён. Попробуйте задать более узкий вопрос.")


@pytest.mark.parametrize("data", [None, [], {}, {"choices": []}])
async def test_invalid_response_structure_is_rejected(llm_case, data):
    # Arrange
    client, _, response, _, _ = llm_case
    response.json.return_value = data

    # Act
    with pytest.raises(LLMError) as error:
        await client.generate([{"role": "user", "content": "Вопрос"}], 0.3)

    # Assert
    assert str(error.value) == ("Модель вернула некорректный ответ. Попробуйте ещё раз.")


async def test_invalid_json_is_rejected(llm_case):
    # Arrange
    client, _, response, _, _ = llm_case
    response.json.side_effect = ValueError("private response body")

    # Act
    with pytest.raises(LLMError) as error:
        await client.generate([{"role": "user", "content": "Вопрос"}], 0.3)

    # Assert
    assert str(error.value) == ("Модель вернула некорректный ответ. Попробуйте ещё раз.")


async def test_error_logs_do_not_contain_sensitive_data(llm_case, caplog):
    # Arrange
    client, _, _, context, _ = llm_case
    context.__aenter__.side_effect = aiohttp.ClientConnectionError(
        "fake-key-for-tests private-user-text"
    )
    caplog.set_level(logging.INFO, logger="app.llm")

    # Act
    with pytest.raises(LLMError):
        await client.generate([{"role": "user", "content": "private-user-text"}], 0.3)

    # Assert
    assert "Начало вызова LLM" in caplog.text
    assert "Ошибка LLM" in caplog.text
    assert "request_id=" in caplog.text
    assert "model=openai/gpt-oss-120b" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "fake-key-for-tests" not in caplog.text
    assert "private-user-text" not in caplog.text


@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.7, 1.0])
async def test_qwen_request_preserves_messages_and_parameters(llm_case, temperature):
    # Arrange
    original, session, _, _, _ = llm_case
    settings = replace(
        original.settings,
        model="qwen/qwen3.8-27b",
        max_completion_tokens=4096,
    )
    client = LLMClient(settings, session)
    messages = [
        {"role": "system", "content": "Объясняй программирование."},
        {"role": "user", "content": "Что такое переменная?"},
        {"role": "assistant", "content": "Это имя, связанное со значением."},
        {"role": "user", "content": "Приведи пример."},
    ]

    # Act
    result = await client.generate(messages, temperature)

    # Assert
    session.post.assert_called_once()
    assert session.post.call_args.kwargs["json"] == {
        "model": "qwen/qwen3.8-27b",
        "messages": messages,
        "temperature": temperature,
        "reasoning_effort": "low",
        "max_completion_tokens": 4096,
        "reasoning_format": "hidden",
    }
    assert result.text == "Ответ модели"
