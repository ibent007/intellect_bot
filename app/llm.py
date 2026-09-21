import logging
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

import aiohttp

from app.config import LLMSettings

logger = logging.getLogger("app.llm")

ERROR_MESSAGES = {
    "timeout": "Модель не успела ответить. Попробуйте ещё раз чуть позже.",
    "unavailable": "Сервис ИИ временно недоступен. Попробуйте позже.",
    "authorization": "Не удалось подключиться к сервису ИИ. Сообщите администратору.",
    "rate_limit": "Достигнут лимит запросов к ИИ. Попробуйте позже.",
    "invalid_response": "Модель вернула некорректный ответ. Попробуйте ещё раз.",
    "empty_response": "Модель вернула пустой ответ. Попробуйте переформулировать вопрос.",
    "incomplete_response": "Ответ модели не завершён. Попробуйте задать более узкий вопрос.",
    "invalid_temperature": "Допустимые значения temperature: 0.0, 0.3, 0.7, 1.0.",
}


class LLMError(Exception):
    """Ошибка с безопасным текстом для пользователя."""

    def __init__(self, code: str):
        self.code = code
        self.user_message = ERROR_MESSAGES[code]
        super().__init__(self.user_message)


@dataclass(frozen=True)
class LLMResponse:
    text: str = field(repr=False)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient:
    def __init__(self, settings: LLMSettings, session: aiohttp.ClientSession):
        self.settings = settings
        self.session = session

    async def generate(self, messages: list[dict[str, str]], temperature: float) -> LLMResponse:
        if isinstance(temperature, bool) or temperature not in (0.0, 0.3, 0.7, 1.0):
            raise LLMError("invalid_temperature")

        request_id = uuid4().hex
        started = perf_counter()

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "reasoning_effort": self.settings.reasoning_effort,
            "include_reasoning": False,
            "max_completion_tokens": self.settings.max_completion_tokens,
        }

        logger.info(
            "Начало вызова LLM: request_id=%s model=%s",
            request_id,
            self.settings.model,
        )

        try:
            data = await self._request(payload)
            result = self._parse_response(data)
        except LLMError as error:
            logger.warning(
                "Ошибка LLM: request_id=%s model=%s code=%s duration_ms=%.0f",
                request_id,
                self.settings.model,
                error.code,
                (perf_counter() - started) * 1000,
            )
            raise

        logger.info(
            "Успешный вызов LLM: request_id=%s model=%s duration_ms=%.0f",
            request_id,
            self.settings.model,
            (perf_counter() - started) * 1000,
        )
        return result

    async def _request(self, payload: dict) -> object:
        try:
            async with self.session.post(
                f"{self.settings.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                },
                timeout=aiohttp.ClientTimeout(total=self.settings.timeout_seconds),
                allow_redirects=False,
            ) as response:
                if response.status in (401, 403):
                    raise LLMError("authorization")
                if response.status == 429:
                    raise LLMError("rate_limit")
                if response.status != 200:
                    raise LLMError("unavailable")

                return await response.json()

        except TimeoutError:
            raise LLMError("timeout") from None
        except aiohttp.ContentTypeError:
            raise LLMError("invalid_response") from None
        except aiohttp.ClientError:
            raise LLMError("unavailable") from None
        except ValueError:
            raise LLMError("invalid_response") from None

    @staticmethod
    def _parse_response(data: object) -> LLMResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message["content"]
            finish_reason = choice["finish_reason"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("invalid_response") from None

        if finish_reason == "length":
            raise LLMError("incomplete_response")
        if finish_reason != "stop":
            raise LLMError("invalid_response")
        if text is None or (isinstance(text, str) and not text.strip()):
            raise LLMError("empty_response")
        if not isinstance(text, str):
            raise LLMError("invalid_response")

        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        def token_count(name: str) -> int | None:
            value = usage.get(name)
            if type(value) is int and value >= 0:
                return value
            return None

        return LLMResponse(
            text=text,
            prompt_tokens=token_count("prompt_tokens"),
            completion_tokens=token_count("completion_tokens"),
        )
