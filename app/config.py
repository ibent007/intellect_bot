"""Единственное место чтения и проверки настроек приложения."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080

    @classmethod
    def load(
        cls, env_file: Path | str = ".env", *, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        proxy = value("TELEGRAM_PROXY_URL")
        if proxy:
            try:
                parsed = urlsplit(proxy)
                if (
                    parsed.scheme not in {"http", "socks5"}
                    or not parsed.hostname
                    or not parsed.port
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    "TELEGRAM_PROXY_URL: нужен http://host:port или socks5://host:port; "
                    "при необходимости добавьте user:password@."
                ) from None
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        return cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
        )


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    reasoning_effort: str = "low"
    timeout_seconds: int = 60
    max_completion_tokens: int = 2048
    history_max_messages: int = 12
    context_max_chars: int = 6000

    @classmethod
    def load(
        cls, env_file: Path | str = ".env", *, environ: Mapping[str, str] | None = None
    ) -> "LLMSettings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def required(name: str) -> str:
            result = (values.get(name) or "").strip()
            if not result:
                raise ConfigError(f"{name}: обязательная настройка не задана.")
            return result

        def positive_integer(name: str, default: str) -> int:
            raw = values.get(name)
            try:
                result = int(default if raw is None else raw)
                if result <= 0:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{name}: нужно положительное целое число.") from None

        base_url = required("LLM_BASE_URL").rstrip("/")
        try:
            parsed = urlsplit(base_url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or any(character.isspace() for character in base_url)
            ):
                raise ValueError
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError
        except ValueError:
            raise ConfigError(
                "LLM_BASE_URL: нужен HTTPS-адрес API без логина, пароля и параметров."
            ) from None

        api_key = required("LLM_API_KEY")
        model = required("LLM_MODEL")
        effort = (values.get("LLM_REASONING_EFFORT") or "low").strip()

        if effort not in {"low", "medium", "high"}:
            raise ConfigError("LLM_REASONING_EFFORT: используйте low, medium или high.")

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            reasoning_effort=effort,
            timeout_seconds=positive_integer("LLM_TIMEOUT_SECONDS", "60"),
            max_completion_tokens=positive_integer("LLM_MAX_COMPLETION_TOKENS", "2048"),
            history_max_messages=positive_integer("LLM_HISTORY_MAX_MESSAGES", "12"),
            context_max_chars=positive_integer("LLM_CONTEXT_MAX_CHARS", "6000"),
        )
