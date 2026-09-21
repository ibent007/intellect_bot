import logging

import pytest

from app.config import ConfigError, LLMSettings, Settings
from app.logging_setup import configure_logging


@pytest.fixture
def llm_environment():
    return {
        "LLM_BASE_URL": "https://api.groq.com/openai/v1/",
        "LLM_API_KEY": "fake-key-for-tests",
        "LLM_MODEL": "openai/gpt-oss-120b",
        "LLM_REASONING_EFFORT": "low",
    }


def test_llm_settings_load_valid_configuration(tmp_path, llm_environment):
    # Arrange
    path = tmp_path / ".env"

    # Act
    settings = LLMSettings.load(path, environ=llm_environment)

    # Assert
    assert settings.base_url == "https://api.groq.com/openai/v1"
    assert settings.model == "openai/gpt-oss-120b"
    assert settings.reasoning_effort == "low"
    assert settings.timeout_seconds == 60
    assert settings.max_completion_tokens == 2048
    assert settings.api_key == "fake-key-for-tests"
    assert settings.api_key not in repr(settings)


@pytest.mark.parametrize("name", ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"])
def test_missing_required_setting_is_rejected(tmp_path, llm_environment, name):
    # Arrange
    del llm_environment[name]

    # Act
    with pytest.raises(ConfigError) as error:
        LLMSettings.load(tmp_path / ".env", environ=llm_environment)

    # Assert
    assert str(error.value) == f"{name}: обязательная настройка не задана."


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_TIMEOUT_SECONDS", "0"),
        ("LLM_TIMEOUT_SECONDS", "abc"),
        ("LLM_MAX_COMPLETION_TOKENS", "-1"),
        ("LLM_MAX_COMPLETION_TOKENS", "1.5"),
    ],
)
def test_invalid_numeric_setting_is_rejected(tmp_path, llm_environment, name, value):
    # Arrange
    llm_environment[name] = value

    # Act
    with pytest.raises(ConfigError) as error:
        LLMSettings.load(tmp_path / ".env", environ=llm_environment)

    # Assert
    assert str(error.value) == f"{name}: нужно положительное целое число."


def test_invalid_reasoning_effort_is_rejected(tmp_path, llm_environment):
    # Arrange
    llm_environment["LLM_REASONING_EFFORT"] = "none"

    # Act
    with pytest.raises(ConfigError) as error:
        LLMSettings.load(tmp_path / ".env", environ=llm_environment)

    # Assert
    assert str(error.value) == ("LLM_REASONING_EFFORT: используйте low, medium или high.")


def test_url_credentials_are_rejected_without_leaking(tmp_path, llm_environment):
    # Arrange
    llm_environment["LLM_BASE_URL"] = "https://user:private-password@example.com"

    # Act
    with pytest.raises(ConfigError) as error:
        LLMSettings.load(tmp_path / ".env", environ=llm_environment)

    # Assert
    assert str(error.value) == (
        "LLM_BASE_URL: нужен HTTPS-адрес API без логина, пароля и параметров."
    )
    assert "private-password" not in str(error.value)


def test_environment_overrides_llm_file_settings(tmp_path, llm_environment):
    # Arrange
    path = tmp_path / ".env"
    path.write_text("LLM_TIMEOUT_SECONDS=15\n", encoding="utf-8")
    llm_environment["LLM_TIMEOUT_SECONDS"] = "30"

    # Act
    settings = LLMSettings.load(path, environ=llm_environment)

    # Assert
    assert settings.timeout_seconds == 30


def test_logging_hides_llm_key(tmp_path, llm_environment, capsys):
    # Arrange
    settings = Settings(bot_token="fake-token", postgres_password="fake-password")
    llm_settings = LLMSettings.load(tmp_path / ".env", environ=llm_environment)
    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    previous_level = root.level

    try:
        configure_logging(settings, llm_settings)

        # Act
        logging.getLogger("app.llm").error("Проверка: %s", llm_settings.api_key)
        output = capsys.readouterr().err

        # Assert
        assert llm_settings.api_key not in output
        assert "Проверка: [скрыто]" in output
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)
