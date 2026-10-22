import pytest

from umai.config.settings import Settings


def test_daily_user_limit_is_disabled_when_env_var_is_missing(monkeypatch):
    monkeypatch.delenv("UMAI_DAILY_USER_LIMIT_ENABLED", raising=False)
    monkeypatch.delenv("UMAI_DAILY_USER_LIMIT_REQUESTS", raising=False)

    settings = Settings.from_env()

    assert settings.umai_daily_user_limit_enabled is False
    assert settings.umai_daily_user_limit_requests == 150


def test_daily_user_limit_can_be_enabled_from_env(monkeypatch):
    monkeypatch.setenv("UMAI_DAILY_USER_LIMIT_ENABLED", "true")

    settings = Settings.from_env()

    assert settings.umai_daily_user_limit_enabled is True


def test_log_level_is_normalized_from_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = Settings.from_env()

    assert settings.log_level == "DEBUG"


def test_invalid_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    settings = Settings.from_env()

    with pytest.raises(ValueError, match="LOG_LEVEL must be one of"):
        settings.validate()


def test_archive_options_accept_empty_port_and_explicit_database(monkeypatch):
    monkeypatch.setenv("DB_ARCHIVE_PORT", "")
    monkeypatch.setenv("DB_ARCHIVE_HOST", "archive.internal")
    monkeypatch.setenv("DB_ARCHIVE_DATABASE", "archive_data")
    monkeypatch.setenv("DB_ARCHIVE_USERNAME", "archive_reader")
    monkeypatch.setenv("DB_ARCHIVE_PASSWORD", "")
    settings = Settings.from_env()
    assert settings.db_archive_port is None
    assert settings.db_archive_host == "archive.internal"
    assert settings.db_archive_name == "archive_data"
    assert settings.db_archive_user == "archive_reader"
    assert settings.db_archive_password is None
