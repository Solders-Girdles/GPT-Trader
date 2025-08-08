from bot.config import settings


def test_settings_defaults() -> None:
    assert settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
