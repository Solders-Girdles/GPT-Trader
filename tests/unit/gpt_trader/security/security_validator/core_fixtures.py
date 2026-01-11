from __future__ import annotations

from typing import Any

import pytest

from gpt_trader.app.config import BotConfig


@pytest.fixture
def validator_bot_config(bot_config_factory) -> BotConfig:
    """Bot config tailored for security_validator tests."""
    return bot_config_factory()


@pytest.fixture
def security_validator() -> Any:
    """SecurityValidator instance for testing."""
    from gpt_trader.security.security_validator import SecurityValidator

    return SecurityValidator()
