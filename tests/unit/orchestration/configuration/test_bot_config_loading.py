from decimal import Decimal
from pathlib import Path

import pytest

from gpt_trader.orchestration.configuration.bot_config.bot_config import BotConfig
from gpt_trader.orchestration.configuration.bot_config.validator import validate_config

# Path to the actual paper profile
PAPER_PROFILE_PATH = Path("config/profiles/paper.yaml")


def test_load_paper_profile():
    """Test loading the actual paper.yaml profile."""
    if not PAPER_PROFILE_PATH.exists():
        pytest.skip(f"Profile not found at {PAPER_PROFILE_PATH}")

    config = BotConfig.from_yaml(PAPER_PROFILE_PATH)

    # Verify top-level mappings
    assert config.perps_paper_trading is True
    assert "BTC-USD" in config.symbols

    # Verify Risk mappings
    # paper.yaml: max_position_pct: 0.10
    assert config.risk.position_fraction == Decimal("0.10")

    # Verify Strategy mappings
    # paper.yaml: short_window: 5, long_window: 20
    assert config.strategy.short_ma_period == 5
    assert config.strategy.long_ma_period == 20


def test_validate_paper_profile():
    """Test validating the loaded paper profile."""
    if not PAPER_PROFILE_PATH.exists():
        pytest.skip(f"Profile not found at {PAPER_PROFILE_PATH}")

    config = BotConfig.from_yaml(PAPER_PROFILE_PATH)
    errors = validate_config(config)

    assert not errors, f"Validation failed with errors: {errors}"


def test_validation_errors():
    """Test that the validator catches errors."""
    config = BotConfig()
    config.risk.position_fraction = Decimal("1.5")  # Invalid > 1
    config.strategy.short_ma_period = 20
    config.strategy.long_ma_period = 10  # Invalid short >= long

    errors = validate_config(config)
    assert len(errors) >= 2
    assert any("position_fraction" in e for e in errors)
    assert any("short_ma_period" in e for e in errors)
