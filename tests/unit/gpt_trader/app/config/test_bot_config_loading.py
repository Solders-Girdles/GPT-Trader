from decimal import Decimal
from pathlib import Path

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.config.validation import validate_config

# Path to the actual paper profile
PAPER_PROFILE_PATH = Path("config/profiles/paper.yaml")


def test_load_paper_profile():
    """Test loading the actual paper.yaml profile."""
    if not PAPER_PROFILE_PATH.exists():
        pytest.skip(f"Profile not found at {PAPER_PROFILE_PATH}")

    with pytest.warns(DeprecationWarning, match=r"Legacy profile-style YAML mapping"):
        config = BotConfig.from_yaml(PAPER_PROFILE_PATH)

    # Verify top-level mappings
    assert "BTC-USD" in config.symbols
    assert config.dry_run is True
    assert config.mock_broker is True

    # Verify Risk mappings
    # paper.yaml: position_fraction: 0.10
    assert config.risk.position_fraction == Decimal("0.10")

    # Verify Strategy mappings
    # paper.yaml: short_ma_period: 10, long_ma_period: 50
    assert config.strategy.short_ma_period == 10
    assert config.strategy.long_ma_period == 50


def test_validate_paper_profile():
    """Test validating the loaded paper profile."""
    if not PAPER_PROFILE_PATH.exists():
        pytest.skip(f"Profile not found at {PAPER_PROFILE_PATH}")

    with pytest.warns(DeprecationWarning, match=r"Legacy profile-style YAML mapping"):
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
