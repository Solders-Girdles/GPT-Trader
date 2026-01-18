"""Tests for derivatives_enabled helpers in gpt_trader.features.live_trade.symbols."""

from __future__ import annotations

from unittest.mock import Mock

from gpt_trader.config.types import Profile
from gpt_trader.features.live_trade import symbols
from tests.unit.gpt_trader.features.live_trade.symbols_test_helpers import make_bot_config


def test_derivatives_enabled_with_spot_profile() -> None:
    """Test derivatives_enabled returns False for SPOT profile."""
    # Test with actual Profile.SPOT enum
    config = make_bot_config(derivatives_enabled=False)
    result = symbols.derivatives_enabled(Profile.SPOT, config=config)
    assert result is False

    # Test with string profile value
    string_profile = Mock()
    string_profile.value = "spot"
    result = symbols.derivatives_enabled(string_profile, config=config)
    assert result is False

    # Test with plain string
    result = symbols.derivatives_enabled("spot", config=config)
    assert result is False


def test_derivatives_enabled_with_config_override() -> None:
    """Test derivatives_enabled respects config setting."""
    # Test when config enables derivatives
    config = make_bot_config(derivatives_enabled=True)
    result = symbols.derivatives_enabled(Profile.PROD, config=config)  # Use PROD, not SPOT
    assert result is True  # Config override should take effect

    # Test when config disables derivatives
    config = make_bot_config(derivatives_enabled=False)
    result = symbols.derivatives_enabled(Profile.PROD, config=config)
    assert result is False  # Config override should take effect


def test_derivatives_enabled_edge_cases() -> None:
    """Test derivatives_enabled with edge cases and import failures."""
    config = make_bot_config(derivatives_enabled=False)

    # Test with None profile
    result = symbols.derivatives_enabled(None, config=config)
    assert result is False  # Disabled in config

    # Test with profile object without value attribute
    profile_no_value = Mock()
    del profile_no_value.value
    result = symbols.derivatives_enabled(profile_no_value, config=config)
    assert result is False  # Disabled in config
