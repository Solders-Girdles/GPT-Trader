"""Tests for normalize_symbols in gpt_trader.features.live_trade.symbols."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from gpt_trader.config.types import Profile
from gpt_trader.features.live_trade import symbols
from tests.unit.gpt_trader.features.live_trade.symbols_test_helpers import make_bot_config


def test_normalize_symbols_integration() -> None:
    """Test normalize_symbols function with logging integration."""
    config = make_bot_config(
        coinbase_default_quote="EUR",
        derivatives_enabled=True,
    )

    # Mock logger to capture log calls
    with pytest.MonkeyPatch().context() as m:
        mock_log = Mock()
        m.setattr(symbols.logger, "log", mock_log)

        symbols_list = ["btc-perp", "invalid-perp"]
        result, derivatives_enabled = symbols.normalize_symbols(
            Profile.PROD, symbols_list, config=config
        )

        assert derivatives_enabled is True
        assert "BTC-PERP" in result
        assert "INVALID-PERP" not in result

        # Verify logger was called
        assert mock_log.call_count == 1  # Warning for invalid perp


def test_normalize_symbols_custom_quote_and_config() -> None:
    """Test normalize_symbols with custom quote and config."""
    config = make_bot_config(
        coinbase_default_quote="USD",
        derivatives_enabled=False,
    )

    result, derivatives_enabled = symbols.normalize_symbols(
        Profile.SPOT,
        ["BTC-PERP"],
        quote="JPY",
        config=config,  # Custom quote override
    )

    assert derivatives_enabled is False  # SPOT profile
    assert result == ["BTC-JPY"]  # Custom quote used
