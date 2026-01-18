"""Tests for derivatives mode helpers and constants in gpt_trader.features.live_trade.symbols."""

from __future__ import annotations

import logging

from gpt_trader.config.types import Profile
from gpt_trader.features.live_trade import symbols
from tests.unit.gpt_trader.features.live_trade.symbols_test_helpers import (
    make_bot_config_extended,
)


def test_symbol_normalization_log_dataclass() -> None:
    """Test SymbolNormalizationLog dataclass."""
    log = symbols.SymbolNormalizationLog(
        level=logging.WARNING, message="Test message %s", args=("test_arg",)
    )

    assert log.level == logging.WARNING
    assert log.message == "Test message %s"
    assert log.args == ("test_arg",)

    # Test with default args
    log_default = symbols.SymbolNormalizationLog(level=logging.INFO, message="Test message")

    assert log_default.args == ()


def test_perps_allowlist_constant() -> None:
    """Test PERPS_ALLOWLIST constant contains expected symbols."""
    expected_symbols = {"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"}
    assert symbols.PERPS_ALLOWLIST == expected_symbols
    assert isinstance(symbols.PERPS_ALLOWLIST, frozenset)


def test_us_futures_allowlist_constant() -> None:
    """Test US_FUTURES_ALLOWLIST constant contains expected symbols."""
    expected_symbols = {"BTC-FUTURES", "ETH-FUTURES", "SOL-FUTURES", "XRP-FUTURES"}
    assert symbols.US_FUTURES_ALLOWLIST == expected_symbols


class TestUsFuturesEnabled:
    """Tests for us_futures_enabled function."""

    def test_returns_false_when_derivatives_disabled(self) -> None:
        config = make_bot_config_extended(derivatives_enabled=False)
        result = symbols.us_futures_enabled(Profile.PROD, config=config)
        assert result is False

    def test_returns_true_when_us_futures_flag_enabled(self) -> None:
        config = make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_us_futures_enabled=True,
        )
        result = symbols.us_futures_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_true_when_derivatives_type_is_us_futures(self) -> None:
        config = make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_derivatives_type="us_futures",
        )
        result = symbols.us_futures_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_false_by_default(self) -> None:
        config = make_bot_config_extended(derivatives_enabled=True)
        result = symbols.us_futures_enabled(Profile.PROD, config=config)
        assert result is False


class TestIntxPerpetualsEnabled:
    """Tests for intx_perpetuals_enabled function."""

    def test_returns_false_when_derivatives_disabled(self) -> None:
        config = make_bot_config_extended(derivatives_enabled=False)
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is False

    def test_returns_true_when_intx_flag_enabled(self) -> None:
        config = make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_intx_perpetuals_enabled=True,
        )
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_true_when_derivatives_type_is_intx_perps(self) -> None:
        config = make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_derivatives_type="intx_perps",
        )
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_true_when_derivatives_type_is_perpetuals(self) -> None:
        config = make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_derivatives_type="perpetuals",
        )
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_true_by_default(self) -> None:
        config = make_bot_config_extended(derivatives_enabled=True)
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is True
