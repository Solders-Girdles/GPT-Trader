"""Tests for BotConfig.from_env with RISK_* prefix support."""

from __future__ import annotations

import os
from decimal import Decimal
from unittest import mock


class TestBotConfigEnvAliasing:
    """Tests for RISK_* prefixed environment variable support."""

    def test_risk_max_leverage_with_prefix(self) -> None:
        """Test RISK_MAX_LEVERAGE is read correctly."""
        with mock.patch.dict(os.environ, {"RISK_MAX_LEVERAGE": "7"}, clear=False):
            from gpt_trader.app.config.bot_config import BotConfig

            config = BotConfig.from_env()
            assert config.risk.max_leverage == 7

    def test_risk_position_fraction_from_pct_per_symbol(self) -> None:
        """Test RISK_MAX_POSITION_PCT_PER_SYMBOL maps to position_fraction."""
        with mock.patch.dict(os.environ, {"RISK_MAX_POSITION_PCT_PER_SYMBOL": "0.15"}, clear=False):
            from gpt_trader.app.config.bot_config import BotConfig

            config = BotConfig.from_env()
            assert config.risk.position_fraction == Decimal("0.15")

    def test_risk_daily_loss_limit_pct(self) -> None:
        """Test RISK_DAILY_LOSS_LIMIT_PCT is read correctly."""
        with mock.patch.dict(os.environ, {"RISK_DAILY_LOSS_LIMIT_PCT": "0.08"}, clear=False):
            from gpt_trader.app.config.bot_config import BotConfig

            config = BotConfig.from_env()
            assert config.risk.daily_loss_limit_pct == 0.08

    def test_trading_symbols_alias(self) -> None:
        """Test TRADING_SYMBOLS is read correctly."""
        with mock.patch.dict(os.environ, {"TRADING_SYMBOLS": "BTC-USD,SOL-USD"}, clear=False):
            from gpt_trader.app.config.bot_config import BotConfig

            config = BotConfig.from_env()
            assert config.symbols == ["BTC-USD", "SOL-USD"]
