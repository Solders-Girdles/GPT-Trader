"""Tests for BotConfig.from_env with RISK_* prefix support."""

from __future__ import annotations

from decimal import Decimal

import pytest


class TestBotConfigEnvAliasing:
    """Tests for RISK_* prefixed environment variable support."""

    def test_risk_max_leverage_with_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test RISK_MAX_LEVERAGE is read correctly."""
        monkeypatch.setenv("RISK_MAX_LEVERAGE", "7")
        from gpt_trader.app.config.bot_config import BotConfig

        config = BotConfig.from_env()
        assert config.risk.max_leverage == 7

    def test_risk_position_fraction_from_pct_per_symbol(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test RISK_MAX_POSITION_PCT_PER_SYMBOL maps to position_fraction."""
        monkeypatch.setenv("RISK_MAX_POSITION_PCT_PER_SYMBOL", "0.15")
        from gpt_trader.app.config.bot_config import BotConfig

        config = BotConfig.from_env()
        assert config.risk.position_fraction == Decimal("0.15")

    def test_risk_daily_loss_limit_pct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test RISK_DAILY_LOSS_LIMIT_PCT is read correctly."""
        monkeypatch.setenv("RISK_DAILY_LOSS_LIMIT_PCT", "0.08")
        from gpt_trader.app.config.bot_config import BotConfig

        config = BotConfig.from_env()
        assert config.risk.daily_loss_limit_pct == 0.08

    def test_trading_symbols_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test TRADING_SYMBOLS is read correctly."""
        monkeypatch.setenv("TRADING_SYMBOLS", "BTC-USD,SOL-USD")
        from gpt_trader.app.config.bot_config import BotConfig

        config = BotConfig.from_env()
        assert config.symbols == ["BTC-USD", "SOL-USD"]
