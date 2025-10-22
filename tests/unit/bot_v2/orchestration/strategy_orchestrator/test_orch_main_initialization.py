"""Tests for StrategyOrchestrator initialization, strategy selection, and configuration.

This module tests:
- Orchestrator initialization with/without SpotProfileService  
- Strategy initialization (single perps vs per-symbol spot)
- Position fraction overrides
- Leverage configuration
- Strategy retrieval logic
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.strategies.perps_baseline import BaselinePerpsStrategy
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator


class TestStrategyOrchestratorInitialization:
    """Test StrategyOrchestrator initialization."""

    def test_initialization_with_bot(self, mock_bot, mock_spot_profile_service):
        """Test orchestrator initializes with bot reference."""
        orchestrator = StrategyOrchestrator(
            bot=mock_bot, spot_profile_service=mock_spot_profile_service
        )

        assert orchestrator._bot == mock_bot
        assert orchestrator._spot_profiles == mock_spot_profile_service

    def test_initialization_creates_default_spot_profile_service(self, mock_bot):
        """Test creates default SpotProfileService when not provided."""
        orchestrator = StrategyOrchestrator(bot=mock_bot)

        assert orchestrator._bot == mock_bot
        assert orchestrator._spot_profiles is not None


class TestInitStrategy:
    """Test init_strategy method."""

    def test_initializes_perps_strategy_when_not_spot(self, orchestrator, mock_bot):
        """Test initializes single strategy for non-SPOT profile."""
        mock_bot.config.profile = Profile.PROD

        orchestrator.init_strategy()

        # Should create single strategy on bot.strategy
        assert isinstance(mock_bot.runtime_state.strategy, BaselinePerpsStrategy)
        assert mock_bot.runtime_state.strategy.config.short_ma_period == 10
        assert mock_bot.runtime_state.strategy.config.long_ma_period == 30
        assert mock_bot.runtime_state.strategy.config.target_leverage == 2

    def test_initializes_spot_strategy_per_symbol(
        self, orchestrator, mock_bot, mock_spot_profile_service
    ):
        """Test initializes per-symbol strategies for SPOT profile."""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.config.symbols = ["BTC-PERP", "ETH-PERP"]
        mock_spot_profile_service.load.return_value = {
            "BTC-PERP": {"short_window": 5, "long_window": 15},
            "ETH-PERP": {"short_window": 7, "long_window": 20},
        }

        orchestrator.init_strategy()

        # Should create per-symbol strategies
        symbol_map = mock_bot.runtime_state.symbol_strategies
        assert "BTC-PERP" in symbol_map
        assert "ETH-PERP" in symbol_map

        btc_strat = symbol_map["BTC-PERP"]
        assert btc_strat.config.short_ma_period == 5
        assert btc_strat.config.long_ma_period == 15
        assert btc_strat.config.enable_shorts is False

    def test_applies_position_fraction_override(self, orchestrator, mock_bot):
        """Test applies position fraction from config."""
        mock_bot.config.perps_position_fraction = 0.3

        orchestrator.init_strategy()

        assert mock_bot.runtime_state.strategy.config.position_fraction == 0.3

    def test_disables_leverage_when_derivatives_disabled(self, orchestrator, mock_bot):
        """Test sets leverage to 1 when derivatives disabled."""
        mock_bot.config.derivatives_enabled = False
        mock_bot.config.target_leverage = 5

        orchestrator.init_strategy()

        assert mock_bot.runtime_state.strategy.config.target_leverage == 1
        assert mock_bot.runtime_state.strategy.config.enable_shorts is False


class TestGetStrategy:
    """Test get_strategy method."""

    def test_returns_bot_strategy_for_perps_profile(self, orchestrator, mock_bot):
        """Test returns bot.strategy for non-SPOT profile."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.runtime_state.strategy = Mock(spec=BaselinePerpsStrategy)

        strategy = orchestrator.get_strategy("BTC-PERP")

        assert strategy == mock_bot.runtime_state.strategy

    def test_returns_symbol_strategy_for_spot_profile(self, orchestrator, mock_bot):
        """Test returns symbol-specific strategy for SPOT profile."""
        mock_bot.config.profile = Profile.SPOT
        btc_strategy = Mock(spec=BaselinePerpsStrategy)
        mock_bot.runtime_state.symbol_strategies["BTC-PERP"] = btc_strategy

        strategy = orchestrator.get_strategy("BTC-PERP")

        assert strategy == btc_strategy

    def test_creates_default_strategy_for_unknown_symbol(self, orchestrator, mock_bot):
        """Test creates default strategy when symbol not in map."""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.runtime_state.symbol_strategies.clear()

        strategy = orchestrator.get_strategy("NEW-SYMBOL")

        assert isinstance(strategy, BaselinePerpsStrategy)
        assert "NEW-SYMBOL" in mock_bot.runtime_state.symbol_strategies
