"""
Tests for initialization in StrategyOrchestrator.

The configuration/initialization tests for spot vs. perps strategy setup.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator

from .conftest import (
    test_balance,
    test_position,
    test_product,
)


class TestInitialization:
    """Test initialization functionality."""

    def test_orchestrator_initialization_with_perps_bot(
        self, fake_perps_bot, fake_spot_profile_service
    ):
        """Test StrategyOrchestrator initialization with perps bot."""
        orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service
        )

        # Verify basic attributes
        assert orchestrator._bot == fake_perps_bot
        assert orchestrator._spot_profiles == fake_spot_profile_service

        # Verify bot configuration is preserved
        assert fake_perps_bot.config.profile == Profile.PROD
        assert fake_perps_bot.config.derivatives_enabled is True

    def test_orchestrator_initialization_with_spot_bot(
        self, fake_perps_bot, fake_spot_profile_service
    ):
        """Test StrategyOrchestrator initialization with spot bot."""
        # Configure bot for spot
        fake_perps_bot.config.profile = Profile.SPOT
        fake_perps_bot.config.derivatives_enabled = False

        orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service
        )

        # Verify basic attributes
        assert orchestrator._bot == fake_perps_bot
        assert orchestrator._spot_profiles == fake_spot_profile_service

        # Verify spot configuration is preserved
        assert fake_perps_bot.config.profile == Profile.SPOT
        assert fake_perps_bot.config.derivatives_enabled is False

    def test_orchestrator_initialization_preserves_bot_config(
        self, fake_perps_bot, fake_spot_profile_service
    ):
        """Test that orchestrator initialization preserves all bot config."""
        # Set specific config values
        fake_perps_bot.config.short_ma = 5
        fake_perps_bot.config.long_ma = 15
        fake_perps_bot.config.target_leverage = 3
        fake_perps_bot.config.trailing_stop_pct = Decimal("0.03")
        fake_perps_bot.config.enable_shorts = False
        fake_perps_bot.config.symbols = ["ETH-PERP"]

        orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service
        )

        # Verify config values are preserved
        assert fake_perps_bot.config.short_ma == 5
        assert fake_perps_bot.config.long_ma == 15
        assert fake_perps_bot.config.target_leverage == 3
        assert fake_perps_bot.config.trailing_stop_pct == Decimal("0.03")
        assert fake_perps_bot.config.enable_shorts is False
        assert fake_perps_bot.config.symbols == ["ETH-PERP"]
        assert orchestrator._bot.config.symbols == ["ETH-PERP"]

    def test_orchestrator_initialization_with_custom_spot_service(self, fake_perps_bot):
        """Test StrategyOrchestrator initialization with custom spot service."""
        custom_spot_service = Mock()
        custom_spot_service.load = Mock(return_value={"custom": "config"})

        orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=custom_spot_service
        )

        # Verify custom service is used
        assert orchestrator._spot_profiles == custom_spot_service

    def test_orchestrator_initialization_preserves_runtime_state(
        self, fake_perps_bot, fake_spot_profile_service
    ):
        """Test that orchestrator initialization preserves runtime state."""
        # Set some runtime state
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 10
        fake_perps_bot.runtime_state.last_decisions["ETH-PERP"] = Mock()

        orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service
        )

        # Verify runtime state is preserved
        assert "BTC-PERP" in fake_perps_bot.runtime_state.mark_windows
        assert "ETH-PERP" in fake_perps_bot.runtime_state.last_decisions
        assert orchestrator._bot.runtime_state is fake_perps_bot.runtime_state

    def test_orchestrator_initialization_preserves_broker_methods(
        self, fake_perps_bot, fake_spot_profile_service
    ):
        """Test that orchestrator initialization preserves broker methods."""
        # Set up broker methods
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.get_product.return_value = test_product

        orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service
        )

        # Verify broker methods are preserved
        assert fake_perps_bot.broker.list_balances is not None
        assert fake_perps_bot.broker.list_positions is not None
        assert fake_perps_bot.get_product is not None
        assert orchestrator._bot.broker is fake_perps_bot.broker

    def test_orchestrator_initialization_preserves_risk_manager(
        self, fake_perps_bot, fake_spot_profile_service
    ):
        """Test that orchestrator initialization preserves risk manager."""
        # Set up risk manager
        fake_perps_bot.risk_manager.config.kill_switch_enabled = True

        orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service
        )

        # Verify risk manager is preserved
        assert fake_perps_bot.risk_manager.config.kill_switch_enabled is True
        assert fake_perps_bot.risk_manager.check_volatility_circuit_breaker is not None
        assert fake_perps_bot.risk_manager.check_mark_staleness is not None
        assert orchestrator._bot.risk_manager is fake_perps_bot.risk_manager

    def test_orchestrator_initialization_preserves_execution_methods(
        self, fake_perps_bot, fake_spot_profile_service
    ):
        """Test that orchestrator initialization preserves execution methods."""
        # Set up execution method
        fake_perps_bot.execute_decision = Mock(return_value="executed")

        orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service
        )

        # Verify execution method is preserved
        assert fake_perps_bot.execute_decision is not None
        assert orchestrator._bot.execute_decision is fake_perps_bot.execute_decision

    @pytest.mark.asyncio
    async def test_orchestrator_spot_vs_perps_strategy_setup(
        self, fake_perps_bot, fake_spot_profile_service
    ):
        """Test that orchestrator correctly handles spot vs perps strategy setup."""
        # Test with perps config
        fake_perps_bot.config.profile = Profile.PROD
        fake_perps_bot.config.derivatives_enabled = True

        perps_orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service
        )
        assert perps_orchestrator._bot.config.derivatives_enabled is True

        # Change to spot config
        fake_perps_bot.config.profile = Profile.SPOT
        fake_perps_bot.config.derivatives_enabled = False

        spot_orchestrator = StrategyOrchestrator(
            bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service
        )
        assert spot_orchestrator._bot.config.derivatives_enabled is False
