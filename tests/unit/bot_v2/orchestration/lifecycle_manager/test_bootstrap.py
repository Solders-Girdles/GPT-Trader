"""
Bootstrap tests for lifecycle_manager.py.

Tests bootstrap initialization including:
- Coordinator initialization success/failure
- Broker and risk manager assignment
- Strategy orchestrator initialization
- Extras assignment from registry
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_v2.orchestration.lifecycle_manager import LifecycleManager


class TestLifecycleManagerBootstrap:
    """Test bootstrap method in LifecycleManager."""

    def test_bootstrap_success(
        self, fake_bot, fake_coordinator_registry, fake_strategy_orchestrator
    ):
        """Test bootstrap initializes coordinators successfully."""
        # Setup
        updated_context = SimpleNamespace(
            registry=SimpleNamespace(extras={}),
            broker=None,
            risk_manager=None,
        )
        fake_coordinator_registry.initialize_all.return_value = updated_context

        manager = LifecycleManager(fake_bot)

        # Execute
        manager.bootstrap()

        # Verify
        fake_coordinator_registry.initialize_all.assert_called_once()
        fake_strategy_orchestrator.init_strategy.assert_called_once()
        assert fake_bot._coordinator_context == updated_context

    def test_bootstrap_coordinator_init_failure(self, fake_bot, fake_coordinator_registry):
        """Test bootstrap handles coordinator initialization failures."""
        fake_coordinator_registry.initialize_all.side_effect = Exception("Init failed")

        manager = LifecycleManager(fake_bot)

        with pytest.raises(Exception, match="Init failed"):
            manager.bootstrap()

    def test_bootstrap_broker_assignment_failure(self, fake_bot, fake_coordinator_registry):
        """Test bootstrap handles broker assignment failures."""
        updated_context = SimpleNamespace(
            registry=SimpleNamespace(extras={}),
            broker=Mock(),
            risk_manager=None,
        )
        fake_coordinator_registry.initialize_all.return_value = updated_context
        fake_bot.broker = Mock(side_effect=Exception("Broker assignment failed"))

        manager = LifecycleManager(fake_bot)

        # Should not raise - exception is caught and logged
        manager.bootstrap()

        # Verify broker was attempted to be set
        assert fake_bot.broker == updated_context.broker

    def test_bootstrap_risk_manager_assignment_failure(self, fake_bot, fake_coordinator_registry):
        """Test bootstrap handles risk manager assignment failures."""
        updated_context = SimpleNamespace(
            registry=SimpleNamespace(extras={}),
            broker=None,
            risk_manager=Mock(),
        )
        fake_coordinator_registry.initialize_all.return_value = updated_context
        fake_bot.risk_manager = Mock(side_effect=Exception("Risk manager assignment failed"))

        manager = LifecycleManager(fake_bot)

        # Should not raise - exception is caught and logged
        manager.bootstrap()

        # Verify risk manager was attempted to be set
        assert fake_bot.risk_manager == updated_context.risk_manager

    def test_bootstrap_strategy_init_failure(
        self, fake_bot, fake_coordinator_registry, fake_strategy_orchestrator
    ):
        """Test bootstrap handles strategy orchestrator initialization failures."""
        updated_context = SimpleNamespace(
            registry=SimpleNamespace(extras={}),
            broker=None,
            risk_manager=None,
        )
        fake_coordinator_registry.initialize_all.return_value = updated_context
        fake_strategy_orchestrator.init_strategy.side_effect = Exception("Strategy init failed")

        manager = LifecycleManager(fake_bot)

        # Should not raise - exception is caught and logged
        manager.bootstrap()

        fake_strategy_orchestrator.init_strategy.assert_called_once()

    def test_bootstrap_assigns_extras(self, fake_bot, fake_coordinator_registry):
        """Test bootstrap assigns extras from registry."""
        account_manager = Mock()
        account_telemetry = Mock()
        market_monitor = Mock()

        updated_context = SimpleNamespace(
            registry=SimpleNamespace(
                extras={
                    "account_manager": account_manager,
                    "account_telemetry": account_telemetry,
                    "market_monitor": market_monitor,
                }
            ),
            broker=None,
            risk_manager=None,
        )
        fake_coordinator_registry.initialize_all.return_value = updated_context

        manager = LifecycleManager(fake_bot)

        manager.bootstrap()

        assert fake_bot.account_manager == account_manager
        assert fake_bot.account_telemetry == account_telemetry
        assert fake_bot.market_monitor == market_monitor
        assert fake_bot._market_monitor == market_monitor
