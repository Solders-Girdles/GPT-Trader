"""Orchestration fixtures for testing without direct live_trade imports.

This module provides fixtures that expose orchestration components (PerpsBot,
build_bot, etc.) to tests, allowing them to test against the public API rather
than importing internal implementation details from bot_v2.features.live_trade.

Usage:
    # Instead of:
    from bot_v2.features.live_trade.risk import LiveRiskManager

    # Do this:
    def test_something(risk_manager):
        # Use the fixture which provides risk_manager via bot.risk_manager
        ...
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.orchestration import BotConfig, PerpsBot, Profile, build_bot


@pytest.fixture
def test_bot_config() -> BotConfig:
    """Create minimal bot configuration for testing.

    Provides a safe test configuration with:
    - Canary profile (non-production)
    - Single test symbol
    - Conservative risk limits
    - Dry run enabled by default
    """
    return BotConfig(
        profile=Profile.CANARY,
        symbols=["BTC-USD"],
        max_leverage=Decimal("2"),
        daily_loss_limit=Decimal("100"),
        dry_run=True,
        update_interval=60,
        short_ma=10,
        long_ma=20,
    )


@pytest.fixture
def perps_bot(test_bot_config: BotConfig) -> PerpsBot:
    """Create a PerpsBot instance for testing.

    The bot is fully initialized with all orchestration components but
    configured for safe testing (dry run, canary profile).

    Access components via properties:
    - bot.risk_manager -> LiveRiskManager
    - bot.broker -> IBrokerage
    - bot.exec_engine -> AdvancedExecutionEngine
    """
    bot, _ = build_bot(test_bot_config)
    return bot


@pytest.fixture
def risk_manager(perps_bot: PerpsBot):
    """Provide LiveRiskManager via bot property instead of direct import.

    This replaces:
        from bot_v2.features.live_trade.risk import LiveRiskManager
        risk = LiveRiskManager(config)

    With:
        def test_something(risk_manager):
            # risk_manager is already configured via bot
            ...
    """
    return perps_bot.risk_manager


@pytest.fixture
def execution_engine(perps_bot: PerpsBot):
    """Provide execution engine via bot property.

    This replaces:
        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

    With:
        def test_something(execution_engine):
            # engine is configured and ready
            ...
    """
    return perps_bot.exec_engine


@pytest.fixture
def mock_broker():
    """Create a mock broker for testing without live connections.

    Provides all IBrokerage interface methods with sensible defaults.
    Configure specific behaviors in individual tests.
    """
    broker = Mock()
    broker.list_balances.return_value = []
    broker.list_positions.return_value = []
    broker.get_quote.return_value = Mock(last=Decimal("50000"), ts=None)
    broker.place_order.return_value = Mock(
        order_id="test-order-123",
        status="open",
        filled_size=Decimal("0"),
    )
    return broker


# Strategy type fixtures - for tests that need sample strategy objects


@pytest.fixture
def sample_action():
    """Sample Action enum value for strategy tests.

    Provides a concrete Action without importing from live_trade.strategies.
    """
    # Import locally to avoid module-level dependency
    from bot_v2.features.live_trade.strategies.perps_baseline import Action

    return Action.BUY


@pytest.fixture
def sample_decision():
    """Sample Decision object for strategy tests.

    Provides a minimal Decision for testing execution logic.
    """
    # Import locally to avoid module-level dependency
    from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

    return Decision(
        action=Action.BUY,
        symbol="BTC-USD",
        size=Decimal("0.01"),
        reasoning="Test decision",
        confidence=Decimal("0.8"),
    )


# Error type fixtures - for tests that need exception types


@pytest.fixture
def validation_error_class():
    """Provide ValidationError class without direct import.

    Use when you need to catch/raise ValidationError in tests:

        def test_validation(validation_error_class):
            with pytest.raises(validation_error_class):
                # test code that should raise
                ...
    """
    from bot_v2.features.live_trade.risk import ValidationError

    return ValidationError


@pytest.fixture
def guard_error_classes():
    """Provide guard error classes for exception testing.

    Returns a dict of error classes:
        - RiskGuardError
        - OrderValidationError
        - etc.
    """
    from bot_v2.features.live_trade.guard_errors import (
        OrderValidationError,
        RiskGuardError,
    )

    return {
        "RiskGuardError": RiskGuardError,
        "OrderValidationError": OrderValidationError,
    }


# Runtime state fixtures


@pytest.fixture
def circuit_breaker_action():
    """Provide CircuitBreakerAction enum without direct import."""
    from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction

    return CircuitBreakerAction


@pytest.fixture
def sample_risk_state():
    """Provide a sample RiskRuntimeState for testing."""
    from bot_v2.features.live_trade.risk import RiskRuntimeState

    return RiskRuntimeState(
        total_exposure=Decimal("1000"),
        daily_pnl=Decimal("50"),
        open_positions=1,
        reduce_only_mode=False,
    )
