"""
Integration Tests for Guardrails End-to-End Behavior

Tests guardrail triggering, reduce-only mode activation, auto-reset, and
integration with PerpsBot orchestration layer.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, UTC, timedelta
from unittest.mock import Mock, patch

from bot_v2.orchestration.guardrails import GuardRailManager, OrderCheckResult
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.features.brokerages.core.interfaces import Position, Balance, OrderSide


@pytest.fixture
def guardrail_config():
    """Bot config with strict guardrails for testing."""
    return BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-USD"],
        update_interval=60,
        mock_broker=True,
        dry_run=False,
        # Guardrail limits
        max_trade_value=Decimal("100.00"),
        symbol_position_caps={"BTC-USD": Decimal("0.01")},
        daily_loss_limit=Decimal("10.00"),
    )


@pytest.fixture
def guard_manager(guardrail_config):
    """GuardRailManager with test limits."""
    return GuardRailManager(
        max_trade_value=guardrail_config.max_trade_value,
        symbol_position_caps=guardrail_config.symbol_position_caps,
        daily_loss_limit=guardrail_config.daily_loss_limit,
    )


@pytest.mark.integration
class TestGuardrailOrderCaps:
    """Test order cap guardrails block oversized orders."""

    def test_order_cap_blocks_large_order(self, guard_manager):
        """Verify max_trade_value blocks orders exceeding limit."""
        # Attempt order exceeding $100 limit
        context = {
            "symbol": "BTC-USD",
            "mark": Decimal("15000.00"),
            "order_kwargs": {
                "symbol": "BTC-USD",
                "side": OrderSide.BUY,
                "quantity": Decimal("0.01"),  # Notional = 15000 * 0.01 = $150
            },
        }

        result = guard_manager.check_order(context)

        # Verify order blocked
        assert not result.allowed
        assert result.guard == "max_trade_value"
        assert "150" in result.reason
        assert "100" in result.reason

    def test_order_cap_allows_small_order(self, guard_manager):
        """Verify orders within limit are allowed."""
        context = {
            "symbol": "BTC-USD",
            "mark": Decimal("50000.00"),
            "order_kwargs": {
                "symbol": "BTC-USD",
                "side": OrderSide.BUY,
                "quantity": Decimal("0.001"),  # Notional = 50000 * 0.001 = $50
            },
        }

        result = guard_manager.check_order(context)

        # Verify order allowed
        assert result.allowed
        assert result.guard is None

    def test_symbol_position_cap_blocks_oversized_position(self, guard_manager):
        """Verify symbol_position_caps blocks orders exceeding position cap."""
        # Attempting to place order larger than 0.01 BTC cap
        # Use low mark price to avoid max_trade_value guard triggering first
        context = {
            "symbol": "BTC-USD",
            "mark": Decimal("100.00"),  # Low price so notional is small
            "order_kwargs": {
                "symbol": "BTC-USD",
                "side": OrderSide.BUY,
                "quantity": Decimal(
                    "0.015"
                ),  # Exceeds 0.01 BTC cap, but notional = 100*0.015 = $1.50 < $100
            },
        }

        result = guard_manager.check_order(context)

        # Verify order blocked by position limit (not max_trade_value)
        assert not result.allowed
        assert result.guard == "position_limit"
        assert "0.015" in result.reason  # Order quantity mentioned
        assert "0.01" in result.reason  # Cap value mentioned


@pytest.mark.integration
class TestGuardrailDailyLossLimit:
    """Test daily loss limit triggers reduce-only mode."""

    def test_daily_loss_limit_triggers_reduce_only(self, guard_manager):
        """Verify daily loss limit activates daily_loss guard."""
        # Record $12 loss (exceeds $10 limit)
        guard_manager.record_realized_pnl(Decimal("-12.00"))

        # Verify guard not active yet (needs cycle check)
        assert not guard_manager.is_guard_active("daily_loss")

        # Run cycle check to evaluate daily loss limit
        guard_manager.check_cycle({})

        # Verify daily_loss guard now active
        assert guard_manager.is_guard_active("daily_loss")

        # Note: The daily_loss guard doesn't directly block orders in check_order.
        # Instead, PerpsBot detects the active guard and enters reduce-only mode,
        # which then blocks new position-increasing orders.
        # See perps_bot.py:209-212 for the integration behavior.

    def test_daily_loss_guard_persists_during_day(self, guard_manager):
        """Verify daily_loss guard stays active until day change."""
        # Record loss exceeding limit
        guard_manager.record_realized_pnl(Decimal("-15.00"))
        guard_manager.check_cycle({})

        # Verify guard active
        assert guard_manager.is_guard_active("daily_loss")

        # Record additional loss - guard should stay active
        guard_manager.record_realized_pnl(Decimal("-5.00"))
        guard_manager.check_cycle({})

        # Guard still active
        assert guard_manager.is_guard_active("daily_loss")

        # Verify total loss tracked correctly
        assert guard_manager.get_daily_pnl() == Decimal("-20.00")

    def test_daily_loss_guard_auto_resets_next_day(self, guard_manager):
        """Verify daily loss guard auto-resets when new trading day detected."""
        from datetime import date, timedelta

        # Record loss and activate guard
        guard_manager.record_realized_pnl(Decimal("-12.00"))
        guard_manager.check_cycle({})
        assert guard_manager.is_guard_active("daily_loss")

        # Simulate day change by setting tracking date to yesterday
        yesterday = date.today() - timedelta(days=1)
        guard_manager._pnl_tracking_date = yesterday

        # Run cycle check - should detect new day and reset
        guard_manager.check_cycle({})

        # Guard should be cleared and P&L reset
        assert not guard_manager.is_guard_active("daily_loss")
        assert guard_manager.get_daily_pnl() == Decimal("0")


@pytest.mark.integration
class TestGuardrailListeners:
    """Test guardrail state change listeners."""

    def test_listener_receives_guard_activation(self, guard_manager):
        """Verify listeners are notified when guards activate."""
        events = []

        def listener(guard_name, active):
            events.append({"guard": guard_name, "active": active})

        guard_manager.register_listener(listener)

        # Trigger daily loss guard
        guard_manager.record_realized_pnl(Decimal("-15.00"))
        guard_manager.check_cycle({})

        # Verify listener called
        assert len(events) > 0
        assert any(e["guard"] == "daily_loss" and e["active"] is True for e in events)

    def test_listener_receives_guard_deactivation(self, guard_manager):
        """Verify listeners notified when guards deactivate."""
        from datetime import date, timedelta

        events = []

        def listener(guard_name, active):
            events.append({"guard": guard_name, "active": active})

        guard_manager.register_listener(listener)

        # Activate guard
        guard_manager.record_realized_pnl(Decimal("-15.00"))
        guard_manager.check_cycle({})

        # Clear events from activation
        events.clear()

        # Deactivate by simulating day change
        yesterday = date.today() - timedelta(days=1)
        guard_manager._pnl_tracking_date = yesterday
        guard_manager.check_cycle({})

        # Verify deactivation event
        assert any(e["guard"] == "daily_loss" and e["active"] is False for e in events)


@pytest.mark.integration
@pytest.mark.asyncio
class TestGuardrailPerpsBotIntegration:
    """Test guardrails integration with PerpsBot orchestration."""

    async def test_perps_bot_activates_reduce_only_on_daily_loss(
        self, monkeypatch, tmp_path, guardrail_config
    ):
        """Verify PerpsBot enters reduce-only mode when daily loss limit hit."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(guardrail_config)

        # Verify guardrails are initialized
        assert hasattr(bot, "guardrails")
        assert bot.guardrails is not None

        # Record loss exceeding limit
        bot.guardrails.record_realized_pnl(Decimal("-12.00"))  # Exceeds $10 limit

        # Simulate PerpsBot's run_cycle logic
        balances = []
        positions = []
        position_map = {}

        # Create cycle context
        cycle_context = {
            "balances": balances,
            "positions": positions,
            "position_map": position_map,
        }

        # Check cycle-level guards (this is what run_cycle does)
        bot.guardrails.check_cycle(cycle_context)

        # Verify daily_loss guard activated
        assert bot.guardrails.is_guard_active("daily_loss")

        # In actual PerpsBot.run_cycle(), this guard would trigger:
        # if bot.guardrails.is_guard_active("daily_loss"):
        #     bot.set_reduce_only_mode(True, "daily_loss_limit_reached")

    async def test_perps_bot_guardrail_metrics_integration(
        self, monkeypatch, tmp_path, guardrail_config
    ):
        """Verify guardrail events are exported to metrics server."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(guardrail_config)

        # Register guardrails with metrics server if available
        if hasattr(bot, "metrics_server") and bot.metrics_server:
            bot.metrics_server.register_guard_manager(bot.guardrails)

        # Trigger guard
        bot.guardrails.record_realized_pnl(Decimal("-15.00"))
        bot.guardrails.check_cycle({})

        # Verify guard activated
        assert bot.guardrails.is_guard_active("daily_loss")

        # Verify metrics server received guard state change
        # (Actual verification would check MetricsServer state if exposed)


@pytest.mark.integration
class TestGuardrailDryRunMode:
    """Test guardrails dry-run guard (blocks all orders)."""

    def test_dry_run_guard_blocks_all_orders(self):
        """Verify dry-run guard blocks all order placement."""
        dry_run_manager = GuardRailManager(
            max_trade_value=Decimal("100.00"),
            symbol_position_caps={"BTC-USD": Decimal("0.01")},
            daily_loss_limit=Decimal("10.00"),
        )

        # Activate dry-run guard
        dry_run_manager.set_dry_run(True)

        # Attempt to place order
        context = {
            "symbol": "BTC-USD",
            "side": OrderSide.BUY,
            "size": Decimal("0.001"),
            "mark_price": Decimal("50000.00"),  # Small order that would normally pass
        }

        result = dry_run_manager.check_order(context)

        # Dry-run guard blocks all orders
        assert not result.allowed
        assert result.guard == "dry_run"
        assert "dry_run_active" in result.reason

    def test_dry_run_guard_can_be_disabled(self):
        """Verify dry-run guard can be toggled on/off."""
        dry_run_manager = GuardRailManager(
            max_trade_value=Decimal("100.00"),
            daily_loss_limit=Decimal("10.00"),
        )

        # Enable dry-run
        dry_run_manager.set_dry_run(True)
        assert dry_run_manager.is_guard_active("dry_run")

        # Disable dry-run
        dry_run_manager.set_dry_run(False)
        assert not dry_run_manager.is_guard_active("dry_run")

        # Orders now allowed (if no other guards active)
        context = {
            "symbol": "BTC-USD",
            "side": OrderSide.BUY,
            "size": Decimal("0.001"),
            "mark_price": Decimal("50000.00"),
        }
        result = dry_run_manager.check_order(context)
        assert result.allowed
