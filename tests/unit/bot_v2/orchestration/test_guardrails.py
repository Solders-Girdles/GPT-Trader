"""Tests for GuardRailManager and guard-integrated order placement."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import pytest
from prometheus_client import CollectorRegistry

from bot_v2.monitoring.metrics_server import MetricsServer
from bot_v2.orchestration.guardrails import GuardRailManager
from bot_v2.orchestration.execution.order_placement import OrderPlacementService
from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.features.brokerages.core.interfaces import OrderType


class DummyExecEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def place_order(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        raise AssertionError("place_order should not be called when guard blocks orders")


class DummyOrdersStore:
    def upsert(self, order: object) -> None:  # pragma: no cover - not used in dry-run path
        pass


@pytest.mark.asyncio
async def test_dry_run_guard_blocks_orders() -> None:
    guardrails = GuardRailManager()
    guardrails.set_dry_run(True)

    registry = CollectorRegistry()
    metrics = MetricsServer(host="127.0.0.1", port=0, registry=registry)
    metrics.register_guard_manager(guardrails, profile="test")

    order_stats: dict[str, int] = {"attempted": 0, "successful": 0, "failed": 0}
    service = OrderPlacementService(
        orders_store=DummyOrdersStore(),
        order_stats=order_stats,
        dry_run=False,
        metrics_server=metrics,
        guardrails=guardrails,
        profile="test",
    )

    exec_engine = DummyExecEngine()
    decision = type(
        "Decision",
        (),
        {
            "action": Action.BUY,
            "reduce_only": False,
            "leverage": None,
            "order_type": OrderType.MARKET,
            "limit_price": None,
            "stop_trigger": None,
            "time_in_force": None,
            "quantity": Decimal("0.01"),
        },
    )()

    product = type("Product", (), {"symbol": "BTC-USD"})()

    await service.execute_decision(
        symbol="BTC-USD",
        decision=decision,
        mark=Decimal("50000"),
        product=product,
        position_state=None,
        exec_engine=exec_engine,
        reduce_only_mode=False,
        default_time_in_force="GTC",
    )

    # Order should never reach exec_engine
    assert exec_engine.calls == []

    # Order stats reflect guard-handled success
    assert order_stats["attempted"] == 1
    assert order_stats["successful"] == 1
    assert order_stats["failed"] == 0

    # Guard trip recorded in metrics
    guard_value = metrics.guard_state_gauge.labels(profile="test", guard="dry_run")._value.get()
    assert guard_value == 1.0

    counter_value = None
    for metric in metrics.guard_trip_counter.collect():
        for sample in metric.samples:
            if sample.name == "bot_guard_trips_total" and sample.labels.get("guard") == "dry_run":
                counter_value = sample.value
    assert counter_value == 1.0


def test_guard_listener_updates_state() -> None:
    guardrails = GuardRailManager()
    states: list[tuple[str, bool]] = []
    guardrails.register_listener(lambda name, active: states.append((name, active)))

    guardrails.set_dry_run(True)
    guardrails.set_dry_run(False)

    assert states == [("dry_run", True), ("dry_run", False)]


def test_error_streak_triggers_circuit_breaker() -> None:
    guardrails = GuardRailManager(error_threshold=2, error_cooldown_seconds=60)

    streak, triggered = guardrails.record_error("order_failure")
    assert streak == 1
    assert triggered is False
    assert not guardrails.is_guard_active("circuit_breaker")

    streak, triggered = guardrails.record_error("order_failure")
    assert streak == 2
    assert triggered is True
    assert guardrails.is_guard_active("circuit_breaker")

    guardrails.record_success()
    assert guardrails.get_error_streak() == 0
    assert not guardrails.is_guard_active("circuit_breaker")


def test_error_streak_cooldown_resets_guard() -> None:
    guardrails = GuardRailManager(error_threshold=1, error_cooldown_seconds=10)

    guardrails.record_error("order_failure")
    assert guardrails.is_guard_active("circuit_breaker")

    guardrails._last_error_timestamp = time.time() - 20  # Force cooldown expiry
    guardrails.check_cycle({})

    assert guardrails.get_error_streak() == 0
    assert not guardrails.is_guard_active("circuit_breaker")


def test_max_trade_value_cap_blocks_large_orders() -> None:
    """Test that max_trade_value cap blocks orders exceeding the limit."""
    guardrails = GuardRailManager(max_trade_value=Decimal("10000"))

    # Order within limit should pass
    context = {
        "symbol": "BTC-USD",
        "mark": Decimal("50000"),
        "order_kwargs": {"quantity": Decimal("0.1")},  # $5,000 notional
    }
    result = guardrails.check_order(context)
    assert result.allowed is True

    # Order exceeding limit should be blocked
    context = {
        "symbol": "BTC-USD",
        "mark": Decimal("50000"),
        "order_kwargs": {"quantity": Decimal("0.5")},  # $25,000 notional
    }
    result = guardrails.check_order(context)
    assert result.allowed is False
    assert result.guard == "max_trade_value"
    assert "25000" in result.reason
    assert "10000" in result.reason


def test_max_trade_value_cap_at_exact_limit() -> None:
    """Test that orders at exact limit are allowed."""
    guardrails = GuardRailManager(max_trade_value=Decimal("10000"))

    context = {
        "symbol": "BTC-USD",
        "mark": Decimal("50000"),
        "order_kwargs": {"quantity": Decimal("0.2")},  # Exactly $10,000
    }
    result = guardrails.check_order(context)
    assert result.allowed is True


def test_max_trade_value_zero_disables_check() -> None:
    """Test that max_trade_value=0 disables the check."""
    guardrails = GuardRailManager(max_trade_value=Decimal("0"))

    context = {
        "symbol": "BTC-USD",
        "mark": Decimal("50000"),
        "order_kwargs": {"quantity": Decimal("100")},  # $5,000,000 notional
    }
    result = guardrails.check_order(context)
    assert result.allowed is True


def test_max_trade_value_missing_price_allows_order() -> None:
    """Test that missing mark price fails open (allows order)."""
    guardrails = GuardRailManager(max_trade_value=Decimal("10000"))

    context = {
        "symbol": "BTC-USD",
        "mark": None,  # Missing price
        "order_kwargs": {"quantity": Decimal("1.0")},
    }
    result = guardrails.check_order(context)
    assert result.allowed is True


def test_symbol_position_cap_blocks_large_quantities() -> None:
    """Test that symbol position caps block orders exceeding per-symbol limits."""
    guardrails = GuardRailManager(
        symbol_position_caps={"BTC-USD": Decimal("1.0"), "ETH-USD": Decimal("10.0")}
    )

    # Order within cap should pass
    context = {
        "symbol": "BTC-USD",
        "order_kwargs": {"quantity": Decimal("0.5")},
    }
    result = guardrails.check_order(context)
    assert result.allowed is True

    # Order exceeding cap should be blocked
    context = {
        "symbol": "BTC-USD",
        "order_kwargs": {"quantity": Decimal("2.0")},
    }
    result = guardrails.check_order(context)
    assert result.allowed is False
    assert result.guard == "position_limit"
    assert "BTC-USD" in result.reason
    assert "2.0" in result.reason or "2" in result.reason
    assert "1.0" in result.reason or "1" in result.reason


def test_symbol_position_cap_at_exact_limit() -> None:
    """Test that orders at exact cap are allowed."""
    guardrails = GuardRailManager(symbol_position_caps={"BTC-USD": Decimal("1.0")})

    context = {
        "symbol": "BTC-USD",
        "order_kwargs": {"quantity": Decimal("1.0")},
    }
    result = guardrails.check_order(context)
    assert result.allowed is True


def test_symbol_position_cap_negative_quantity() -> None:
    """Test that negative quantities are treated as absolute values."""
    guardrails = GuardRailManager(symbol_position_caps={"BTC-USD": Decimal("1.0")})

    context = {
        "symbol": "BTC-USD",
        "order_kwargs": {"quantity": Decimal("-2.0")},  # Sell order
    }
    result = guardrails.check_order(context)
    assert result.allowed is False
    assert result.guard == "position_limit"


def test_symbol_position_cap_uncapped_symbol_allows_order() -> None:
    """Test that symbols without caps allow any quantity."""
    guardrails = GuardRailManager(symbol_position_caps={"BTC-USD": Decimal("1.0")})

    context = {
        "symbol": "ETH-USD",  # No cap for this symbol
        "order_kwargs": {"quantity": Decimal("1000.0")},
    }
    result = guardrails.check_order(context)
    assert result.allowed is True


def test_update_limits_changes_behavior() -> None:
    """Test that update_limits() dynamically changes guard behavior."""
    guardrails = GuardRailManager(max_trade_value=Decimal("10000"))

    # Initially blocks at $10k limit
    context = {
        "symbol": "BTC-USD",
        "mark": Decimal("50000"),
        "order_kwargs": {"quantity": Decimal("0.5")},  # $25,000
    }
    result = guardrails.check_order(context)
    assert result.allowed is False

    # Update to higher limit
    guardrails.update_limits(max_trade_value=Decimal("30000"))

    # Same order now allowed
    result = guardrails.check_order(context)
    assert result.allowed is True


def test_combined_caps_both_enforced() -> None:
    """Test that both max_trade_value and symbol caps are checked (value checked first)."""
    guardrails = GuardRailManager(
        max_trade_value=Decimal("10000"), symbol_position_caps={"BTC-USD": Decimal("0.1")}
    )

    # Order blocked by value cap first (checked before position cap)
    context = {
        "symbol": "BTC-USD",
        "mark": Decimal("50000"),
        "order_kwargs": {"quantity": Decimal("0.5")},  # $25,000 notional, 0.5 BTC
    }
    result = guardrails.check_order(context)
    assert result.allowed is False
    assert result.guard == "max_trade_value"

    # Order passes value cap but blocked by position cap
    context = {
        "symbol": "BTC-USD",
        "mark": Decimal("50000"),
        "order_kwargs": {"quantity": Decimal("0.15")},  # $7,500 notional, 0.15 BTC > 0.1 cap
    }
    result = guardrails.check_order(context)
    assert result.allowed is False
    assert result.guard == "position_limit"


def test_daily_loss_tracking_accumulates_pnl() -> None:
    """Test that daily P&L accumulates correctly."""
    guardrails = GuardRailManager()

    # Record some P&L
    guardrails.record_realized_pnl(Decimal("100"))  # Profit
    assert guardrails.get_daily_pnl() == Decimal("100")
    assert guardrails.get_daily_loss() == Decimal("0")

    guardrails.record_realized_pnl(Decimal("-50"))  # Loss
    assert guardrails.get_daily_pnl() == Decimal("50")
    assert guardrails.get_daily_loss() == Decimal("0")  # Still net positive

    guardrails.record_realized_pnl(Decimal("-200"))  # Bigger loss
    assert guardrails.get_daily_pnl() == Decimal("-150")
    assert guardrails.get_daily_loss() == Decimal("150")  # Now in loss


def test_daily_loss_limit_triggers_guard() -> None:
    """Test that exceeding daily loss limit activates the guard."""
    guardrails = GuardRailManager(daily_loss_limit=Decimal("100"))

    # Record losses below limit
    guardrails.record_realized_pnl(Decimal("-50"))
    guardrails.check_cycle({})
    assert not guardrails.is_guard_active("daily_loss")

    # Record losses exceeding limit
    guardrails.record_realized_pnl(Decimal("-60"))  # Total: -110
    guardrails.check_cycle({})
    assert guardrails.is_guard_active("daily_loss")


def test_daily_loss_guard_at_exact_limit() -> None:
    """Test that hitting exactly the limit triggers the guard."""
    guardrails = GuardRailManager(daily_loss_limit=Decimal("100"))

    guardrails.record_realized_pnl(Decimal("-100"))
    guardrails.check_cycle({})
    assert guardrails.is_guard_active("daily_loss")


def test_daily_loss_zero_disables_check() -> None:
    """Test that daily_loss_limit=0 disables the check."""
    guardrails = GuardRailManager(daily_loss_limit=Decimal("0"))

    guardrails.record_realized_pnl(Decimal("-10000"))
    guardrails.check_cycle({})
    assert not guardrails.is_guard_active("daily_loss")


def test_daily_loss_resets_on_new_day() -> None:
    """Test that daily loss tracking resets on new day."""
    from datetime import date
    from unittest.mock import patch

    guardrails = GuardRailManager(daily_loss_limit=Decimal("100"))

    # Record loss on day 1
    day1 = date(2024, 1, 1)
    with patch("bot_v2.orchestration.guardrails.date") as mock_date:
        mock_date.today.return_value = day1
        guardrails.record_realized_pnl(Decimal("-80"))
        guardrails.check_cycle({})
        assert guardrails.get_daily_loss() == Decimal("80")
        assert not guardrails.is_guard_active("daily_loss")

    # Move to next day
    day2 = date(2024, 1, 2)
    with patch("bot_v2.orchestration.guardrails.date") as mock_date:
        mock_date.today.return_value = day2
        # Check cycle should reset tracking
        guardrails.check_cycle({})
        assert guardrails.get_daily_pnl() == Decimal("0")
        assert guardrails.get_daily_loss() == Decimal("0")
        assert not guardrails.is_guard_active("daily_loss")


def test_daily_loss_guard_clears_on_new_day() -> None:
    """Test that active daily_loss guard clears on new day."""
    from datetime import date
    from unittest.mock import patch

    guardrails = GuardRailManager(daily_loss_limit=Decimal("100"))

    # Trigger guard on day 1
    day1 = date(2024, 1, 1)
    with patch("bot_v2.orchestration.guardrails.date") as mock_date:
        mock_date.today.return_value = day1
        guardrails.record_realized_pnl(Decimal("-150"))
        guardrails.check_cycle({})
        assert guardrails.is_guard_active("daily_loss")

    # Move to next day
    day2 = date(2024, 1, 2)
    with patch("bot_v2.orchestration.guardrails.date") as mock_date:
        mock_date.today.return_value = day2
        # Check cycle should clear guard
        guardrails.check_cycle({})
        assert not guardrails.is_guard_active("daily_loss")


def test_daily_loss_profits_dont_trigger_guard() -> None:
    """Test that profits don't trigger daily loss guard."""
    guardrails = GuardRailManager(daily_loss_limit=Decimal("100"))

    guardrails.record_realized_pnl(Decimal("1000"))  # Big profit
    guardrails.check_cycle({})
    assert not guardrails.is_guard_active("daily_loss")
    assert guardrails.get_daily_loss() == Decimal("0")


def test_update_limits_changes_daily_loss_limit() -> None:
    """Test that update_limits() changes daily loss limit."""
    guardrails = GuardRailManager(daily_loss_limit=Decimal("100"))

    # Loss doesn't trigger with high limit
    guardrails.record_realized_pnl(Decimal("-80"))
    guardrails.check_cycle({})
    assert not guardrails.is_guard_active("daily_loss")

    # Lower the limit
    guardrails.update_limits(daily_loss_limit=Decimal("50"))

    # Same loss now triggers guard
    guardrails.check_cycle({})
    assert guardrails.is_guard_active("daily_loss")
