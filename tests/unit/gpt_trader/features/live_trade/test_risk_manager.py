"""Tests for LiveRiskManager and related classes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

from gpt_trader.features.live_trade.risk.manager import (
    LiveRiskManager,
    ValidationError,
    VolatilityCheckOutcome,
)


@pytest.fixture(autouse=True)
def mock_load_state():
    """Prevent LiveRiskManager from loading state during tests."""
    with patch("gpt_trader.features.live_trade.risk.manager.LiveRiskManager._load_state"):
        yield


# ============================================================
# Test: ValidationError exception
# ============================================================


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_is_exception(self) -> None:
        """Test ValidationError inherits from Exception."""
        assert issubclass(ValidationError, Exception)

    def test_validation_error_with_message(self) -> None:
        """Test ValidationError can be raised with a message."""
        with pytest.raises(ValidationError, match="test error"):
            raise ValidationError("test error")

    def test_validation_error_without_message(self) -> None:
        """Test ValidationError can be raised without a message."""
        with pytest.raises(ValidationError):
            raise ValidationError()


# ============================================================
# Test: VolatilityCheckOutcome dataclass
# ============================================================


class TestVolatilityCheckOutcome:
    """Tests for VolatilityCheckOutcome dataclass."""

    def test_default_values(self) -> None:
        """Test default values for VolatilityCheckOutcome."""
        outcome = VolatilityCheckOutcome()

        assert outcome.triggered is False
        assert outcome.symbol == ""
        assert outcome.reason == ""

    def test_custom_values(self) -> None:
        """Test creating VolatilityCheckOutcome with custom values."""
        outcome = VolatilityCheckOutcome(
            triggered=True,
            symbol="BTC-USD",
            reason="High volatility detected",
        )

        assert outcome.triggered is True
        assert outcome.symbol == "BTC-USD"
        assert outcome.reason == "High volatility detected"

    def test_to_payload_not_triggered(self) -> None:
        """Test to_payload for non-triggered outcome."""
        outcome = VolatilityCheckOutcome()
        payload = outcome.to_payload()

        assert payload == {
            "triggered": False,
            "symbol": "",
            "reason": "",
        }

    def test_to_payload_triggered(self) -> None:
        """Test to_payload for triggered outcome."""
        outcome = VolatilityCheckOutcome(
            triggered=True,
            symbol="ETH-USD",
            reason="Volatility exceeded 5%",
        )
        payload = outcome.to_payload()

        assert payload == {
            "triggered": True,
            "symbol": "ETH-USD",
            "reason": "Volatility exceeded 5%",
        }


# ============================================================
# Test: LiveRiskManager initialization
# ============================================================


class TestLiveRiskManagerInit:
    """Tests for LiveRiskManager initialization."""

    def test_init_no_config(self) -> None:
        """Test initialization without config."""
        manager = LiveRiskManager()

        assert manager.config is None
        assert manager.event_store is None
        assert manager._reduce_only_mode is False
        assert manager._reduce_only_reason == ""
        assert manager._daily_pnl_triggered is False
        assert manager._risk_metrics == []
        assert manager._start_of_day_equity is None

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = {"max_leverage": 10}
        manager = LiveRiskManager(config=config)

        assert manager.config == {"max_leverage": 10}

    def test_init_with_event_store(self) -> None:
        """Test initialization with event store."""
        event_store = object()
        manager = LiveRiskManager(event_store=event_store)

        assert manager.event_store is event_store

    def test_positions_default_dict(self) -> None:
        """Test positions is a defaultdict."""
        manager = LiveRiskManager()

        # Access non-existent key should return empty dict
        assert manager.positions["BTC-USD"] == {}

    def test_last_mark_update_empty(self) -> None:
        """Test last_mark_update starts empty."""
        manager = LiveRiskManager()

        assert manager.last_mark_update == {}


# ============================================================
# Test: check_order and update_position stubs
# ============================================================


class TestLiveRiskManagerStubs:
    """Tests for stub methods."""

    def test_check_order_returns_true(self) -> None:
        """Test check_order always returns True."""
        manager = LiveRiskManager()

        assert manager.check_order(None) is True
        assert manager.check_order({"symbol": "BTC-USD"}) is True
        assert manager.check_order(object()) is True

    def test_update_position_is_noop(self) -> None:
        """Test update_position is a no-op."""
        manager = LiveRiskManager()

        # Should not raise
        manager.update_position(None)
        manager.update_position({"symbol": "BTC-USD"})
        assert manager.positions == {}


# ============================================================
# Test: check_liquidation_buffer
# ============================================================


@dataclass
class MockPosition:
    """Mock position object for testing."""

    liquidation_price: Decimal | None = None
    mark_price: Decimal | None = None
    mark: Decimal | None = None


@dataclass
class MockConfig:
    """Mock config for testing."""

    min_liquidation_buffer_pct: Decimal = Decimal("0.1")
    max_leverage: Decimal = Decimal("10")
    daily_loss_limit_pct: Decimal | None = None
    mark_staleness_threshold: float = 120.0
    volatility_threshold_pct: Decimal | None = None
    max_exposure_pct: float = 100.0  # 10000% - effectively no limit for leverage tests


class TestCheckLiquidationBuffer:
    """Tests for check_liquidation_buffer method."""

    def test_no_config_returns_false(self) -> None:
        """Test returns False when no config."""
        manager = LiveRiskManager()

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 110}, Decimal("1000")
        )

        assert result is False

    def test_dict_position_no_liquidation_price(self) -> None:
        """Test with dict position missing liquidation_price."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer("BTC-USD", {"mark": 110}, Decimal("1000"))

        assert result is False

    def test_dict_position_no_mark_price(self) -> None:
        """Test with dict position missing mark price."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100}, Decimal("1000")
        )

        assert result is False

    def test_dict_position_with_mark_key(self) -> None:
        """Test dict position using 'mark' key."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)

        # Buffer = |110 - 100| / 110 = 0.0909... (9.09%), threshold 5%
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 110}, Decimal("1000")
        )

        assert result is False  # Buffer > threshold, not dangerous

    def test_dict_position_with_mark_price_key(self) -> None:
        """Test dict position using 'mark_price' key."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark_price": 110}, Decimal("1000")
        )

        assert result is False

    def test_object_position_with_mark_price(self) -> None:
        """Test object position with mark_price attribute."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        position = MockPosition(liquidation_price=Decimal("100"), mark_price=Decimal("110"))

        result = manager.check_liquidation_buffer("BTC-USD", position, Decimal("1000"))

        assert result is False

    def test_object_position_with_mark_fallback(self) -> None:
        """Test object position falling back to mark attribute."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        position = MockPosition(liquidation_price=Decimal("100"), mark=Decimal("110"))

        result = manager.check_liquidation_buffer("BTC-USD", position, Decimal("1000"))

        assert result is False

    def test_triggers_reduce_only_when_buffer_too_small(self) -> None:
        """Test triggers reduce_only when buffer is below threshold."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.15"))
        manager = LiveRiskManager(config=config)

        # Buffer = |110 - 100| / 110 = 0.0909... (9.09%), threshold 15%
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 110}, Decimal("1000")
        )

        assert result is True
        assert manager.positions["BTC-USD"]["reduce_only"] is True

    def test_zero_mark_price_returns_false(self) -> None:
        """Test returns False when mark price is zero."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 0}, Decimal("1000")
        )

        assert result is False

    def test_handles_attribute_error(self) -> None:
        """Test handles AttributeError gracefully."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        # Object without expected attributes
        result = manager.check_liquidation_buffer("BTC-USD", object(), Decimal("1000"))

        assert result is False

    def test_handles_none_values(self) -> None:
        """Test handles None values gracefully."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        # Position with None liquidation_price
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": None, "mark": 110}, Decimal("1000")
        )

        assert result is False

    def test_handles_value_error(self) -> None:
        """Test handles ValueError gracefully (empty string conversion)."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        # Empty strings that pass the falsy check but fail Decimal conversion
        # Note: empty string is falsy, so this hits the early return
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": "", "mark": 110}, Decimal("1000")
        )

        assert result is False


# ============================================================
# Test: pre_trade_validate
# ============================================================


class TestPreTradeValidate:
    """Tests for pre_trade_validate method."""

    def test_no_config_passes(self) -> None:
        """Test validation passes when no config."""
        manager = LiveRiskManager()

        result = manager.pre_trade_validate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal("50000"),
            product=None,
            equity=Decimal("10000"),
            current_positions={},
        )
        assert result is None

    def test_leverage_within_limit(self) -> None:
        """Test validation passes when leverage within limit."""
        config = MockConfig(max_leverage=Decimal("10"))
        manager = LiveRiskManager(config=config)

        # Notional = 1 * 50000 = 50000, Leverage = 50000 / 10000 = 5x
        result = manager.pre_trade_validate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal("50000"),
            product=None,
            equity=Decimal("10000"),
            current_positions={},
        )
        assert result is None

    def test_leverage_exceeds_limit_raises(self) -> None:
        """Test validation raises when leverage exceeds limit."""
        config = MockConfig(max_leverage=Decimal("5"))
        manager = LiveRiskManager(config=config)

        # Notional = 2 * 50000 = 100000, Leverage = 100000 / 10000 = 10x
        with pytest.raises(ValidationError, match="Leverage .* exceeds max 5"):
            manager.pre_trade_validate(
                symbol="BTC-USD",
                side="buy",
                quantity=Decimal("2"),
                price=Decimal("50000"),
                product=None,
                equity=Decimal("10000"),
                current_positions={},
            )

    def test_zero_equity_skips_validation(self) -> None:
        """Test validation skipped when equity is zero."""
        config = MockConfig(max_leverage=Decimal("5"))
        manager = LiveRiskManager(config=config)

        # Should not raise even with high notional
        result = manager.pre_trade_validate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("50000"),
            product=None,
            equity=Decimal("0"),
            current_positions={},
        )
        assert result is None

    def test_negative_equity_skips_validation(self) -> None:
        """Test validation skipped when equity is negative."""
        config = MockConfig(max_leverage=Decimal("5"))
        manager = LiveRiskManager(config=config)

        # Negative equity means the condition equity > 0 fails
        result = manager.pre_trade_validate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("50000"),
            product=None,
            equity=Decimal("-1000"),
            current_positions={},
        )
        assert result is None


# ============================================================
# Test: track_daily_pnl
# ============================================================


class TestTrackDailyPnl:
    """Tests for track_daily_pnl method."""

    def test_first_call_sets_start_equity(self) -> None:
        """Test first call sets start of day equity."""
        manager = LiveRiskManager()

        result = manager.track_daily_pnl(Decimal("10000"), {})

        assert result is False
        assert manager._start_of_day_equity == Decimal("10000")

    def test_no_config_returns_false(self) -> None:
        """Test returns False when no config."""
        manager = LiveRiskManager()
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is False

    def test_no_daily_loss_limit_returns_false(self) -> None:
        """Test returns False when no daily_loss_limit_pct in config."""
        config = MockConfig(daily_loss_limit_pct=None)
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is False

    def test_loss_within_limit(self) -> None:
        """Test returns False when loss within limit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.10"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        # 5% loss (9500 from 10000)
        result = manager.track_daily_pnl(Decimal("9500"), {})

        assert result is False
        assert manager._daily_pnl_triggered is False

    def test_loss_exceeds_limit_triggers(self) -> None:
        """Test triggers when loss exceeds limit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        # 10% loss (9000 from 10000)
        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is True
        assert manager._daily_pnl_triggered is True
        assert manager._reduce_only_mode is True
        assert manager._reduce_only_reason == "daily_loss_limit_breached"

    def test_profit_does_not_trigger(self) -> None:
        """Test profit does not trigger loss limit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        # 10% profit
        result = manager.track_daily_pnl(Decimal("11000"), {})

        assert result is False

    def test_zero_start_equity_skips(self) -> None:
        """Test skips when start of day equity is zero."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("0")

        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is False


# ============================================================
# Test: check_mark_staleness
# ============================================================


class TestCheckMarkStaleness:
    """Tests for check_mark_staleness method."""

    def test_no_update_is_stale(self) -> None:
        """Test returns True when no update recorded."""
        manager = LiveRiskManager()

        assert manager.check_mark_staleness("BTC-USD") is True

    def test_recent_update_not_stale(self) -> None:
        """Test returns False when update is recent."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time()

        assert manager.check_mark_staleness("BTC-USD") is False

    def test_old_update_is_stale(self) -> None:
        """Test returns True when update is old."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time() - 200  # 200 seconds ago

        assert manager.check_mark_staleness("BTC-USD") is True

    def test_custom_staleness_threshold(self) -> None:
        """Test uses config staleness threshold."""
        config = MockConfig(mark_staleness_threshold=30.0)
        manager = LiveRiskManager(config=config)
        manager.last_mark_update["BTC-USD"] = time.time() - 50  # 50 seconds ago

        # 50 > 30, so stale
        assert manager.check_mark_staleness("BTC-USD") is True

    def test_default_threshold_without_config(self) -> None:
        """Test default 120 second threshold without config."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time() - 100  # 100 seconds ago

        # 100 < 120, so not stale
        assert manager.check_mark_staleness("BTC-USD") is False

    @patch("time.time")
    def test_exact_boundary(self, mock_time: Any) -> None:
        """Test behavior at exact threshold boundary."""
        mock_time.return_value = 1000.0
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = 880.0  # Exactly 120 seconds ago

        # 1000 - 880 = 120, and 120 > 120 is False
        assert manager.check_mark_staleness("BTC-USD") is False


# ============================================================
# Test: append_risk_metrics
# ============================================================


class TestAppendRiskMetrics:
    """Tests for append_risk_metrics method."""

    @patch("time.time")
    def test_appends_metrics(self, mock_time: Any) -> None:
        """Test appends metrics with timestamp."""
        mock_time.return_value = 12345.0
        manager = LiveRiskManager()

        manager.append_risk_metrics(Decimal("10000"), {"BTC-USD": {"pnl": Decimal("100")}})

        assert len(manager._risk_metrics) == 1
        assert manager._risk_metrics[0]["timestamp"] == 12345.0
        assert manager._risk_metrics[0]["equity"] == "10000"
        assert manager._risk_metrics[0]["positions"] == {"BTC-USD": {"pnl": "100"}}
        assert manager._risk_metrics[0]["reduce_only_mode"] is False

    def test_captures_reduce_only_mode(self) -> None:
        """Test captures reduce_only_mode state."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True

        manager.append_risk_metrics(Decimal("10000"), {})

        assert manager._risk_metrics[0]["reduce_only_mode"] is True

    def test_limits_to_100_metrics(self) -> None:
        """Test keeps only last 100 metrics."""
        manager = LiveRiskManager()

        for i in range(150):
            manager.append_risk_metrics(Decimal(str(i)), {})

        assert len(manager._risk_metrics) == 100
        # First metric should be #50 (0-49 removed)
        assert manager._risk_metrics[0]["equity"] == "50"
        assert manager._risk_metrics[-1]["equity"] == "149"

    def test_converts_nested_decimals_to_strings(self) -> None:
        """Test converts nested Decimal values to strings."""
        manager = LiveRiskManager()
        positions = {
            "BTC-USD": {
                "pnl": Decimal("123.456"),
                "size": Decimal("-0.5"),
            },
            "ETH-USD": {
                "pnl": Decimal("0"),
            },
        }

        manager.append_risk_metrics(Decimal("9999.99"), positions)

        result_positions = manager._risk_metrics[0]["positions"]
        assert result_positions["BTC-USD"]["pnl"] == "123.456"
        assert result_positions["BTC-USD"]["size"] == "-0.5"
        assert result_positions["ETH-USD"]["pnl"] == "0"


# ============================================================
# Test: check_volatility_circuit_breaker
# ============================================================


class TestCheckVolatilityCircuitBreaker:
    """Tests for check_volatility_circuit_breaker method."""

    def test_empty_closes_not_triggered(self) -> None:
        """Test returns not triggered for empty closes."""
        manager = LiveRiskManager()

        result = manager.check_volatility_circuit_breaker("BTC-USD", [])

        assert result.triggered is False
        assert result.symbol == "BTC-USD"

    def test_too_few_closes_not_triggered(self) -> None:
        """Test returns not triggered for fewer than 5 closes."""
        manager = LiveRiskManager()
        closes = [Decimal("100"), Decimal("101"), Decimal("102"), Decimal("103")]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_no_config_not_triggered(self) -> None:
        """Test returns not triggered without config."""
        manager = LiveRiskManager()
        closes = [Decimal(str(i)) for i in range(100, 110)]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_no_volatility_threshold_not_triggered(self) -> None:
        """Test returns not triggered without volatility_threshold_pct."""
        config = MockConfig(volatility_threshold_pct=None)
        manager = LiveRiskManager(config=config)
        closes = [Decimal(str(i)) for i in range(100, 110)]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_low_volatility_not_triggered(self) -> None:
        """Test returns not triggered when volatility is below threshold."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.10"))
        manager = LiveRiskManager(config=config)
        # Small variance - max deviation ~2.5% from mean
        closes = [Decimal("100"), Decimal("101"), Decimal("99"), Decimal("100"), Decimal("100")]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_high_volatility_triggers(self) -> None:
        """Test triggers when volatility exceeds threshold."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        # High variance - includes value far from mean
        closes = [Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100"), Decimal("150")]
        # Mean = 110, max deviation = 40, volatility = 40/110 ≈ 36%

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is True
        assert result.symbol == "BTC-USD"
        assert "exceeds threshold" in result.reason
        assert manager._reduce_only_mode is True
        assert "volatility_breaker_BTC-USD" in manager._reduce_only_reason

    def test_zero_average_not_triggered(self) -> None:
        """Test returns not triggered when average is zero."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        closes = [Decimal("0")] * 5

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False


# ============================================================
# Test: set_reduce_only_mode / is_reduce_only_mode
# ============================================================


class TestReduceOnlyMode:
    """Tests for reduce-only mode methods."""

    def test_default_not_reduce_only(self) -> None:
        """Test default state is not reduce-only."""
        manager = LiveRiskManager()

        assert manager.is_reduce_only_mode() is False

    def test_set_reduce_only_true(self) -> None:
        """Test setting reduce-only to True."""
        manager = LiveRiskManager()

        manager.set_reduce_only_mode(True)

        assert manager.is_reduce_only_mode() is True

    def test_set_reduce_only_false(self) -> None:
        """Test setting reduce-only to False."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True

        manager.set_reduce_only_mode(False)

        assert manager.is_reduce_only_mode() is False

    def test_set_reduce_only_with_reason(self) -> None:
        """Test setting reduce-only with reason."""
        manager = LiveRiskManager()

        manager.set_reduce_only_mode(True, reason="liquidation_warning")

        assert manager._reduce_only_mode is True
        assert manager._reduce_only_reason == "liquidation_warning"

    def test_set_reduce_only_clears_reason_when_false(self) -> None:
        """Test setting reduce-only to False clears reason."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True
        manager._reduce_only_reason = "some_reason"

        manager.set_reduce_only_mode(False, reason="")

        assert manager._reduce_only_mode is False
        assert manager._reduce_only_reason == ""


# ============================================================
# Test: reset_daily_tracking
# ============================================================


class TestResetDailyTracking:
    """Tests for reset_daily_tracking method."""

    def test_resets_all_daily_state(self) -> None:
        """Test resets all daily tracking state."""
        manager = LiveRiskManager()
        manager._start_of_day_equity = Decimal("10000")
        manager._daily_pnl_triggered = True
        manager._reduce_only_mode = True
        manager._reduce_only_reason = "daily_loss_limit"

        manager.reset_daily_tracking()

        assert manager._start_of_day_equity is None
        assert manager._daily_pnl_triggered is False
        assert manager._reduce_only_mode is False
        assert manager._reduce_only_reason == ""

    def test_reset_is_idempotent(self) -> None:
        """Test multiple resets are safe."""
        manager = LiveRiskManager()

        manager.reset_daily_tracking()
        manager.reset_daily_tracking()

        assert manager._start_of_day_equity is None


# ============================================================
# Test: Integration scenarios
# ============================================================


class TestLiveRiskManagerIntegration:
    """Integration tests for LiveRiskManager."""

    def test_daily_workflow(self) -> None:
        """Test typical daily workflow."""
        config = MockConfig(
            max_leverage=Decimal("10"),
            daily_loss_limit_pct=Decimal("0.05"),
            min_liquidation_buffer_pct=Decimal("0.10"),
        )
        manager = LiveRiskManager(config=config)

        # Start of day
        manager.reset_daily_tracking()
        assert manager._start_of_day_equity is None

        # First equity update
        manager.track_daily_pnl(Decimal("10000"), {})
        assert manager._start_of_day_equity == Decimal("10000")

        # Normal trading
        manager.pre_trade_validate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            product=None,
            equity=Decimal("10000"),
            current_positions={},
        )

        # Record metrics
        manager.append_risk_metrics(Decimal("10000"), {})
        assert len(manager._risk_metrics) == 1

        # Small loss - should not trigger
        assert manager.track_daily_pnl(Decimal("9800"), {}) is False

        # Large loss - should trigger
        assert manager.track_daily_pnl(Decimal("9000"), {}) is True
        assert manager.is_reduce_only_mode() is True

        # End of day reset
        manager.reset_daily_tracking()
        assert manager.is_reduce_only_mode() is False

    def test_volatility_triggers_reduce_only(self) -> None:
        """Test volatility breaker triggers reduce-only mode."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)

        assert manager.is_reduce_only_mode() is False

        # High volatility closes
        closes = [Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100"), Decimal("200")]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is True
        assert manager.is_reduce_only_mode() is True

    def test_liquidation_buffer_triggers_reduce_only(self) -> None:
        """Test liquidation buffer check triggers reduce-only for symbol."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.20"))
        manager = LiveRiskManager(config=config)

        # Buffer = |105 - 100| / 105 ≈ 4.76%, threshold 20%
        triggered = manager.check_liquidation_buffer(
            "BTC-PERP",
            {"liquidation_price": 100, "mark": 105},
            Decimal("1000"),
        )

        assert triggered is True
        assert manager.positions["BTC-PERP"]["reduce_only"] is True
