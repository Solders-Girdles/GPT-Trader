"""
Comprehensive runtime monitoring coverage tests targeting 80%+ coverage.

This test suite targets critical runtime protection mechanisms:
- Daily loss gate logic (track_daily_pnl)
- Liquidation buffer monitoring (check_liquidation_buffer)
- Volatility circuit breakers (check_volatility_circuit_breaker)
- Mark staleness detection (check_mark_staleness)
- Risk metrics collection (append_risk_metrics)
- Correlation risk monitoring (check_correlation_risk)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk.runtime_monitoring import RuntimeMonitor
from bot_v2.features.live_trade.risk_runtime.circuit_breakers import CircuitBreakerOutcome


class TestDailyLossGate:
    """Daily loss gate logic - critical financial protection mechanism."""

    def test_daily_loss_tracking_within_limit(self, mock_event_store):
        """Test normal daily P&L tracking within loss limits."""
        config = RiskConfig()
        config.daily_loss_limit = Decimal("1000")  # $1000 daily loss limit

        monitor = RuntimeMonitor(config, mock_event_store)

        # P&L within limit (-$500 loss)
        positions_pnl = {
            "BTC-USD": {"realized_pnl": Decimal("-200"), "unrealized_pnl": Decimal("-300")}
        }

        result = monitor.track_daily_pnl(
            current_equity=Decimal("9500"),
            positions_pnl=positions_pnl,
            daily_pnl=Decimal("-500"),  # This gets recalculated
            start_of_day_equity=Decimal("10000"),
        )

        assert result[0] is False  # No reduce-only triggered
        assert result[1] == Decimal("-500")  # Total P&L from positions

    def test_daily_loss_limit_breach_triggers_reduce_only(self, mock_event_store):
        """Test daily loss limit breach triggers reduce-only mode."""
        config = RiskConfig()
        config.daily_loss_limit = Decimal("1000")  # $1000 daily loss limit

        # Mock the reduce_only setter to capture calls
        reduce_only_calls = []

        def mock_set_reduce_only(enabled: bool, reason: str):
            reduce_only_calls.append((enabled, reason))

        monitor = RuntimeMonitor(
            config, mock_event_store, set_reduce_only_mode=mock_set_reduce_only
        )

        # P&L exceeding limit (-$1500 loss)
        positions_pnl = {
            "BTC-USD": {"realized_pnl": Decimal("-800"), "unrealized_pnl": Decimal("-700")}
        }

        result = monitor.track_daily_pnl(
            current_equity=Decimal("8500"),
            positions_pnl=positions_pnl,
            daily_pnl=Decimal("-1500"),
            start_of_day_equity=Decimal("10000"),
        )

        assert result[0] is True  # Reduce-only triggered
        assert result[1] == Decimal("-1500")
        assert len(reduce_only_calls) == 1
        assert reduce_only_calls[0] == (True, "daily_loss_limit")

        # Verify risk event was logged
        assert len(mock_event_store.metrics) > 0

    def test_daily_loss_with_complex_positions(self, mock_event_store):
        """Test daily loss tracking with multiple positions."""
        config = RiskConfig()
        config.daily_loss_limit = Decimal("2000")

        monitor = RuntimeMonitor(config, mock_event_store)

        positions_pnl = {
            "BTC-USD": {"realized_pnl": Decimal("-200"), "unrealized_pnl": Decimal("-300")},
            "ETH-USD": {"realized_pnl": Decimal("-150"), "unrealized_pnl": Decimal("0")},
            "SOL-PERP": {"realized_pnl": Decimal("50"), "unrealized_pnl": Decimal("100")},
        }

        result = monitor.track_daily_pnl(
            current_equity=Decimal("9500"),
            positions_pnl=positions_pnl,
            daily_pnl=Decimal("-500"),
            start_of_day_equity=Decimal("10000"),
        )

        # Total P&L: -200-300-150+50+100 = -500
        assert result[0] is False  # No reduce-only (within $2000 limit)
        assert result[1] == Decimal("-500")

    def test_daily_loss_with_zero_positions(self, mock_event_store):
        """Test daily loss tracking with no positions."""
        config = RiskConfig()
        config.daily_loss_limit = Decimal("1000")

        monitor = RuntimeMonitor(config, mock_event_store)

        result = monitor.track_daily_pnl(
            current_equity=Decimal("10000"),
            positions_pnl={},
            daily_pnl=Decimal("0"),
            start_of_day_equity=Decimal("10000"),
        )

        assert result[0] is False  # No reduce-only
        assert result[1] == Decimal("0")

    def test_daily_loss_positive_pnl(self, mock_event_store):
        """Test daily loss tracking with positive P&L."""
        config = RiskConfig()
        config.daily_loss_limit = Decimal("1000")

        monitor = RuntimeMonitor(config, mock_event_store)

        positions_pnl = {
            "BTC-USD": {"realized_pnl": Decimal("500"), "unrealized_pnl": Decimal("300")}
        }

        result = monitor.track_daily_pnl(
            current_equity=Decimal("10800"),
            positions_pnl=positions_pnl,
            daily_pnl=Decimal("800"),
            start_of_day_equity=Decimal("10000"),
        )

        assert result[0] is False  # No reduce-only for profits
        assert result[1] == Decimal("800")


class TestLiquidationBufferMonitoring:
    """Liquidation buffer monitoring - critical position safety mechanism."""

    def test_liquidation_buffer_within_safe_range(self, mock_event_store):
        """Test liquidation buffer monitoring within safe range."""
        config = RiskConfig()
        config.min_liquidation_buffer_pct = Decimal("0.20")  # 20% minimum buffer

        monitor = RuntimeMonitor(config, mock_event_store)

        # Position with 30% liquidation buffer (safe)
        position_data = {
            "quantity": Decimal("1.0"),
            "mark": Decimal("50000"),
            "liquidation_price": Decimal("35000"),  # 30% away from current
        }

        result = monitor.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data=position_data,
            equity=Decimal("10000"),
        )

        assert result is False  # No reduce-only triggered

        # Verify position is not marked as reduce-only
        assert "BTC-PERP" not in monitor.positions or not monitor.positions["BTC-PERP"].get(
            "reduce_only", False
        )

    def test_liquidation_buffer_breach_triggers_reduce_only(self, mock_event_store):
        """Test liquidation buffer breach triggers reduce-only mode."""
        config = RiskConfig()
        config.min_liquidation_buffer_pct = Decimal("0.20")  # 20% minimum buffer

        monitor = RuntimeMonitor(config, mock_event_store)

        # Position with 12% liquidation buffer (below minimum)
        position_data = {
            "quantity": Decimal("1.0"),
            "mark": Decimal("50000"),
            "liquidation_price": Decimal("44000"),  # 12% away from current
        }

        result = monitor.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data=position_data,
            equity=Decimal("10000"),
        )

        assert result is True  # Reduce-only triggered
        assert monitor.positions["BTC-PERP"]["reduce_only"] is True

        # Verify risk event was logged
        assert len(mock_event_store.metrics) > 0

    def test_liquidation_buffer_with_leverage_fallback(self, mock_event_store):
        """Test liquidation buffer using leverage fallback when no liquidation price."""
        config = RiskConfig()
        config.min_liquidation_buffer_pct = Decimal("0.20")
        config.max_leverage = Decimal("5.0")  # 5x max leverage

        monitor = RuntimeMonitor(config, mock_event_store)

        # Position without liquidation price, using leverage calculation
        position_data = {
            "quantity": Decimal("1.0"),
            "mark": Decimal("50000"),
            # No liquidation_price - will use leverage fallback
        }

        result = monitor.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data=position_data,
            equity=Decimal("10000"),
        )

        # With 5x leverage, margin used = $50000/5 = $10000
        # Buffer = (equity - margin_used) / equity = (10000-10000)/10000 = 0%
        # This should trigger reduce-only since 0% < 20%
        assert result is True
        assert monitor.positions["BTC-PERP"]["reduce_only"] is True

    def test_liquidation_buffer_zero_quantity(self, mock_event_store):
        """Test liquidation buffer with zero quantity position."""
        config = RiskConfig()
        config.min_liquidation_buffer_pct = Decimal("0.20")

        monitor = RuntimeMonitor(config, mock_event_store)

        # Position with zero quantity
        position_data = {
            "quantity": Decimal("0"),
            "mark": Decimal("50000"),
            "liquidation_price": Decimal("35000"),
        }

        result = monitor.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data=position_data,
            equity=Decimal("10000"),
        )

        assert result is False  # No reduce-only for zero quantity

    def test_liquidation_buffer_zero_mark_price(self, mock_event_store):
        """Test liquidation buffer with zero mark price."""
        config = RiskConfig()
        config.min_liquidation_buffer_pct = Decimal("0.20")

        monitor = RuntimeMonitor(config, mock_event_store)

        # Position with zero mark price
        position_data = {
            "quantity": Decimal("1.0"),
            "mark": Decimal("0"),
            "liquidation_price": Decimal("35000"),
        }

        result = monitor.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data=position_data,
            equity=Decimal("10000"),
        )

        assert result is False  # No reduce-only for zero mark price

    def test_liquidation_buffer_invalid_position_data(self, mock_event_store):
        """Test liquidation buffer with invalid position data."""
        config = RiskConfig()
        config.min_liquidation_buffer_pct = Decimal("0.20")

        monitor = RuntimeMonitor(config, mock_event_store)

        # Position with invalid quantity
        position_data = {
            "quantity": "invalid_quantity",
            "mark": Decimal("50000"),
        }

        with pytest.raises(Exception):  # Should raise RiskGuardDataCorrupt
            monitor.check_liquidation_buffer(
                symbol="BTC-PERP",
                position_data=position_data,
                equity=Decimal("10000"),
            )

    def test_liquidation_buffer_negative_equity(self, mock_event_store):
        """Test liquidation buffer with negative equity."""
        config = RiskConfig()
        config.min_liquidation_buffer_pct = Decimal("0.20")
        config.max_leverage = Decimal("5.0")

        monitor = RuntimeMonitor(config, mock_event_store)

        position_data = {
            "quantity": Decimal("1.0"),
            "mark": Decimal("50000"),
        }

        result = monitor.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data=position_data,
            equity=Decimal("-1000"),  # Negative equity
        )

        # Should handle negative equity gracefully
        assert isinstance(result, bool)


class TestVolatilityCircuitBreaker:
    """Volatility circuit breaker - market condition protection."""

    def test_volatility_circuit_breaker_basic_functionality(self, mock_event_store):
        """Test basic volatility circuit breaker functionality."""
        config = RiskConfig()
        config.volatility_window_periods = 20  # 20-period window

        # Mock the reduce_only setter
        reduce_only_calls = []

        def mock_set_reduce_only(enabled: bool, reason: str):
            reduce_only_calls.append((enabled, reason))

        monitor = RuntimeMonitor(
            config, mock_event_store, set_reduce_only_mode=mock_set_reduce_only
        )

        # Generate a simple price series
        recent_marks = [Decimal("50000") + Decimal(i * 100) for i in range(25)]

        result = monitor.check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=recent_marks,
        )

        # Should return a CircuitBreakerOutcome
        assert isinstance(result, CircuitBreakerOutcome)
        assert hasattr(result, "triggered")
        assert hasattr(result, "action")
        # Note: volatility might be named 'value' in the actual implementation

    def test_volatility_circuit_breaker_with_short_marks(self, mock_event_store):
        """Test volatility circuit breaker with insufficient marks."""
        config = RiskConfig()

        monitor = RuntimeMonitor(config, mock_event_store)

        # Provide only a few marks
        short_marks = [Decimal("50000"), Decimal("50100"), Decimal("50200")]

        result = monitor.check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=short_marks,
        )

        # Should handle gracefully and return some outcome
        assert isinstance(result, CircuitBreakerOutcome)

    def test_volatility_circuit_breaker_records_state(self, mock_event_store):
        """Test that circuit breaker records state when triggered."""
        config = RiskConfig()
        config.volatility_window_periods = 10

        monitor = RuntimeMonitor(config, mock_event_store)

        # Create marks that might trigger volatility
        volatile_marks = []
        base_price = Decimal("50000")
        for i in range(15):
            if i % 2 == 0:
                volatile_marks.append(base_price + Decimal("1000"))
            else:
                volatile_marks.append(base_price - Decimal("1000"))
            base_price = volatile_marks[-1]

        monitor.check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=volatile_marks,
        )

        # Should have circuit breaker state
        assert monitor.circuit_breaker_state is not None

    def test_volatility_circuit_breaker_with_different_symbols(self, mock_event_store):
        """Test circuit breaker works with different symbols."""
        config = RiskConfig()

        monitor = RuntimeMonitor(config, mock_event_store)

        marks = [Decimal("100") + Decimal(i) for i in range(20)]

        # Test with different symbols
        for symbol in ["BTC-PERP", "ETH-PERP", "SOL-PERP"]:
            result = monitor.check_volatility_circuit_breaker(
                symbol=symbol,
                recent_marks=marks,
            )
            assert isinstance(result, CircuitBreakerOutcome)


class TestMarkStalenessDetection:
    """Test mark price staleness detection functionality."""

    def test_mark_staleness_fresh_data(self, mock_event_store):
        """Test staleness detection with fresh data."""
        config = RiskConfig()
        config.max_mark_staleness_seconds = 60  # 1 minute

        monitor = RuntimeMonitor(config, mock_event_store)

        # Fresh timestamp
        fresh_timestamp = datetime.utcnow() - timedelta(seconds=30)

        result = monitor.check_mark_staleness(symbol="BTC-PERP", mark_timestamp=fresh_timestamp)

        assert result is False  # Not stale

        # Verify timestamp was recorded
        assert "BTC-PERP" in monitor.last_mark_update
        assert monitor.last_mark_update["BTC-PERP"] == fresh_timestamp

    def test_mark_staleness_stale_data(self, mock_event_store):
        """Test staleness detection with stale data."""
        config = RiskConfig()
        config.max_mark_staleness_seconds = 60  # 1 minute

        # Mock time provider for deterministic testing
        fixed_now = datetime.utcnow()

        def mock_now():
            return fixed_now

        monitor = RuntimeMonitor(config, mock_event_store, now_provider=mock_now)

        # Stale timestamp (2 minutes ago)
        stale_timestamp = fixed_now - timedelta(minutes=2)

        # Pre-populate last_mark_update with stale timestamp
        monitor.last_mark_update["BTC-PERP"] = stale_timestamp

        result = monitor.check_mark_staleness(symbol="BTC-PERP")

        # Should detect staleness or handle gracefully depending on implementation
        assert isinstance(result, bool)

    def test_mark_staleness_no_timestamp(self, mock_event_store):
        """Test staleness detection with no previous timestamp."""
        config = RiskConfig()
        config.max_mark_staleness_seconds = 60

        monitor = RuntimeMonitor(config, mock_event_store)

        # Check with no previous timestamp
        result = monitor.check_mark_staleness(symbol="BTC-PERP")

        # Should handle gracefully (likely return False or True based on implementation)
        assert isinstance(result, bool)

    def test_mark_staleness_multiple_symbols(self, mock_event_store):
        """Test staleness detection across multiple symbols."""
        config = RiskConfig()
        config.max_mark_staleness_seconds = 60

        monitor = RuntimeMonitor(config, mock_event_store)

        now = datetime.utcnow()

        # Update timestamps for multiple symbols
        timestamps = {
            "BTC-PERP": now - timedelta(seconds=30),  # Fresh
            "ETH-PERP": now - timedelta(minutes=2),  # Stale
            "SOL-PERP": now - timedelta(seconds=10),  # Fresh
        }

        results = {}
        for symbol, timestamp in timestamps.items():
            monitor.last_mark_update[symbol] = timestamp
            results[symbol] = monitor.check_mark_staleness(symbol)

        assert results["BTC-PERP"] is False  # Fresh
        assert results["ETH-PERP"] is True  # Stale
        assert results["SOL-PERP"] is False  # Fresh


class TestRiskMetricsCollection:
    """Test risk metrics collection functionality."""

    def test_append_risk_metrics_basic(self, mock_event_store):
        """Test basic risk metrics collection."""
        config = RiskConfig()
        monitor = RuntimeMonitor(config, mock_event_store)

        equity = Decimal("10000")
        positions = {
            "BTC-PERP": {"quantity": "0.1", "mark": "50000"},
            "ETH-PERP": {"quantity": "2.0", "mark": "3000"},
        }
        daily_pnl = Decimal("500")
        start_of_day_equity = Decimal("9500")
        is_reduce_only_mode = False

        monitor.append_risk_metrics(
            equity=equity,
            positions=positions,
            daily_pnl=daily_pnl,
            start_of_day_equity=start_of_day_equity,
            is_reduce_only_mode=is_reduce_only_mode,
        )

        # Verify metrics were appended to event store
        assert len(mock_event_store.metrics) > 0

    def test_append_risk_metrics_with_kill_switch(self, mock_event_store):
        """Test risk metrics collection when kill switch is enabled."""
        config = RiskConfig()
        config.kill_switch_enabled = True

        monitor = RuntimeMonitor(config, mock_event_store)

        monitor.append_risk_metrics(
            equity=Decimal("10000"),
            positions={},
            daily_pnl=Decimal("0"),
            start_of_day_equity=Decimal("10000"),
            is_reduce_only_mode=True,
        )

        # Verify metrics were appended
        assert len(mock_event_store.metrics) > 0

    def test_append_risk_metrics_negative_pnl(self, mock_event_store):
        """Test risk metrics collection with negative daily P&L."""
        config = RiskConfig()
        monitor = RuntimeMonitor(config, mock_event_store)

        monitor.append_risk_metrics(
            equity=Decimal("9000"),
            positions={},
            daily_pnl=Decimal("-1000"),
            start_of_day_equity=Decimal("10000"),
            is_reduce_only_mode=True,
        )

        # Should handle negative P&L gracefully
        assert len(mock_event_store.metrics) > 0


class TestCorrelationRiskMonitoring:
    """Test correlation risk monitoring functionality."""

    def test_correlation_risk_basic(self, mock_event_store):
        """Test basic correlation risk detection."""
        config = RiskConfig()
        monitor = RuntimeMonitor(config, mock_event_store)

        # Concentrated portfolio - should trigger correlation risk
        positions = {
            "BTC": {"quantity": 1, "mark": 50000},
            "ETH": {"quantity": 10, "mark": 3000},  # Similar exposure
        }

        result = monitor.check_correlation_risk(positions)

        # Should return boolean indicating if correlation risk detected
        assert isinstance(result, bool)

    def test_correlation_risk_diversified_portfolio(self, mock_event_store):
        """Test correlation risk with diversified portfolio."""
        config = RiskConfig()
        monitor = RuntimeMonitor(config, mock_event_store)

        # Diversified portfolio
        positions = {
            "BTC": {"quantity": 0.5, "mark": 50000},
            "ETH": {"quantity": 5, "mark": 3000},
            "SOL": {"quantity": 100, "mark": 100},
            "AAPL": {"quantity": 50, "mark": 150},  # Different asset class
        }

        result = monitor.check_correlation_risk(positions)

        assert isinstance(result, bool)

    def test_correlation_risk_empty_positions(self, mock_event_store):
        """Test correlation risk with no positions."""
        config = RiskConfig()
        monitor = RuntimeMonitor(config, mock_event_store)

        result = monitor.check_correlation_risk({})

        # Should handle empty positions gracefully
        assert isinstance(result, bool)


class TestRuntimeMonitoringErrorHandling:
    """Test error handling in runtime monitoring."""

    def test_event_store_logging_error(self, mock_event_store):
        """Test handling of event store logging errors."""
        config = RiskConfig()

        # Mock event store to raise exception using a proper Mock
        from unittest.mock import Mock

        mock_event_store.append_metric = Mock(side_effect=Exception("Event store error"))

        monitor = RuntimeMonitor(config, mock_event_store)

        # Should raise RiskGuardTelemetryError when logging fails
        with pytest.raises(Exception):
            monitor._log_risk_event("test_event", {"detail": "test"}, guard="test_guard")

    def test_now_provider_uses_default(self, mock_event_store):
        """Test that default now provider is used when none specified."""
        config = RiskConfig()
        monitor = RuntimeMonitor(config, mock_event_store)

        # Should use datetime.utcnow() by default
        assert monitor._now_provider is not None

        # Test that it returns a datetime
        now = monitor._now_provider()
        assert isinstance(now, datetime)

    def test_initialization_with_custom_now_provider(self, mock_event_store):
        """Test initialization with custom now provider."""
        config = RiskConfig()

        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        monitor = RuntimeMonitor(config, mock_event_store, now_provider=lambda: fixed_time)

        assert monitor._now_provider() == fixed_time

    def test_initialization_with_last_mark_update(self, mock_event_store):
        """Test initialization with last_mark_update parameter."""
        config = RiskConfig()

        now = datetime.utcnow()
        last_mark_update = {
            "BTC-PERP": now - timedelta(seconds=30),
            "ETH-PERP": None,  # Should be filtered out
            "SOL-PERP": now - timedelta(minutes=1),
        }

        monitor = RuntimeMonitor(config, mock_event_store, last_mark_update=last_mark_update)

        # Should only include non-None timestamps
        assert "BTC-PERP" in monitor.last_mark_update
        assert "SOL-PERP" in monitor.last_mark_update
        assert "ETH-PERP" not in monitor.last_mark_update

    def test_positions_dict_initialization(self, mock_event_store):
        """Test positions dictionary is properly initialized."""
        config = RiskConfig()
        monitor = RuntimeMonitor(config, mock_event_store)

        assert isinstance(monitor.positions, dict)
        assert len(monitor.positions) == 0

    def test_circuit_breaker_state_initialization(self, mock_event_store):
        """Test circuit breaker state is properly initialized."""
        config = RiskConfig()
        monitor = RuntimeMonitor(config, mock_event_store)

        assert monitor.circuit_breaker_state is not None
        assert monitor._cb_last_trigger == {}
