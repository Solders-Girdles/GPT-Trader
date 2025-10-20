"""
Tests for guard actions including cancellation, invalidation, volatility guard, and circuit breaker behavior.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import Candle
from bot_v2.features.live_trade.guard_errors import (
    RiskGuardActionError,
    RiskGuardComputationError,
    RiskGuardDataCorrupt,
    RiskGuardDataUnavailable,
    RiskGuardTelemetryError,
)


class TestGuardActions:
    """Test individual guard actions and their behavior."""

    @pytest.mark.asyncio
    async def test_guard_daily_loss_triggered(
        self,
        guard_manager,
        fake_risk_manager,
        runtime_guard_state,
        cancel_orders_callback,
        invalidate_cache_callback,
    ):
        """Test guard_daily_loss triggers order cancellation."""
        fake_risk_manager.track_daily_pnl.return_value = True

        guard_manager.guard_daily_loss(runtime_guard_state)

        cancel_orders_callback.assert_called_once()
        invalidate_cache_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_daily_loss_cancel_failure(
        self, guard_manager, fake_risk_manager, runtime_guard_state, cancel_orders_callback
    ):
        """Test guard_daily_loss handles cancel order failures."""
        fake_risk_manager.track_daily_pnl.return_value = True
        cancel_orders_callback.side_effect = Exception("Cancel failed")

        with pytest.raises(RiskGuardActionError) as exc_info:
            guard_manager.guard_daily_loss(runtime_guard_state)

        assert "Failed to cancel orders" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_guard_liquidation_buffers_success(
        self, guard_manager, fake_broker, fake_risk_manager, runtime_guard_state
    ):
        """Test guard_liquidation_buffers runs successfully."""
        fake_broker.get_position_risk.return_value = {"liquidation_price": "45000"}

        guard_manager.guard_liquidation_buffers(runtime_guard_state, incremental=False)

        fake_risk_manager.check_liquidation_buffer.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_liquidation_buffers_data_corrupt(self, guard_manager, runtime_guard_state):
        """Test guard_liquidation_buffers handles data corruption."""
        # Corrupt position data by setting mark_price to None
        runtime_guard_state.positions[0].mark_price = None

        with pytest.raises(RiskGuardDataCorrupt) as exc_info:
            guard_manager.guard_liquidation_buffers(runtime_guard_state, incremental=False)

        assert "Position payload missing numeric fields" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_guard_liquidation_buffers_risk_fetch_failure(
        self, guard_manager, fake_broker, runtime_guard_state
    ):
        """Test guard_liquidation_buffers handles risk fetch failures."""
        fake_broker.get_position_risk.side_effect = Exception("Risk fetch failed")

        with pytest.raises(RiskGuardDataUnavailable) as exc_info:
            guard_manager.guard_liquidation_buffers(runtime_guard_state, incremental=False)

        assert "Failed to fetch position risk" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_guard_mark_staleness_success(
        self, guard_manager, fake_broker, fake_risk_manager, runtime_guard_state
    ):
        """Test guard_mark_staleness runs successfully."""
        fake_broker._mark_cache.get_mark.return_value = None  # This triggers check_mark_staleness

        guard_manager.guard_mark_staleness(runtime_guard_state)

        fake_risk_manager.check_mark_staleness.assert_called_once_with("BTC-PERP")

    @pytest.mark.asyncio
    async def test_guard_mark_staleness_cache_failure(
        self, guard_manager, fake_broker, runtime_guard_state
    ):
        """Test guard_mark_staleness handles cache failures."""
        fake_broker._mark_cache.get_mark.side_effect = Exception("Cache failed")

        with pytest.raises(RiskGuardDataUnavailable) as exc_info:
            guard_manager.guard_mark_staleness(runtime_guard_state)

        assert "Failed to refresh mark data" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_guard_risk_metrics_success(
        self, guard_manager, fake_risk_manager, runtime_guard_state
    ):
        """Test guard_risk_metrics runs successfully."""
        guard_manager.guard_risk_metrics(runtime_guard_state)

        fake_risk_manager.append_risk_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_risk_metrics_failure(
        self, guard_manager, fake_risk_manager, runtime_guard_state
    ):
        """Test guard_risk_metrics handles failures."""
        fake_risk_manager.append_risk_metrics.side_effect = Exception("Metrics failed")

        with pytest.raises(RiskGuardTelemetryError) as exc_info:
            guard_manager.guard_risk_metrics(runtime_guard_state)

        assert "Failed to append risk metrics" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_guard_correlation_success(
        self, guard_manager, fake_risk_manager, runtime_guard_state
    ):
        """Test guard_correlation runs successfully."""
        guard_manager.guard_correlation(runtime_guard_state)

        fake_risk_manager.check_correlation_risk.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_correlation_failure(
        self, guard_manager, fake_risk_manager, runtime_guard_state
    ):
        """Test guard_correlation handles failures."""
        fake_risk_manager.check_correlation_risk.side_effect = Exception("Correlation failed")

        with pytest.raises(RiskGuardComputationError) as exc_info:
            guard_manager.guard_correlation(runtime_guard_state)

        assert "Correlation risk check failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_guard_volatility_success(
        self, guard_manager, fake_broker, fake_risk_manager, runtime_guard_state
    ):
        """Test guard_volatility runs successfully."""
        candles = [Mock(spec=Candle, close=Decimal("50000")) for _ in range(25)]
        # Mock the async call to return the candles directly
        fake_broker.get_candles = Mock(return_value=candles)

        guard_manager.guard_volatility(runtime_guard_state)

        fake_broker.get_candles.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_volatility_circuit_breaker_triggered(
        self, guard_manager, fake_broker, fake_risk_manager, runtime_guard_state
    ):
        """Test guard_volatility triggers circuit breaker."""
        candles = [Mock(spec=Candle, close=Decimal("50000")) for _ in range(25)]
        fake_broker.get_candles = Mock(return_value=candles)
        fake_risk_manager.check_volatility_circuit_breaker.return_value = Mock(
            triggered=True, to_payload=lambda: {"triggered": True}
        )

        guard_manager.guard_volatility(runtime_guard_state)

        assert len(runtime_guard_state.guard_events) == 1
        assert runtime_guard_state.guard_events[0]["triggered"] is True

    @pytest.mark.asyncio
    async def test_guard_volatility_candle_fetch_failure(
        self, guard_manager, fake_broker, runtime_guard_state
    ):
        """Test guard_volatility handles candle fetch failures."""
        fake_broker.get_candles = Mock(side_effect=Exception("Candle fetch failed"))

        with pytest.raises(RiskGuardDataUnavailable) as exc_info:
            guard_manager.guard_volatility(runtime_guard_state)

        assert "Failed to fetch candles" in str(exc_info.value)
