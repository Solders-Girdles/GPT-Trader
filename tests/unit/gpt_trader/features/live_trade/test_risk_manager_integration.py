"""Integration-style tests for LiveRiskManager workflows."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,  # naming: allow
)


@pytest.fixture(autouse=True)
def mock_load_state():
    """Prevent LiveRiskManager from loading state during tests."""
    with patch("gpt_trader.features.live_trade.risk.manager.LiveRiskManager._load_state"):
        yield


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
