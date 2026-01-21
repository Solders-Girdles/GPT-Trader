"""Tests for LiveRiskManager pre-trade validation and workflows."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager, ValidationError
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,  # naming: allow
)


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch):
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", lambda self: None)


class TestPreTradeValidate:
    """Tests for pre_trade_validate method."""

    def test_no_config_passes(self) -> None:
        """Should pass validation when no config."""
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
        """Should pass when leverage within limit."""
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
        """Should raise when leverage exceeds limit."""
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
        """Should skip validation when equity is zero."""
        config = MockConfig(max_leverage=Decimal("5"))
        manager = LiveRiskManager(config=config)

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
        """Should skip validation when equity is negative."""
        config = MockConfig(max_leverage=Decimal("5"))
        manager = LiveRiskManager(config=config)

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


class TestLiveRiskManagerWorkflows:
    """Workflow tests for LiveRiskManager."""

    def test_daily_workflow(self) -> None:
        """Should handle typical daily workflow."""
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
        """Should trigger reduce-only mode on high volatility."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)

        assert manager.is_reduce_only_mode() is False

        closes = [
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("200"),
        ]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is True
        assert manager.is_reduce_only_mode() is True

    def test_liquidation_buffer_triggers_reduce_only(self) -> None:
        """Should trigger reduce-only for symbol when buffer low."""
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
