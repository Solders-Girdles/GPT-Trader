"""Tests for LiveRiskManager core behavior, validation, and staleness."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager, ValidationError
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,  # naming: allow
)


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", MagicMock())


class TestLiveRiskManagerInit:
    """Tests for LiveRiskManager initialization."""

    def test_init_no_config(self) -> None:
        """Should initialize with default values when no config provided."""
        manager = LiveRiskManager()

        assert manager.config is None
        assert manager.event_store is None
        assert manager._reduce_only_mode is False
        assert manager._reduce_only_reason == ""
        assert manager._daily_pnl_triggered is False
        assert manager._risk_metrics == []
        assert manager._start_of_day_equity is None

    def test_init_with_config(self) -> None:
        """Should accept config dictionary."""
        config = {"max_leverage": 10}
        manager = LiveRiskManager(config=config)

        assert manager.config == {"max_leverage": 10}

    def test_init_with_event_store(self) -> None:
        """Should accept event store reference."""
        event_store = object()
        manager = LiveRiskManager(event_store=event_store)

        assert manager.event_store is event_store

    def test_positions_default_dict(self) -> None:
        """Should return empty dict for non-existent position keys."""
        manager = LiveRiskManager()

        assert manager.positions["BTC-USD"] == {}

    def test_last_mark_update_empty(self) -> None:
        """Should start with empty last_mark_update."""
        manager = LiveRiskManager()

        assert manager.last_mark_update == {}


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_is_exception(self) -> None:
        """Should inherit from Exception."""
        assert issubclass(ValidationError, Exception)

    def test_validation_error_with_message(self) -> None:
        """Should accept error message."""
        with pytest.raises(ValidationError, match="test error"):
            raise ValidationError("test error")

    def test_validation_error_without_message(self) -> None:
        """Should work without message."""
        with pytest.raises(ValidationError):
            raise ValidationError()


class TestLiveRiskManagerStubs:
    """Tests for stub methods."""

    def test_check_order_returns_true(self) -> None:
        """Should always return True."""
        manager = LiveRiskManager()

        assert manager.check_order(None) is True
        assert manager.check_order({"symbol": "BTC-USD"}) is True
        assert manager.check_order(object()) is True

    def test_update_position_is_noop(self) -> None:
        """Should not modify positions."""
        manager = LiveRiskManager()

        manager.update_position(None)
        manager.update_position({"symbol": "BTC-USD"})
        assert manager.positions == {}


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


class TestCheckMarkStaleness:
    """Tests for check_mark_staleness method."""

    def test_no_update_is_stale(self) -> None:
        """Should return True when no update recorded."""
        manager = LiveRiskManager()

        assert manager.check_mark_staleness("BTC-USD") is True

    def test_recent_update_not_stale(self) -> None:
        """Should return False when update is recent."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time()

        assert manager.check_mark_staleness("BTC-USD") is False

    def test_old_update_is_stale(self) -> None:
        """Should return True when update is old."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time() - 200

        assert manager.check_mark_staleness("BTC-USD") is True

    def test_custom_staleness_threshold(self) -> None:
        """Should use config staleness threshold."""
        config = MockConfig(mark_staleness_threshold=30.0)
        manager = LiveRiskManager(config=config)
        manager.last_mark_update["BTC-USD"] = time.time() - 50

        assert manager.check_mark_staleness("BTC-USD") is True  # 50 > 30

    def test_default_threshold_without_config(self) -> None:
        """Should use default 30 second threshold."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time() - 20

        assert manager.check_mark_staleness("BTC-USD") is False  # 20 < 30

    def test_exact_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should handle exact threshold boundary."""
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = 970.0  # Exactly 30 seconds ago

        assert manager.check_mark_staleness("BTC-USD") is False  # 30 > 30 is False
