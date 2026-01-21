"""Tests for LiveRiskManager initialization, ValidationError, stubs, and risk metrics."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.risk.manager as risk_manager_module
from gpt_trader.features.live_trade.risk.manager import LiveRiskManager, ValidationError


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


class TestAppendRiskMetrics:
    """Tests for append_risk_metrics method."""

    def test_appends_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should append metrics with timestamp."""
        monkeypatch.setattr(risk_manager_module.time, "time", lambda: 12345.0)
        manager = LiveRiskManager()

        manager.append_risk_metrics(Decimal("10000"), {"BTC-USD": {"pnl": Decimal("100")}})

        assert len(manager._risk_metrics) == 1
        assert manager._risk_metrics[0]["timestamp"] == 12345.0
        assert manager._risk_metrics[0]["equity"] == "10000"
        assert manager._risk_metrics[0]["positions"] == {"BTC-USD": {"pnl": "100"}}
        assert manager._risk_metrics[0]["reduce_only_mode"] is False

    def test_captures_reduce_only_mode(self) -> None:
        """Should capture reduce_only_mode state."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True

        manager.append_risk_metrics(Decimal("10000"), {})

        assert manager._risk_metrics[0]["reduce_only_mode"] is True

    def test_limits_to_100_metrics(self) -> None:
        """Should keep only last 100 metrics."""
        manager = LiveRiskManager()

        for i in range(150):
            manager.append_risk_metrics(Decimal(str(i)), {})

        assert len(manager._risk_metrics) == 100
        assert manager._risk_metrics[0]["equity"] == "50"
        assert manager._risk_metrics[-1]["equity"] == "149"

    def test_converts_nested_decimals_to_strings(self) -> None:
        """Should convert nested Decimal values to strings."""
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
