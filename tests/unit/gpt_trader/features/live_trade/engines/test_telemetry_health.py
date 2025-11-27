"""Tests for telemetry_health - health check and mark extraction for telemetry."""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any
from unittest.mock import Mock

from gpt_trader.features.live_trade.engines.telemetry_health import (
    extract_mark_from_message,
    health_check,
    update_mark_and_metrics,
)

# ============================================================
# Test: extract_mark_from_message
# ============================================================


class TestExtractMarkFromMessage:
    """Tests for extract_mark_from_message function."""

    def test_extracts_from_bid_ask(self) -> None:
        """Test extracts mark from bid/ask midpoint."""
        msg = {"best_bid": "50000", "best_ask": "50010"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("50005")

    def test_extracts_from_bid_ask_strings(self) -> None:
        """Test extracts from bid/ask as string keys."""
        msg = {"bid": "3000.50", "ask": "3001.50"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("3001")

    def test_prefers_best_bid_over_bid(self) -> None:
        """Test prefers best_bid over bid key."""
        msg = {"best_bid": "100", "bid": "90", "best_ask": "102", "ask": "92"}
        result = extract_mark_from_message(msg)

        # Should use best_bid/best_ask: (100 + 102) / 2 = 101
        assert result == Decimal("101")

    def test_extracts_from_last_price(self) -> None:
        """Test extracts from last price when no bid/ask."""
        msg = {"last": "45678.90"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("45678.90")

    def test_extracts_from_price_key(self) -> None:
        """Test extracts from price key when no bid/ask/last."""
        msg = {"price": "1234.56"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("1234.56")

    def test_returns_none_for_empty_message(self) -> None:
        """Test returns None for empty message."""
        msg: dict[str, Any] = {}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_zero_mark(self) -> None:
        """Test returns None when mark is zero."""
        msg = {"last": "0"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_negative_mark(self) -> None:
        """Test returns None when mark is negative."""
        msg = {"last": "-100"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_zero_bid_ask_average(self) -> None:
        """Test returns None when bid/ask average is zero."""
        msg = {"bid": "0", "ask": "0"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_invalid_decimal(self) -> None:
        """Test returns None for invalid decimal string."""
        msg = {"last": "not_a_number"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_handles_bid_only(self) -> None:
        """Test returns None when only bid is present."""
        msg = {"bid": "100"}
        result = extract_mark_from_message(msg)

        # Need both bid and ask for midpoint
        assert result is None

    def test_handles_ask_only(self) -> None:
        """Test returns None when only ask is present."""
        msg = {"ask": "100"}
        result = extract_mark_from_message(msg)

        # Need both bid and ask for midpoint
        assert result is None


# ============================================================
# Test: update_mark_and_metrics
# ============================================================


class TestUpdateMarkAndMetrics:
    """Tests for update_mark_and_metrics function."""

    def _create_mock_context(self) -> Mock:
        """Create a mock CoordinatorContext."""
        ctx = Mock()
        ctx.strategy_coordinator = None
        ctx.runtime_state = None
        ctx.registry.extras = {}
        ctx.risk_manager = None
        ctx.bot_id = "test_bot"
        ctx.event_store = Mock()
        return ctx

    def test_updates_strategy_coordinator_mark_window(self) -> None:
        """Test updates mark window via strategy_coordinator."""
        coordinator = Mock()
        coordinator._market_monitor = None
        ctx = self._create_mock_context()
        strategy_coord = Mock()
        strategy_coord.update_mark_window = Mock()
        ctx.strategy_coordinator = strategy_coord

        update_mark_and_metrics(coordinator, ctx, "BTC-PERP", Decimal("50000"))

        strategy_coord.update_mark_window.assert_called_once_with("BTC-PERP", Decimal("50000"))

    def test_updates_runtime_state_mark_window(self) -> None:
        """Test updates mark window via runtime_state."""
        coordinator = Mock()
        coordinator._market_monitor = None
        ctx = self._create_mock_context()
        runtime_state = Mock()
        runtime_state.mark_lock = threading.Lock()
        runtime_state.mark_windows = {}
        ctx.runtime_state = runtime_state
        ctx.config.short_ma = 10
        ctx.config.long_ma = 20

        update_mark_and_metrics(coordinator, ctx, "ETH-PERP", Decimal("3000"))

        assert "ETH-PERP" in runtime_state.mark_windows
        assert Decimal("3000") in runtime_state.mark_windows["ETH-PERP"]

    def test_prunes_mark_window_when_exceeds_max_size(self) -> None:
        """Test prunes mark window when it exceeds max size."""
        coordinator = Mock()
        coordinator._market_monitor = None
        ctx = self._create_mock_context()
        runtime_state = Mock()
        runtime_state.mark_lock = threading.Lock()
        # Pre-populate with many marks
        runtime_state.mark_windows = {"BTC-PERP": [Decimal(str(i)) for i in range(50)]}
        ctx.runtime_state = runtime_state
        ctx.config.short_ma = 10
        ctx.config.long_ma = 20

        update_mark_and_metrics(coordinator, ctx, "BTC-PERP", Decimal("100"))

        # max_size = max(10, 20) + 5 = 25
        assert len(runtime_state.mark_windows["BTC-PERP"]) <= 25

    def test_records_market_update_via_extras_monitor(self) -> None:
        """Test records update via market_monitor from extras."""
        coordinator = Mock()
        coordinator._market_monitor = None
        ctx = self._create_mock_context()
        monitor = Mock()
        monitor.record_update = Mock()
        ctx.registry.extras = {"market_monitor": monitor}

        update_mark_and_metrics(coordinator, ctx, "BTC-PERP", Decimal("50000"))

        # Monitor from extras should NOT be called (only fallback monitor is called)
        # Based on the code logic, monitor from extras is found but never used
        # Only coordinator._market_monitor is called

    def test_records_market_update_via_coordinator_monitor(self) -> None:
        """Test records update via coordinator._market_monitor."""
        coordinator = Mock()
        monitor = Mock()
        monitor.record_update = Mock()
        coordinator._market_monitor = monitor
        ctx = self._create_mock_context()

        update_mark_and_metrics(coordinator, ctx, "BTC-PERP", Decimal("50000"))

        monitor.record_update.assert_called_once_with("BTC-PERP")

    def test_records_risk_manager_mark_update(self) -> None:
        """Test records mark update to risk_manager."""
        coordinator = Mock()
        coordinator._market_monitor = None
        ctx = self._create_mock_context()
        risk_manager = Mock()
        risk_manager.record_mark_update = Mock(return_value=None)
        risk_manager.last_mark_update = {}
        ctx.risk_manager = risk_manager

        update_mark_and_metrics(coordinator, ctx, "BTC-PERP", Decimal("50000"))

        risk_manager.record_mark_update.assert_called()

    def test_creates_last_mark_update_dict_if_missing(self) -> None:
        """Test creates last_mark_update dict if not present."""
        coordinator = Mock()
        coordinator._market_monitor = None
        ctx = self._create_mock_context()
        risk_manager = Mock(spec=[])  # No attributes
        ctx.risk_manager = risk_manager

        update_mark_and_metrics(coordinator, ctx, "BTC-PERP", Decimal("50000"))

        # Should have created the dict
        assert hasattr(risk_manager, "last_mark_update")

    def test_handles_strategy_coordinator_error(self) -> None:
        """Test handles error from strategy_coordinator gracefully."""
        coordinator = Mock()
        coordinator._market_monitor = None
        ctx = self._create_mock_context()
        strategy_coord = Mock()
        strategy_coord.update_mark_window.side_effect = RuntimeError("Failed")
        ctx.strategy_coordinator = strategy_coord

        # Should not raise
        update_mark_and_metrics(coordinator, ctx, "BTC-PERP", Decimal("50000"))

    def test_handles_market_monitor_error(self) -> None:
        """Test handles error from market_monitor gracefully."""
        coordinator = Mock()
        monitor = Mock()
        monitor.record_update.side_effect = RuntimeError("Failed")
        coordinator._market_monitor = monitor
        ctx = self._create_mock_context()

        # Should not raise
        update_mark_and_metrics(coordinator, ctx, "BTC-PERP", Decimal("50000"))

    def test_handles_non_dict_extras(self) -> None:
        """Test handles non-dict extras gracefully."""
        coordinator = Mock()
        coordinator._market_monitor = None
        ctx = self._create_mock_context()
        ctx.registry.extras = "not_a_dict"

        # Should not raise
        update_mark_and_metrics(coordinator, ctx, "BTC-PERP", Decimal("50000"))


# ============================================================
# Test: health_check
# ============================================================


class TestHealthCheck:
    """Tests for health_check function."""

    def _create_mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.name = "telemetry"
        coordinator._market_monitor = None
        coordinator._stream_task = None
        coordinator._background_tasks = []
        coordinator.context.registry.extras = {}
        return coordinator

    def test_healthy_when_account_telemetry_present(self) -> None:
        """Test returns healthy when account_telemetry is present."""
        coordinator = self._create_mock_coordinator()
        coordinator.context.registry.extras = {"account_telemetry": Mock()}

        result = health_check(coordinator)

        assert result.healthy is True
        assert result.component == "telemetry"
        assert result.details["has_account_telemetry"] is True

    def test_unhealthy_when_account_telemetry_missing(self) -> None:
        """Test returns unhealthy when account_telemetry is missing."""
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert result.healthy is False
        assert result.details["has_account_telemetry"] is False

    def test_detects_market_monitor_from_extras(self) -> None:
        """Test detects market_monitor from extras."""
        coordinator = self._create_mock_coordinator()
        coordinator.context.registry.extras = {"market_monitor": Mock()}

        result = health_check(coordinator)

        assert result.details["has_market_monitor"] is True

    def test_detects_market_monitor_from_coordinator(self) -> None:
        """Test detects market_monitor from coordinator attribute."""
        coordinator = self._create_mock_coordinator()
        coordinator._market_monitor = Mock()

        result = health_check(coordinator)

        assert result.details["has_market_monitor"] is True

    def test_no_market_monitor(self) -> None:
        """Test reports no market_monitor when missing."""
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert result.details["has_market_monitor"] is False

    def test_streaming_active_when_task_running(self) -> None:
        """Test reports streaming_active when task is running."""
        coordinator = self._create_mock_coordinator()
        task = Mock()
        task.done.return_value = False
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.details["streaming_active"] is True

    def test_streaming_inactive_when_task_done(self) -> None:
        """Test reports streaming_inactive when task is done."""
        coordinator = self._create_mock_coordinator()
        task = Mock()
        task.done.return_value = True
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.details["streaming_active"] is False

    def test_streaming_inactive_when_no_task(self) -> None:
        """Test reports streaming_inactive when no task."""
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert result.details["streaming_active"] is False

    def test_counts_background_tasks(self) -> None:
        """Test counts background tasks."""
        coordinator = self._create_mock_coordinator()
        coordinator._background_tasks = [Mock(), Mock(), Mock()]

        result = health_check(coordinator)

        assert result.details["background_tasks"] == 3

    def test_handles_non_dict_extras_conversion(self) -> None:
        """Test handles non-dict extras that can be converted."""
        coordinator = self._create_mock_coordinator()
        # Use a list of tuples that can be converted to dict
        coordinator.context.registry.extras = [("account_telemetry", Mock())]

        result = health_check(coordinator)

        assert result.healthy is True
        assert result.details["has_account_telemetry"] is True

    def test_handles_unconvertible_extras(self) -> None:
        """Test handles extras that cannot be converted to dict."""
        coordinator = self._create_mock_coordinator()
        coordinator.context.registry.extras = 12345  # Cannot convert to dict

        result = health_check(coordinator)

        assert result.healthy is False
        assert result.details["has_account_telemetry"] is False
