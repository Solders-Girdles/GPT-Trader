"""Tests for telemetry_health mark extraction and health checks."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import Mock

from gpt_trader.features.live_trade.engines.telemetry_health import (
    extract_mark_from_message,
    health_check,
)


class TestExtractMarkFromMessage:
    def test_extracts_from_bid_ask(self) -> None:
        msg = {"best_bid": "50000", "best_ask": "50010"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("50005")

    def test_extracts_from_bid_ask_strings(self) -> None:
        msg = {"bid": "3000.50", "ask": "3001.50"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("3001")

    def test_prefers_best_bid_over_bid(self) -> None:
        msg = {"best_bid": "100", "bid": "90", "best_ask": "102", "ask": "92"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("101")

    def test_extracts_from_last_price(self) -> None:
        msg = {"last": "45678.90"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("45678.90")

    def test_extracts_from_price_key(self) -> None:
        msg = {"price": "1234.56"}
        result = extract_mark_from_message(msg)

        assert result == Decimal("1234.56")

    def test_returns_none_for_empty_message(self) -> None:
        msg: dict[str, Any] = {}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_zero_mark(self) -> None:
        msg = {"last": "0"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_negative_mark(self) -> None:
        msg = {"last": "-100"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_zero_bid_ask_average(self) -> None:
        msg = {"bid": "0", "ask": "0"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_returns_none_for_invalid_decimal(self) -> None:
        msg = {"last": "not_a_number"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_handles_bid_only(self) -> None:
        msg = {"bid": "100"}
        result = extract_mark_from_message(msg)

        assert result is None

    def test_handles_ask_only(self) -> None:
        msg = {"ask": "100"}
        result = extract_mark_from_message(msg)

        assert result is None


class TestHealthCheck:
    def _create_mock_coordinator(self) -> Mock:
        coordinator = Mock()
        coordinator.name = "telemetry"
        coordinator._market_monitor = None
        coordinator._stream_task = None
        coordinator._background_tasks = []
        coordinator.context.registry.extras = {}
        return coordinator

    def test_healthy_when_market_monitor_present(self) -> None:
        coordinator = self._create_mock_coordinator()
        coordinator._market_monitor = Mock()

        result = health_check(coordinator)

        assert result.healthy is True
        assert result.component == "telemetry"
        assert result.details["has_market_monitor"] is True

    def test_unhealthy_when_no_monitor_or_streaming(self) -> None:
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert result.healthy is False
        assert result.details["has_market_monitor"] is False
        assert result.details["streaming_active"] is False

    def test_healthy_when_streaming_active(self) -> None:
        coordinator = self._create_mock_coordinator()
        task = Mock()
        task.done.return_value = False
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.healthy is True
        assert result.details["streaming_active"] is True

    def test_detects_market_monitor_from_coordinator(self) -> None:
        coordinator = self._create_mock_coordinator()
        coordinator._market_monitor = Mock()

        result = health_check(coordinator)

        assert result.details["has_market_monitor"] is True

    def test_no_market_monitor(self) -> None:
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert result.details["has_market_monitor"] is False

    def test_streaming_active_when_task_running(self) -> None:
        coordinator = self._create_mock_coordinator()
        task = Mock()
        task.done.return_value = False
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.details["streaming_active"] is True

    def test_streaming_inactive_when_task_done(self) -> None:
        coordinator = self._create_mock_coordinator()
        task = Mock()
        task.done.return_value = True
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.details["streaming_active"] is False

    def test_streaming_inactive_when_no_task(self) -> None:
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert result.details["streaming_active"] is False

    def test_counts_background_tasks(self) -> None:
        coordinator = self._create_mock_coordinator()
        coordinator._background_tasks = [Mock(), Mock(), Mock()]

        result = health_check(coordinator)

        assert result.details["background_tasks"] == 3

    def test_health_with_both_monitor_and_streaming(self) -> None:
        coordinator = self._create_mock_coordinator()
        coordinator._market_monitor = Mock()
        task = Mock()
        task.done.return_value = False
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.healthy is True
        assert result.details["has_market_monitor"] is True
        assert result.details["streaming_active"] is True

    def test_health_details_always_present(self) -> None:
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert "has_market_monitor" in result.details
        assert "streaming_active" in result.details
        assert "background_tasks" in result.details
