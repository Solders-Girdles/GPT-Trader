"""Tests for telemetry_streaming metrics + gating helpers."""

from __future__ import annotations

from unittest.mock import Mock, patch

from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    _emit_metric,
    _should_enable_streaming,
)


class TestEmitMetric:
    """Tests for _emit_metric function."""

    @patch("gpt_trader.utilities.telemetry.emit_metric")
    def test_emit_metric_calls_utility(self, mock_emit: Mock) -> None:
        """Test _emit_metric calls the utility emit_metric."""
        event_store = Mock()
        bot_id = "test_bot"
        payload = {"event_type": "test"}

        _emit_metric(event_store, bot_id, payload)

        mock_emit.assert_called_once_with(event_store, bot_id, payload)


class TestShouldEnableStreaming:
    """Tests for _should_enable_streaming function."""

    def test_returns_false_for_test_profile(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "test"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is False

    def test_returns_false_when_streaming_disabled(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "prod"
        coordinator.context.config.perps_enable_streaming = False

        result = _should_enable_streaming(coordinator)

        assert result is False

    def test_returns_true_for_prod_with_streaming_enabled(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "prod"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is True

    def test_returns_true_for_canary_with_streaming_enabled(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "canary"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is True

    def test_returns_false_for_dev_profile(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = "dev"
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is False

    def test_handles_profile_with_value_attribute(self) -> None:
        coordinator = Mock()
        profile_enum = Mock()
        profile_enum.value = "prod"
        coordinator.context.config.profile = profile_enum
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is True

    def test_handles_none_profile(self) -> None:
        coordinator = Mock()
        coordinator.context.config.profile = None
        coordinator.context.config.perps_enable_streaming = True

        result = _should_enable_streaming(coordinator)

        assert result is False
