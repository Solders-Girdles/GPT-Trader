"""Tests for streaming restart lifecycle and configuration changes.

This module tests:
- Streaming restart behavior with configuration changes
- Detection of relevant vs irrelevant config changes
- Full restart cycles with streaming enabled/disabled
- Error handling for stop/start failures
- Symbol and stream level change handling
- Concurrent configuration changes
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator


class TestStreamingRestartLifecycle:
    """Test streaming restart functionality with various configuration changes."""

    def test_restart_streaming_if_needed_ignores_irrelevant_config_changes(
        self, make_context
    ) -> None:
        """Test that restart is ignored for irrelevant configuration changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the streaming methods to track if they're called
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock()
        coordinator._schedule_coroutine = Mock()

        # Test irrelevant config changes
        irrelevant_diffs = [
            {"some_other_field": "value"},
            {"unrelated_config": True},
            {"random_setting": 123},
            {"not_streaming_related": "ignored"},
        ]

        for diff in irrelevant_diffs:
            coordinator.restart_streaming_if_needed(diff)

        # Should not schedule any coroutine for irrelevant changes
        coordinator._schedule_coroutine.assert_not_called()

    def test_restart_streaming_if_needed_handles_streaming_config_changes(
        self, make_context
    ) -> None:
        """Test restart behavior for streaming-related configuration changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the streaming methods
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock()
        coordinator._schedule_coroutine = Mock()

        # Test streaming-related config changes
        streaming_diffs = [
            {"perps_enable_streaming": True},
            {"perps_stream_level": 2},
            {"symbols": ["BTC-PERP", "ETH-PERP"]},
            {"perps_enable_streaming": False, "symbols": ["ETH-PERP"]},
        ]

        for diff in streaming_diffs:
            coordinator.restart_streaming_if_needed(diff)

        # Should schedule coroutine for each streaming-related change
        assert coordinator._schedule_coroutine.call_count == len(streaming_diffs)

    def test_restart_streaming_if_needed_with_mixed_config_changes(self, make_context) -> None:
        """Test restart behavior with mixed relevant and irrelevant config changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the streaming methods
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock()
        coordinator._schedule_coroutine = Mock()

        # Test mixed config changes
        mixed_diff = {
            "perps_enable_streaming": True,  # relevant
            "some_other_field": "value",  # irrelevant
            "perps_stream_level": 3,  # relevant
            "random_setting": 123,  # irrelevant
        }

        coordinator.restart_streaming_if_needed(mixed_diff)

        # Should schedule coroutine once for any relevant changes
        coordinator._schedule_coroutine.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_streaming_full_cycle_with_streaming_enabled(
        self, make_context
    ) -> None:
        """Test complete restart cycle when streaming should be enabled."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)

        # Update context to enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock streaming methods
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock(return_value=Mock())

        # Test restart with streaming enabled
        diff = {"perps_enable_streaming": True}
        coordinator.restart_streaming_if_needed(diff)

        # Allow async operations to complete
        await asyncio.sleep(0.1)

        # Should have stopped and started streaming
        coordinator._stop_streaming.assert_called_once()
        coordinator._start_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_streaming_full_cycle_with_streaming_disabled(
        self, make_context
    ) -> None:
        """Test complete restart cycle when streaming should be disabled."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)

        # Update context to disable streaming
        updated_config = context.config.model_copy(update={"perps_enable_streaming": False})
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock streaming methods
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock(return_value=Mock())

        # Test restart with streaming disabled
        diff = {"perps_enable_streaming": False}
        coordinator.restart_streaming_if_needed(diff)

        # Allow async operations to complete
        await asyncio.sleep(0.1)

        # Should have stopped but not started streaming
        coordinator._stop_streaming.assert_called_once()
        coordinator._start_streaming.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_streaming_handles_stop_failure(self, make_context) -> None:
        """Test restart when stop streaming operation fails."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)

        # Enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock streaming to fail on stop
        coordinator._stop_streaming = AsyncMock(side_effect=Exception("Stop failed"))
        coordinator._start_streaming = AsyncMock(return_value=Mock())

        # Should still attempt restart despite stop failure
        diff = {"perps_enable_streaming": True}
        coordinator.restart_streaming_if_needed(diff)

        # Allow async operations to complete
        await asyncio.sleep(0.1)

        # Should have attempted both operations
        coordinator._stop_streaming.assert_called_once()
        coordinator._start_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_streaming_handles_start_failure(self, make_context) -> None:
        """Test restart when start streaming operation fails."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)

        # Enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock streaming to fail on start
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock(side_effect=Exception("Start failed"))

        # Should handle start failure gracefully
        diff = {"perps_enable_streaming": True}
        coordinator.restart_streaming_if_needed(diff)

        # Allow async operations to complete
        await asyncio.sleep(0.1)

        # Should have attempted both operations
        coordinator._stop_streaming.assert_called_once()
        coordinator._start_streaming.assert_called_once()

    def test_restart_streaming_with_symbols_change(self, make_context) -> None:
        """Test restart behavior when symbols configuration changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the streaming methods
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock()
        coordinator._schedule_coroutine = Mock()

        # Test symbols change
        diff = {"symbols": ["ETH-PERP", "SOL-PERP"]}
        coordinator.restart_streaming_if_needed(diff)

        # Should schedule restart for symbols change
        coordinator._schedule_coroutine.assert_called_once()

    def test_restart_streaming_with_stream_level_change(self, make_context) -> None:
        """Test restart behavior when stream level changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the streaming methods
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock()
        coordinator._schedule_coroutine = Mock()

        # Test stream level change
        diff = {"perps_stream_level": 5}
        coordinator.restart_streaming_if_needed(diff)

        # Should schedule restart for stream level change
        coordinator._schedule_coroutine.assert_called_once()

    def test_restart_streaming_with_empty_config_diff(self, make_context) -> None:
        """Test restart behavior with empty configuration diff."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the streaming methods
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock()
        coordinator._schedule_coroutine = Mock()

        # Test empty diff
        diff = {}
        coordinator.restart_streaming_if_needed(diff)

        # Should not schedule restart for empty diff
        coordinator._schedule_coroutine.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_streaming_with_concurrent_changes(self, make_context) -> None:
        """Test restart behavior when multiple config changes happen concurrently."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = make_context(broker=broker)

        # Enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock streaming methods
        coordinator._stop_streaming = AsyncMock()
        coordinator._start_streaming = AsyncMock(return_value=Mock())

        # Simulate multiple concurrent config changes
        diffs = [
            {"perps_enable_streaming": True},
            {"perps_stream_level": 2},
            {"symbols": ["BTC-PERP", "ETH-PERP"]},
        ]

        for diff in diffs:
            coordinator.restart_streaming_if_needed(diff)

        # Allow async operations to complete
        await asyncio.sleep(0.2)

        # Should have scheduled multiple restarts
        # The exact behavior depends on implementation, but it shouldn't crash
        assert coordinator._stop_streaming.call_count >= 1
        assert coordinator._start_streaming.call_count >= 1
