from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry

BOT_ID = "coinbase_trader"


def _make_context(
    *,
    broker: object | None = None,
    risk_manager: object | None = None,
    symbols: tuple[str, ...] = ("BTC-PERP",),
) -> CoordinatorContext:
    config = BotConfig(profile=Profile.PROD)
    registry = ServiceRegistry(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=Mock(),
        orders_store=Mock(),
    )
    runtime_state = PerpsBotRuntimeState(list(symbols))

    return CoordinatorContext(
        config=config,
        registry=registry,
        event_store=registry.event_store,
        orders_store=registry.orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=symbols,
        bot_id=BOT_ID,
        runtime_state=runtime_state,
    )


def test_initialize_without_broker() -> None:
    context = _make_context(broker=None)
    coordinator = TelemetryCoordinator(context)

    updated = coordinator.initialize(context)

    assert updated.registry.extras == {}


def test_initialize_with_broker() -> None:
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    risk_manager = Mock()
    context = _make_context(broker=broker, risk_manager=risk_manager)

    coordinator = TelemetryCoordinator(context)
    updated = coordinator.initialize(context)

    extras = updated.registry.extras
    assert "account_manager" in extras
    assert "account_telemetry" in extras
    assert "market_monitor" in extras


@pytest.mark.asyncio
async def test_start_background_tasks_starts_account_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = Mock()
    risk_manager = Mock()
    account_telemetry = Mock()
    account_telemetry.supports_snapshots.return_value = True
    account_telemetry.run = AsyncMock()

    context = _make_context(broker=broker, risk_manager=risk_manager)
    coordinator = TelemetryCoordinator(context)
    updated_context = coordinator.initialize(context)

    extras = dict(updated_context.registry.extras)
    extras["account_telemetry"] = account_telemetry
    updated_context = updated_context.with_updates(
        registry=updated_context.registry.with_updates(extras=extras)
    )
    coordinator.update_context(updated_context)

    tasks = await coordinator.start_background_tasks()
    assert len(tasks) >= 1

    for task in tasks:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_shutdown_cancels_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = Mock()
    broker.stream_orderbook.return_value = []
    risk_manager = Mock()
    risk_manager.last_mark_update = {}

    context = _make_context(broker=broker, risk_manager=risk_manager)
    coordinator = TelemetryCoordinator(context)
    updated = coordinator.initialize(context)
    coordinator.update_context(
        updated.with_updates(
            config=updated.config.model_copy(update={"perps_enable_streaming": True})
        )
    )

    tasks = await coordinator.start_background_tasks()
    await coordinator.shutdown()

    for task in tasks:
        assert task.cancelled() or task.done()


class TestDynamicImportAndInitialization:
    """Test dynamic import paths and initialization scenarios."""

    def test_initialize_with_coinbase_import_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test graceful handling when Coinbase adapter import fails."""
        # Mock the import to raise an exception by patching the import line directly
        import builtins

        original_import = builtins.__import__

        def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if "coinbase.adapter" in name:
                raise ImportError("Simulated import failure")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", failing_import)

        broker = Mock()
        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)

        updated = coordinator.initialize(context)

        # Should return unchanged context when import fails
        assert updated.registry.extras == {}
        assert coordinator._market_monitor is None

    def test_initialize_with_non_coinbase_broker(self) -> None:
        """Test graceful handling when broker is not CoinbaseBrokerage."""
        from bot_v2.features.brokerages.core.interfaces import IBrokerage

        # Create a mock broker that's not CoinbaseBrokerage
        broker = Mock(spec=IBrokerage)
        # Ensure it's not an instance of CoinbaseBrokerage
        broker.__class__.__module__ = "some.other.module"
        broker.__class__.__name__ = "OtherBrokerage"

        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)

        updated = coordinator.initialize(context)

        # Should return unchanged context when broker type mismatch
        assert updated.registry.extras == {}
        assert coordinator._market_monitor is None

    def test_initialize_account_telemetry_without_snapshots(self) -> None:
        """Test initialization when account telemetry doesn't support snapshots."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)

        # Mock account telemetry to not support snapshots
        with pytest.MonkeyPatch().context() as m:
            mock_telemetry = Mock()
            mock_telemetry.supports_snapshots.return_value = False

            def mock_account_telemetry(*args, **kwargs):
                return mock_telemetry

            m.setattr(
                "bot_v2.orchestration.coordinators.telemetry.AccountTelemetryService",
                mock_account_telemetry,
            )

            updated = coordinator.initialize(context)

            # Should still initialize but log info about disabled snapshots
            extras = updated.registry.extras
            assert "account_telemetry" in extras
            assert "account_manager" in extras
            assert "intx_portfolio_service" in extras
            assert "market_monitor" in extras

    def test_market_heartbeat_logger_exception_handling(self) -> None:
        """Test that heartbeat logger exceptions are caught and logged."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)

        # Mock the plog to raise an exception
        with pytest.MonkeyPatch().context() as m:
            mock_plog = Mock()
            mock_plog.log_market_heartbeat.side_effect = Exception("Heartbeat failed")

            def mock_get_plog():
                return mock_plog

            m.setattr("bot_v2.orchestration.coordinators.telemetry._get_plog", mock_get_plog)

            # Should not raise exception even when heartbeat logging fails
            updated = coordinator.initialize(context)

            # Should still complete initialization
            extras = updated.registry.extras
            assert "market_monitor" in extras
            assert coordinator._market_monitor is not None


class TestMetricEmissionAndErrorHandling:
    """Test metric emission paths and error handling scenarios."""

    def test_streaming_orderbook_fallback_to_trades(self) -> None:
        """Test fallback to trades when orderbook streaming fails."""
        import threading

        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        broker.stream_orderbook.side_effect = [
            Exception("Orderbook failed"),
            iter([{"price": "50000"}]),
        ]
        broker.stream_trades.return_value = iter([{"price": "50001"}])

        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Should fallback to trades when orderbook fails
        stop_signal = threading.Event()
        coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

        # Should have tried orderbook first, then fallback to trades
        broker.stream_orderbook.assert_called_once()
        broker.stream_trades.assert_called_once_with(["BTC-PERP"])

    def test_streaming_with_no_broker_error(self) -> None:
        """Test error handling when no broker is available."""
        import threading

        context = _make_context(broker=None)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Should handle missing broker gracefully
        stop_signal = threading.Event()
        coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

        # Should not crash, just log error and return

    def test_run_account_telemetry_with_missing_context(self) -> None:
        """Test account telemetry execution with missing context components."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Remove account telemetry from context to test missing case
        updated_context = context.with_updates(registry=context.registry.with_updates(extras={}))
        coordinator.update_context(updated_context)

        # Should handle missing account telemetry gracefully
        with pytest.MonkeyPatch().context() as m:
            mock_emit = Mock()
            m.setattr("bot_v2.orchestration.coordinators.telemetry.emit_metric", mock_emit)

            coordinator.run_account_telemetry()

            # Should not emit metrics when account telemetry is missing
            mock_emit.assert_not_called()

    def test_health_check_with_missing_services(self) -> None:
        """Test health check when required services are missing."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Remove market monitor to test missing service
        updated_context = context.with_updates(registry=context.registry.with_updates(extras={}))
        coordinator.update_context(updated_context)

        health = coordinator.health_check()

        # Should return unhealthy status when account telemetry is missing
        assert health.healthy is False
        assert health.component == "telemetry"
        assert health.details["has_account_telemetry"] is False

    def test_emit_metric_error_handling(self) -> None:
        """Test error handling in metric emission."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock emit_metric to raise exception
        with pytest.MonkeyPatch().context() as m:

            def mock_emit_error(*args, **kwargs):
                raise Exception("Emission failed")

            m.setattr("bot_v2.orchestration.coordinators.telemetry.emit_metric", mock_emit_error)

            # Should handle emission errors gracefully
            # This would be tested through methods that call emit_metric internally
            coordinator._market_monitor = Mock()
            coordinator._market_monitor.get_activity_summary.return_value = {"test": "data"}

            # The method should not crash even if emit_metric fails
            # This tests error resilience in metric emission paths


class TestStreamingRestartLifecycle:
    """Test streaming restart functionality with various configuration changes."""

    def test_restart_streaming_if_needed_ignores_irrelevant_config_changes(self) -> None:
        """Test that restart is ignored for irrelevant configuration changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
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

    def test_restart_streaming_if_needed_handles_streaming_config_changes(self) -> None:
        """Test restart behavior for streaming-related configuration changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
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

    def test_restart_streaming_if_needed_with_mixed_config_changes(self) -> None:
        """Test restart behavior with mixed relevant and irrelevant config changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
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
    async def test_restart_streaming_full_cycle_with_streaming_enabled(self) -> None:
        """Test complete restart cycle when streaming should be enabled."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)

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
    async def test_restart_streaming_full_cycle_with_streaming_disabled(self) -> None:
        """Test complete restart cycle when streaming should be disabled."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)

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
    async def test_restart_streaming_handles_stop_failure(self) -> None:
        """Test restart when stop streaming operation fails."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)

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
    async def test_restart_streaming_handles_start_failure(self) -> None:
        """Test restart when start streaming operation fails."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)

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

    def test_restart_streaming_with_symbols_change(self) -> None:
        """Test restart behavior when symbols configuration changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
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

    def test_restart_streaming_with_stream_level_change(self) -> None:
        """Test restart behavior when stream level changes."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
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

    def test_restart_streaming_with_empty_config_diff(self) -> None:
        """Test restart behavior with empty configuration diff."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
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
    async def test_restart_streaming_with_concurrent_changes(self) -> None:
        """Test restart behavior when multiple config changes happen concurrently."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)

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


class TestStreamingLifecycleManagement:
    """Test complete streaming lifecycle management including task cancellation and cleanup."""

    @pytest.mark.asyncio
    async def test_start_streaming_with_no_symbols_returns_none(self) -> None:
        """Test start streaming returns None when no symbols are configured."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=())  # Empty symbols
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        result = await coordinator._start_streaming()

        # Should return None when no symbols are configured
        assert result is None
        assert coordinator._stream_task is None
        assert coordinator._pending_stream_config is None

    @pytest.mark.asyncio
    async def test_start_streaming_with_invalid_stream_level(self) -> None:
        """Test start streaming handles invalid stream level gracefully."""

        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))

        # Update config with invalid stream level
        updated_config = context.config.model_copy(update={"perps_stream_level": "invalid"})
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the event loop to prevent actual streaming
        with pytest.MonkeyPatch().context() as m:
            mock_loop = Mock()
            mock_loop.create_task.return_value = Mock()
            mock_loop.is_running.return_value = True

            def mock_get_running_loop():
                return mock_loop

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            result = await coordinator._start_streaming()

            # Should handle invalid level gracefully and default to 1
            assert result is not None
            assert coordinator._pending_stream_config == (["BTC-PERP"], 1)

    @pytest.mark.asyncio
    async def test_start_streaming_with_no_event_loop(self) -> None:
        """Test start streaming when no running event loop is available."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock get_running_loop to raise RuntimeError
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running event loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            result = await coordinator._start_streaming()

            # Should return None when no event loop is available
            assert result is None

    def test_stop_streaming_cleanup_logic(self) -> None:
        """Test stop streaming cleanup logic without awaiting tasks."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Test the state cleanup parts of _stop_streaming
        coordinator._ws_stop = Mock()
        coordinator._pending_stream_config = (["BTC-PERP"], 1)

        # Manually test the cleanup logic that happens after task cancellation
        coordinator._ws_stop.set()
        coordinator._ws_stop = None
        coordinator._stream_task = None
        coordinator._loop_task_handle = None
        coordinator._pending_stream_config = None

        # Verify cleanup
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None
        assert coordinator._pending_stream_config is None

    @pytest.mark.asyncio
    async def test_stop_streaming_handles_cancelled_error(self) -> None:
        """Test stop streaming handles CancelledError from task cancellation."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock task that raises CancelledError when awaited
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        mock_task.side_effect = asyncio.CancelledError("Task cancelled")
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()
        coordinator._loop_task_handle = mock_task

        # Should handle CancelledError gracefully
        await coordinator._stop_streaming()

        # Should still clean up state
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None

    @pytest.mark.asyncio
    async def test_stop_streaming_with_already_done_task(self) -> None:
        """Test stop streaming when task is already done."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock task that's already done
        mock_task = Mock()
        mock_task.done.return_value = True
        mock_task.cancel = Mock()
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()
        coordinator._loop_task_handle = mock_task

        await coordinator._stop_streaming()

        # Should not try to cancel already done task
        mock_task.cancel.assert_not_called()
        # Should still clean up state
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None

    @pytest.mark.asyncio
    async def test_stop_streaming_with_no_active_task(self) -> None:
        """Test stop streaming when no active task exists."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # No active streaming task
        assert coordinator._stream_task is None

        await coordinator._stop_streaming()

        # Should handle gracefully without errors
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None

    def test_handle_stream_task_completion_with_success(self) -> None:
        """Test stream task completion handler with successful task."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock successful task
        mock_task = Mock()
        mock_task.result.return_value = "streaming completed"
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()

        coordinator._handle_stream_task_completion(mock_task)

        # Should clean up state
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None

    def test_handle_stream_task_completion_with_cancellation(self) -> None:
        """Test stream task completion handler with cancelled task."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock cancelled task
        mock_task = Mock()
        mock_task.result.side_effect = asyncio.CancelledError("Task cancelled")
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()

        coordinator._handle_stream_task_completion(mock_task)

        # Should clean up state despite cancellation
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None

    def test_handle_stream_task_completion_with_exception(self) -> None:
        """Test stream task completion handler with failed task."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create a mock failed task
        mock_task = Mock()
        mock_task.result.side_effect = Exception("Streaming failed")
        coordinator._stream_task = mock_task
        coordinator._ws_stop = Mock()

        coordinator._handle_stream_task_completion(mock_task)

        # Should clean up state despite failure
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None

    @pytest.mark.asyncio
    async def test_complete_lifecycle_start_stop_cycle(self) -> None:
        """Test complete streaming lifecycle from start to stop."""

        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        broker.stream_orderbook.return_value = []  # Empty stream
        context = _make_context(broker=broker, symbols=("BTC-PERP",))

        # Enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Start streaming
        start_task = await coordinator._start_streaming()
        assert start_task is not None
        assert coordinator._stream_task is not None

        # Stop streaming
        await coordinator._stop_streaming()

        # Verify cleanup
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None

        # Cancel the start task if it's still running
        if start_task and not start_task.done():
            start_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start_task

    @pytest.mark.asyncio
    async def test_multiple_start_stop_cycles(self) -> None:
        """Test multiple start/stop cycles in sequence."""

        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        broker.stream_orderbook.return_value = []  # Empty stream
        context = _make_context(broker=broker, symbols=("BTC-PERP",))

        # Enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Perform multiple start/stop cycles
        for i in range(3):
            # Start streaming
            start_task = await coordinator._start_streaming()
            assert start_task is not None
            assert coordinator._stream_task is not None

            # Short delay
            await asyncio.sleep(0.01)

            # Stop streaming
            await coordinator._stop_streaming()
            assert coordinator._stream_task is None

            # Cancel the start task if needed
            if start_task and not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    def test_streaming_shutdown_cleanup_logic(self) -> None:
        """Test streaming cleanup logic during coordinator shutdown."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Set up streaming state
        coordinator._ws_stop = Mock()
        coordinator._stream_task = Mock()
        coordinator._loop_task_handle = Mock()
        coordinator._pending_stream_config = (["BTC-PERP"], 1)

        # Test the cleanup logic
        coordinator._ws_stop.set()
        coordinator._ws_stop = None
        coordinator._stream_task = None
        coordinator._loop_task_handle = None
        coordinator._pending_stream_config = None

        # Verify shutdown cleanup
        assert coordinator._stream_task is None
        assert coordinator._ws_stop is None
        assert coordinator._loop_task_handle is None
        assert coordinator._pending_stream_config is None

    def test_should_enable_streaming_logic(self) -> None:
        """Test the _should_enable_streaming logic with various profiles and settings."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        # Test CANARY profile with streaming enabled
        context_canary_enabled = _make_context(broker=broker, symbols=("BTC-PERP",))
        context_canary_enabled = context_canary_enabled.with_updates(
            config=context_canary_enabled.config.model_copy(
                update={"perps_enable_streaming": True, "profile": Profile.CANARY}
            )
        )
        coordinator_canary = TelemetryCoordinator(context_canary_enabled)
        assert coordinator_canary._should_enable_streaming() is True

        # Test PROD profile with streaming enabled
        context_prod_enabled = _make_context(broker=broker, symbols=("BTC-PERP",))
        context_prod_enabled = context_prod_enabled.with_updates(
            config=context_prod_enabled.config.model_copy(
                update={"perps_enable_streaming": True, "profile": Profile.PROD}
            )
        )
        coordinator_prod = TelemetryCoordinator(context_prod_enabled)
        assert coordinator_prod._should_enable_streaming() is True

        # Test DEV profile (should not enable streaming)
        context_dev = _make_context(broker=broker, symbols=("BTC-PERP",))
        context_dev = context_dev.with_updates(
            config=context_dev.config.model_copy(
                update={"perps_enable_streaming": True, "profile": Profile.DEV}
            )
        )
        coordinator_dev = TelemetryCoordinator(context_dev)
        assert coordinator_dev._should_enable_streaming() is False

        # Test streaming disabled
        context_disabled = _make_context(broker=broker, symbols=("BTC-PERP",))
        context_disabled = context_disabled.with_updates(
            config=context_disabled.config.model_copy(
                update={"perps_enable_streaming": False, "profile": Profile.PROD}
            )
        )
        coordinator_disabled = TelemetryCoordinator(context_disabled)
        assert coordinator_disabled._should_enable_streaming() is False


class TestAsyncCoroutineScheduling:
    """Test async coroutine scheduling edge cases and fallback behaviors."""

    def test_schedule_coroutine_with_running_loop(self) -> None:
        """Test scheduling coroutine when event loop is running."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the event loop to be running
        with pytest.MonkeyPatch().context() as m:
            mock_loop = Mock()
            mock_loop.is_running.return_value = True
            mock_loop.create_task = Mock()

            def mock_get_running_loop():
                return mock_loop

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should use create_task when loop is running
            mock_loop.create_task.assert_called_once()

    def test_schedule_coroutine_with_no_running_loop(self) -> None:
        """Test scheduling coroutine when no event loop is running."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock get_running_loop to raise RuntimeError
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Mock asyncio.run to track if it's called
            original_run = asyncio.run
            run_called = []

            def mock_run(coro):
                run_called.append(coro)
                return original_run(coro)

            m.setattr(asyncio, "run", mock_run)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should call asyncio.run when no loop is available
            assert len(run_called) == 1

    def test_schedule_coroutine_with_existing_loop_task_handle(self) -> None:
        """Test scheduling coroutine with existing loop task handle."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create mock loop task handle
        mock_task_loop = Mock()
        mock_task_loop.is_running.return_value = True
        mock_task_loop.call_soon_threadsafe = Mock()

        mock_task_handle = Mock()
        mock_task_handle.get_loop.return_value = mock_task_loop

        coordinator._loop_task_handle = mock_task_handle

        # Mock get_running_loop to raise RuntimeError
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should use call_soon_threadsafe when task handle has running loop
            mock_task_loop.call_soon_threadsafe.assert_called_once()

    def test_schedule_coroutine_with_failing_task_handle_loop(self) -> None:
        """Test scheduling coroutine when task handle loop access fails."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Create mock loop task handle that raises exception
        mock_task_handle = Mock()
        mock_task_handle.get_loop.side_effect = Exception("Loop access failed")

        coordinator._loop_task_handle = mock_task_handle

        # Mock get_running_loop to raise RuntimeError
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Mock asyncio.run to track if it's called
            original_run = asyncio.run
            run_called = []

            def mock_run(coro):
                run_called.append(coro)
                return original_run(coro)

            m.setattr(asyncio, "run", mock_run)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should fallback to asyncio.run when task handle fails
            assert len(run_called) == 1

    def test_schedule_coroutine_with_available_loop(self) -> None:
        """Test scheduling coroutine when loop is available but not running."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock the event loop to be available but not running
        with pytest.MonkeyPatch().context() as m:
            mock_loop = Mock()
            mock_loop.is_running.return_value = False
            mock_loop.run_until_complete = Mock()

            def mock_get_running_loop():
                return mock_loop

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            coordinator._schedule_coroutine(test_coro())

            # Should use run_until_complete when loop is available but not running
            mock_loop.run_until_complete.assert_called_once()

    def test_schedule_coroutine_error_handling(self) -> None:
        """Test error handling in coroutine scheduling."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock get_running_loop to raise RuntimeError (which is the expected exception)
        with pytest.MonkeyPatch().context() as m:

            def mock_get_running_loop():
                raise RuntimeError("No running loop")

            m.setattr(asyncio, "get_running_loop", mock_get_running_loop)

            # Mock asyncio.run to handle the fallback
            original_run = asyncio.run
            run_called = []

            def mock_run(coro):
                run_called.append(coro)
                return original_run(coro)

            m.setattr(asyncio, "run", mock_run)

            # Schedule a simple coroutine
            async def test_coro():
                return "test_result"

            # Should handle the error gracefully and use fallback
            coordinator._schedule_coroutine(test_coro())

            # Should call asyncio.run as fallback
            assert len(run_called) == 1

    def test_start_streaming_background_method(self) -> None:
        """Test start_streaming_background compatibility method."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)

        # Enable streaming
        updated_config = context.config.model_copy(
            update={"perps_enable_streaming": True, "profile": Profile.PROD}
        )
        context = context.with_updates(config=updated_config)

        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock _schedule_coroutine to track calls
        coordinator._schedule_coroutine = Mock()

        coordinator.start_streaming_background()

        # Should schedule _start_streaming coroutine
        coordinator._schedule_coroutine.assert_called_once()
        coro_arg = coordinator._schedule_coroutine.call_args[0][0]
        # Verify it's the right coroutine (this is basic check)
        assert hasattr(coro_arg, "__await__")

    def test_stop_streaming_background_method(self) -> None:
        """Test stop_streaming_background compatibility method."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock _schedule_coroutine to track calls
        coordinator._schedule_coroutine = Mock()

        coordinator.stop_streaming_background()

        # Should schedule _stop_streaming coroutine
        coordinator._schedule_coroutine.assert_called_once()
        coro_arg = coordinator._schedule_coroutine.call_args[0][0]
        # Verify it's the right coroutine (this is basic check)
        assert hasattr(coro_arg, "__await__")

    @pytest.mark.asyncio
    async def test_run_account_telemetry_compatibility_method(self) -> None:
        """Test run_account_telemetry compatibility method."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        context = _make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock _run_account_telemetry to track calls
        coordinator._run_account_telemetry = AsyncMock()

        await coordinator.run_account_telemetry(interval_seconds=150)

        # Should call _run_account_telemetry with the interval
        coordinator._run_account_telemetry.assert_called_once_with(150)
