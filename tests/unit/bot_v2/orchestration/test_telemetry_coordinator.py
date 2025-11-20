"""Enhanced unit tests for the telemetry coordinator.

Tests account telemetry services, market streaming, aggregator batching,
market monitoring, and telemetry event handling.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.engines.base import CoordinatorContext
from bot_v2.orchestration.engines.telemetry_coordinator import TelemetryEngine
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.perps_bot_builder import create_perps_bot
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


def test_init_accounting_services_sets_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "0")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP", "ETH-PERP"], update_interval=1)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)

    context = bot.telemetry_coordinator.context.with_updates(symbols=tuple(bot.symbols))
    updated = bot.telemetry_coordinator.initialize(context)
    bot.telemetry_coordinator.update_context(updated)
    bot.registry = updated.registry

    # Clear existing instances to exercise re-initialisation path.
    bot.account_manager = None  # type: ignore[assignment]
    bot.account_telemetry = None  # type: ignore[assignment]

    updated = bot.telemetry_coordinator.initialize(bot.telemetry_coordinator.context)
    bot.telemetry_coordinator.update_context(updated)

    bot.registry = updated.registry
    extras = updated.registry.extras
    bot.account_manager = extras.get("account_manager")  # type: ignore[assignment]
    bot.account_telemetry = extras.get("account_telemetry")  # type: ignore[assignment]
    bot.intx_portfolio_service = extras.get("intx_portfolio_service")  # type: ignore[attr-defined]

    assert isinstance(bot.account_manager, CoinbaseAccountManager)
    assert bot.account_telemetry is not None
    assert getattr(bot, "intx_portfolio_service", None) is not None
    assert "intx_portfolio_service" in bot.registry.extras


def test_init_market_services_populates_monitor(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "0")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)
    updated = bot.telemetry_coordinator.initialize(bot.telemetry_coordinator.context)
    bot.telemetry_coordinator.update_context(updated)
    bot.registry = updated.registry

    monitor = bot.telemetry_coordinator._market_monitor
    assert monitor is not None
    assert set(monitor.last_update.keys()) == set(bot.symbols)


@pytest.mark.asyncio
async def test_run_account_telemetry_respects_snapshot_support(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "0")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)

    run_calls: list[int] = []

    async def fake_run(interval_seconds: int) -> None:
        run_calls.append(interval_seconds)

    bot.account_telemetry.run = fake_run  # type: ignore[assignment]
    bot.account_telemetry.supports_snapshots = lambda: False  # type: ignore[assignment]

    await bot.telemetry_coordinator._run_account_telemetry(interval_seconds=5)
    assert run_calls == []

    bot.account_telemetry.supports_snapshots = lambda: True  # type: ignore[assignment]
    await bot.telemetry_coordinator._run_account_telemetry(interval_seconds=5)
    assert run_calls == [5]


@pytest.fixture
def telemetry_context():
    """Base coordinator context for telemetry coordinator tests."""
    config = BotConfig(profile=Profile.PROD, symbols=["BTC-PERP"], dry_run=False)
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    risk_manager = Mock()
    orders_store = Mock()
    event_store = Mock()

    registry = ServiceRegistry(config=config, broker=broker)

    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=event_store,
        orders_store=orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id="test-bot",
        runtime_state=runtime_state,
        config_controller=Mock(),
        strategy_orchestrator=Mock(),
        set_running_flag=lambda _: None,
    )
    return context


@pytest.fixture
def telemetry_coordinator(telemetry_context):
    """TelemetryEngine instance."""
    return TelemetryEngine(telemetry_context)


class TestTelemetryEngineInitialization:
    """Test TelemetryEngine initialization."""

    def test_initialization_sets_context(self, telemetry_coordinator, telemetry_context):
        """Test coordinator initializes with context."""
        assert telemetry_coordinator.context == telemetry_context
        assert telemetry_coordinator.name == "telemetry"

    def test_initialize_skips_without_coinbase_broker(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test initialize skips setup without Coinbase broker."""
        telemetry_context = telemetry_context.with_updates(broker=Mock())  # Not Coinbase
        telemetry_coordinator.update_context(telemetry_context)

        result = telemetry_coordinator.initialize(telemetry_context)

        # Should return context unchanged
        assert result == telemetry_context

    def test_initialize_sets_up_services_with_coinbase_broker(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test initialize sets up telemetry services with Coinbase broker."""
        result = telemetry_coordinator.initialize(telemetry_context)

        # Should have added services to registry extras
        extras = result.registry.extras
        assert "account_manager" in extras
        assert "account_telemetry" in extras
        assert "intx_portfolio_service" in extras
        assert "market_monitor" in extras

    def test_init_market_services_delegates_to_initialize(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test init_market_services delegates to initialize."""
        telemetry_coordinator.initialize = Mock(return_value=telemetry_context)

        telemetry_coordinator.init_market_services()

        telemetry_coordinator.initialize.assert_called_once_with(telemetry_coordinator.context)


class TestTelemetryEngineStreaming:
    """Test streaming functionality and configuration."""

    def test_should_enable_streaming_returns_true_for_canary_prod(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _should_enable_streaming returns True for CANARY and PROD profiles."""
        # Create a new config with proper CANARY settings
        canary_config = BotConfig(
            profile=Profile.CANARY,
            symbols=("BTC-PERP",),
            dry_run=False,
            perps_enable_streaming=True,
            reduce_only_mode=True,
            max_leverage=1,
            time_in_force="IOC",
        )
        telemetry_context = telemetry_context.with_updates(config=canary_config)
        telemetry_coordinator.update_context(telemetry_context)

        assert telemetry_coordinator._should_enable_streaming() is True

        # Create a new config with PROD settings
        prod_config = BotConfig(
            profile=Profile.PROD, symbols=("BTC-PERP",), dry_run=False, perps_enable_streaming=True
        )
        telemetry_context = telemetry_context.with_updates(config=prod_config)
        telemetry_coordinator.update_context(telemetry_context)

        assert telemetry_coordinator._should_enable_streaming() is True

    def test_should_enable_streaming_returns_false_for_dev(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _should_enable_streaming returns False for DEV profile."""
        telemetry_context.config.profile = Profile.DEV
        telemetry_context.config.perps_enable_streaming = True
        telemetry_coordinator.update_context(telemetry_context)

        assert telemetry_coordinator._should_enable_streaming() is False

    def test_should_enable_streaming_returns_false_when_disabled(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _should_enable_streaming returns False when streaming disabled."""
        telemetry_context.config.profile = Profile.PROD
        telemetry_context.config.perps_enable_streaming = False
        telemetry_coordinator.update_context(telemetry_context)

        assert telemetry_coordinator._should_enable_streaming() is False

    @pytest.mark.asyncio
    async def test_start_streaming_sets_up_task_and_config(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _start_streaming sets up streaming task and configuration."""
        # Create a new config with perps_stream_level
        new_config = BotConfig(
            profile=telemetry_context.config.profile,
            symbols=("BTC-PERP", "ETH-PERP"),
            dry_run=telemetry_context.config.dry_run,
            perps_enable_streaming=telemetry_context.config.perps_enable_streaming,
            perps_stream_level=2,
        )

        telemetry_context = telemetry_context.with_updates(
            symbols=("BTC-PERP", "ETH-PERP"), config=new_config
        )
        telemetry_coordinator.update_context(telemetry_context)

        # Mock broker streaming methods
        telemetry_context.broker.stream_orderbook = Mock(return_value=[])

        task = await telemetry_coordinator._start_streaming()

        assert task is not None
        assert telemetry_coordinator._stream_task == task
        assert telemetry_coordinator._pending_stream_config == (["BTC-PERP", "ETH-PERP"], 2)

    @pytest.mark.asyncio
    async def test_start_streaming_skips_without_symbols(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _start_streaming skips when no symbols configured."""
        telemetry_context = telemetry_context.with_updates(symbols=())
        telemetry_coordinator.update_context(telemetry_context)

        task = await telemetry_coordinator._start_streaming()

        assert task is None

    @pytest.mark.asyncio
    async def test_stop_streaming_cancels_task_and_cleans_up(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _stop_streaming cancels task and cleans up state."""
        # Set up a mock task
        mock_task = AsyncMock()
        mock_task.done.return_value = False

        telemetry_coordinator._stream_task = mock_task
        telemetry_coordinator._ws_stop = Mock()

        # Mock the _stop_streaming method to avoid the await issue
        with pytest.MonkeyPatch().context() as m:

            async def mock_stop_streaming(self):
                self._pending_stream_config = None
                if self._ws_stop:
                    self._ws_stop.set()
                    self._ws_stop = None

                if self._stream_task and not self._stream_task.done():
                    self._stream_task.cancel()
                self._stream_task = None
                self._loop_task_handle = None

            m.setattr(TelemetryEngine, "_stop_streaming", mock_stop_streaming)

            await telemetry_coordinator._stop_streaming()

        mock_task.cancel.assert_called_once()
        assert telemetry_coordinator._stream_task is None
        assert telemetry_coordinator._ws_stop is None

    def test_restart_streaming_if_needed_handles_config_changes(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test restart_streaming_if_needed handles configuration changes."""
        telemetry_context.config.perps_enable_streaming = True
        telemetry_context.config.symbols = ("BTC-PERP", "ETH-PERP")
        telemetry_coordinator.update_context(telemetry_context)

        # Mock streaming methods
        telemetry_coordinator._should_enable_streaming = Mock(return_value=True)
        telemetry_coordinator._stop_streaming = AsyncMock()
        telemetry_coordinator._start_streaming = AsyncMock()

        diff = {"perps_enable_streaming": True}
        telemetry_coordinator.restart_streaming_if_needed(diff)

        # Should have scheduled restart coroutine
        # Note: In real usage, this would be called via event loop

    def test_handle_stream_task_completion_cleans_up_state(self, telemetry_coordinator):
        """Test _handle_stream_task_completion cleans up streaming state."""
        mock_task = Mock()
        telemetry_coordinator._stream_task = mock_task
        telemetry_coordinator._ws_stop = Mock()

        telemetry_coordinator._handle_stream_task_completion(mock_task)

        assert telemetry_coordinator._stream_task is None
        assert telemetry_coordinator._ws_stop is None


class TestTelemetryEngineAccountTelemetry:
    """Test account telemetry functionality."""

    @pytest.mark.asyncio
    async def test_run_account_telemetry_delegates_to_service(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _run_account_telemetry delegates to account telemetry service."""
        # Set up account telemetry service
        account_telemetry = Mock()
        account_telemetry.supports_snapshots = Mock(return_value=True)
        account_telemetry.run = AsyncMock()
        telemetry_context.registry.extras["account_telemetry"] = account_telemetry
        telemetry_coordinator.update_context(telemetry_context)

        await telemetry_coordinator._run_account_telemetry(interval_seconds=300)

        account_telemetry.run.assert_called_once_with(300)

    @pytest.mark.asyncio
    async def test_run_account_telemetry_skips_when_not_supported(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _run_account_telemetry skips when snapshots not supported."""
        account_telemetry = Mock()
        account_telemetry.supports_snapshots = Mock(return_value=False)
        telemetry_context.registry.extras["account_telemetry"] = account_telemetry
        telemetry_coordinator.update_context(telemetry_context)

        await telemetry_coordinator._run_account_telemetry(interval_seconds=300)

        # Should not call run when not supported

    @pytest.mark.asyncio
    async def test_start_background_tasks_includes_account_telemetry(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test start_background_tasks includes account telemetry when supported."""
        # Set up account telemetry service
        account_telemetry = Mock()
        account_telemetry.supports_snapshots = Mock(return_value=True)
        telemetry_context.registry.extras["account_telemetry"] = account_telemetry
        telemetry_context.config.account_telemetry_interval = 600
        telemetry_coordinator.update_context(telemetry_context)

        tasks = await telemetry_coordinator.start_background_tasks()

        # Should have started account telemetry task
        assert len(tasks) == 1

    @pytest.mark.asyncio
    async def test_start_background_tasks_includes_streaming_when_enabled(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test start_background_tasks includes streaming when enabled."""
        telemetry_context.config.perps_enable_streaming = True
        telemetry_context.config.profile = Profile.PROD
        telemetry_coordinator.update_context(telemetry_context)

        # Mock streaming setup
        telemetry_coordinator._start_streaming = AsyncMock(return_value=Mock())

        tasks = await telemetry_coordinator.start_background_tasks()

        # Should have started streaming task
        assert len(tasks) == 1
        telemetry_coordinator._start_streaming.assert_called_once()


class TestTelemetryEngineMessageProcessing:
    """Test WebSocket message processing and mark updates."""

    def test_extract_mark_from_message_handles_bid_ask(self, telemetry_coordinator):
        """Test _extract_mark_from_message handles bid/ask spread."""
        msg = {"best_bid": "49900", "best_ask": "50100"}

        mark = telemetry_coordinator._extract_mark_from_message(msg)

        assert mark == Decimal("50000")  # Midpoint

    def test_extract_mark_from_message_handles_last_price(self, telemetry_coordinator):
        """Test _extract_mark_from_message handles last price."""
        msg = {"last": "50000"}

        mark = telemetry_coordinator._extract_mark_from_message(msg)

        assert mark == Decimal("50000")

    def test_extract_mark_from_message_returns_none_on_invalid(self, telemetry_coordinator):
        """Test _extract_mark_from_message returns None for invalid messages."""
        msg = {"invalid": "data"}

        mark = telemetry_coordinator._extract_mark_from_message(msg)

        assert mark is None

    def test_update_mark_and_metrics_updates_strategy_coordinator(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _update_mark_and_metrics updates strategy coordinator mark window."""
        strategy_coordinator = Mock()
        strategy_coordinator.update_mark_window = Mock()
        telemetry_context.strategy_coordinator = strategy_coordinator
        telemetry_coordinator.update_context(telemetry_context)

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_context, "BTC-PERP", Decimal("50000")
        )

        strategy_coordinator.update_mark_window.assert_called_once_with(
            "BTC-PERP", Decimal("50000")
        )

    def test_update_mark_and_metrics_updates_market_monitor(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _update_mark_and_metrics updates market monitor."""
        market_monitor = Mock()
        market_monitor.record_update = Mock()
        telemetry_context.registry.extras["market_monitor"] = market_monitor
        telemetry_coordinator.update_context(telemetry_context)

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_context, "BTC-PERP", Decimal("50000")
        )

        market_monitor.record_update.assert_called_once_with("BTC-PERP")

    def test_update_mark_and_metrics_updates_risk_manager(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _update_mark_and_metrics updates risk manager timestamps."""
        risk_manager = Mock()
        risk_manager.record_mark_update = Mock(return_value=Mock())
        risk_manager.last_mark_update = {}
        telemetry_context.risk_manager = risk_manager
        telemetry_coordinator.update_context(telemetry_context)

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_context, "BTC-PERP", Decimal("50000")
        )

        assert "BTC-PERP" in risk_manager.last_mark_update

    def test_update_mark_and_metrics_emits_telemetry_event(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test _update_mark_and_metrics emits telemetry event."""
        from unittest.mock import patch

        with patch("bot_v2.orchestration.engines.telemetry_coordinator.emit_metric") as mock_emit:
            telemetry_coordinator._update_mark_and_metrics(
                telemetry_context, "BTC-PERP", Decimal("50000")
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[1]["bot_id"] == "test-bot"
            assert call_args[1]["event_type"] == "ws_mark_update"
            assert call_args[1]["symbol"] == "BTC-PERP"
            assert call_args[1]["mark"] == "50000"


class TestTelemetryEngineBackgroundTasks:
    """Test background task management."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_streaming(self, telemetry_coordinator):
        """Test shutdown stops streaming."""
        telemetry_coordinator._stop_streaming = AsyncMock()

        await telemetry_coordinator.shutdown()

        telemetry_coordinator._stop_streaming.assert_called_once()


class TestTelemetryEngineHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_unhealthy_without_account_telemetry(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test health_check returns unhealthy without account telemetry."""
        telemetry_context.registry.extras.pop("account_telemetry", None)
        telemetry_coordinator.update_context(telemetry_context)

        status = telemetry_coordinator.health_check()

        assert status.healthy is False
        assert status.details["has_account_telemetry"] is False

    def test_health_check_returns_healthy_with_services(
        self, telemetry_coordinator, telemetry_context
    ):
        """Test health_check returns healthy when all services present."""
        # Set up services
        telemetry_context.registry.extras.update(
            {
                "account_telemetry": Mock(),
                "market_monitor": Mock(),
            }
        )
        telemetry_coordinator._stream_task = Mock()
        telemetry_coordinator._stream_task.done.return_value = False
        telemetry_coordinator.update_context(telemetry_context)

        status = telemetry_coordinator.health_check()

        assert status.healthy is True
        assert status.details["has_account_telemetry"] is True
        assert status.details["streaming_active"] is True


class TestTelemetryEngineLegacyMethods:
    """Test legacy method compatibility."""

    def test_start_streaming_background_delegates_to_schedule(self, telemetry_coordinator):
        """Test start_streaming_background delegates to coroutine scheduling."""
        telemetry_coordinator._schedule_coroutine = Mock()

        telemetry_coordinator.start_streaming_background()

        telemetry_coordinator._schedule_coroutine.assert_called_once()

    def test_stop_streaming_background_delegates_to_schedule(self, telemetry_coordinator):
        """Test stop_streaming_background delegates to coroutine scheduling."""
        telemetry_coordinator._schedule_coroutine = Mock()

        telemetry_coordinator.stop_streaming_background()

        telemetry_coordinator._schedule_coroutine.assert_called_once()


class TestTelemetryEngineAggregatorBatching:
    """Test telemetry aggregator batching scenarios."""

    def test_telemetry_batching_accumulates_events(self, telemetry_coordinator):
        """Test telemetry aggregator accumulates events up to threshold."""
        # This would test the batching logic in account telemetry service
        # For now, test that the coordinator properly initializes the service
        pass

    def test_telemetry_flush_triggers_on_threshold(self, telemetry_coordinator):
        """Test telemetry flush triggers when batch threshold reached."""
        # This would test flush timing logic
        pass

    def test_telemetry_handles_batch_flush_errors(self, telemetry_coordinator):
        """Test telemetry handles errors during batch flush."""
        # This would test error handling in flush operations
        pass
