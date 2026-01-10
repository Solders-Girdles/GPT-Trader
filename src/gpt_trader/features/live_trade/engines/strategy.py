"""
Simplified Strategy Engine.
Replaces the 608-line enterprise coordinator with a simple loop.

State Recovery:
On startup, reads `price_tick` events from EventStore to restore price history.
During operation, persists price ticks to EventStore for crash recovery.

Streaming Lifecycle:
When enabled, starts WebSocket streaming for real-time market data.
Includes WS health watchdog that monitors staleness and triggers degradation.
"""

import asyncio
import threading
import time
from collections import deque
from decimal import Decimal
from typing import Any

from gpt_trader.app.health_server import HealthState
from gpt_trader.config.constants import HEALTH_CHECK_INTERVAL_SECONDS
from gpt_trader.core import OrderSide, OrderType, Position, Product
from gpt_trader.features.live_trade.degradation import DegradationState
from gpt_trader.features.live_trade.engines.base import (
    BaseEngine,
    CoordinatorContext,
    HealthStatus,
)
from gpt_trader.features.live_trade.engines.equity_calculator import (
    EquityCalculator,
)
from gpt_trader.features.live_trade.engines.price_tick_store import (
    EVENT_PRICE_TICK,
    PriceTickStore,
)
from gpt_trader.features.live_trade.engines.system_maintenance import (
    SystemMaintenanceService,
)
from gpt_trader.features.live_trade.engines.telemetry_health import (
    extract_mark_from_message,
    update_mark_and_metrics,
)
from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    _handle_stream_task_completion,
    _run_stream_loop,
    _run_stream_loop_async,
    _schedule_coroutine,
    _should_enable_streaming,
    _start_streaming,
    _stop_streaming,
    start_streaming_background,
    stop_streaming_background,
)
from gpt_trader.features.live_trade.execution.guard_manager import GuardManager
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.features.live_trade.execution.state_collection import StateCollector
from gpt_trader.features.live_trade.execution.submission_result import (
    OrderSubmissionResult,
    OrderSubmissionStatus,
)
from gpt_trader.features.live_trade.execution.validation import (
    OrderValidator,
    get_failure_tracker,
)
from gpt_trader.features.live_trade.factory import create_strategy
from gpt_trader.features.live_trade.guard_errors import GuardError
from gpt_trader.features.live_trade.lifecycle import (
    ENGINE_TRANSITIONS,
    EngineState,
    LifecycleStateMachine,
)
from gpt_trader.features.live_trade.risk.manager import ValidationError
from gpt_trader.features.live_trade.strategies.perps_baseline import (
    Action,
    Decision,
)
from gpt_trader.logging.correlation import correlation_context
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.health_checks import HealthCheckRunner
from gpt_trader.monitoring.heartbeat import HeartbeatService
from gpt_trader.monitoring.metrics_collector import record_histogram
from gpt_trader.monitoring.profiling import profile_span
from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.observability.tracing import trace_span
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="trading_engine")

# Re-export for backward compatibility
__all__ = ["TradingEngine", "EVENT_PRICE_TICK"]


class TradingEngine(BaseEngine):
    """
    Simple trading loop that fetches data and executes strategy.

    Supports state recovery via EventStore persistence.
    """

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self._lifecycle = LifecycleStateMachine(
            initial_state=EngineState.INIT,
            entity="trading_engine",
            transitions=ENGINE_TRANSITIONS,
            logger=logger,
        )
        # Create strategy via factory (supports baseline and mean_reversion)
        self.strategy = create_strategy(self.context.config)
        self._current_positions: dict[str, Position] = {}
        self._rehydrated = False
        self._cycle_count = 0

        # Initialize price tick store for state recovery
        self._price_tick_store = PriceTickStore(
            event_store=context.event_store,
            symbols=list(context.config.symbols),
            bot_id=context.bot_id,
        )

        # Initialize heartbeat service
        self._heartbeat = HeartbeatService(
            event_store=context.event_store,
            ping_url=getattr(context.config, "heartbeat_url", None),
            interval_seconds=getattr(context.config, "heartbeat_interval", 60),
            bot_id=context.bot_id,
            enabled=getattr(context.config, "heartbeat_enabled", True),
        )

        # Initialize status reporter
        self._status_reporter = StatusReporter(
            status_file=getattr(context.config, "status_file", "status.json"),
            update_interval=getattr(context.config, "status_interval", 10),
            bot_id=context.bot_id,
            enabled=getattr(context.config, "status_enabled", True),
        )
        self._status_reporter.set_heartbeat_service(self._heartbeat)

        # Initialize system maintenance service (health reporting + pruning)
        self._system_maintenance = SystemMaintenanceService(
            status_reporter=self._status_reporter,
            event_store=context.event_store,
        )

        # System health tracking
        self._last_latency = 0.0
        self._connection_status = "UNKNOWN"

        # Initialize graceful degradation state
        self._degradation = DegradationState()

        # Initialize equity calculator (extracted for reusability)
        self._equity_calculator = EquityCalculator(
            config=context.config,
            degradation=self._degradation,
            risk_manager=context.risk_manager,
            price_history=self._price_tick_store.price_history,
        )

        # Initialize health check runner for active /health probes
        health_state = context.container.health_state if context.container else HealthState()
        self._health_check_runner = HealthCheckRunner(
            health_state=health_state,
            broker=context.broker,
            degradation_state=self._degradation,
            risk_manager=context.risk_manager,
            interval_seconds=HEALTH_CHECK_INTERVAL_SECONDS,
            message_stale_seconds=getattr(context.config, "ws_message_stale_seconds", 60.0),
            heartbeat_stale_seconds=getattr(context.config, "ws_heartbeat_stale_seconds", 120.0),
        )

        # Initialize streaming lifecycle attributes
        self._ws_stop: threading.Event | None = None
        self._pending_stream_config: tuple[list[str], int] | None = None
        self._stream_task: asyncio.Task[Any] | None = None
        self._loop_task_handle: asyncio.Task[Any] | None = None
        self._market_monitor: Any = None  # Market monitor for telemetry

        # WS health watchdog attributes
        self._ws_health_task: asyncio.Task[Any] | None = None
        self._ws_reconnect_attempts: int = 0
        self._ws_reconnect_delay: float = 1.0
        self._ws_last_health_check: float = 0.0

        # Concurrency control for REST API calls (prevents burst overload)
        # Default to 5 concurrent calls; configurable via config if needed
        max_concurrent_rest = getattr(context.config, "max_concurrent_rest_calls", 5)
        self._rest_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_rest)

        # Initialize pre-trade guard stack (Option A: embedded guards)
        self._init_guard_stack()

    def _init_guard_stack(self) -> None:
        """Initialize StateCollector, OrderValidator, OrderSubmitter for pre-trade guards."""
        # Event store fallback
        event_store = self.context.event_store or EventStore()
        bot_id = str(self.context.bot_id or self.context.config.profile or "live")

        # Broker and risk manager must exist
        broker = self.context.broker
        risk_manager = self.context.risk_manager

        # Track open orders
        self._open_orders: list[str] = []

        # StateCollector: needs broker, config
        self._state_collector = StateCollector(
            broker=broker,  # type: ignore[arg-type]
            config=self.context.config,
            integration_mode=False,
        )

        # OrderSubmitter: broker + event store + bot_id + open_orders
        self._order_submitter = OrderSubmitter(
            broker=broker,  # type: ignore[arg-type]
            event_store=event_store,
            bot_id=bot_id,
            open_orders=self._open_orders,
            integration_mode=False,
        )

        # Failure tracker (global) with escalation callback for graceful degradation
        failure_tracker = get_failure_tracker()

        # Wire escalation callback: on repeated validation failures, pause + reduce-only
        def _on_validation_escalation() -> None:
            """Handle validation infrastructure failures by pausing and setting reduce-only."""
            if risk_manager is None:
                return

            risk_manager.set_reduce_only_mode(True, reason="validation_failures")
            cooldown = 180
            if risk_manager.config is not None:
                cooldown = risk_manager.config.validation_failure_cooldown_seconds
            self._degradation.pause_all(
                seconds=cooldown,
                reason="validation_failures",
                allow_reduce_only=True,
            )
            logger.warning(
                "Validation escalation triggered - pausing trading",
                cooldown_seconds=cooldown,
                operation="degradation",
                stage="validation_escalation",
            )

        failure_tracker.escalation_callback = _on_validation_escalation

        # OrderValidator: broker + risk_manager + preview config + callbacks + tracker
        self._order_validator: OrderValidator | None = None
        if risk_manager is not None:
            self._order_validator = OrderValidator(
                broker=broker,  # type: ignore[arg-type]
                risk_manager=risk_manager,
                enable_order_preview=self.context.config.enable_order_preview,
                record_preview_callback=self._order_submitter.record_preview,
                record_rejection_callback=self._order_submitter.record_rejection,
                failure_tracker=failure_tracker,
            )

        # GuardManager: runtime guards (daily loss, liquidation buffer, volatility)
        self._guard_manager: GuardManager | None = None
        if broker is not None and risk_manager is not None:
            self._guard_manager = GuardManager(
                broker=broker,  # type: ignore[arg-type]
                risk_manager=risk_manager,
                equity_calculator=self._state_collector.calculate_equity_from_balances,
                open_orders=self._open_orders,
                invalidate_cache_callback=lambda: None,
            )

    # =========================================================================
    # Streaming Lifecycle Methods
    # =========================================================================

    def _should_enable_streaming(self) -> bool:
        """Check if streaming should be enabled based on config."""
        return _should_enable_streaming(self)

    def _schedule_coroutine(self, coro: Any) -> None:
        """Schedule a coroutine for execution."""
        _schedule_coroutine(self, coro)

    async def _start_streaming(self) -> asyncio.Task[Any] | None:
        """Start WebSocket streaming."""
        return await _start_streaming(self)

    async def _stop_streaming(self) -> None:
        """Stop WebSocket streaming."""
        await _stop_streaming(self)

    def _handle_stream_task_completion(self, task: asyncio.Task[Any]) -> None:
        """Handle stream task completion callback."""
        _handle_stream_task_completion(self, task)

    async def _run_stream_loop_async(
        self,
        symbols: list[str],
        level: int,
        stop_signal: threading.Event | None,
    ) -> None:
        """Run streaming loop asynchronously."""
        await _run_stream_loop_async(self, symbols, level, stop_signal)

    def _run_stream_loop(
        self,
        symbols: list[str],
        level: int,
        stop_signal: threading.Event | None,
    ) -> None:
        """Run streaming loop synchronously (called from executor)."""
        _run_stream_loop(self, symbols, level, stop_signal)

    def _extract_mark_from_message(self, msg: dict[str, Any]) -> Decimal | None:
        """Extract mark price from WebSocket message."""
        return extract_mark_from_message(msg)

    def _update_mark_and_metrics(
        self,
        ctx: CoordinatorContext,
        symbol: str,
        mark: Decimal,
    ) -> None:
        """Update mark price and related metrics."""
        update_mark_and_metrics(self, ctx, symbol, mark)

    @property
    def status_reporter(self) -> StatusReporter:
        return self._status_reporter

    @property
    def price_history(self) -> dict[str, deque[Decimal]]:
        """Access price history via PriceTickStore."""
        return self._price_tick_store.price_history

    async def _notify(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Send notification if service is available."""
        if self.context.notification_service is None:
            return
        try:
            await self.context.notification_service.notify(
                title=title,
                message=message,
                severity=severity,
                source="TradingEngine",
                context=context,
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    @property
    def name(self) -> str:
        return "strategy"

    @property
    def state(self) -> EngineState:
        return self._lifecycle.state

    @property
    def running(self) -> bool:
        return self.state in (EngineState.STARTING, EngineState.RUNNING)

    @running.setter
    def running(self, value: bool) -> None:
        target = EngineState.RUNNING if value else EngineState.STOPPED
        self._lifecycle.transition(
            target,
            reason="running_override",
            details={"via": "running_set"},
            force=True,
        )

    def _transition_state(
        self,
        target: EngineState,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        return self._lifecycle.transition(
            target,
            reason=reason,
            details=details,
            force=force,
        )

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        """Start the main trading loop and heartbeat service.

        Before starting, attempts to rehydrate state from EventStore.
        """
        self._transition_state(EngineState.STARTING, reason="start_background_tasks")
        # Rehydrate state from EventStore before starting
        if not self._rehydrated:
            self._rehydrate_from_events()
            self._rehydrated = True

        self._transition_state(EngineState.RUNNING, reason="tasks_scheduled")

        tasks: list[asyncio.Task[Any]] = []

        # Start main trading loop
        trading_task = asyncio.create_task(self._run_loop())
        self._register_background_task(trading_task)
        tasks.append(trading_task)

        # Start heartbeat service
        heartbeat_task = await self._heartbeat.start()
        if heartbeat_task:
            self._register_background_task(heartbeat_task)
            tasks.append(heartbeat_task)

        # Start status reporter
        status_task = await self._status_reporter.start()
        if status_task:
            self._register_background_task(status_task)
            tasks.append(status_task)

        # Start health check runner for active /health probes
        await self._health_check_runner.start()

        # Start database pruning task via system maintenance service
        prune_task = await self._system_maintenance.start_prune_loop()
        self._register_background_task(prune_task)
        tasks.append(prune_task)

        # Start runtime guard sweep (daily loss, liquidation buffer, volatility)
        if self._guard_manager is not None:
            guard_task = asyncio.create_task(
                self._runtime_guard_sweep(), name="runtime_guard_sweep"
            )
            self._register_background_task(guard_task)
            tasks.append(guard_task)

        # Start WebSocket streaming if enabled
        if self._should_enable_streaming():
            start_streaming_background(self)
            logger.info(
                "Started WebSocket streaming",
                operation="streaming",
                stage="start",
            )

        # Start WS health watchdog
        self._ws_health_task = asyncio.create_task(
            self._monitor_ws_health(), name="ws_health_watchdog"
        )
        self._register_background_task(self._ws_health_task)
        tasks.append(self._ws_health_task)
        logger.info(
            "Started WS health watchdog",
            operation="ws_health",
            stage="start",
        )

        return tasks

    def _rehydrate_from_events(self) -> int:
        """Restore price history from persisted events.

        Delegates to PriceTickStore for the actual rehydration logic.

        Returns:
            Number of price ticks restored
        """
        # Prepare strategy rehydration callback if strategy supports it
        strategy_callback = None
        if hasattr(self.strategy, "rehydrate"):
            strategy_callback = self.strategy.rehydrate

        return self._price_tick_store.rehydrate(strategy_rehydrate_callback=strategy_callback)

    async def _runtime_guard_sweep(self) -> None:
        """Periodically run runtime guards to check risk limits.

        Runs on a cadence to proactively detect risk breaches (daily loss,
        liquidation buffer, volatility) rather than only at order time.

        On guard failure, triggers graceful degradation (pause + reduce-only).
        """
        interval = getattr(self.context.config, "runtime_guard_interval", 60)
        while self.running:
            try:
                if self._guard_manager is not None:
                    # Use run_runtime_guards directly to catch GuardError for degradation
                    self._guard_manager.run_runtime_guards()

            except GuardError as err:
                # Trigger graceful degradation on guard failure
                await self._handle_guard_failure(err)

            except Exception:
                logger.exception("Runtime guard sweep failed", operation="runtime_guards")

            await asyncio.sleep(interval)

    async def _handle_guard_failure(self, err: GuardError) -> None:
        """Handle guard failure by triggering graceful degradation."""
        risk_manager = self.context.risk_manager
        config = risk_manager.config if risk_manager else None

        # Determine cooldown from config
        cooldown_seconds = 300  # Default 5 minutes
        if config is not None:
            cooldown_seconds = config.api_health_cooldown_seconds

        # Set reduce-only mode
        if risk_manager is not None:
            risk_manager.set_reduce_only_mode(True, reason=f"guard_failure:{err.guard_name}")

        # Cancel all open orders
        if self._guard_manager is not None:
            cancelled = self._guard_manager.cancel_all_orders()
            logger.warning(
                "Guard failure triggered order cancellation",
                guard_name=err.guard_name,
                cancelled_orders=cancelled,
                operation="degradation",
                stage="cancel_orders",
            )

        # Pause all trading
        self._degradation.pause_all(
            seconds=cooldown_seconds,
            reason=f"guard_failure:{err.guard_name}",
            allow_reduce_only=True,
        )

        # Record rejection for telemetry
        self._order_submitter.record_rejection(
            symbol="*",
            side="*",
            quantity=Decimal("0"),
            price=None,
            reason=f"guard_failure:{err.guard_name}",
        )

        # Notify
        await self._notify(
            title="Guard Failure - Trading Paused",
            message=f"Runtime guard '{err.guard_name}' failed: {err.message}. "
            f"Trading paused for {cooldown_seconds}s. Reduce-only mode activated.",
            severity=AlertSeverity.ERROR,
            context={
                "guard_name": err.guard_name,
                "message": err.message,
                "cooldown_seconds": cooldown_seconds,
                "recoverable": err.recoverable,
            },
        )

    async def _monitor_ws_health(self) -> None:
        """Monitor WebSocket health and trigger degradation on staleness.

        Periodically polls WS health metrics from the broker. If messages
        or heartbeats are stale beyond configured thresholds, triggers:
        - Reduce-only mode for affected symbols
        - Symbol pause for configured cooldown
        - Notification alerts

        On reconnect, pauses briefly to allow state synchronization.
        """
        risk_manager = self.context.risk_manager
        config = risk_manager.config if risk_manager else None

        # Get thresholds from config or use defaults
        interval = config.ws_health_interval_seconds if config else 5
        message_stale_threshold = config.ws_message_stale_seconds if config else 15
        heartbeat_stale_threshold = config.ws_heartbeat_stale_seconds if config else 30
        reconnect_pause = config.ws_reconnect_pause_seconds if config else 30

        last_reconnect_count = 0

        while self.running:
            try:
                # Get WS health from broker (if it supports the method)
                broker = self.context.broker
                ws_health: dict[str, Any] = {}

                if broker is not None and hasattr(broker, "get_ws_health"):
                    try:
                        ws_health = broker.get_ws_health()
                    except Exception as exc:
                        logger.debug(
                            "Failed to get WS health",
                            error=str(exc),
                            operation="ws_health",
                            stage="poll",
                        )

                if not ws_health:
                    # No WS connection or broker doesn't support health check
                    await asyncio.sleep(interval)
                    continue

                current_time = time.time()
                last_message_ts = ws_health.get("last_message_ts")
                last_heartbeat_ts = ws_health.get("last_heartbeat_ts")
                reconnect_count = ws_health.get("reconnect_count", 0)
                gap_count = ws_health.get("gap_count", 0)
                connected = ws_health.get("connected", False)

                # Check for reconnect event
                if reconnect_count > last_reconnect_count:
                    logger.warning(
                        "WebSocket reconnected - pausing for state sync",
                        reconnect_count=reconnect_count,
                        pause_seconds=reconnect_pause,
                        operation="ws_health",
                        stage="reconnect",
                    )
                    last_reconnect_count = reconnect_count

                    # Reset reconnect attempts on successful reconnect
                    self._ws_reconnect_attempts = 0
                    self._ws_reconnect_delay = 1.0

                    # Pause all symbols briefly after reconnect
                    self._degradation.pause_all(
                        seconds=reconnect_pause,
                        reason="ws_reconnect",
                        allow_reduce_only=True,
                    )

                    await self._notify(
                        title="WebSocket Reconnected",
                        message=f"Trading paused for {reconnect_pause}s for state sync.",
                        severity=AlertSeverity.WARNING,
                        context={"reconnect_count": reconnect_count},
                    )

                    await asyncio.sleep(interval)
                    continue

                # Check message staleness
                is_message_stale = False
                if last_message_ts is not None:
                    message_age = current_time - last_message_ts
                    is_message_stale = message_age > message_stale_threshold

                # Check heartbeat staleness
                is_heartbeat_stale = False
                if last_heartbeat_ts is not None:
                    heartbeat_age = current_time - last_heartbeat_ts
                    is_heartbeat_stale = heartbeat_age > heartbeat_stale_threshold

                # Trigger degradation if stale
                if is_message_stale or is_heartbeat_stale:
                    stale_reason = "ws_message_stale" if is_message_stale else "ws_heartbeat_stale"
                    stale_age = (
                        (current_time - last_message_ts)
                        if is_message_stale and last_message_ts
                        else (current_time - last_heartbeat_ts if last_heartbeat_ts else 0)
                    )

                    logger.warning(
                        "WebSocket data stale - triggering degradation",
                        reason=stale_reason,
                        stale_age_seconds=stale_age,
                        message_stale=is_message_stale,
                        heartbeat_stale=is_heartbeat_stale,
                        connected=connected,
                        gap_count=gap_count,
                        operation="ws_health",
                        stage="degradation",
                    )

                    # Set reduce-only mode
                    if risk_manager is not None:
                        risk_manager.set_reduce_only_mode(True, reason=stale_reason)

                    # Pause all trading (allow reduce-only)
                    cooldown = reconnect_pause
                    self._degradation.pause_all(
                        seconds=cooldown,
                        reason=stale_reason,
                        allow_reduce_only=True,
                    )

                    await self._notify(
                        title="WebSocket Stale - Trading Paused",
                        message=f"No WS data for {stale_age:.1f}s. Reduce-only mode enabled.",
                        severity=AlertSeverity.WARNING,
                        context={
                            "reason": stale_reason,
                            "stale_age_seconds": stale_age,
                            "cooldown_seconds": cooldown,
                        },
                    )

                # Log gap detection warnings
                if gap_count > 0 and self._cycle_count % 60 == 0:
                    logger.info(
                        "WebSocket sequence gaps detected",
                        gap_count=gap_count,
                        operation="ws_health",
                        stage="info",
                    )

                # Update status reporter with WS health
                self._status_reporter.update_ws_health(ws_health)

            except Exception:
                logger.exception("WS health watchdog error", operation="ws_health")

            await asyncio.sleep(interval)

    async def _run_loop(self) -> None:
        logger.info("Starting strategy loop...")
        while self.running:
            try:
                await self._cycle()
                # Record successful cycle
                self._status_reporter.record_cycle()
            except Exception as e:
                logger.error(f"Error in strategy cycle: {e}", exc_info=True)
                # Record error in status reporter
                self._status_reporter.record_error(str(e))
                await self._notify(
                    title="Strategy Cycle Error",
                    message=f"Error during trading cycle: {e}",
                    severity=AlertSeverity.ERROR,
                    context={"error": str(e)},
                )

            await asyncio.sleep(self.context.config.interval)

    def _report_system_status(self) -> None:
        """Collect and report system health metrics.

        Delegates to SystemMaintenanceService for the actual reporting.
        """
        self._system_maintenance.report_system_status(
            latency_seconds=self._last_latency,
            connection_status=self._connection_status,
        )

    async def _cycle(self) -> None:
        """One trading cycle."""
        assert self.context.broker is not None, "Broker not initialized"
        self._cycle_count += 1

        # Wrap entire cycle in correlation context and trace span
        start_time = time.perf_counter()
        result = "ok"
        with correlation_context(cycle=self._cycle_count):
            with trace_span("cycle", {"cycle": self._cycle_count}) as span:
                try:
                    await self._cycle_inner()
                except Exception:
                    result = "error"
                    if span:
                        span.set_attribute("error", True)
                    raise
                finally:
                    duration = time.perf_counter() - start_time
                    if span:
                        span.set_attribute("duration_seconds", duration)
                        span.set_attribute("result", result)
                    record_histogram(
                        "gpt_trader_cycle_duration_seconds",
                        duration,
                        labels={"result": result},
                    )

    async def _cycle_inner(self) -> None:
        """Inner cycle logic wrapped in correlation context."""
        logger.info(f"=== CYCLE {self._cycle_count} START ===")

        # Report system status at start of cycle
        self._report_system_status()
        broker = self.context.broker
        if broker is None:
            logger.error("Broker not initialized", operation="cycle")
            self._connection_status = "DISCONNECTED"
            return

        # 1. Fetch positions and audit orders in parallel (independent operations)
        logger.info("Step 1: Fetching positions and auditing orders (parallel)...")
        positions_task = asyncio.create_task(self._fetch_positions())
        audit_task = asyncio.create_task(self._audit_orders())

        # Await positions first (needed for equity calculation)
        with profile_span("fetch_positions") as _pos_span:
            positions = await positions_task
        self._current_positions = positions
        logger.info(f"Fetched {len(positions)} positions")

        # Update status reporter with positions (complete data for TUI)
        self._status_reporter.update_positions(self._positions_to_status_format(positions))

        # 2. Calculate total equity including unrealized PnL
        logger.info("Step 2: Calculating total equity...")
        with profile_span("equity_computation") as _eq_span:
            equity = await self._fetch_total_equity(positions)
        if equity is None:
            logger.error(
                "Failed to fetch equity - cannot continue cycle. "
                "Check logs above for balance fetch errors."
            )
            # Update status reporter with error state
            self._status_reporter.record_error("Failed to fetch equity")
            # Ensure audit completes even on equity failure
            try:
                await audit_task
            except Exception as e:
                logger.warning(f"Order audit failed during equity error path: {e}")
            return

        # Ensure audit task completes (should be done by now, but be explicit)
        try:
            await audit_task
        except Exception as e:
            logger.warning(f"Order audit failed: {e}")

        logger.info(f"Successfully calculated equity: ${equity}")
        # Update status reporter with equity
        self._status_reporter.update_equity(equity)
        logger.info("Equity updated in status reporter")

        # Track daily PnL for risk management
        if self.context.risk_manager:
            triggered = self.context.risk_manager.track_daily_pnl(equity, {})
            if triggered:
                logger.warning("Daily loss limit triggered! Reduce-only mode activated.")

            # Update status reporter with risk metrics
            rm = self.context.risk_manager
            # Calculate current daily loss pct if possible
            # Assuming rm tracks start_of_day_equity
            daily_loss_pct = 0.0
            start_equity = getattr(rm, "_start_of_day_equity", 0)
            if start_equity and start_equity > 0:
                daily_pnl = equity - start_equity
                daily_loss_pct = float(-daily_pnl / start_equity)

            self._status_reporter.update_risk(
                max_leverage=float(getattr(rm.config, "max_leverage", 0.0) if rm.config else 0.0),
                daily_loss_limit=float(
                    getattr(rm.config, "daily_loss_limit_pct", 0.0) if rm.config else 0.0
                ),
                current_daily_loss=daily_loss_pct,
                reduce_only=getattr(rm, "_reduce_only_mode", False),
                reduce_reason=getattr(rm, "_reduce_only_reason", ""),
            )

        # 3. Batch fetch tickers for all symbols (reduces API calls in Advanced mode)
        symbols = self.context.config.symbols
        tickers: dict[str, dict[str, Any]] = {}
        batch_start = time.time()

        # Only use batch fetch if broker has get_tickers method (not a mock)
        get_tickers_method = getattr(broker, "get_tickers", None)
        if get_tickers_method is not None and callable(get_tickers_method):
            try:
                async with self._rest_semaphore:
                    result = await asyncio.to_thread(get_tickers_method, symbols)
                # Validate result is a dict (protects against mocks)
                if isinstance(result, dict):
                    tickers = result
                    logger.debug(
                        f"Batch ticker fetch: {len(tickers)}/{len(symbols)} symbols "
                        f"in {time.time() - batch_start:.3f}s"
                    )
            except Exception as e:
                logger.warning(f"Batch ticker fetch failed, falling back to individual: {e}")

        # 4. Process Symbols - fetch candles and any missing tickers
        for symbol in symbols:
            ticker = tickers.get(symbol)
            candles: list[Any] = []
            start_time = time.time()

            # If ticker not in batch result, fetch individually
            if ticker is None:
                try:
                    async with self._rest_semaphore:
                        ticker = await asyncio.to_thread(broker.get_ticker, symbol)
                except Exception as e:
                    logger.error(f"Failed to fetch ticker for {symbol}: {e}")
                    self._connection_status = "DISCONNECTED"
                    continue

            if ticker is None or not ticker.get("price"):
                logger.error(f"No ticker data for {symbol}")
                self._connection_status = "DISCONNECTED"
                continue

            # Fetch candles (always individual per-symbol)
            try:
                async with self._rest_semaphore:
                    candles_result = await asyncio.to_thread(
                        broker.get_candles, symbol, granularity="ONE_MINUTE"
                    )
                if isinstance(candles_result, Exception):
                    logger.warning(f"Failed to fetch candles for {symbol}: {candles_result}")
                else:
                    candles = candles_result or []
            except Exception as e:
                logger.warning(f"Failed to fetch candles for {symbol}: {e}")

            # Update latency and connection status
            self._last_latency = time.time() - start_time
            self._connection_status = "CONNECTED"

            price = Decimal(str(ticker.get("price", 0)))
            logger.info(f"{symbol} price: {price}")

            # Seed mark staleness timestamp from REST fetch (prevents deadlock on startup)
            if self.context.risk_manager is not None:
                self.context.risk_manager.last_mark_update[symbol] = time.time()

            # Update status reporter with price
            self._status_reporter.update_price(symbol, price)

            # Record price tick (updates in-memory history + persists for crash recovery)
            self._record_price_tick(symbol, price)

            position_state = self._build_position_state(symbol, positions)

            with profile_span("strategy_decision", {"symbol": symbol}) as _strat_span:
                decision = self.strategy.decide(
                    symbol=symbol,
                    current_mark=price,
                    position_state=position_state,
                    recent_marks=self.price_history[symbol],
                    equity=equity,
                    product=None,  # Future: fetch from broker
                    candles=candles,
                )

            logger.info(f"Strategy Decision for {symbol}: {decision.action} ({decision.reason})")

            # Report strategy decision
            active_strats = getattr(
                self.strategy, "active_strategies", [self.strategy.__class__.__name__]
            )
            decision_record = {
                "symbol": symbol,
                "action": decision.action.value,
                "reason": decision.reason,
                "confidence": str(decision.confidence),
                "timestamp": time.time(),
            }
            self._status_reporter.update_strategy(active_strats, [decision_record])

            if decision.action in (Action.BUY, Action.SELL):
                logger.info(
                    "Executing order",
                    symbol=symbol,
                    action=decision.action.value,
                    operation="order_placement",
                    stage="start",
                )
                try:
                    with profile_span(
                        "order_placement", {"symbol": symbol, "action": decision.action.value}
                    ):
                        result = await self._validate_and_place_order(
                            symbol=symbol,
                            decision=decision,
                            price=price,
                            equity=equity,
                        )
                    if result.blocked:
                        logger.warning(
                            "Order blocked",
                            symbol=symbol,
                            action=decision.action.value,
                            reason=result.reason,
                            operation="order_placement",
                            stage="blocked",
                        )
                    elif result.failed:
                        logger.error(
                            "Order submission failed",
                            symbol=symbol,
                            action=decision.action.value,
                            error_message=result.error,
                            operation="order_placement",
                            stage="failed",
                        )
                except Exception as e:
                    logger.error(
                        "Order placement failed",
                        symbol=symbol,
                        action=decision.action.value,
                        error_message=str(e),
                        operation="order_placement",
                        stage="failed",
                    )
                    await self._notify(
                        title="Order Placement Failed",
                        message=f"Failed to execute {decision.action} for {symbol}: {e}",
                        severity=AlertSeverity.ERROR,
                        context={
                            "symbol": symbol,
                            "action": decision.action.value,
                            "error": str(e),
                        },
                    )

            elif decision.action == Action.CLOSE and position_state:
                # Handle CLOSE action separately if needed, or integrate into place_order
                # For now, logging it as per original logic, or we can implement close logic here
                logger.info(f"CLOSE signal for {symbol} - not fully implemented yet")

    async def _fetch_total_equity(self, positions: dict[str, Position]) -> Decimal | None:
        """Fetch total equity = collateral + unrealized PnL."""
        return await self._equity_calculator.calculate_total_equity(self.context.broker, positions)

    async def _fetch_positions(self) -> dict[str, Position]:
        """Fetch current positions as a lookup dict."""
        assert self.context.broker is not None
        start_time = time.perf_counter()
        result = "ok"
        try:
            positions_list = await asyncio.to_thread(self.context.broker.list_positions)
            # Success: reset broker failure counter
            self._degradation.reset_broker_failures()
            return {p.symbol: p for p in positions_list}
        except Exception as e:
            result = "error"
            logger.error(f"Failed to fetch positions: {e}")
            # Track broker failure for degradation
            config = self.context.risk_manager.config if self.context.risk_manager else None
            if config is not None:
                self._degradation.record_broker_failure(config)
            return {}
        finally:
            duration = time.perf_counter() - start_time
            record_histogram(
                "gpt_trader_positions_fetch_seconds",
                duration,
                labels={"result": result},
            )

    def _build_position_state(
        self, symbol: str, positions: dict[str, Position]
    ) -> dict[str, Any] | None:
        """Build position state dict for strategy.decide()."""
        if symbol not in positions:
            return None
        pos = positions[symbol]
        return {
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "side": pos.side,
            # Add other fields if needed by strategy
        }

    def _record_price_tick(self, symbol: str, price: Decimal) -> None:
        """Persist price tick to EventStore for crash recovery.

        Delegates to PriceTickStore which handles both in-memory
        history update and EventStore persistence.
        """
        self._price_tick_store.record_price_tick(symbol, price)

    def _positions_to_risk_format(
        self, positions: dict[str, Position]
    ) -> dict[str, dict[str, Any]]:
        """Convert Position objects to dict format expected by risk manager."""
        return {
            symbol: {
                "quantity": pos.quantity,
                "mark": pos.mark_price,
            }
            for symbol, pos in positions.items()
        }

    def _positions_to_status_format(
        self, positions: dict[str, Position]
    ) -> dict[str, dict[str, Any]]:
        """Convert Position objects to dict format for StatusReporter with complete TUI data."""
        return {
            symbol: {
                "quantity": str(pos.quantity),
                "mark_price": str(pos.mark_price),
                "entry_price": str(pos.entry_price),
                "unrealized_pnl": str(pos.unrealized_pnl),
                "realized_pnl": str(pos.realized_pnl),
                "side": pos.side,
            }
            for symbol, pos in positions.items()
        }

    def _calculate_order_quantity(
        self,
        symbol: str,
        price: Decimal,
        equity: Decimal,
        product: Product | None,
        *,
        quantity_override: Decimal | None = None,
    ) -> Decimal:
        """Calculate order size based on equity and position_fraction."""
        # External override (submit_order) bypasses dynamic sizing.
        if quantity_override is not None:
            return quantity_override

        # 1. Determine fraction
        fraction = Decimal("0.1")  # Default
        if hasattr(self.strategy, "config") and self.strategy.config.position_fraction:
            fraction = Decimal(str(self.strategy.config.position_fraction))
        elif (
            hasattr(self.context.config, "perps_position_fraction")
            and self.context.config.perps_position_fraction is not None
        ):
            fraction = Decimal(str(self.context.config.perps_position_fraction))

        # 2. Calculate raw quantity
        if price == 0:
            return Decimal("0")

        target_notional = equity * fraction
        quantity = target_notional / price

        # 3. Apply constraints
        if product and product.min_size:
            if quantity < product.min_size:
                logger.warning(f"Quantity {quantity} below min size {product.min_size}")
                return Decimal("0")

            # Round to step size if needed (simplified)
            # quantity = (quantity // product.step_size) * product.step_size

        return quantity

    def _is_reduce_only_order(self, current_pos: Position | None, side: OrderSide) -> bool:
        """Determine if an order would reduce an existing position."""
        if current_pos is None:
            return False

        # Handle Position objects
        if hasattr(current_pos, "side") and hasattr(current_pos, "quantity"):
            pos_side = current_pos.side.lower() if current_pos.side else ""
            pos_qty = current_pos.quantity
            # Reducing = LONG + SELL or SHORT + BUY
            return (pos_side == "long" and side == OrderSide.SELL and pos_qty > 0) or (
                pos_side == "short" and side == OrderSide.BUY and pos_qty > 0
            )

        # Handle dict format
        if isinstance(current_pos, dict):
            pos_side = str(current_pos.get("side", "")).lower()
            pos_qty = Decimal(str(current_pos.get("quantity", 0)))
            if pos_side in ("long", "short"):
                return (pos_side == "long" and side == OrderSide.SELL and pos_qty > 0) or (
                    pos_side == "short" and side == OrderSide.BUY and pos_qty > 0
                )
            # Legacy: quantity sign indicates direction
            return (pos_qty > 0 and side == OrderSide.SELL) or (
                pos_qty < 0 and side == OrderSide.BUY
            )

        return False

    async def _validate_and_place_order(
        self,
        symbol: str,
        decision: Decision,
        price: Decimal,
        equity: Decimal,
        quantity_override: Decimal | None = None,
        reduce_only_requested: bool = False,
    ) -> OrderSubmissionResult:
        """Validate and submit an order through the guard stack.

        Returns:
            OrderSubmissionResult describing success/blocked/failed.
        """
        side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL

        # Early check: is this order reduce-only? (needed for degradation check)
        current_pos = self._current_positions.get(symbol)
        is_reducing = self._is_reduce_only_order(current_pos, side)
        reduce_only_flag = reduce_only_requested or is_reducing

        # Gate: Check degradation state before proceeding
        if self._degradation.is_paused(symbol, is_reduce_only=reduce_only_flag):
            pause_reason = self._degradation.get_pause_reason(symbol) or "unknown"
            logger.warning(
                f"Order blocked: trading paused for {symbol}",
                symbol=symbol,
                side=side.value,
                reason=pause_reason,
                operation="degradation",
                stage="order_blocked",
            )
            self._order_submitter.record_rejection(
                symbol, side.value, Decimal("0"), price, f"paused:{pause_reason}"
            )
            await self._notify(
                title="Order Blocked - Trading Paused",
                message=f"Cannot place {side.value} order for {symbol}: {pause_reason}",
                severity=AlertSeverity.WARNING,
                context={"symbol": symbol, "side": side.value, "reason": pause_reason},
            )
            return OrderSubmissionResult(
                status=OrderSubmissionStatus.BLOCKED,
                reason=f"paused:{pause_reason}",
            )

        # Dynamic position sizing
        quantity = self._calculate_order_quantity(
            symbol,
            price,
            equity,
            product=None,
            quantity_override=quantity_override,
        )

        if quantity <= 0:
            logger.warning(f"Calculated quantity is {quantity}, skipping order")
            return OrderSubmissionResult(
                status=OrderSubmissionStatus.BLOCKED,
                reason="quantity_zero",
            )

        # Security Validation (Hard Limits)
        from gpt_trader.security.security_validator import get_validator

        security_order = {
            "symbol": symbol,
            "side": side.value,
            "quantity": float(quantity),
            "price": float(price),
            "type": "MARKET",
        }

        # Construct dynamic limits from config
        limits = {}
        if hasattr(self.context.config, "risk"):
            risk = self.context.config.risk
            if risk:
                limits["max_position_size"] = float(getattr(risk, "max_position_pct", 0.05))
                limits["max_leverage"] = float(getattr(risk, "max_leverage", 2.0))
                limits["max_daily_loss"] = float(getattr(risk, "daily_loss_limit_pct", 0.02))
                # Map other fields if available or use defaults

        security_result = get_validator().validate_order_request(
            security_order, account_value=float(equity), limits=limits
        )

        if not security_result.is_valid:
            error_msg = f"Security validation failed: {', '.join(security_result.errors)}"
            logger.error(error_msg)
            await self._notify(
                title="Security Validation Failed",
                message=error_msg,
                severity=AlertSeverity.ERROR,
                context=security_order,
            )
            return OrderSubmissionResult(
                status=OrderSubmissionStatus.BLOCKED,
                reason=error_msg,
            )

        # Run pre-trade validation if risk manager is available
        risk_manager = self.context.risk_manager
        if risk_manager is not None:
            try:
                risk_manager.pre_trade_validate(
                    symbol=symbol,
                    side=side.value,
                    quantity=quantity,
                    price=price,
                    product=None,
                    equity=equity,
                    current_positions=self._positions_to_risk_format(self._current_positions),
                )
            except ValidationError as exc:
                logger.warning(
                    "Risk validation failed",
                    symbol=symbol,
                    side=side.value,
                    error_message=str(exc),
                    operation="order_validation",
                    stage="risk_manager",
                )
                await self._notify(
                    title="Risk Validation Failed",
                    message=f"Order blocked by risk manager: {exc}",
                    severity=AlertSeverity.WARNING,
                    context={
                        "symbol": symbol,
                        "side": side.value,
                        "reason": str(exc),
                    },
                )
                return OrderSubmissionResult(
                    status=OrderSubmissionStatus.BLOCKED,
                    reason=str(exc),
                )
            except Exception as exc:
                logger.error(
                    "Risk validation error",
                    symbol=symbol,
                    side=side.value,
                    error_message=str(exc),
                    operation="order_validation",
                    stage="risk_manager_error",
                )
                await self._notify(
                    title="Risk Validation Failed",
                    message=f"Order validation failed for {symbol}: {exc}",
                    severity=AlertSeverity.ERROR,
                    context={
                        "symbol": symbol,
                        "side": side.value,
                        "error": str(exc),
                    },
                )
                return OrderSubmissionResult(
                    status=OrderSubmissionStatus.FAILED,
                    error=str(exc),
                )

            logger.info(f"Risk validation passed for {symbol} {side.value}")

            # In reduce-only mode, clamp quantity to prevent position flips
            daily_pnl_triggered = bool(getattr(risk_manager, "_daily_pnl_triggered", False))
            reduce_only_mode = risk_manager.is_reduce_only_mode()
            reduce_only_active = reduce_only_mode or daily_pnl_triggered
            if reduce_only_active and is_reducing and current_pos is not None:
                # Get current position quantity
                if hasattr(current_pos, "quantity"):
                    current_qty = abs(current_pos.quantity)
                elif isinstance(current_pos, dict):
                    current_qty = abs(Decimal(str(current_pos.get("quantity", 0))))
                else:
                    current_qty = Decimal("0")

                # Clamp order quantity to current position size
                if quantity > current_qty:
                    logger.warning(
                        f"Reduce-only: clamping order from {quantity} to {current_qty} "
                        f"to prevent position flip for {symbol}"
                    )
                    quantity = current_qty

                # If clamped to zero, skip the order
                if quantity <= 0:
                    logger.info(f"Reduce-only: no position to reduce for {symbol}, skipping order")
                    return OrderSubmissionResult(
                        status=OrderSubmissionStatus.BLOCKED,
                        reason="reduce_only_empty_position",
                    )

            # Create order dict for check_order
            order_for_check = {
                "symbol": symbol,
                "side": side.value,
                "quantity": float(quantity),
                "reduce_only": reduce_only_flag,
            }

            if not risk_manager.check_order(order_for_check):
                error_msg = (
                    f"Order blocked by risk manager: "
                    f"reduce_only_mode={reduce_only_mode}, "
                    f"daily_pnl_triggered={daily_pnl_triggered}"
                )
                logger.warning(error_msg)
                await self._notify(
                    title="Order Blocked - Reduce Only Mode",
                    message=f"Cannot open new {side.value} position for {symbol} while in reduce-only mode",
                    severity=AlertSeverity.WARNING,
                    context=order_for_check,
                )
                return OrderSubmissionResult(
                    status=OrderSubmissionStatus.BLOCKED,
                    reason=error_msg,
                )
        else:
            logger.warning("No risk manager configured - skipping validation")

        # Guard: Check mark price staleness before placing order
        if self.context.risk_manager is not None:
            if self.context.risk_manager.check_mark_staleness(symbol):
                # Trigger degradation: pause symbol for staleness cooldown
                config = self.context.risk_manager.config
                if config is not None:
                    allow_reduce = config.mark_staleness_allow_reduce_only
                    cooldown = config.mark_staleness_cooldown_seconds
                    self._degradation.pause_symbol(
                        symbol=symbol,
                        seconds=cooldown,
                        reason="mark_staleness",
                        allow_reduce_only=allow_reduce,
                    )
                    # If reduce-only allowed and this is a reduce order, let it through
                    if allow_reduce and reduce_only_flag:
                        logger.info(
                            f"Mark stale for {symbol} but allowing reduce-only order",
                            operation="degradation",
                        )
                    else:
                        logger.warning(f"Order blocked: mark price stale for {symbol}")
                        self._order_submitter.record_rejection(
                            symbol, side.value, quantity, price, "mark_staleness"
                        )
                        await self._notify(
                            title="Order Blocked - Stale Mark Price",
                            message=f"Cannot place order for {symbol}: mark price data is stale",
                            severity=AlertSeverity.WARNING,
                            context={"symbol": symbol, "side": side.value},
                        )
                        return OrderSubmissionResult(
                            status=OrderSubmissionStatus.BLOCKED,
                            reason="mark_staleness",
                        )
                else:
                    logger.warning(f"Order blocked: mark price stale for {symbol}")
                    await self._notify(
                        title="Order Blocked - Stale Mark Price",
                        message=f"Cannot place order for {symbol}: mark price data is stale",
                        severity=AlertSeverity.WARNING,
                        context={"symbol": symbol, "side": side.value},
                    )
                    return OrderSubmissionResult(
                        status=OrderSubmissionStatus.BLOCKED,
                        reason="mark_staleness",
                    )

        # Pre-trade guards via OrderValidator (exchange rules, slippage, preview)
        effective_price = price
        if self._order_validator is not None:
            try:
                with profile_span("pre_trade_validation", {"symbol": symbol}) as _val_span:
                    # Get product for exchange rules validation
                    product = self._state_collector.require_product(symbol, product=None)

                    # Resolve effective price via StateCollector
                    effective_price = self._state_collector.resolve_effective_price(
                        symbol, side.value.lower(), price, product
                    )

                    # Exchange rules + quantization
                    quantity, _ = self._order_validator.validate_exchange_rules(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        order_quantity=quantity,
                        price=None,
                        effective_price=effective_price,
                        product=product,
                    )

                    # Slippage guard with degradation tracking
                    try:
                        self._order_validator.enforce_slippage_guard(
                            symbol, side, quantity, effective_price
                        )
                        # Success: reset slippage failure count
                        self._degradation.reset_slippage_failures(symbol)
                    except ValidationError as slippage_exc:
                        # Track slippage failures and potentially pause symbol
                        config = (
                            self.context.risk_manager.config if self.context.risk_manager else None
                        )
                        if config is not None:
                            self._degradation.record_slippage_failure(symbol, config)
                        raise slippage_exc

                    # Pre-trade validation via OrderValidator (leverage/exposure)
                    current_positions_dict = self._state_collector.build_positions_dict(
                        list(self._current_positions.values())
                    )
                    self._order_validator.run_pre_trade_validation(
                        symbol=symbol,
                        side=side,
                        order_quantity=quantity,
                        effective_price=effective_price,
                        product=product,
                        equity=equity,
                        current_positions=current_positions_dict,
                    )

                    # Order preview (if enabled) - with auto-disable on repeated failures
                    failure_tracker = get_failure_tracker()
                    config = self.context.risk_manager.config if self.context.risk_manager else None
                    preview_disable_threshold = (
                        config.preview_failure_disable_after if config else 5
                    )

                    # Check if preview should be auto-disabled due to repeated failures
                    if (
                        self._order_validator.enable_order_preview
                        and failure_tracker.get_failure_count("order_preview")
                        >= preview_disable_threshold
                    ):
                        logger.warning(
                            "Auto-disabling order preview due to repeated failures",
                            consecutive_failures=failure_tracker.get_failure_count("order_preview"),
                            threshold=preview_disable_threshold,
                            operation="degradation",
                            stage="preview_disable",
                        )
                        self._order_validator.enable_order_preview = False

                    self._order_validator.maybe_preview_order(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        order_quantity=quantity,
                        effective_price=effective_price,
                        stop_price=None,
                        tif=self.context.config.time_in_force,
                        reduce_only=reduce_only_flag,
                        leverage=None,
                    )

                    # Finalize reduce-only flag (risk manager may have triggered it)
                    reduce_only_flag = self._order_validator.finalize_reduce_only_flag(
                        reduce_only_flag, symbol
                    )

            except ValidationError as exc:
                logger.warning(f"Pre-trade guard rejected order: {exc}")
                self._order_submitter.record_rejection(
                    symbol, side.value, quantity, effective_price, str(exc)
                )
                await self._notify(
                    title="Order Blocked - Guard Rejection",
                    message=f"Cannot place order for {symbol}: {exc}",
                    severity=AlertSeverity.WARNING,
                    context={"symbol": symbol, "side": side.value, "reason": str(exc)},
                )
                return OrderSubmissionResult(
                    status=OrderSubmissionStatus.BLOCKED,
                    reason=str(exc),
                )
            except Exception as exc:
                # Non-validation errors: log + record metrics but still block order (fail-closed)
                logger.error(f"Guard check error: {exc}")
                self._order_submitter.record_rejection(
                    symbol, side.value, quantity, price, f"guard_error: {exc}"
                )
                await self._notify(
                    title="Order Blocked - Guard Error",
                    message=f"Cannot place order for {symbol}: guard check failed",
                    severity=AlertSeverity.ERROR,
                    context={"symbol": symbol, "side": side.value, "error": str(exc)},
                )
                return OrderSubmissionResult(
                    status=OrderSubmissionStatus.FAILED,
                    error=str(exc),
                )

        # Place order via OrderSubmitter for proper ID tracking and telemetry
        order_id = await asyncio.to_thread(
            self._order_submitter.submit_order,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            order_quantity=quantity,
            price=None,  # Market order
            effective_price=effective_price,
            stop_price=None,
            tif=self.context.config.time_in_force,
            reduce_only=reduce_only_flag,
            leverage=None,
            client_order_id=None,  # Let OrderSubmitter generate stable ID
        )

        # Notify on successful order placement
        if order_id is not None:
            await self._notify(
                title="Order Executed",
                message=f"{side.value} {quantity} {symbol} at ~{price}",
                severity=AlertSeverity.INFO,
                context={
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(quantity),
                    "price": str(price),
                    "order_id": order_id,
                },
            )

            # Record trade in status reporter
            self._status_reporter.add_trade(
                {
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(quantity),
                    "price": str(price),
                    "order_id": order_id,
                }
            )
            return OrderSubmissionResult(
                status=OrderSubmissionStatus.SUCCESS,
                order_id=order_id,
            )
        else:
            # Order was rejected by broker
            logger.warning(
                "Order submission returned None - order may have been rejected",
                symbol=symbol,
                side=side.value,
                operation="order_submit",
                stage="rejected",
            )
            return OrderSubmissionResult(
                status=OrderSubmissionStatus.FAILED,
                error="broker_rejected",
            )

    def reset_daily_tracking(self) -> None:
        """Reset daily PnL tracking and guard cache (start of trading day)."""
        try:
            broker = self.context.broker
            if broker is None:
                logger.warning(
                    "Cannot reset daily tracking without broker",
                    operation="daily_tracking",
                    stage="missing_broker",
                )
                return

            balances = broker.list_balances()
            equity, _, _ = self._state_collector.calculate_equity_from_balances(balances)

            if self.context.risk_manager is not None:
                self.context.risk_manager.reset_daily_tracking()

            if self._guard_manager is not None:
                self._guard_manager.invalidate_cache()

            logger.info(
                "Daily tracking reset",
                operation="daily_tracking",
                stage="reset",
                equity=float(equity),
            )
        except Exception as exc:
            logger.error(
                "Failed to reset daily tracking",
                error_message=str(exc),
                operation="daily_tracking",
                stage="reset",
            )

    # =========================================================================
    # PUBLIC SUBMISSION ENTRYPOINT
    # =========================================================================
    # This is the canonical order submission path. All order execution should
    # route through this method to ensure the full guard stack is applied:
    # degradation gate → sizing → security → risk → staleness → validator
    # =========================================================================

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        equity: Decimal,
        *,
        quantity_override: Decimal | None = None,
        reduce_only: bool = False,
        reason: str = "external_submission",
        confidence: float = 1.0,
    ) -> OrderSubmissionResult:
        """Public entrypoint for order submission through the canonical guard stack.

        This method provides external callers (OrderRouter, TUI actions, etc.) access
        to the full pre-trade validation pipeline. All orders should route through
        here to ensure consistent guard enforcement.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD").
            side: Order side (BUY or SELL).
            price: Current market price for validation.
            equity: Current account equity for position sizing.
            quantity_override: If provided, uses this quantity instead of dynamic sizing.
            reduce_only: If True, order is reduce-only (affects degradation gate).
            reason: Reason for the order (for logging/telemetry).
            confidence: Decision confidence score (0.0-1.0).

        Note:
            This method delegates to _validate_and_place_order after constructing
            a Decision object. The full guard stack is applied.

        Returns:
            OrderSubmissionResult describing the outcome.
        """
        # Construct Decision from inputs
        action = Action.BUY if side == OrderSide.BUY else Action.SELL
        decision = Decision(
            action=action,
            reason=reason,
            confidence=confidence,
        )

        # Pass quantity override through to guard stack sizing.
        return await self._validate_and_place_order(
            symbol,
            decision,
            price,
            equity,
            quantity_override=quantity_override,
            reduce_only_requested=reduce_only,
        )

    async def shutdown(self) -> None:
        self._transition_state(EngineState.STOPPING, reason="shutdown_called")

        # Stop WS health watchdog
        if self._ws_health_task is not None and not self._ws_health_task.done():
            self._ws_health_task.cancel()
            try:
                await self._ws_health_task
            except asyncio.CancelledError:
                pass
            self._ws_health_task = None
            logger.info(
                "Stopped WS health watchdog",
                operation="ws_health",
                stage="stop",
            )

        # Stop streaming
        stop_streaming_background(self)
        logger.info(
            "Stopped WebSocket streaming",
            operation="streaming",
            stage="stop",
        )

        # Stop health check runner
        await self._health_check_runner.stop()

        await self._system_maintenance.stop()
        await self._status_reporter.stop()
        await self._heartbeat.stop()
        await super().shutdown()
        self._transition_state(EngineState.STOPPED, reason="shutdown_complete")

    def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self.running, component=self.name)

    async def _audit_orders(self) -> None:
        """Audit open orders for reconciliation."""
        assert self.context.broker is not None
        try:
            # Fetch open orders
            # Note: Coinbase API uses 'order_status' for filtering
            # Use getattr to safely call list_orders (not part of base BrokerProtocol)
            list_orders = getattr(self.context.broker, "list_orders", None)
            if list_orders:
                response = await asyncio.to_thread(list_orders, order_status="OPEN")
                orders = response.get("orders", [])
            else:
                orders = []

            if orders:
                logger.info(f"AUDIT: Found {len(orders)} OPEN orders")
                for order in orders:
                    logger.info(
                        f"  Order {order.get('order_id')}: {order.get('side')} "
                        f"{order.get('product_id')} {order.get('order_configuration')}"
                    )

            # Update status reporter
            self._status_reporter.update_orders(orders)

            # Update Account Metrics (every 60 cycles ~ 1 minute)
            if self._cycle_count % 60 == 0:
                try:
                    balances = self.context.broker.list_balances()
                    # Check if broker supports transaction summary (Coinbase specific)
                    summary = {}
                    if hasattr(self.context.broker, "client") and hasattr(
                        self.context.broker.client, "get_transaction_summary"
                    ):
                        try:
                            summary = self.context.broker.client.get_transaction_summary()
                        except Exception:
                            pass  # Feature might not be available or API mode issue

                    self._status_reporter.update_account(balances, summary)
                except Exception as e:
                    logger.warning(f"Failed to update account metrics: {e}")

        except Exception as e:
            logger.warning(f"Failed to audit orders: {e}")
