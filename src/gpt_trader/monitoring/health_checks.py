"""
Active health check implementations.

Provides concrete health check functions that probe system components
and return structured results for the health server.

Each check returns (bool, dict) where:
- bool: True if healthy, False if unhealthy
- dict: Details including severity, latency, error messages
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

from gpt_trader.monitoring.metrics_collector import record_counter
from gpt_trader.monitoring.profiling import profile_span
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.time_provider import TimeProvider, get_clock

if TYPE_CHECKING:
    from gpt_trader.app.health_server import HealthState
    from gpt_trader.features.brokerages.core.protocols import BrokerProtocol
    from gpt_trader.features.live_trade.degradation import DegradationState
    from gpt_trader.utilities.async_tools.bounded_to_thread import BoundedToThread
from gpt_trader.features.brokerages.core.protocols import (
    TickerFreshnessProvider,
    TickerFreshnessProviderSource,
)
from gpt_trader.features.live_trade.risk.protocols import RiskManagerProtocol

logger = get_logger(__name__, component="health_checks")

TickerFreshnessService = TickerFreshnessProviderSource | TickerFreshnessProvider

# Metrics emitted by the ticker freshness health check.
TICKER_FRESHNESS_CHECKS_COUNTER = "gpt_trader_ticker_freshness_checks_total"
TICKER_CACHE_UNAVAILABLE_COUNTER = "gpt_trader_ticker_cache_unavailable_total"
TICKER_STALE_SYMBOLS_COUNTER = "gpt_trader_ticker_stale_symbols_total"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    healthy: bool
    details: dict[str, Any]


HealthCheckOutcome = tuple[bool, dict[str, Any]]
HealthCheckMode = Literal["blocking", "fast"]


@dataclass(frozen=True)
class HealthCheckDescriptor:
    """Descriptor for a registered health check."""

    name: str
    mode: HealthCheckMode
    run: Callable[[], HealthCheckOutcome]


def check_broker_ping(broker: BrokerProtocol) -> tuple[bool, dict[str, Any]]:
    """
    Check broker connectivity by making a lightweight API call.

    Uses get_time() if available (cheapest), otherwise falls back to
    list_balances().

    Args:
        broker: Broker protocol instance.

    Returns:
        Tuple of (healthy, details) where details includes:
            - latency_ms: Round-trip time in milliseconds
            - error: Exception string if failed
            - method: Which method was used for the check
    """
    start = time.perf_counter()
    details: dict[str, Any] = {"severity": "critical"}

    try:
        # Prefer get_time() as it's the lightest-weight call
        if hasattr(broker, "get_time") and callable(broker.get_time):
            broker.get_time()
            method = "get_time"
        else:
            # Fall back to list_balances
            broker.list_balances()
            method = "list_balances"

        latency_ms = (time.perf_counter() - start) * 1000
        details.update(
            {
                "latency_ms": round(latency_ms, 2),
                "method": method,
            }
        )

        # Warn if latency is high (>2s is concerning for trading)
        if latency_ms > 2000:
            details["severity"] = "warning"
            details["warning"] = "High latency detected"

        return True, details

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        details.update(
            {
                "latency_ms": round(latency_ms, 2),
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )
        logger.warning(
            "Broker ping failed",
            operation="health_check",
            error=str(exc),
            latency_ms=latency_ms,
        )
        return False, details


def check_ws_freshness(
    broker: BrokerProtocol,
    message_stale_seconds: float = 60.0,
    heartbeat_stale_seconds: float = 120.0,
    time_provider: TimeProvider | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check WebSocket connection health and message freshness.

    Args:
        broker: Broker protocol instance (must have get_ws_health method).
        message_stale_seconds: Max age of last message before considered stale.
        heartbeat_stale_seconds: Max age of last heartbeat before considered stale.
        time_provider: Optional time provider for deterministic freshness checks.

    Returns:
        Tuple of (healthy, details) where details includes:
            - connected: Whether WS is currently connected
            - last_message_age_seconds: Age of last message
            - last_heartbeat_age_seconds: Age of last heartbeat
            - stale: Whether data is considered stale
            - max_attempts_triggered: Whether reconnection limit was hit
    """
    details: dict[str, Any] = {"severity": "warning"}

    # Check if broker supports WS health
    if not hasattr(broker, "get_ws_health"):
        details["error"] = "Broker does not support WebSocket health checks"
        details["ws_not_supported"] = True
        # Not having WS is not necessarily unhealthy - it's optional
        return True, details

    try:
        health = broker.get_ws_health()

        if not health:
            # WS not initialized - this is OK, streaming is optional
            details["ws_not_initialized"] = True
            return True, details

        clock = time_provider or get_clock()
        now = clock.time()
        connected = health.get("connected", False)
        last_message_ts = health.get("last_message_ts", 0)
        last_heartbeat_ts = health.get("last_heartbeat_ts", 0)
        max_attempts_triggered = health.get("max_attempts_triggered", False)

        # Calculate ages
        message_age = now - last_message_ts if last_message_ts else float("inf")
        heartbeat_age = now - last_heartbeat_ts if last_heartbeat_ts else float("inf")

        details.update(
            {
                "connected": connected,
                "last_message_age_seconds": (
                    round(message_age, 1) if message_age != float("inf") else None
                ),
                "last_heartbeat_age_seconds": (
                    round(heartbeat_age, 1) if heartbeat_age != float("inf") else None
                ),
                "gap_count": health.get("gap_count", 0),
                "reconnect_count": health.get("reconnect_count", 0),
                "max_attempts_triggered": max_attempts_triggered,
            }
        )

        # Determine health status
        is_stale = False
        if connected:
            if last_message_ts and message_age > message_stale_seconds:
                is_stale = True
                details["stale_reason"] = "message"
            elif last_heartbeat_ts and heartbeat_age > heartbeat_stale_seconds:
                is_stale = True
                details["stale_reason"] = "heartbeat"

        details["stale"] = is_stale

        # Max attempts triggered is critical
        if max_attempts_triggered:
            details["severity"] = "critical"
            return False, details

        # Disconnected or stale is a failure
        if not connected or is_stale:
            return False, details

        return True, details

    except Exception as exc:
        details["error"] = str(exc)
        details["error_type"] = type(exc).__name__
        logger.warning(
            "WS health check failed",
            operation="health_check",
            error=str(exc),
        )
        return False, details


def _coerce_symbol_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, (list, tuple, set)):
        return [str(symbol) for symbol in raw if symbol]
    return []


def _extract_market_data_symbols(market_data_service: Any) -> list[str]:
    for attribute_name in ("symbols", "_symbols"):
        raw = getattr(market_data_service, attribute_name, None)
        if raw is not None:
            return _coerce_symbol_list(raw)

    getter = getattr(market_data_service, "get_symbols", None)
    if callable(getter):
        try:
            return _coerce_symbol_list(getter())
        except Exception:
            return []

    return []


def _resolve_ticker_freshness_provider(
    market_data_service: Any,
) -> TickerFreshnessProvider | None:
    if market_data_service is None:
        return None

    # If the service itself is a provider, use it.
    if isinstance(market_data_service, TickerFreshnessProvider):
        return market_data_service

    # If the service explicitly exposes a provider, trust that contract.
    # Important: if it returns None, do NOT guess/fallback to internal attrs.
    if isinstance(market_data_service, TickerFreshnessProviderSource):
        try:
            provider = market_data_service.get_ticker_freshness_provider()
        except Exception:
            return None
        if isinstance(provider, TickerFreshnessProvider):
            return provider
        return None

    # Backwards-compatible attribute-based discovery (used by some tests / simple stubs).
    for attribute_name in ("ticker_cache", "_ticker_cache"):
        candidate = getattr(market_data_service, attribute_name, None)
        if candidate is not None and hasattr(candidate, "is_stale"):
            return cast(TickerFreshnessProvider, candidate)

    # Finally, accept any object that has a callable is_stale.
    if hasattr(market_data_service, "is_stale") and callable(
        getattr(market_data_service, "is_stale", None)
    ):
        return cast(TickerFreshnessProvider, market_data_service)

    return None


def check_ticker_freshness(
    market_data_service: TickerFreshnessService | None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check market data ticker freshness using a freshness provider.

    Args:
        market_data_service: Market data service or provider that exposes ticker freshness.

    Returns:
        Tuple of (healthy, details) where details includes:
            - symbols_checked: Symbols evaluated
            - stale_symbols: Symbols with stale or missing tickers
            - stale_count: Count of stale symbols
            - symbol_count: Total symbol count

    Metrics:
        - gpt_trader_profile_duration_seconds{phase="ticker_freshness"}: duration histogram
          recorded via profile_span.
        - gpt_trader_ticker_freshness_checks_total{result="ok"|"error"}: check outcome.
        - gpt_trader_ticker_cache_unavailable_total: increments when the market data
          provider or ticker cache is unavailable.
        - gpt_trader_ticker_stale_symbols_total: increments by the number of stale
          symbols observed per check.
    """
    details: dict[str, Any] = {"severity": "warning"}

    def record_outcome(healthy: bool, details: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        result = "ok" if healthy else "error"
        record_counter(TICKER_FRESHNESS_CHECKS_COUNTER, labels={"result": result})
        return healthy, details

    with profile_span("ticker_freshness"):
        if market_data_service is None:
            record_counter(TICKER_CACHE_UNAVAILABLE_COUNTER)
            details.update(
                {
                    "skipped": True,
                    "reason": "market_data_service_unavailable",
                }
            )
            return record_outcome(True, details)

        symbols = _extract_market_data_symbols(market_data_service)
        details["symbol_count"] = len(symbols)
        details["symbols_checked"] = list(symbols)

        if not symbols:
            details.update(
                {
                    "stale_symbols": [],
                    "stale_count": 0,
                    "stale": False,
                }
            )
            return record_outcome(True, details)

        ticker_freshness_provider = _resolve_ticker_freshness_provider(market_data_service)
        if ticker_freshness_provider is None:
            record_counter(TICKER_CACHE_UNAVAILABLE_COUNTER)
            details.update(
                {
                    "ticker_cache_unavailable": True,
                    "ticker_freshness_provider_unavailable": True,
                    "skipped": True,
                    "reason": "ticker_freshness_provider_unavailable",
                }
            )
            # No provider means we cannot evaluate staleness. Treat this as skipped/healthy so /health doesn't permanently fail.
            return record_outcome(True, details)

        stale_symbols: list[str] = []
        fresh_symbols: list[str] = []
        try:
            for symbol in symbols:
                if ticker_freshness_provider.is_stale(symbol):
                    stale_symbols.append(symbol)
                else:
                    fresh_symbols.append(symbol)
        except Exception as exc:
            details.update(
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "severity": "critical",
                }
            )
            logger.warning(
                "Ticker freshness check failed",
                operation="health_check",
                error=str(exc),
            )
            return record_outcome(False, details)

        details.update(
            {
                "fresh_symbols": fresh_symbols,
                "stale_symbols": stale_symbols,
                "stale_count": len(stale_symbols),
                "stale": bool(stale_symbols),
            }
        )

        if stale_symbols:
            record_counter(TICKER_STALE_SYMBOLS_COUNTER, increment=len(stale_symbols))
            if len(stale_symbols) == len(symbols):
                details["severity"] = "critical"
            return record_outcome(False, details)

        return record_outcome(True, details)


def check_degradation_state(
    degradation_state: DegradationState,
    risk_manager: RiskManagerProtocol | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check trading degradation state.

    Args:
        degradation_state: DegradationState instance.
        risk_manager: Optional LiveRiskManager to check reduce-only mode.

    Returns:
        Tuple of (healthy, details) where details includes:
            - global_paused: Whether all trading is paused
            - global_pause_reason: Reason for global pause
            - paused_symbol_count: Number of symbols with individual pauses
            - reduce_only_mode: Whether reduce-only is active
            - reduce_only_reason: Reason for reduce-only mode
    """
    details: dict[str, Any] = {"severity": "warning"}

    try:
        status = degradation_state.get_status()

        global_paused = status.get("global_paused", False)
        global_reason = status.get("global_reason")
        paused_symbols = status.get("paused_symbols", {})
        if not isinstance(paused_symbols, dict):
            paused_symbols = {}

        details.update(
            {
                "global_paused": global_paused,
                "global_pause_reason": global_reason,
                "global_remaining_seconds": status.get("global_remaining_seconds", 0),
                "paused_symbol_count": len(paused_symbols),
            }
        )

        if paused_symbols:
            details["paused_symbols"] = list(paused_symbols.keys())

        # Check reduce-only mode from risk manager
        reduce_only = False
        reduce_only_reason = None

        if risk_manager is not None:
            if hasattr(risk_manager, "is_reduce_only_mode"):
                reduce_only = risk_manager.is_reduce_only_mode()
            elif hasattr(risk_manager, "_reduce_only_mode"):
                reduce_only = risk_manager._reduce_only_mode

            if reduce_only and hasattr(risk_manager, "_reduce_only_reason"):
                reduce_only_reason = risk_manager._reduce_only_reason

            # Also check CFM reduce-only mode
            cfm_reduce_only = False
            if hasattr(risk_manager, "is_cfm_reduce_only_mode"):
                cfm_reduce_only = risk_manager.is_cfm_reduce_only_mode()
            elif hasattr(risk_manager, "_cfm_reduce_only_mode"):
                cfm_reduce_only = risk_manager._cfm_reduce_only_mode

            if cfm_reduce_only:
                reduce_only = True
                if hasattr(risk_manager, "_cfm_reduce_only_reason"):
                    reduce_only_reason = risk_manager._cfm_reduce_only_reason

        details["reduce_only_mode"] = reduce_only
        if reduce_only_reason:
            details["reduce_only_reason"] = reduce_only_reason

        # Global pause is critical
        if global_paused:
            details["severity"] = "critical"
            return False, details

        # Reduce-only or symbol pauses are warnings but not failures
        if reduce_only or paused_symbols:
            details["severity"] = "warning"
            # Return healthy but with warning details
            return True, details

        return True, details

    except Exception as exc:
        details["error"] = str(exc)
        details["error_type"] = type(exc).__name__
        details["severity"] = "critical"
        logger.warning(
            "Degradation state check failed",
            operation="health_check",
            error=str(exc),
        )
        return False, details


class HealthCheckRunner:
    """
    Executes health checks on a configurable cadence.

    Runs checks in a background task and updates the HealthState
    with results for the health server to expose.
    """

    def __init__(
        self,
        health_state: HealthState,
        broker: BrokerProtocol | None = None,
        degradation_state: DegradationState | None = None,
        risk_manager: RiskManagerProtocol | None = None,
        market_data_service: TickerFreshnessService | None = None,
        interval_seconds: float = 30.0,
        message_stale_seconds: float = 60.0,
        heartbeat_stale_seconds: float = 120.0,
        broker_calls: BoundedToThread | None = None,
        time_provider: TimeProvider | None = None,
    ) -> None:
        """
        Initialize the health check runner.

        Args:
            health_state: HealthState instance to update with results.
            broker: Broker for connectivity checks.
            degradation_state: DegradationState for pause/reduce-only checks.
            risk_manager: RiskManager for reduce-only mode checks.
            market_data_service: Market data service or provider for ticker freshness checks.
            interval_seconds: How often to run checks (default 30s).
            message_stale_seconds: WS message staleness threshold.
            heartbeat_stale_seconds: WS heartbeat staleness threshold.
            time_provider: Optional time provider for deterministic staleness checks.
        """
        self._health_state = health_state
        self._broker = broker
        self._degradation_state = degradation_state
        self._risk_manager = risk_manager
        self._market_data_service = market_data_service
        self._interval = interval_seconds
        self._message_stale_seconds = message_stale_seconds
        self._heartbeat_stale_seconds = heartbeat_stale_seconds
        self._broker_calls = broker_calls
        self._time_provider = time_provider
        self._running = False
        self._task: Any = None

    def set_broker(self, broker: BrokerProtocol) -> None:
        """Set the broker for connectivity checks."""
        self._broker = broker

    def set_degradation_state(self, state: DegradationState) -> None:
        """Set the degradation state for pause checks."""
        self._degradation_state = state

    def set_risk_manager(self, manager: RiskManagerProtocol) -> None:
        """Set the risk manager for reduce-only checks."""
        self._risk_manager = manager

    def set_market_data_service(self, market_data_service: TickerFreshnessService) -> None:
        """Set the market data service or provider for ticker freshness checks."""
        self._market_data_service = market_data_service

    def _health_check_registry(self) -> tuple[HealthCheckDescriptor, ...]:
        """Build the registry of health checks to execute."""
        checks: list[HealthCheckDescriptor] = []

        broker = self._broker
        if broker is not None:
            checks.append(
                HealthCheckDescriptor(
                    name="broker",
                    mode="blocking",
                    run=partial(check_broker_ping, broker),
                )
            )
            checks.append(
                HealthCheckDescriptor(
                    name="websocket",
                    mode="blocking",
                    run=partial(
                        check_ws_freshness,
                        broker,
                        message_stale_seconds=self._message_stale_seconds,
                        heartbeat_stale_seconds=self._heartbeat_stale_seconds,
                        time_provider=self._time_provider,
                    ),
                )
            )

        market_data_service = self._market_data_service
        if market_data_service is not None:
            checks.append(
                HealthCheckDescriptor(
                    name="ticker_freshness",
                    mode="fast",
                    run=partial(check_ticker_freshness, market_data_service),
                )
            )

        degradation_state = self._degradation_state
        if degradation_state is not None:
            checks.append(
                HealthCheckDescriptor(
                    name="degradation",
                    mode="fast",
                    run=partial(check_degradation_state, degradation_state, self._risk_manager),
                )
            )

        return tuple(checks)

    async def start(self) -> None:
        """Start the health check runner background task."""
        import asyncio

        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Health check runner started",
            operation="health_check_runner",
            interval_seconds=self._interval,
        )

    async def stop(self) -> None:
        """Stop the health check runner."""
        import asyncio

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
            except Exception:
                pass
            self._task = None
        logger.info("Health check runner stopped", operation="health_check_runner")

    async def _run_loop(self) -> None:
        """Main loop that executes checks periodically."""
        import asyncio

        while self._running:
            try:
                await self._execute_checks()
            except Exception as exc:
                logger.warning(
                    "Health check execution failed",
                    operation="health_check_runner",
                    error=str(exc),
                )

            await asyncio.sleep(self._interval)

    async def _execute_checks(self) -> None:
        """Execute all configured health checks."""
        import asyncio

        health_state = self._health_state
        broker_calls = self._broker_calls
        if broker_calls is not None and not asyncio.iscoroutinefunction(
            getattr(broker_calls, "__call__", None)
        ):
            broker_calls = None

        async def run_blocking(
            callable_obj: Callable[[], HealthCheckOutcome],
        ) -> HealthCheckOutcome:
            if broker_calls is None:
                return await asyncio.to_thread(callable_obj)
            return await broker_calls(callable_obj)

        for check in self._health_check_registry():
            try:
                if check.mode == "blocking":
                    healthy, details = await run_blocking(check.run)
                else:
                    healthy, details = check.run()
                health_state.add_check(check.name, healthy, details)
            except Exception as exc:
                health_state.add_check(check.name, False, {"error": str(exc)})

    def run_checks_sync(self) -> dict[str, tuple[bool, dict[str, Any]]]:
        """
        Run all checks synchronously and return results.

        Useful for testing or one-off health checks.

        Returns:
            Dict mapping check name to (healthy, details) tuple.
        """
        results: dict[str, HealthCheckOutcome] = {}

        for check in self._health_check_registry():
            results[check.name] = check.run()

        return results


def compute_execution_health_signals(
    thresholds: HealthThresholds | None = None,
) -> HealthSummary:
    """Compute health signals from execution metrics.

    Uses the metrics collector to derive health signals for:
    - Order submission error rate
    - Order retry rate
    - Broker call latency (p95)
    - Guard trip frequency
    - Missing decision_id count

    Args:
        thresholds: Optional custom thresholds. Uses defaults if None.

    Returns:
        HealthSummary with computed signals.
    """
    from gpt_trader.monitoring.health_signals import (
        HealthSummary,
        HealthThresholds,
    )
    from gpt_trader.monitoring.metrics_collector import get_metrics_collector

    if thresholds is None:
        thresholds = HealthThresholds()

    counters, histograms = _extract_execution_metrics(get_metrics_collector().get_metrics_summary())
    signals = _build_execution_health_signals(counters, histograms, thresholds)
    return HealthSummary.from_signals(signals)


def _extract_execution_metrics(
    summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    counters_raw = summary.get("counters", {})
    histograms_raw = summary.get("histograms", {})
    counters = counters_raw if isinstance(counters_raw, dict) else {}
    histograms = histograms_raw if isinstance(histograms_raw, dict) else {}
    return counters, histograms


def _build_execution_health_signals(
    counters: dict[str, Any],
    histograms: dict[str, Any],
    thresholds: HealthThresholds,
) -> list[HealthSignal]:
    from gpt_trader.monitoring.health_signals import HealthSignal

    total_submissions, failed_submissions = _count_order_submissions(counters)
    error_rate = _ratio_or_zero(failed_submissions, total_submissions)

    broker_call_count, broker_call_failures = _count_broker_calls(histograms)
    retry_rate = _ratio_or_zero(broker_call_count - total_submissions, total_submissions)
    if broker_call_count <= total_submissions:
        retry_rate = 0.0

    latency_p95_ms = _compute_broker_latency_p95_ms(histograms)
    guard_trip_count = _count_guard_trips(counters)
    missing_decision_id_count = _count_missing_decision_ids(counters)

    return [
        HealthSignal.from_value(
            name="order_error_rate",
            value=error_rate,
            threshold_warn=thresholds.order_error_rate_warn,
            threshold_crit=thresholds.order_error_rate_crit,
            unit="ratio",
            details={
                "total_submissions": total_submissions,
                "failed_submissions": failed_submissions,
            },
        ),
        HealthSignal.from_value(
            name="order_retry_rate",
            value=retry_rate,
            threshold_warn=thresholds.order_retry_rate_warn,
            threshold_crit=thresholds.order_retry_rate_crit,
            unit="ratio",
            details={
                "broker_call_count": broker_call_count,
                "broker_call_failures": broker_call_failures,
                "submission_count": total_submissions,
            },
        ),
        HealthSignal.from_value(
            name="broker_latency_p95",
            value=latency_p95_ms,
            threshold_warn=thresholds.broker_latency_ms_warn,
            threshold_crit=thresholds.broker_latency_ms_crit,
            unit="ms",
            details={},
        ),
        HealthSignal.from_value(
            name="guard_trip_count",
            value=float(guard_trip_count),
            threshold_warn=float(thresholds.guard_trip_count_warn),
            threshold_crit=float(thresholds.guard_trip_count_crit),
            unit="count",
            details={},
        ),
        HealthSignal.from_value(
            name="missing_decision_id_count",
            value=float(missing_decision_id_count),
            threshold_warn=float(thresholds.missing_decision_id_count_warn),
            threshold_crit=float(thresholds.missing_decision_id_count_crit),
            unit="count",
            details={},
        ),
    ]


def _count_order_submissions(counters: dict[str, Any]) -> tuple[int, int]:
    total_submissions = 0
    failed_submissions = 0
    for key, count in counters.items():
        if key.startswith("gpt_trader_order_submission_total"):
            total_submissions += count
            if "result=failed" in key or "result=rejected" in key:
                failed_submissions += count
    return total_submissions, failed_submissions


def _count_broker_calls(histograms: dict[str, Any]) -> tuple[int, int]:
    broker_call_count = 0
    broker_call_failures = 0
    for key, hist_data in histograms.items():
        if key.startswith("gpt_trader_broker_call_latency_seconds"):
            if isinstance(hist_data, dict):
                broker_call_count += hist_data.get("count", 0)
                if "outcome=failure" in key:
                    broker_call_failures += hist_data.get("count", 0)
    return broker_call_count, broker_call_failures


def _compute_broker_latency_p95_ms(histograms: dict[str, Any]) -> float:
    for key, hist_data in histograms.items():
        if key.startswith("gpt_trader_broker_call_latency_seconds") and "outcome=success" in key:
            if isinstance(hist_data, dict):
                return _approximate_p95_ms(hist_data)
            break
    return 0.0


def _approximate_p95_ms(hist_data: dict[str, Any]) -> float:
    total_count = hist_data.get("count", 0)
    if total_count <= 0:
        return 0.0

    buckets = hist_data.get("buckets", {})
    target = total_count * 0.95
    for bucket_str, bucket_count in sorted(buckets.items(), key=lambda x: float(x[0])):
        if bucket_count >= target:
            return float(bucket_str) * 1000

    mean = hist_data.get("mean", 0.0)
    return float(mean) * 1000 * 1.5


def _count_guard_trips(counters: dict[str, Any]) -> int:
    guard_trip_count = 0
    for key, count in counters.items():
        lowered = key.lower()
        if "guard" in lowered and ("failure" in lowered or "trip" in lowered):
            guard_trip_count += count
    return guard_trip_count


def _count_missing_decision_ids(counters: dict[str, Any]) -> int:
    missing_count = 0
    for key, count in counters.items():
        if key.startswith("gpt_trader_order_missing_decision_id_total"):
            missing_count += count
    return missing_count


def _ratio_or_zero(numerator: int | float, denominator: int | float) -> float:
    if denominator > 0:
        return float(numerator) / float(denominator)
    return 0.0


def check_ws_staleness_signal(
    broker: BrokerProtocol,
    thresholds: HealthThresholds | None = None,
) -> HealthSignal:
    """Check WebSocket staleness as a health signal.

    Args:
        broker: Broker protocol instance.
        thresholds: Optional custom thresholds.

    Returns:
        HealthSignal for WebSocket staleness.
    """
    from gpt_trader.monitoring.health_signals import (
        HealthSignal,
        HealthStatus,
        HealthThresholds,
    )

    if thresholds is None:
        thresholds = HealthThresholds()

    # Check if broker supports WS health
    if not hasattr(broker, "get_ws_health"):
        return HealthSignal(
            name="ws_staleness",
            status=HealthStatus.UNKNOWN,
            value=0.0,
            threshold_warn=thresholds.ws_staleness_seconds_warn,
            threshold_crit=thresholds.ws_staleness_seconds_crit,
            unit="seconds",
            details={"ws_not_supported": True},
        )

    try:
        health = broker.get_ws_health()
        if not health:
            return HealthSignal(
                name="ws_staleness",
                status=HealthStatus.OK,
                value=0.0,
                threshold_warn=thresholds.ws_staleness_seconds_warn,
                threshold_crit=thresholds.ws_staleness_seconds_crit,
                unit="seconds",
                details={"ws_not_initialized": True},
            )

        now = time.time()
        last_message_ts = health.get("last_message_ts", 0)
        staleness = now - last_message_ts if last_message_ts else float("inf")

        # Cap staleness at a reasonable max for display
        if staleness == float("inf"):
            staleness = 9999.0

        return HealthSignal.from_value(
            name="ws_staleness",
            value=staleness,
            threshold_warn=thresholds.ws_staleness_seconds_warn,
            threshold_crit=thresholds.ws_staleness_seconds_crit,
            unit="seconds",
            details={
                "connected": health.get("connected", False),
                "last_message_ts": last_message_ts,
            },
        )

    except Exception as exc:
        return HealthSignal(
            name="ws_staleness",
            status=HealthStatus.UNKNOWN,
            value=0.0,
            threshold_warn=thresholds.ws_staleness_seconds_warn,
            threshold_crit=thresholds.ws_staleness_seconds_crit,
            unit="seconds",
            details={"error": str(exc)},
        )


# Import for type hints
if TYPE_CHECKING:
    from gpt_trader.monitoring.health_signals import (
        HealthSignal,
        HealthSummary,
        HealthThresholds,
    )


__all__ = [
    "HealthCheckResult",
    "HealthCheckRunner",
    "TICKER_FRESHNESS_CHECKS_COUNTER",
    "TICKER_CACHE_UNAVAILABLE_COUNTER",
    "TICKER_STALE_SYMBOLS_COUNTER",
    "check_broker_ping",
    "check_ticker_freshness",
    "check_ws_freshness",
    "check_degradation_state",
    "compute_execution_health_signals",
    "check_ws_staleness_signal",
]
