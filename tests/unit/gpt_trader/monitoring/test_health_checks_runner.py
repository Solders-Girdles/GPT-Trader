"""Tests for the health check runner and result types."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.app.health_server import HealthState
from gpt_trader.monitoring import health_checks
from gpt_trader.monitoring.health_checks import (
    HealthCheckCycleError,
    HealthCheckDescriptor,
    HealthCheckPlanner,
    HealthCheckResult,
    HealthCheckRunner,
)
from gpt_trader.monitoring.interfaces import HealthCheckDependency


class FakeTickerCache:
    def __init__(
        self,
        stale_symbols: set[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._stale_symbols = stale_symbols or set()
        self._error = error

    def is_stale(self, symbol: str) -> bool:
        if self._error is not None:
            raise self._error
        return symbol in self._stale_symbols


class FakeMarketDataService:
    def __init__(self, symbols: list[str], ticker_cache: FakeTickerCache) -> None:
        self._symbols = symbols
        self.ticker_cache = ticker_cache


class TestHealthCheckRunner:
    """Tests for HealthCheckRunner class."""

    def test_initialization(self) -> None:
        """Test runner initialization with default values."""
        health_state = HealthState()
        runner = HealthCheckRunner(health_state=health_state)

        assert runner._health_state is health_state
        assert runner._broker is None
        assert runner._degradation_state is None
        assert runner._risk_manager is None
        assert runner._market_data_service is None
        assert runner._interval == 30.0
        assert runner._running is False

    def test_set_broker(self) -> None:
        """Test setting broker after initialization."""
        runner = HealthCheckRunner(health_state=HealthState())
        broker = MagicMock()

        runner.set_broker(broker)

        assert runner._broker is broker

    def test_set_degradation_state(self) -> None:
        """Test setting degradation state."""
        runner = HealthCheckRunner(health_state=HealthState())
        state = MagicMock()

        runner.set_degradation_state(state)

        assert runner._degradation_state is state

    def test_register_health_check_records_result(self) -> None:
        """Test canonical registration stores typed results."""
        health_state = HealthState()

        def check_custom() -> HealthCheckResult:
            return HealthCheckResult(healthy=True, details={"source": "registry"})

        result = health_checks.register_health_check(health_state, "custom", check_custom)

        assert result.healthy is True
        assert health_state.checks["custom"] is result
        assert health_state.checks_payload()["custom"]["status"] == "pass"

    def test_run_checks_sync(self) -> None:
        """Test synchronous check execution."""
        broker = MagicMock()
        broker.get_time.return_value = {"epoch": 123}
        broker.get_ws_health.return_value = None

        ticker_cache = FakeTickerCache()
        market_data_service = FakeMarketDataService(["BTC-USD"], ticker_cache)

        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {},
            "global_remaining_seconds": 0,
        }

        runner = HealthCheckRunner(
            health_state=HealthState(),
            broker=broker,
            degradation_state=degradation_state,
            market_data_service=market_data_service,
        )

        results = runner.run_checks_sync()

        assert "broker" in results
        assert "websocket" in results
        assert "ticker_freshness" in results
        assert "degradation" in results
        assert results["broker"].healthy is True
        assert results["ticker_freshness"].healthy is True
        assert results["degradation"].healthy is True

    def test_run_checks_sync_ticker_freshness_exception(self) -> None:
        """Test that ticker freshness errors are captured in results."""
        ticker_error = RuntimeError("ticker failure")
        ticker_cache = FakeTickerCache(error=ticker_error)
        market_data_service = FakeMarketDataService(["BTC-USD"], ticker_cache)

        runner = HealthCheckRunner(
            health_state=HealthState(),
            market_data_service=market_data_service,
        )

        results = runner.run_checks_sync()

        assert "ticker_freshness" in results
        result = results["ticker_freshness"]
        assert result.healthy is False
        assert result.details["error"] == "ticker failure"

    def test_run_checks_sync_degradation_exception(self) -> None:
        """Test that degradation errors are captured in results."""
        degradation_state = MagicMock()
        degradation_state.get_status.side_effect = RuntimeError("degradation failure")

        runner = HealthCheckRunner(
            health_state=HealthState(),
            degradation_state=degradation_state,
        )

        results = runner.run_checks_sync()

        assert "degradation" in results
        result = results["degradation"]
        assert result.healthy is False
        assert result.details["error"] == "degradation failure"

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        """Test async start and stop."""
        import asyncio

        runner = HealthCheckRunner(
            health_state=HealthState(),
            interval_seconds=60.0,
        )  # Long interval to avoid loop execution

        await runner.start()
        assert runner._running is True
        assert runner._task is not None

        # Give the task a moment to start the loop
        await asyncio.sleep(0.01)

        await runner.stop()
        assert runner._running is False
        assert runner._task is None

    @pytest.mark.asyncio
    async def test_execute_checks_updates_health_state(self) -> None:
        """Test that execute_checks updates the global health state."""
        broker = MagicMock()
        broker.get_time.return_value = {"epoch": 123}
        broker.get_ws_health.return_value = None

        ticker_cache = FakeTickerCache()
        market_data_service = FakeMarketDataService(["BTC-USD"], ticker_cache)

        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {},
            "global_remaining_seconds": 0,
        }

        health_state = MagicMock()
        runner = HealthCheckRunner(
            health_state=health_state,
            broker=broker,
            market_data_service=market_data_service,
            degradation_state=degradation_state,
            interval_seconds=60.0,
        )

        await runner._execute_checks()

        # Verify checks were added
        assert health_state.add_check.call_count >= 2
        call_args = [call[0][0] for call in health_state.add_check.call_args_list]
        assert "broker" in call_args
        assert "websocket" in call_args
        assert "ticker_freshness" in call_args
        assert "degradation" in call_args

    @pytest.mark.asyncio
    async def test_registry_drives_sync_and_async_paths(self) -> None:
        """Ensure both sync and async paths use the registry."""
        health_state = MagicMock()
        runner = HealthCheckRunner(health_state=health_state)
        calls: list[str] = []

        def custom_check() -> HealthCheckResult:
            calls.append("called")
            return HealthCheckResult(healthy=True, details={"source": "registry"})

        registry = (
            health_checks.HealthCheckDescriptor(
                name="custom_check",
                mode="fast",
                run=custom_check,
            ),
        )
        runner._health_check_registry = MagicMock(return_value=registry)

        sync_results = runner.run_checks_sync()

        assert sync_results["custom_check"].healthy is True
        assert sync_results["custom_check"].details["source"] == "registry"

        await runner._execute_checks()

        assert runner._health_check_registry.call_count == 2
        assert calls == ["called", "called"]
        expected_result = HealthCheckResult(healthy=True, details={"source": "registry"})
        health_state.add_check.assert_called_with(
            "custom_check",
            expected_result,
        )


class TestHealthCheckPlanner:
    """Tests for HealthCheckPlanner ordering and diagnostics."""

    @staticmethod
    def _make_descriptor(
        name: str,
        dependencies: tuple[HealthCheckDependency, ...] = (),
    ) -> HealthCheckDescriptor:
        def check() -> HealthCheckResult:
            return HealthCheckResult(healthy=True, details={"name": name})

        return HealthCheckDescriptor(
            name=name,
            mode="fast",
            run=check,
            dependencies=dependencies,
        )

    def test_planner_orders_by_dependencies_and_name(self) -> None:
        checks = (
            self._make_descriptor(
                "alpha",
                dependencies=(HealthCheckDependency(name="zeta"),),
            ),
            self._make_descriptor("beta"),
            self._make_descriptor("zeta"),
        )

        planner = HealthCheckPlanner(checks)
        ordered = [check.name for check in planner.build_order()]

        assert ordered == ["beta", "zeta", "alpha"]

    def test_planner_skips_missing_optional_dependencies(self) -> None:
        checks = (
            self._make_descriptor(
                "alpha",
                dependencies=(HealthCheckDependency.optional("missing"),),
            ),
        )

        planner = HealthCheckPlanner(checks)
        ordered = [check.name for check in planner.build_order()]

        assert ordered == ["alpha"]

    def test_planner_detects_cycles_with_diagnostics(self) -> None:
        checks = (
            self._make_descriptor(
                "alpha",
                dependencies=(HealthCheckDependency(name="beta"),),
            ),
            self._make_descriptor(
                "beta",
                dependencies=(HealthCheckDependency(name="alpha"),),
            ),
        )

        planner = HealthCheckPlanner(checks)

        with pytest.raises(HealthCheckCycleError) as exc_info:
            planner.build_order()

        error_text = str(exc_info.value)
        assert "alpha" in error_text
        assert "beta" in error_text

    def test_runner_falls_back_to_name_order_on_plan_error(self) -> None:
        runner = HealthCheckRunner(health_state=HealthState())

        registry = (
            self._make_descriptor(
                "beta",
                dependencies=(HealthCheckDependency(name="alpha"),),
            ),
            self._make_descriptor(
                "alpha",
                dependencies=(HealthCheckDependency(name="beta"),),
            ),
        )
        runner._health_check_registry = MagicMock(return_value=registry)

        planned = runner._planned_checks()

        assert [check.name for check in planned] == ["alpha", "beta"]


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a health check result."""
        result = HealthCheckResult(healthy=True, details={"latency_ms": 50})

        assert result.healthy is True
        assert result.status == "pass"
        assert result.details["latency_ms"] == 50
        assert result.to_payload() == {"status": "pass", "details": {"latency_ms": 50}}

    def test_unhealthy_result(self) -> None:
        """Test creating an unhealthy result."""
        result = HealthCheckResult(
            healthy=False,
            details={"error": "connection refused", "severity": "critical"},
        )

        assert result.healthy is False
        assert result.status == "fail"
        assert result.details["error"] == "connection refused"
