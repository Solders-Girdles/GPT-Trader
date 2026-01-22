"""Tests for SystemMonitorWidget updates, caching, and watch behavior."""

from contextlib import asynccontextmanager

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import ResilienceState, SystemStatus
from gpt_trader.tui.widgets.dashboard import SystemMonitorWidget
from gpt_trader.tui.widgets.primitives import ProgressBarWidget


class SystemMonitorTestApp(App):
    def compose(self) -> ComposeResult:
        yield SystemMonitorWidget(id="test-system-monitor")


@asynccontextmanager
async def _run_system_monitor():
    app = SystemMonitorTestApp()
    async with app.run_test():
        yield app, app.query_one(SystemMonitorWidget)


def _state_with(
    *,
    system_data: SystemStatus | None = None,
    resilience_data: ResilienceState | None = None,
    execution_data=None,
    running: bool = True,
    connection_healthy: bool = True,
    data_source_mode: str | None = None,
    degraded_mode: bool = False,
    degraded_reason: str = "",
) -> TuiState:
    state = TuiState(validation_enabled=False, delta_updates_enabled=False)
    state.system_data = system_data
    state.resilience_data = resilience_data
    state.execution_data = execution_data
    state.running = running
    state.connection_healthy = connection_healthy
    state.data_source_mode = data_source_mode
    state.degraded_mode = degraded_mode
    state.degraded_reason = degraded_reason
    return state


def _state_system_only() -> TuiState:
    return _state_with(system_data=SystemStatus(cpu_usage="50%", api_latency=100.0))


def _state_degraded() -> TuiState:
    return _state_with(
        degraded_mode=True,
        degraded_reason="StatusReporter unavailable",
        system_data=SystemStatus(),
    )


def _state_unhealthy() -> TuiState:
    return _state_with(
        running=False,
        connection_healthy=False,
        data_source_mode="live",
        system_data=SystemStatus(connection_status="DISCONNECTED"),
    )


def _state_resilience_uninitialized() -> TuiState:
    return _state_with(
        system_data=SystemStatus(cpu_usage="25%"),
        resilience_data=ResilienceState(
            latency_p50_ms=50.0,
            latency_p95_ms=100.0,
            error_rate=0.01,
            cache_hit_rate=0.8,
            any_circuit_open=False,
            last_update=0,
        ),
    )


def _signature_state() -> TuiState:
    return _state_with(
        system_data=SystemStatus(cpu_usage="50%", api_latency=100.0),
        running=True,
    )


def _change_cpu(state: TuiState) -> None:
    state.system_data = SystemStatus(cpu_usage="75%", api_latency=100.0)


def _toggle_running(state: TuiState) -> None:
    state.running = False


@pytest.mark.asyncio
async def test_initial_state() -> None:
    async with _run_system_monitor() as (app, widget):
        assert widget.cpu_usage == 0.0
        assert widget.memory_usage == "0MB"
        assert widget.latency == 0.0
        assert widget.connection_status == "CONNECTING"
        assert widget.rate_limit == "0%"
        assert "SYSTEM" in str(app.query_one(".sys-header", Label).render())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "system_data, expected",
    [
        (
            SystemStatus(
                api_latency=123.45,
                connection_status="CONNECTED",
                rate_limit_usage="15%",
                memory_usage="256MB",
                cpu_usage="45%",
            ),
            {
                "cpu_usage": 45.0,
                "latency": 123.45,
                "memory_usage": "256MB",
                "connection_status": "CONNECTED",
                "rate_limit": "15%",
            },
        ),
        (SystemStatus(cpu_usage=75.5), {"cpu_usage": 75.5}),
        (None, {"cpu_usage": 0.0}),
    ],
)
async def test_on_state_updated_system_data(system_data, expected) -> None:
    async with _run_system_monitor() as (_, widget):
        widget.on_state_updated(_state_with(system_data=system_data))
        for key, value in expected.items():
            assert getattr(widget, key) == value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory, expected",
    [
        (_state_system_only, {"cpu_usage": 50.0, "latency": 100.0}),
        (_state_degraded, {"cpu_usage": 0.0}),
        (_state_unhealthy, {"connection_status": "STOPPED"}),
        (_state_resilience_uninitialized, {"latency_p50": 0.0, "latency_p95": 0.0}),
    ],
)
async def test_partial_state_handling(factory, expected) -> None:
    async with _run_system_monitor() as (_, widget):
        widget.on_state_updated(factory())
        for key, value in expected.items():
            assert getattr(widget, key) == value


@pytest.mark.asyncio
async def test_signature_caching_skips_on_same_state() -> None:
    async with _run_system_monitor() as (_, widget):
        state = _signature_state()
        widget.on_state_updated(state)
        cached_sig = widget._last_display_signature
        widget.on_state_updated(state)
        assert widget._last_display_signature == cached_sig


@pytest.mark.asyncio
@pytest.mark.parametrize("mutator", [_change_cpu, _toggle_running])
async def test_signature_changes_on_update(mutator) -> None:
    async with _run_system_monitor() as (_, widget):
        state = _signature_state()
        widget.on_state_updated(state)
        original_sig = widget._last_display_signature
        mutator(state)
        widget.on_state_updated(state)
        assert widget._last_display_signature != original_sig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attr, value, selector, expected",
    [
        ("cpu_usage", 65.0, "#pb-cpu", 65.0),
        ("rate_limit", "25%", "#pb-rate", 25.0),
        ("rate_limit", "90%", "#pb-rate", 90.0),
        ("memory_usage", "512MB", "#pb-memory", 50.0),
    ],
)
async def test_watch_progress_bars(attr, value, selector, expected) -> None:
    async with _run_system_monitor() as (app, widget):
        setattr(widget, attr, value)
        bar = app.query_one(selector, ProgressBarWidget)
        assert bar.percentage == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "latency, expected",
    [
        (25.0, "green"),
        (100.0, "yellow"),
        (300.0, "red"),
    ],
)
async def test_watch_latency_thresholds(latency: float, expected: str) -> None:
    async with _run_system_monitor() as (app, widget):
        widget.latency = latency
        rendered = str(app.query_one("#lbl-latency", Label).render()).lower()
        assert expected in rendered or f"{int(latency)}ms" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, expected_class, expected_text",
    [
        ("CONNECTED", "status-ok", "connected"),
        ("DISCONNECTED", "status-critical", "disconnected"),
        ("CONNECTING", "status-warning", None),
    ],
)
async def test_watch_connection_status(
    status: str, expected_class: str, expected_text: str | None
) -> None:
    async with _run_system_monitor() as (app, widget):
        widget.connection_status = status
        label = app.query_one("#lbl-conn", Label)
        assert label.has_class(expected_class)
        if expected_text:
            assert expected_text in str(label.render()).lower()
