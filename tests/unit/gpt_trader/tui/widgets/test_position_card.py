"""Tests for PositionCardWidget behavior and caching."""

from contextlib import asynccontextmanager
from decimal import Decimal

import pytest
from textual.app import App, ComposeResult

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import PortfolioSummary, Position, SystemStatus
from gpt_trader.tui.widgets.dashboard_position import PositionCardWidget


class PositionCardTestApp(App):
    def compose(self) -> ComposeResult:
        yield PositionCardWidget(id="test-position-card")


@asynccontextmanager
async def _run_position_card():
    app = PositionCardTestApp()
    async with app.run_test():
        yield app, app.query_one(PositionCardWidget)


def _make_state(
    *,
    running: bool = True,
    position_data: PortfolioSummary | None = None,
    system_data: SystemStatus | None = None,
    degraded_mode: bool = False,
    degraded_reason: str = "",
    connection_healthy: bool = True,
) -> TuiState:
    state = TuiState(validation_enabled=False, delta_updates_enabled=False)
    state.running = running
    state.position_data = position_data
    state.system_data = system_data
    state.degraded_mode = degraded_mode
    state.degraded_reason = degraded_reason
    state.connection_healthy = connection_healthy
    return state


def _base_signature_state() -> TuiState:
    return _make_state(
        running=True,
        system_data=SystemStatus(connection_status="CONNECTED"),
        position_data=PortfolioSummary(
            positions={
                "BTC-USD": Position(
                    symbol="BTC-USD",
                    quantity=Decimal("0.1"),
                    unrealized_pnl=Decimal("100.00"),
                    mark_price=Decimal("95100.00"),
                )
            }
        ),
    )


def _state_no_positions() -> TuiState:
    return _make_state(
        position_data=PortfolioSummary(positions={}),
        system_data=SystemStatus(connection_status="CONNECTED"),
    )


def _state_missing_system() -> TuiState:
    return _make_state(
        position_data=PortfolioSummary(
            positions={
                "ETH-USD": Position(
                    symbol="ETH-USD",
                    quantity=Decimal("1.0"),
                    unrealized_pnl=Decimal("-100.00"),
                    mark_price=Decimal("3550.00"),
                )
            }
        ),
        system_data=None,
    )


def _state_degraded() -> TuiState:
    return _make_state(
        position_data=PortfolioSummary(positions={}),
        system_data=SystemStatus(connection_status="CONNECTED"),
        degraded_mode=True,
        degraded_reason="StatusReporter unavailable",
    )


def _state_unhealthy() -> TuiState:
    return _make_state(
        position_data=PortfolioSummary(positions={}),
        system_data=SystemStatus(connection_status="DISCONNECTED"),
        connection_healthy=False,
    )


def _state_stopped() -> TuiState:
    return _make_state(
        running=False,
        position_data=PortfolioSummary(positions={}),
        system_data=SystemStatus(connection_status="IDLE"),
    )


def _update_pnl(state: TuiState) -> None:
    state.position_data = PortfolioSummary(
        positions={
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("0.1"),
                unrealized_pnl=Decimal("200.00"),
                mark_price=Decimal("95200.00"),
            )
        }
    )


def _update_connection(state: TuiState) -> None:
    state.system_data = SystemStatus(connection_status="RECONNECTING")


@pytest.mark.asyncio
async def test_initial_state() -> None:
    async with _run_position_card() as (_, widget):
        assert not widget._has_received_update


@pytest.mark.asyncio
async def test_on_state_updated_with_position() -> None:
    async with _run_position_card() as (_, widget):
        state = _make_state(
            running=True,
            position_data=PortfolioSummary(
                positions={
                    "BTC-USD": Position(
                        symbol="BTC-USD",
                        quantity=Decimal("0.1"),
                        unrealized_pnl=Decimal("500.00"),
                        mark_price=Decimal("96000.00"),
                    )
                }
            ),
            system_data=SystemStatus(connection_status="CONNECTED"),
        )

        widget.on_state_updated(state)

        assert widget._has_received_update
        assert widget.position_data is not None
        assert widget.position_data["symbol"] == "BTC-USD"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_factory, expected_symbol, expected_connection, expected_bot_running",
    [
        (_state_no_positions, None, None, None),
        (_state_missing_system, "ETH-USD", "", None),
        (_state_degraded, None, None, None),
        (_state_unhealthy, None, "DISCONNECTED", None),
        (_state_stopped, None, "IDLE", False),
    ],
)
async def test_partial_state_handling(
    state_factory,
    expected_symbol: str | None,
    expected_connection: str | None,
    expected_bot_running: bool | None,
) -> None:
    async with _run_position_card() as (_, widget):
        state = state_factory()
        widget.on_state_updated(state)
        if expected_symbol is None:
            assert widget.position_data is None
        else:
            assert widget.position_data is not None
            assert widget.position_data["symbol"] == expected_symbol
        if expected_connection is not None:
            assert widget._connection_status == expected_connection
        if expected_bot_running is not None:
            assert widget._bot_running is expected_bot_running


@pytest.mark.asyncio
async def test_signature_caching_skips_on_same_state() -> None:
    async with _run_position_card() as (_, widget):
        state = _base_signature_state()
        widget.on_state_updated(state)
        cached_sig = widget._last_display_signature
        widget.on_state_updated(state)
        assert widget._last_display_signature == cached_sig


@pytest.mark.asyncio
@pytest.mark.parametrize("mutator", [_update_pnl, _update_connection])
async def test_signature_changes_on_update(mutator) -> None:
    async with _run_position_card() as (_, widget):
        state = _base_signature_state()
        widget.on_state_updated(state)
        original_sig = widget._last_display_signature
        mutator(state)
        widget.on_state_updated(state)
        assert widget._last_display_signature != original_sig


@pytest.mark.asyncio
async def test_mount_registers_with_state_registry() -> None:
    class MockStateRegistry:
        def __init__(self):
            self.registered = []

        def register(self, widget):
            self.registered.append(widget)

        def unregister(self, widget):
            if widget in self.registered:
                self.registered.remove(widget)

    class TestAppWithRegistry(App):
        def __init__(self):
            super().__init__()
            self.state_registry = MockStateRegistry()

        def compose(self) -> ComposeResult:
            yield PositionCardWidget(id="test-widget")

    app = TestAppWithRegistry()

    async with app.run_test():
        widget = app.query_one(PositionCardWidget)
        assert widget in app.state_registry.registered
