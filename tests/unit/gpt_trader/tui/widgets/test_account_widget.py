"""Tests for AccountWidget behavior and caching."""

from contextlib import asynccontextmanager
from decimal import Decimal

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import AccountBalance, AccountSummary, PortfolioSummary
from gpt_trader.tui.widgets.account import AccountWidget

EMPTY_ACCOUNT = AccountSummary(balances=[])
EMPTY_PORTFOLIO = PortfolioSummary()


class AccountWidgetTestApp(App):
    def __init__(self, compact_mode: bool):
        super().__init__()
        self._compact_mode = compact_mode

    def compose(self) -> ComposeResult:
        yield AccountWidget(compact_mode=self._compact_mode, id="test-account")


@asynccontextmanager
async def _run_account_app(compact_mode: bool = True):
    app = AccountWidgetTestApp(compact_mode=compact_mode)
    async with app.run_test():
        yield app, app.query_one(AccountWidget)


def _make_state(
    *,
    running: bool = True,
    account_data: AccountSummary | None = None,
    position_data: PortfolioSummary | None = None,
    degraded_mode: bool = False,
    degraded_reason: str = "",
    connection_healthy: bool = True,
) -> TuiState:
    state = TuiState(validation_enabled=False, delta_updates_enabled=False)
    state.running = running
    state.account_data = account_data
    state.position_data = position_data
    state.degraded_mode = degraded_mode
    state.degraded_reason = degraded_reason
    state.connection_healthy = connection_healthy
    return state


def _base_signature_state() -> TuiState:
    return _make_state(
        running=True,
        account_data=AccountSummary(
            balances=[
                AccountBalance(asset="USD", total=Decimal("5000.00"), available=Decimal("5000.00"))
            ]
        ),
        position_data=PortfolioSummary(equity=Decimal("5000.00")),
    )


def _update_balance(state: TuiState) -> None:
    state.account_data = AccountSummary(
        balances=[
            AccountBalance(asset="USD", total=Decimal("6000.00"), available=Decimal("6000.00"))
        ]
    )


def _update_equity(state: TuiState) -> None:
    state.position_data = PortfolioSummary(equity=Decimal("11000.00"))


def _update_degraded(state: TuiState) -> None:
    state.degraded_mode = True


@pytest.mark.asyncio
@pytest.mark.parametrize("compact_mode", [True, False])
async def test_initial_state(compact_mode: bool) -> None:
    async with _run_account_app(compact_mode) as (_, widget):
        assert widget.compact_mode is compact_mode
        assert not widget._has_received_update


@pytest.mark.asyncio
async def test_on_state_updated_with_account_data() -> None:
    async with _run_account_app() as (app, widget):
        state = _make_state(
            running=True,
            account_data=AccountSummary(
                balances=[
                    AccountBalance(
                        asset="USD", total=Decimal("10000.00"), available=Decimal("8000.00")
                    ),
                    AccountBalance(asset="BTC", total=Decimal("0.5"), available=Decimal("0.5")),
                ],
                volume_30d=Decimal("50000.00"),
                fees_30d=Decimal("25.00"),
                fee_tier="Standard",
            ),
            position_data=PortfolioSummary(
                equity=Decimal("15000.00"),
                total_unrealized_pnl=Decimal("500.00"),
                total_realized_pnl=Decimal("200.00"),
                total_fees=Decimal("25.00"),
            ),
        )

        widget.on_state_updated(state)

        assert widget._has_received_update
        rendered = str(app.query_one("#portfolio-value", Label).render())
        assert "15,000" in rendered or "15000" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_kwargs, expect_received, expect_bot_running",
    [
        ({"account_data": None, "position_data": None, "running": True}, True, None),
        (
            {"account_data": EMPTY_ACCOUNT, "position_data": EMPTY_PORTFOLIO, "running": True},
            True,
            None,
        ),
        (
            {
                "account_data": AccountSummary(
                    balances=[
                        AccountBalance(
                            asset="USD", total=Decimal("1000.00"), available=Decimal("1000.00")
                        )
                    ]
                ),
                "position_data": None,
                "running": True,
            },
            True,
            None,
        ),
        (
            {
                "account_data": EMPTY_ACCOUNT,
                "position_data": EMPTY_PORTFOLIO,
                "running": True,
                "degraded_mode": True,
                "degraded_reason": "StatusReporter unavailable",
            },
            True,
            None,
        ),
        (
            {
                "account_data": EMPTY_ACCOUNT,
                "position_data": EMPTY_PORTFOLIO,
                "running": True,
                "connection_healthy": False,
            },
            True,
            None,
        ),
        (
            {
                "account_data": EMPTY_ACCOUNT,
                "position_data": EMPTY_PORTFOLIO,
                "running": False,
            },
            None,
            False,
        ),
    ],
)
async def test_partial_state_handling(
    state_kwargs: dict, expect_received: bool | None, expect_bot_running: bool | None
) -> None:
    async with _run_account_app() as (_, widget):
        state = _make_state(**state_kwargs)
        widget.on_state_updated(state)
        if expect_received is not None:
            assert widget._has_received_update is expect_received
        if expect_bot_running is not None:
            assert widget._bot_running is expect_bot_running


@pytest.mark.asyncio
async def test_signature_caching_skips_on_same_state() -> None:
    async with _run_account_app() as (_, widget):
        state = _base_signature_state()
        widget.on_state_updated(state)
        cached_sig = widget._last_display_signature
        widget.on_state_updated(state)
        assert widget._last_display_signature == cached_sig


@pytest.mark.asyncio
@pytest.mark.parametrize("mutator", [_update_balance, _update_equity, _update_degraded])
async def test_signature_changes_on_update(mutator) -> None:
    async with _run_account_app() as (_, widget):
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
            yield AccountWidget(id="test-widget")

    app = TestAppWithRegistry()

    async with app.run_test():
        widget = app.query_one(AccountWidget)
        assert widget in app.state_registry.registered
