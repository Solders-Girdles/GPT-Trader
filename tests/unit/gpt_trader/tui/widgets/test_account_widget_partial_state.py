"""Tests for AccountWidget resilience with partial/missing state data."""

from decimal import Decimal

import pytest
from textual.app import App, ComposeResult

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import AccountBalance, AccountSummary, PortfolioSummary
from gpt_trader.tui.widgets.account import AccountWidget


class AccountWidgetTestApp(App):
    """Test app for AccountWidget in compact mode."""

    def compose(self) -> ComposeResult:
        yield AccountWidget(compact_mode=True, id="test-account")


class TestAccountWidgetPartialState:
    """Tests for widget resilience with partial/missing state data."""

    @pytest.mark.asyncio
    async def test_handles_no_account_data(self) -> None:
        """Test that widget handles state with no account_data."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.account_data = None  # No account data

            widget.on_state_updated(state)

            assert widget._has_received_update

    @pytest.mark.asyncio
    async def test_handles_empty_balances(self) -> None:
        """Test that widget handles account with empty balances."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.account_data = AccountSummary(balances=[])
            state.position_data = PortfolioSummary()

            widget.on_state_updated(state)

            assert widget._has_received_update

    @pytest.mark.asyncio
    async def test_handles_no_position_data(self) -> None:
        """Test that widget handles state with no position_data."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.account_data = AccountSummary(
                balances=[
                    AccountBalance(
                        asset="USD", total=Decimal("1000.00"), available=Decimal("1000.00")
                    )
                ]
            )
            state.position_data = None

            widget.on_state_updated(state)

            assert widget._has_received_update

    @pytest.mark.asyncio
    async def test_handles_degraded_mode(self) -> None:
        """Test that widget handles degraded mode."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.degraded_mode = True
            state.degraded_reason = "StatusReporter unavailable"
            state.running = True
            state.account_data = AccountSummary(balances=[])
            state.position_data = PortfolioSummary()

            widget.on_state_updated(state)

            assert widget._has_received_update

    @pytest.mark.asyncio
    async def test_handles_connection_unhealthy(self) -> None:
        """Test that widget handles unhealthy connection state."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.connection_healthy = False
            state.running = True
            state.account_data = AccountSummary(balances=[])
            state.position_data = PortfolioSummary()

            widget.on_state_updated(state)

            assert widget._has_received_update

    @pytest.mark.asyncio
    async def test_handles_bot_not_running(self) -> None:
        """Test that widget handles stopped bot state."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = False
            state.account_data = AccountSummary(balances=[])
            state.position_data = PortfolioSummary()

            widget.on_state_updated(state)

            assert not widget._bot_running
