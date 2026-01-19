"""Tests for AccountWidget basic behavior."""

from decimal import Decimal

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import AccountBalance, AccountSummary, PortfolioSummary
from gpt_trader.tui.widgets.account import AccountWidget


class AccountWidgetTestApp(App):
    """Test app for AccountWidget in compact mode."""

    def compose(self) -> ComposeResult:
        yield AccountWidget(compact_mode=True, id="test-account")


class AccountWidgetExpandedTestApp(App):
    """Test app for AccountWidget in expanded mode."""

    def compose(self) -> ComposeResult:
        yield AccountWidget(compact_mode=False, id="test-account-expanded")


class TestAccountWidget:
    """Tests for AccountWidget basic functionality."""

    @pytest.mark.asyncio
    async def test_initial_state_compact(self) -> None:
        """Test that compact widget renders with defaults."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            assert widget.compact_mode is True
            assert not widget._has_received_update

    @pytest.mark.asyncio
    async def test_initial_state_expanded(self) -> None:
        """Test that expanded widget renders with defaults."""
        app = AccountWidgetExpandedTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            assert widget.compact_mode is False
            assert not widget._has_received_update

    @pytest.mark.asyncio
    async def test_on_state_updated_with_account_data(self) -> None:
        """Test that widget correctly updates with account data."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.account_data = AccountSummary(
                balances=[
                    AccountBalance(
                        asset="USD", total=Decimal("10000.00"), available=Decimal("8000.00")
                    ),
                    AccountBalance(asset="BTC", total=Decimal("0.5"), available=Decimal("0.5")),
                ],
                volume_30d=Decimal("50000.00"),
                fees_30d=Decimal("25.00"),
                fee_tier="Standard",
            )
            state.position_data = PortfolioSummary(
                equity=Decimal("15000.00"),
                total_unrealized_pnl=Decimal("500.00"),
                total_realized_pnl=Decimal("200.00"),
                total_fees=Decimal("25.00"),
            )

            widget.on_state_updated(state)

            assert widget._has_received_update

            portfolio_label = app.query_one("#portfolio-value", Label)
            rendered = str(portfolio_label.render())
            assert "15,000" in rendered or "15000" in rendered
