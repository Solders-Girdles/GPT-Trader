"""Tests for AccountWidget display signature caching."""

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


class TestAccountWidgetSignatureCaching:
    """Tests for display signature caching optimization."""

    @pytest.mark.asyncio
    async def test_skips_update_when_signature_unchanged(self) -> None:
        """Test that on_state_updated skips work when state unchanged."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.account_data = AccountSummary(
                balances=[
                    AccountBalance(
                        asset="USD", total=Decimal("5000.00"), available=Decimal("5000.00")
                    )
                ]
            )
            state.position_data = PortfolioSummary(equity=Decimal("5000.00"))

            widget.on_state_updated(state)
            cached_sig = widget._last_display_signature
            assert cached_sig is not None

            widget.on_state_updated(state)
            assert widget._last_display_signature == cached_sig

    @pytest.mark.asyncio
    async def test_updates_when_balance_changes(self) -> None:
        """Test that signature detects balance changes."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.account_data = AccountSummary(
                balances=[
                    AccountBalance(
                        asset="USD", total=Decimal("5000.00"), available=Decimal("5000.00")
                    )
                ]
            )
            state.position_data = PortfolioSummary(equity=Decimal("5000.00"))

            widget.on_state_updated(state)
            original_sig = widget._last_display_signature

            state.account_data = AccountSummary(
                balances=[
                    AccountBalance(
                        asset="USD", total=Decimal("6000.00"), available=Decimal("6000.00")
                    )
                ]
            )

            widget.on_state_updated(state)
            assert widget._last_display_signature != original_sig

    @pytest.mark.asyncio
    async def test_updates_when_equity_changes(self) -> None:
        """Test that signature detects equity changes."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.account_data = AccountSummary(balances=[])
            state.position_data = PortfolioSummary(equity=Decimal("10000.00"))

            widget.on_state_updated(state)
            original_sig = widget._last_display_signature

            state.position_data = PortfolioSummary(equity=Decimal("11000.00"))

            widget.on_state_updated(state)
            assert widget._last_display_signature != original_sig

    @pytest.mark.asyncio
    async def test_signature_includes_degraded_mode(self) -> None:
        """Test that signature changes when degraded_mode changes."""
        app = AccountWidgetTestApp()

        async with app.run_test():
            widget = app.query_one(AccountWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.degraded_mode = False
            state.account_data = AccountSummary(balances=[])
            state.position_data = PortfolioSummary()

            widget.on_state_updated(state)
            sig_normal = widget._last_display_signature

            state.degraded_mode = True

            widget.on_state_updated(state)
            assert widget._last_display_signature != sig_normal
