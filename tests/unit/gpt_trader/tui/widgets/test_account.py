"""
Tests for AccountWidget in account.py.

Tests cover:
- Partial state handling (missing data sections)
- Degraded mode and connection unhealthy states
- Display signature caching optimization
- Compact vs expanded mode behavior
"""

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

            # Check portfolio value label is updated
            portfolio_label = app.query_one("#portfolio-value", Label)
            rendered = str(portfolio_label.render())
            assert "15,000" in rendered or "15000" in rendered


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

            # Should not raise
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

            # Should not raise
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
                    ),
                ]
            )
            state.position_data = None

            # Should not raise
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

            # Should not raise
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

            # Should not raise
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

            # Should not raise
            widget.on_state_updated(state)

            assert not widget._bot_running


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
                    ),
                ]
            )
            state.position_data = PortfolioSummary(
                equity=Decimal("5000.00"),
            )

            # First call should update
            widget.on_state_updated(state)
            cached_sig = widget._last_display_signature
            assert cached_sig is not None

            # Second call with same state should hit cache
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
                    ),
                ]
            )
            state.position_data = PortfolioSummary(equity=Decimal("5000.00"))

            widget.on_state_updated(state)
            original_sig = widget._last_display_signature

            # Change balance
            state.account_data = AccountSummary(
                balances=[
                    AccountBalance(
                        asset="USD", total=Decimal("6000.00"), available=Decimal("6000.00")
                    ),
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

            # Change equity
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

            # Enable degraded mode
            state.degraded_mode = True

            widget.on_state_updated(state)
            assert widget._last_display_signature != sig_normal


class TestAccountWidgetStateRegistry:
    """Tests for StateRegistry integration."""

    @pytest.mark.asyncio
    async def test_mount_registers_with_state_registry(self) -> None:
        """Test that widget registers with StateRegistry on mount."""

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
