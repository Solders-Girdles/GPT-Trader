from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.tui.screens import MainScreen
from gpt_trader.tui.state import TuiState


class TestMainScreen:
    def test_update_ui(self):
        """Test that update_ui correctly sets screen state for StateRegistry broadcast.

        With the StateRegistry pattern, update_ui() sets self.state which triggers
        watch_state() -> StateRegistry.broadcast(). Widgets that implement StateObserver
        receive updates via their on_state_updated() method rather than direct property
        assignment from MainScreen.
        """
        screen = MainScreen()

        screen.query_one = MagicMock(return_value=MagicMock())

        # Create state with Decimal values
        state = TuiState()
        state.running = True
        state.data_source_mode = "demo"
        state.position_data.equity = Decimal("5000.00")
        state.market_data.prices = {"ETH": Decimal("2000.00")}
        state.strategy_data.active_strategies = ["TestStrat"]
        state.risk_data.max_leverage = 5.0
        state.system_data.connection_status = "CONNECTED"

        # Mock check_connection_health for footer update
        state.check_connection_health = MagicMock(return_value=True)

        # Call update_ui
        screen.update_ui(state)

        # Verify screen state was set (triggers watch_state -> StateRegistry.broadcast)
        assert screen.state == state

        # Note: Dashboard widgets (MarketPulseWidget, PositionCardWidget, etc.)
        # now receive updates via StateRegistry.broadcast() when registered.
        # They implement on_state_updated() to handle state changes.
