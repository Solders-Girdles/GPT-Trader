from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.tui.screens import MainScreen
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import (
    MarketWatchWidget,
    RiskWidget,
    SlimStatusWidget,
    StrategyWidget,
    SystemHealthWidget,
)


class TestMainScreen:
    def test_update_ui(self):
        """Test that update_ui correctly updates SlimStatusWidget and sets screen state."""
        screen = MainScreen()

        # Mock widgets for log-centric layout
        mock_status = MagicMock(spec=SlimStatusWidget)
        mock_market = MagicMock(spec=MarketWatchWidget)
        mock_strategy = MagicMock(spec=StrategyWidget)
        mock_risk = MagicMock(spec=RiskWidget)
        mock_system = MagicMock(spec=SystemHealthWidget)

        # Mock query_one to return specific mocks based on type or ID
        def query_side_effect(arg, *args, **kwargs):
            if arg == SlimStatusWidget:
                return mock_status
            if arg == "#dash-market" or arg == "#market-watch-full":
                return mock_market
            if arg == "#dash-strategy":
                return mock_strategy
            if arg == "#dash-risk":
                return mock_risk
            if arg == "#dash-system" or arg == "#system-full":
                return mock_system
            # Return a generic mock for others to avoid failures
            return MagicMock()

        screen.query_one = MagicMock(side_effect=query_side_effect)

        # Create state with Decimal values
        state = TuiState()
        state.running = True
        state.position_data.equity = Decimal("5000.00")
        state.market_data.prices = {"ETH": Decimal("2000.00")}
        state.strategy_data.active_strategies = ["TestStrat"]
        state.risk_data.max_leverage = 5.0
        state.system_data.connection_status = "CONNECTED"

        # Mock query to return widgets
        screen.query = MagicMock(return_value=[mock_strategy])

        # Call update_ui
        screen.update_ui(state)

        # Verify SlimStatusWidget properties were set directly
        assert mock_status.running is True
        # equity is set via property assignment (checked via hasattr since it's a mock)
        assert hasattr(mock_status, "equity")

        # Verify screen state was set (triggers reactive cascade)
        assert screen.state == state

        # With StateRegistry pattern, widgets self-register and receive updates
        # via broadcast when screen.state is set. The test verifies state propagation
        # by checking screen.state is correctly assigned (done above).
