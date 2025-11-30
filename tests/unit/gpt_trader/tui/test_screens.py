from unittest.mock import MagicMock

from gpt_trader.tui.screens import MainScreen
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import (
    BotStatusWidget,
    MarketWatchWidget,
    RiskWidget,
    StrategyWidget,
    SystemHealthWidget,
)


class TestMainScreen:
    def test_update_ui(self):
        screen = MainScreen()

        # Mock widgets
        mock_status = MagicMock(spec=BotStatusWidget)
        mock_market = MagicMock(spec=MarketWatchWidget)
        mock_strategy = MagicMock(spec=StrategyWidget)
        mock_risk = MagicMock(spec=RiskWidget)
        mock_system = MagicMock(spec=SystemHealthWidget)

        # Mock query_one to return specific mocks based on type or ID
        def query_side_effect(arg, *args, **kwargs):
            if arg == BotStatusWidget:
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

        # Create state
        state = TuiState()
        state.running = True
        state.position_data.equity = "5000.00"
        state.market_data.prices = {"ETH": "2000.00"}
        state.strategy_data.active_strategies = ["TestStrat"]
        state.risk_data.max_leverage = 5.0
        state.system_data.connection_status = "CONNECTED"

        # Call update_ui
        screen.update_ui(state)

        # Verify Status Widget updates
        assert mock_status.running is True
        assert mock_status.equity == "5000.00"

        # Verify Market Widget updates
        # Should be called twice (dashboard and full)
        assert mock_market.update_prices.call_count >= 1
        args, _ = mock_market.update_prices.call_args
        assert args[0] == {"ETH": "2000.00"}

        # Verify Strategy Widget updates
        mock_strategy.update_strategy.assert_called_once()
        args, _ = mock_strategy.update_strategy.call_args
        assert args[0].active_strategies == ["TestStrat"]

        # Verify Risk Widget updates
        mock_risk.update_risk.assert_called_once()
        args, _ = mock_risk.update_risk.call_args
        assert args[0].max_leverage == 5.0

        # Verify System Widget updates
        assert mock_system.update_system.call_count >= 1
        args, _ = mock_system.update_system.call_args
        assert args[0].connection_status == "CONNECTED"
