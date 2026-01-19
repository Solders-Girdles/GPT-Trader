from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.tui.services.alert_manager import AlertManager
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import PortfolioSummary, RiskState, SystemStatus


def create_mock_app() -> MagicMock:
    app = MagicMock()
    app.notify = MagicMock()
    return app


def create_alert_manager(app: MagicMock) -> AlertManager:
    return AlertManager(app)


def create_sample_state() -> TuiState:
    state = TuiState()
    state.system_data = SystemStatus(
        connection_status="CONNECTED",
        rate_limit_usage="50%",
    )
    state.risk_data = RiskState(
        reduce_only_mode=False,
        daily_loss_limit_pct=0.05,
        current_daily_loss_pct=0.0,
    )
    state.position_data = PortfolioSummary(
        positions={},
        total_unrealized_pnl=Decimal("0"),
    )
    state.running = True
    return state
