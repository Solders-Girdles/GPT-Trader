import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.widgets.logs import LogWidget


@pytest.mark.asyncio
async def test_data_flow_from_reporter_to_state(mock_app):
    """Verify that status reporter updates are correctly applied to TuiState."""
    # Simulate status update using SimpleNamespace for object access
    # AccountBalance must be an object
    balance = SimpleNamespace(asset="USD", total="1000.00", available="500.00", hold="0.00")
    
    status_update = SimpleNamespace(
        system=SimpleNamespace(
            api_latency=123.45,
            connection_status="CONNECTED",
            memory_usage="512MB",
            cpu_usage="15%",
            rate_limit_usage=0,
        ),
        market=SimpleNamespace(last_prices={"BTC-USD": "50000.00"}, last_price_update=1000000.0, price_history={}),
        account=SimpleNamespace(
            balances=[balance],
            volume_30d="10000.00",
            fees_30d="10.00",
            fee_tier="Taker"
        ),
        positions=SimpleNamespace(
            positions={}, 
            total_unrealized_pnl="0.00",
            equity="1000.00"
        ),
        orders=[],
        trades=[],
        strategy=SimpleNamespace(last_decisions=[], active_strategies=[]),
        risk=SimpleNamespace(
            max_leverage=5.0,
            daily_loss_limit_pct=0.02,
            current_daily_loss_pct=0.01,
            reduce_only_mode=False,
            reduce_only_reason="",
            active_guards=[]
        ),
        observer_interval=1.0
    )

    # Use the correct method on TuiState
    mock_app.tui_state.update_from_bot_status(status_update)

    # Verify State
    from decimal import Decimal
    assert mock_app.tui_state.system_data.api_latency == 123.45
    assert mock_app.tui_state.system_data.connection_status == "CONNECTED"
    assert mock_app.tui_state.market_data.prices["BTC-USD"] == Decimal("50000.00")
    assert mock_app.tui_state.account_data.balances[0].asset == "USD"


@pytest.mark.skip(reason="LogWidget internal filtering logic has moved to TuiLogHandler")
@pytest.mark.asyncio
async def test_log_filtering(mock_app):
    """Verify LogWidget filtering logic."""
    pass
