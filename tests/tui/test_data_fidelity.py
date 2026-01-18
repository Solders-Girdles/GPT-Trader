from decimal import Decimal

import pytest

from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BalanceEntry,
    BotStatus,
    MarketStatus,
    RiskStatus,
    SystemStatus,
)


def make_status(**overrides):
    """Create a complete BotStatus with optional field overrides."""
    status = BotStatus()
    for key, value in overrides.items():
        setattr(status, key, value)
    return status


@pytest.mark.asyncio
async def test_data_flow_from_reporter_to_state(mock_app):
    """Verify that status reporter updates are correctly applied to TuiState."""
    balance = BalanceEntry(
        asset="USD",
        total=Decimal("1000.00"),
        available=Decimal("500.00"),
        hold=Decimal("0.00"),
    )

    status_update = make_status(
        system=SystemStatus(
            api_latency=123.45,
            connection_status="CONNECTED",
            memory_usage="512MB",
            cpu_usage="15%",
            rate_limit_usage="0%",
        ),
        market=MarketStatus(
            last_prices={"BTC-USD": Decimal("50000.00")},
            last_price_update=1000000.0,
            price_history={},
        ),
        account=AccountStatus(
            balances=[balance],
            volume_30d=Decimal("10000.00"),
            fees_30d=Decimal("10.00"),
            fee_tier="Taker",
        ),
        risk=RiskStatus(
            max_leverage=5.0,
            daily_loss_limit_pct=0.02,
            current_daily_loss_pct=0.01,
            reduce_only_mode=False,
            reduce_only_reason="",
            guards=[],
        ),
    )

    # Use the correct method on TuiState
    mock_app.tui_state.update_from_bot_status(status_update)

    # Verify State
    assert mock_app.tui_state.system_data.api_latency == 123.45
    assert mock_app.tui_state.system_data.connection_status == "CONNECTED"
    assert mock_app.tui_state.market_data.prices["BTC-USD"] == Decimal("50000.00")
    assert mock_app.tui_state.account_data.balances[0].asset == "USD"
