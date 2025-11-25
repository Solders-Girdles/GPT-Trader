import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.orchestration.execution.state_collection import StateCollector
from gpt_trader.features.brokerages.core.interfaces import Balance

@pytest.fixture
def mock_broker():
    return MagicMock()

@pytest.fixture
def collector(mock_broker):
    with patch("gpt_trader.orchestration.execution.state_collection.load_runtime_settings") as mock_load:
        mock_load.return_value.raw_env = {"PERPS_COLLATERAL_ASSETS": "USD,USDC"}
        return StateCollector(mock_broker)

def test_calculate_equity_from_balances(collector):
    balances = [
        Balance(asset="USD", total=Decimal("100"), available=Decimal("50")),
        Balance(asset="USDC", total=Decimal("200"), available=Decimal("100")),
        Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")), # Not collateral
    ]
    
    available, collateral, total = collector.calculate_equity_from_balances(balances)
    
    assert available == Decimal("150") # 50 USD + 100 USDC
    assert total == Decimal("300") # 100 USD + 200 USDC
    assert len(collateral) == 2
    assert collateral[0].asset == "USD"
    assert collateral[1].asset == "USDC"

def test_log_collateral_update(collector):
    # Mock production logger
    collector._production_logger = MagicMock()
    
    collateral = [Balance(asset="USD", total=Decimal("100"), available=Decimal("50"))]
    equity = Decimal("100")
    total = Decimal("100")
    all_balances = collateral
    
    # First update (sets last_collateral_available)
    collector.log_collateral_update(collateral, equity, total, all_balances)
    assert collector._last_collateral_available == Decimal("50")
    
    # Second update (no change)
    collector.log_collateral_update(collateral, equity, total, all_balances)
    # Shouldn't log change
    
    # Third update (change)
    new_collateral = [Balance(asset="USD", total=Decimal("110"), available=Decimal("60"))]
    collector.log_collateral_update(new_collateral, equity, Decimal("110"), all_balances)
    
    assert collector._last_collateral_available == Decimal("60")
    # Verify telemetry log
    collector._production_logger.log_balance_update.assert_called()
