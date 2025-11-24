import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.config.symbols = ["BTC-USD"]
    context.config.interval = 0.1
    context.broker.get_ticker.return_value = {"price": "50000.00"}
    return context

@pytest.mark.asyncio
async def test_trading_engine_initialization(mock_context):
    engine = TradingEngine(mock_context)
    assert engine.running is False
    assert engine.strategy is not None
    assert engine.price_history is not None

@pytest.mark.asyncio
async def test_trading_engine_cycle(mock_context):
    engine = TradingEngine(mock_context)
    
    # Mock the strategy to return a specific decision
    engine.strategy.decide = MagicMock(return_value=Decision(Action.BUY, "Test Buy"))
    
    await engine._cycle()
    
    # Verify broker call
    mock_context.broker.get_ticker.assert_called_with("BTC-USD")
    
    # Verify strategy call
    engine.strategy.decide.assert_called_once()
    call_args = engine.strategy.decide.call_args
    assert call_args.kwargs["symbol"] == "BTC-USD"
    assert call_args.kwargs["current_mark"] == Decimal("50000.00")
    
    # Verify price history update
    assert len(engine.price_history["BTC-USD"]) == 1
    assert engine.price_history["BTC-USD"][0] == Decimal("50000.00")

@pytest.mark.asyncio
async def test_trading_engine_cycle_history_limit(mock_context):
    engine = TradingEngine(mock_context)
    engine.strategy.decide = MagicMock(return_value=Decision(Action.HOLD, "Hold"))
    
    # Fill history with 25 items
    for i in range(25):
        mock_context.broker.get_ticker.return_value = {"price": f"{50000 + i}"}
        await engine._cycle()
        
    assert len(engine.price_history["BTC-USD"]) == 20
    assert engine.price_history["BTC-USD"][-1] == Decimal("50024")
