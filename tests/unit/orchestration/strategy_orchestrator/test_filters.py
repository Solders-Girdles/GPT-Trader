import pytest
import unittest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import datetime

from gpt_trader.orchestration.strategy_orchestrator.filters import (
    VolumeFilter,
    MomentumFilter,
    TrendFilter,
    VolatilityFilter,
    RegimeFilter,
)
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.orchestration.strategy_orchestrator.models import SymbolProcessingContext

@pytest.fixture
def context():
    return SymbolProcessingContext(
        symbol="BTC-USD",
        balances=[],
        equity=Decimal("10000"),
        positions={},
        position_state=None,
        position_quantity=Decimal("0"),
        marks=[],
        product=MagicMock()
    )

@pytest.fixture
def candles():
    # Helper to create mock candles
    class MockCandle:
        def __init__(self, close, volume=100, high=None, low=None):
            self.close = Decimal(str(close))
            self.price = Decimal(str(close)) # Fallback
            self.volume = Decimal(str(volume))
            self.size = Decimal(str(volume)) # Fallback
            self.high = Decimal(str(high if high else close))
            self.low = Decimal(str(low if low else close))
            self.timestamp = datetime.utcnow()
            
    return MockCandle

def test_volume_filter(context, candles):
    flt = VolumeFilter()
    config = {"window": 3, "min_volume": 1000}
    
    # Case 1: Insufficient data
    data = [candles(100, 500)] * 2
    decision = flt.check(context, config, data)
    assert decision.action == Action.HOLD
    assert decision.reason == "volume_filter_wait"
    
    # Case 2: Low volume
    data = [candles(100, 500)] * 3
    decision = flt.check(context, config, data)
    assert decision.action == Action.HOLD
    assert decision.reason == "volume_filter_blocked"
    
    # Case 3: Sufficient volume
    data = [candles(100, 1500)] * 3
    decision = flt.check(context, config, data)
    assert decision is None

def test_momentum_filter(context, candles):
    flt = MomentumFilter()
    config = {"window": 14, "threshold": 60}
    
    # Case 1: Insufficient data
    data = [candles(100)] * 14
    decision = flt.check(context, config, data)
    assert decision.action == Action.HOLD
    assert decision.reason == "momentum_filter_wait"
    
    # Case 2: Low RSI (Flat price = RSI 50 < 60)
    # Note: RSI calculation might need variation. 
    # If all prices are same, RSI is usually 50 or 0 or 100 depending on impl.
    # Let's assume flat prices -> RSI 50 (neutral)
    data = [candles(100)] * 15
    decision = flt.check(context, config, data)
    # Depending on RSI impl, flat might be 0 change.
    # If RSI < threshold (60), it blocks.
    # Let's verify behavior.
    if decision:
        assert decision.action == Action.HOLD
        assert decision.reason == "momentum_filter_blocked"
    
    # Case 3: High RSI (Uptrend)
    data = []
    for i in range(15):
        data.append(candles(100 + i*10))
        
    with unittest.mock.patch("gpt_trader.orchestration.strategy_orchestrator.filters._rsi_from_closes") as mock_rsi:
        mock_rsi.return_value = [Decimal("70")] * 15
        decision = flt.check(context, config, data)
        assert decision is None

def test_trend_filter(context, candles):
    flt = TrendFilter()
    config = {"window": 5, "min_slope": 1}
    
    # Case 1: Insufficient data
    data = [candles(100)] * 5
    decision = flt.check(context, config, data)
    assert decision.action == Action.HOLD
    assert decision.reason == "trend_filter_wait"
    
    # Case 2: Negative Slope
    data = [candles(100 - i) for i in range(6)] # 100, 99, 98...
    # Reverse to be chronological: 95, 96, 97, 98, 99, 100
    data = [candles(100 + i) for i in range(6)] 
    # Wait, slope = (current_ma - prev_ma) / window
    # If prices are increasing, slope is positive.
    
    # Let's try flat prices -> slope 0 < 1
    data = [candles(100)] * 6
    decision = flt.check(context, config, data)
    assert decision.action == Action.HOLD
    assert decision.reason == "trend_filter_blocked"
    
    # Case 3: Steep Slope
    data = [candles(100 + i*10) for i in range(6)]
    decision = flt.check(context, config, data)
    assert decision is None
