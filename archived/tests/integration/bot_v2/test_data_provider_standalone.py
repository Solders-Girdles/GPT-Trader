#!/usr/bin/env python3
"""
ARCHIVED: Early standalone tests for the data provider abstraction.
Active coverage now lives in tests/unit/bot_v2/test_data_provider.py
and related feature tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from bot_v2.data_providers import (
    MockProvider as MockDataProvider,
    YFinanceProvider as YfinanceDataProvider,
    get_data_provider as create_data_provider,
)

pytestmark = pytest.mark.integration


def test_mock_data_provider():
    """Test MockDataProvider with strict assertions (no swallowing)."""
    provider = MockDataProvider()

    # Historical data shape for a couple symbols
    symbols = ["AAPL", "MSFT"]
    multi = provider.get_multiple_symbols(symbols, period="30d")
    assert set(multi.keys()) == set(symbols)
    # Single symbol path
    symbol = symbols[0]
    data = provider.get_historical_data(symbol, period="30d")
    assert len(data) > 0
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if hasattr(data, 'columns'):
        for col in required_columns:
            assert col in data.columns
    else:
        for col in required_columns:
            assert col in data.data

    # Current price positive
    price = provider.get_current_price(symbol)
    assert price > 0


def test_yfinance_data_provider_falls_back_cleanly():
    """YFinanceProvider returns data or falls back to mock without raising."""
    provider = YfinanceDataProvider()
    # Historical data (falls back to mock data if needed)
    df = provider.get_historical_data("AAPL", period="5d")
    assert len(df) > 0
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns
    # Price path returns a float
    assert provider.get_current_price("AAPL") > 0



def test_provider_factory():
    """Assert provider factory returns usable providers."""
    # Mock provider
    provider1 = create_data_provider('mock')
    data = provider1.get_historical_data('AAPL', period="5d")
    assert len(data) > 0

    # YFinance provider (falls back to mock data if needed)
    provider2 = create_data_provider('yfinance')
    data2 = provider2.get_historical_data('AAPL', period="5d")
    assert len(data2) > 0


def test_graceful_fallback():
    """Provider factory + invalid symbol path do not raise and return frames."""
    provider = create_data_provider('yfinance')
    assert provider is not None
    df = provider.get_historical_data("INVALID_SYMBOL_XYZ", period="5d")
    # Should return a non-empty frame via fallback
    assert len(df) > 0



def test_data_consistency():
    """Mock data respects OHLCV structure and is deterministic per symbol."""
    provider = MockDataProvider()
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        data = provider.get_historical_data(symbol, period="10d")
        assert len(data) > 0
        highs = data['High']
        lows = data['Low']
        opens = data['Open']
        closes = data['Close']
        volumes = data['Volume']
        assert (highs >= lows).all()
        assert (highs >= opens).all()
        assert (highs >= closes).all()
        assert (lows <= opens).all()
        assert (lows <= closes).all()
        assert (volumes > 0).all()

    # Determinism for same request
    data1 = provider.get_historical_data("AAPL", period="10d")
    data2 = provider.get_historical_data("AAPL", period="10d")
    assert len(data1) == len(data2)
    if hasattr(data1, 'equals'):
        assert data1.equals(data2)


def main():
    """Run all data provider tests."""
    print("🚀 Data Provider Test Suite")
    print("=" * 50)
    
    tests = [
        test_mock_data_provider,
        test_yfinance_data_provider,
        test_provider_factory,
        test_graceful_fallback,
        test_data_consistency
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All data provider tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
