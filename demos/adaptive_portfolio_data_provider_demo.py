#!/usr/bin/env python3
"""
Demo script showcasing the clean data provider abstraction.

Shows how the adaptive portfolio system works with both real data
(when yfinance is available) and mock data (always available).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot_v2.features.adaptive_portfolio import run_adaptive_strategy
from bot_v2.features.adaptive_portfolio.data_providers import (
    MockDataProvider, 
    YfinanceDataProvider,
    create_data_provider,
    get_data_provider_info
)


def demo_data_provider_info():
    """Demo data provider availability and info."""
    print("🔍 Data Provider Information")
    print("=" * 40)
    
    info = get_data_provider_info()
    print(f"📋 Available Providers:")
    print(f"   ✅ Mock Provider: {info['mock_available']} (synthetic data)")
    print(f"   {'✅' if info['yfinance_available'] else '❌'} YFinance Provider: {info['yfinance_available']} (real market data)")
    print(f"   {'✅' if info['pandas_available'] else '❌'} Pandas Support: {info['pandas_available']} (DataFrame features)")
    print()


def demo_mock_provider():
    """Demo MockDataProvider functionality."""
    print("🎭 Mock Data Provider Demo")
    print("=" * 40)
    
    provider = MockDataProvider()
    
    # Show available symbols
    symbols = provider.get_available_symbols()
    print(f"📊 Available symbols: {len(symbols)}")
    print(f"   Sample: {', '.join(symbols[:8])}")
    
    # Generate sample data
    symbol = "AAPL"
    data = provider.get_historical_data(symbol, period="30d")
    print(f"\n📈 Generated {len(data)} days of data for {symbol}")
    
    # Show current price
    current_price = provider.get_current_price(symbol)
    print(f"💰 Current {symbol} price: ${current_price:.2f}")
    
    # Show market status
    market_open = provider.is_market_open()
    print(f"🕐 Market open: {market_open}")
    print()


def demo_yfinance_provider():
    """Demo YfinanceDataProvider if available."""
    print("📈 Real Data Provider Demo")
    print("=" * 40)
    
    info = get_data_provider_info()
    if not info['yfinance_available']:
        print("⚠️  YfinanceDataProvider not available")
        print("   Install with: pip install yfinance")
        print()
        return
    
    try:
        provider = YfinanceDataProvider()
        
        # Show available symbols
        symbols = provider.get_available_symbols()
        print(f"📊 Available symbols: {len(symbols)}")
        print(f"   Sample: {', '.join(symbols[:8])}")
        
        # Get real data
        symbol = "AAPL"
        data = provider.get_historical_data(symbol, period="5d")
        print(f"\n📈 Retrieved {len(data)} days of real data for {symbol}")
        
        # Show current price
        current_price = provider.get_current_price(symbol)
        print(f"💰 Current {symbol} price: ${current_price:.2f}")
        
        # Show market status
        market_open = provider.is_market_open()
        print(f"🕐 Market open: {market_open}")
        
    except Exception as e:
        print(f"⚠️  Real data provider failed: {e}")
        print("   This might be due to network issues or rate limiting")
    
    print()


def demo_adaptive_portfolio_with_providers():
    """Demo adaptive portfolio with different data providers."""
    print("🎯 Adaptive Portfolio with Data Providers")
    print("=" * 40)
    
    test_portfolios = [
        {"capital": 2500, "name": "Small Portfolio", "symbols": ["AAPL", "MSFT"]},
        {"capital": 25000, "name": "Medium Portfolio", "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"]},
    ]
    
    for portfolio in test_portfolios:
        print(f"\n💼 {portfolio['name']} (${portfolio['capital']:,})")
        print("-" * 30)
        
        # Try with mock provider (always works)
        print("🎭 Using Mock Data Provider:")
        mock_provider = MockDataProvider()
        
        result = run_adaptive_strategy(
            current_capital=portfolio["capital"],
            symbols=portfolio["symbols"],
            data_provider=mock_provider,
            prefer_real_data=False
        )
        
        print(f"   Tier: {result.current_tier.value}")
        print(f"   Strategies: {', '.join(result.tier_config.strategies)}")
        print(f"   Signals: {len(result.signals)}")
        
        if result.signals:
            for i, signal in enumerate(result.signals[:2]):  # Show first 2
                print(f"   Signal {i+1}: {signal.action} {signal.symbol} "
                      f"${signal.target_position_size:,.0f} (confidence: {signal.confidence:.2f})")
        
        # Try with automatic provider selection
        print("\n🔄 Using Automatic Provider Selection:")
        result2 = run_adaptive_strategy(
            current_capital=portfolio["capital"],
            symbols=portfolio["symbols"],
            prefer_real_data=True  # Will fallback to mock if needed
        )
        
        print(f"   Signals: {len(result2.signals)}")
        if result2.signals:
            print(f"   Top signal: {result2.signals[0].action} {result2.signals[0].symbol} "
                  f"${result2.signals[0].target_position_size:,.0f}")


def demo_graceful_fallback():
    """Demo graceful fallback behavior."""
    print("🔄 Graceful Fallback Demo")
    print("=" * 40)
    
    # Test provider factory with preferences
    print("🏭 Provider Factory Tests:")
    
    # Force mock provider
    provider1, type1 = create_data_provider(prefer_real_data=False)
    print(f"   prefer_real_data=False → {type1} provider")
    
    # Try real data, fallback to mock if needed
    provider2, type2 = create_data_provider(prefer_real_data=True)
    print(f"   prefer_real_data=True  → {type2} provider")
    
    # Show that both work
    symbols1 = provider1.get_available_symbols()
    symbols2 = provider2.get_available_symbols()
    print(f"   Mock provider symbols: {len(symbols1)}")
    print(f"   Auto provider symbols: {len(symbols2)}")
    
    print("\n✅ System always has a working data provider!")
    print()


def main():
    """Run all demos."""
    print("🚀 Data Provider Abstraction Demo")
    print("=" * 50)
    
    demos = [
        demo_data_provider_info,
        demo_mock_provider,
        demo_yfinance_provider,
        demo_graceful_fallback,
        demo_adaptive_portfolio_with_providers,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"❌ Demo {demo.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("🎉 Demo complete!")
    print("\n💡 Key Benefits:")
    print("   • Clean abstraction eliminates ugly try/except blocks")
    print("   • MockDataProvider works without any external dependencies")
    print("   • YfinanceDataProvider provides real market data when available")
    print("   • Graceful fallback ensures system always works")
    print("   • Easy to test and develop without network dependencies")


if __name__ == "__main__":
    main()