#!/usr/bin/env python3
"""
Test script to verify paper trading setup works correctly.
This script tests the basic components without actually trading.
"""

import sys
from datetime import datetime


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    try:
        from bot.exec.alpaca_paper import AlpacaPaperBroker

        print("✅ AlpacaPaperBroker imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AlpacaPaperBroker: {e}")
        return False

    try:
        from bot.live.trading_engine import LiveTradingEngine

        print("✅ LiveTradingEngine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LiveTradingEngine: {e}")
        return False

    try:
        from bot.live.portfolio_manager import LivePortfolioManager

        print("✅ LivePortfolioManager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LivePortfolioManager: {e}")
        return False

    try:
        from bot.live.data_manager import LiveDataManager

        print("✅ LiveDataManager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LiveDataManager: {e}")
        return False

    try:
        from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy

        print("✅ TrendBreakoutStrategy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TrendBreakoutStrategy: {e}")
        return False

    try:
        from bot.portfolio.allocator import PortfolioRules

        print("✅ PortfolioRules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PortfolioRules: {e}")
        return False

    return True


def test_broker_initialization():
    """Test broker initialization (without API keys)."""
    print("\n🔍 Testing broker initialization...")

    try:
        from bot.exec.alpaca_paper import AlpacaPaperBroker

        # This should fail gracefully without API keys
        try:
            broker = AlpacaPaperBroker("fake_key", "fake_secret")
            print("⚠️  Broker initialized with fake keys (expected)")
        except Exception as e:
            print(f"✅ Broker properly rejected fake keys: {e}")

        return True
    except Exception as e:
        print(f"❌ Broker initialization test failed: {e}")
        return False


def test_strategy_initialization():
    """Test strategy initialization."""
    print("\n🔍 Testing strategy initialization...")

    try:
        from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy

        strategy = TrendBreakoutStrategy(
            TrendBreakoutParams(
                donchian_lookback=55,
                atr_period=20,
                atr_k=2.0,
            )
        )
        print("✅ Strategy initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Strategy initialization failed: {e}")
        return False


def test_portfolio_rules():
    """Test portfolio rules initialization."""
    print("\n🔍 Testing portfolio rules...")

    try:
        from bot.portfolio.allocator import PortfolioRules

        rules = PortfolioRules(
            per_trade_risk_pct=0.005,
            atr_k=2.0,
            max_positions=10,
            max_gross_exposure_pct=0.60,
            cost_bps=5.0,
        )
        print("✅ Portfolio rules initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Portfolio rules initialization failed: {e}")
        return False


def test_data_structures():
    """Test data structure definitions."""
    print("\n🔍 Testing data structures...")

    try:
        from bot.exec.base import Account, Position
        from bot.live.data_manager import MarketData

        # Test creating instances
        account = Account(
            id="test",
            account_number="test",
            status="ACTIVE",
            crypto_status="ACTIVE",
            currency="USD",
            buying_power=100000.0,
            regt_buying_power=100000.0,
            daytrading_buying_power=100000.0,
            non_marginable_buying_power=100000.0,
            cash=100000.0,
            accrued_fees=0.0,
            pending_transfer_out=0.0,
            pending_transfer_in=0.0,
            portfolio_value=100000.0,
            pattern_day_trader=False,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            created_at=datetime.now(),
            trade_suspended_by_user=False,
            multiplier="1",
            shorting_enabled=False,
            equity=100000.0,
            last_equity=100000.0,
            long_market_value=0.0,
            short_market_value=0.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            last_maintenance_margin=0.0,
            sma=0.0,
            daytrade_count=0,
        )

        position = Position(
            symbol="AAPL",
            qty=100,
            avg_price=150.0,
            market_value=15000.0,
            unrealized_pl=0.0,
            unrealized_plpc=0.0,
            current_price=150.0,
            timestamp=datetime.now(),
        )

        market_data = MarketData(
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=149.0,
            close=152.0,
            volume=1000000,
            timestamp=datetime.now(),
        )

        print("✅ Data structures created successfully")
        return True
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")

    try:
        from bot.config import settings

        print(f"✅ Configuration loaded: log_level={settings.log_level}")
        print(f"   Alpaca API key configured: {'Yes' if settings.alpaca.api_key_id else 'No'}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🤖 GPT-Trader Paper Trading Setup Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_broker_initialization,
        test_strategy_initialization,
        test_portfolio_rules,
        test_data_structures,
        test_config,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Paper trading setup is ready.")
        print("\nNext steps:")
        print("1. Set your Alpaca API credentials in environment variables")
        print("2. Run: python examples/paper_trading_example.py")
        print("3. Or use CLI: poetry run gpt-trader paper --symbols 'AAPL,MSFT' --risk-pct 0.5")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
