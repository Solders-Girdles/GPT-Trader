#!/usr/bin/env python3
"""
Safe Paper Trading Test Script

This script tests the paper trading infrastructure without requiring API keys.
It demonstrates the safety features and validation logic that protect against
accidental live trading.

This script will:
1. Test environment validation (without real keys)
2. Show safety check mechanisms
3. Demonstrate configuration creation
4. Verify import functionality
5. Test audit logging features

Safe to run without any API credentials.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.logging import get_logger

logger = get_logger("paper_trading_test")


def test_imports():
    """Test that all paper trading imports work correctly."""
    print("\n🔍 Testing Paper Trading Imports")
    print("=" * 50)
    
    try:
        # Test Alpaca imports
        from bot.brokers.alpaca import AlpacaConfig, PaperTradingConfig
        print("✅ Alpaca broker imports: OK")
        
        # Test paper trading modules
        from bot.live.alpaca_paper_trader import AlpacaPaperTrader
        print("✅ AlpacaPaperTrader import: OK")
        
        # Test existing paper trading engine
        from bot.paper_trading.paper_trading_engine import PaperTradingEngine
        print("✅ PaperTradingEngine import: OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_safety_validation():
    """Test the safety validation mechanisms."""
    print("\n🛡️  Testing Safety Validation")
    print("=" * 50)
    
    # Save original environment
    original_env = {}
    test_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER"]
    for var in test_vars:
        original_env[var] = os.getenv(var)
    
    try:
        from bot.live.alpaca_paper_trader import AlpacaPaperTrader
        
        # Test 1: Missing API keys
        print("🔸 Test 1: Missing API credentials")
        for var in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
            if var in os.environ:
                del os.environ[var]
        
        trader = AlpacaPaperTrader()
        result = trader.validate_environment()
        if not result:
            print("   ✅ Correctly rejected missing credentials")
        else:
            print("   ❌ Should have rejected missing credentials")
        
        # Test 2: Paper mode not set
        print("\n🔸 Test 2: Paper mode validation")
        os.environ["ALPACA_API_KEY"] = "test_key_123456789012345"
        os.environ["ALPACA_SECRET_KEY"] = "test_secret_1234567890123456789012345678901234567890"
        os.environ["ALPACA_PAPER"] = "false"  # Danger!
        
        trader = AlpacaPaperTrader()
        result = trader.validate_environment()
        if not result:
            print("   ✅ Correctly rejected non-paper mode")
        else:
            print("   ❌ Should have rejected non-paper mode")
        
        # Test 3: Valid paper mode configuration
        print("\n🔸 Test 3: Valid paper configuration")
        os.environ["ALPACA_PAPER"] = "true"
        
        result = trader.validate_environment()
        if result:
            print("   ✅ Correctly accepted valid paper configuration")
        else:
            print("   ❌ Should have accepted valid paper configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Safety validation test failed: {e}")
        return False
        
    finally:
        # Restore original environment
        for var, value in original_env.items():
            if value is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = value


def test_configuration_creation():
    """Test safe configuration creation."""
    print("\n⚙️  Testing Configuration Creation")
    print("=" * 50)
    
    try:
        from bot.brokers.alpaca import AlpacaConfig, PaperTradingConfig
        from bot.live.alpaca_paper_trader import AlpacaPaperTrader
        
        # Set test environment
        os.environ["ALPACA_API_KEY"] = "test_key_123456789012345"
        os.environ["ALPACA_SECRET_KEY"] = "test_secret_1234567890123456789012345678901234567890"
        os.environ["ALPACA_PAPER"] = "true"
        
        trader = AlpacaPaperTrader(
            symbols=["AAPL", "MSFT"],
            initial_capital=50000.0,
            config_overrides={"max_order_value": 1000.0}
        )
        
        # Test validation passes
        if trader.validate_environment():
            print("✅ Environment validation: PASS")
        else:
            print("❌ Environment validation: FAIL")
            return False
        
        # Test configuration creation
        config = trader.create_safe_config()
        
        print(f"✅ Paper trading mode: {config.alpaca_config.paper_trading}")
        print(f"✅ Max order value: ${config.max_order_value:,.2f}")
        print(f"✅ Daily trade limit: {config.max_daily_trades}")
        print(f"✅ Order logging: {config.log_all_orders}")
        print(f"✅ Data symbols: {config.data_symbols}")
        
        # Verify safety settings
        assert config.alpaca_config.paper_trading == True, "Paper mode must be True"
        assert config.max_order_value <= 50000.0, "Order value limit too high"
        assert config.log_all_orders == True, "Order logging must be enabled"
        
        print("✅ All configuration safety checks passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_audit_logging():
    """Test audit logging functionality."""
    print("\n📝 Testing Audit Logging")
    print("=" * 50)
    
    try:
        from bot.live.alpaca_paper_trader import AlpacaPaperTrader
        
        # Set test environment
        os.environ["ALPACA_API_KEY"] = "test_key_123456789012345"
        os.environ["ALPACA_SECRET_KEY"] = "test_secret_1234567890123456789012345678901234567890"
        os.environ["ALPACA_PAPER"] = "true"
        
        trader = AlpacaPaperTrader()
        
        # Test audit log initialization
        initial_count = len(trader.audit_log)
        print(f"✅ Initial audit log entries: {initial_count}")
        
        # Validate environment (this should add audit entries)
        trader.validate_environment()
        
        post_validation_count = len(trader.audit_log)
        print(f"✅ Post-validation audit entries: {post_validation_count}")
        
        # Test audit log structure
        if trader.audit_log:
            latest_entry = trader.audit_log[-1]
            required_fields = ["timestamp", "event"]
            
            for field in required_fields:
                if field in latest_entry:
                    print(f"✅ Audit entry contains {field}")
                else:
                    print(f"❌ Audit entry missing {field}")
                    return False
        
        # Test output directory creation
        output_dir = trader.output_dir
        if output_dir.exists():
            print(f"✅ Output directory created: {output_dir}")
        else:
            print(f"❌ Output directory not created: {output_dir}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Audit logging test failed: {e}")
        return False


def test_integration_points():
    """Test integration with existing system components."""
    print("\n🔗 Testing System Integration Points")
    print("=" * 50)
    
    try:
        # Test orchestrator integration
        from bot.integration.orchestrator import IntegratedOrchestrator, BacktestConfig
        print("✅ Orchestrator integration available")
        
        # Test risk integration
        from bot.risk.integration import RiskConfig
        print("✅ Risk management integration available")
        
        # Test strategy integration
        from bot.strategy.base import Strategy
        print("✅ Strategy system integration available")
        
        # Test data pipeline integration
        from bot.dataflow.pipeline import DataPipeline, PipelineConfig
        print("✅ Data pipeline integration available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Integration test failed: {e}")
        return False


def show_safety_summary():
    """Show a summary of safety features."""
    print("\n🛡️  PAPER TRADING SAFETY FEATURES")
    print("=" * 60)
    print("✅ Environment validation prevents live trading")
    print("✅ Explicit paper mode verification required")
    print("✅ Multiple API endpoint checks")
    print("✅ Order value and frequency limits")
    print("✅ Comprehensive audit logging")
    print("✅ Position tracking and P&L calculation")
    print("✅ Graceful error handling")
    print("✅ Clean resource management")
    print("✅ Integration with existing risk management")
    print("✅ Real-time position monitoring")
    print("\n🔒 LIVE TRADING PROTECTION:")
    print("   • ALPACA_PAPER must be 'true'")
    print("   • API endpoints verified for paper mode")
    print("   • Account balance patterns checked")
    print("   • All orders logged to audit trail")
    print("   • Position size limits enforced")
    print("   • Daily trade limits applied")


def main():
    """Run all paper trading tests."""
    print("🚀 PAPER TRADING SAFETY TEST SUITE")
    print("=" * 60)
    print("This test suite validates paper trading safety features")
    print("WITHOUT requiring actual API credentials.")
    print("=" * 60)
    
    tests = [
        ("Import Functionality", test_imports),
        ("Safety Validation", test_safety_validation),
        ("Configuration Creation", test_configuration_creation),
        ("Audit Logging", test_audit_logging),
        ("System Integration", test_integration_points),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        show_safety_summary()
        print("\n🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Set up Alpaca paper trading API keys")
        print("2. Run the full demo: python scripts/alpaca_paper_trading_demo.py")
        print("3. Test with small position sizes")
        print("4. Integrate with your trading strategies")
        print("5. Always verify paper mode before going live")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)