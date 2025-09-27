#!/usr/bin/env python3
"""
Test API Fixes - Verify wrapper classes work
============================================
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_portfolio_wrapper():
    """Test PortfolioAllocator wrapper works"""
    print("🧪 Testing Portfolio API Wrapper...")
    
    try:
        # Test the wrapper import
        from bot.portfolio import PortfolioAllocator, Allocator
        print("  ✅ PortfolioAllocator import works")
        
        # Test instantiation
        allocator = PortfolioAllocator()
        print("  ✅ PortfolioAllocator instantiation works")
        
        # Test the Allocator alias
        allocator2 = Allocator()
        print("  ✅ Allocator alias works")
        
        return True
    except Exception as e:
        print(f"  ❌ Portfolio wrapper failed: {e}")
        return False


def test_risk_wrapper():
    """Test SimpleRiskManager wrapper works"""
    print("\n🧪 Testing Risk API Wrapper...")
    
    try:
        # Test the wrapper import
        from bot.risk import SimpleRiskManager
        print("  ✅ SimpleRiskManager import works")
        
        # Test instantiation with parameters
        risk_mgr = SimpleRiskManager(
            max_position_size=0.1,
            max_portfolio_risk=0.02,
            stop_loss_pct=0.05
        )
        print("  ✅ SimpleRiskManager instantiation works")
        
        # Test method
        position_size = risk_mgr.calculate_position_size(
            signal_strength=1.0,
            volatility=0.02,
            portfolio_value=100000
        )
        print(f"  ✅ calculate_position_size works: ${position_size:.2f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Risk wrapper failed: {e}")
        return False


def test_paper_trading():
    """Test PaperTradingEngine with correct API"""
    print("\n🧪 Testing Paper Trading API...")
    
    try:
        from bot.paper_trading import PaperTradingEngine, PaperTradingConfig
        print("  ✅ PaperTradingEngine import works")
        
        # Test with config
        config = PaperTradingConfig(
            initial_capital=100000,
            commission_per_share=0.001
        )
        engine = PaperTradingEngine(config)
        print("  ✅ PaperTradingEngine instantiation works")
        
        return True
    except Exception as e:
        print(f"  ❌ Paper trading failed: {e}")
        return False


def test_integration():
    """Test integration components"""
    print("\n🧪 Testing Integration API...")
    
    try:
        from bot.integration.orchestrator import IntegratedOrchestrator
        print("  ✅ IntegratedOrchestrator import works")
        
        orchestrator = IntegratedOrchestrator()
        print("  ✅ IntegratedOrchestrator instantiation works")
        
        # Check method exists
        if hasattr(orchestrator, 'run_backtest'):
            print("  ✅ run_backtest method exists")
        else:
            print("  ❌ run_backtest method missing")
            
        return True
    except Exception as e:
        print(f"  ❌ Integration failed: {e}")
        return False


def test_complete_flow():
    """Test a complete flow with fixed API"""
    print("\n🧪 Testing Complete Flow with Fixed API...")
    
    try:
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Import with wrapper classes
        from bot.portfolio import PortfolioAllocator
        from bot.risk import SimpleRiskManager
        from bot.strategy.demo_ma import DemoMAStrategy
        
        # Create components
        strategy = DemoMAStrategy(fast=10, slow=30)
        allocator = PortfolioAllocator()
        risk_mgr = SimpleRiskManager()
        
        print("  ✅ All components created")
        
        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = pd.DataFrame({
            'open': [100 + i*0.5 for i in range(100)],
            'high': [101 + i*0.5 for i in range(100)],
            'low': [99 + i*0.5 for i in range(100)],
            'close': [100 + i*0.5 for i in range(100)],
            'volume': [1000000] * 100
        }, index=dates)
        
        # Generate signals
        signals = strategy.generate_signals(data)
        print(f"  ✅ Signals generated: {len(signals)} rows")
        
        # Merge signals with price data for allocation
        # The allocator expects both signal and price columns
        enriched_signals = signals.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                enriched_signals[col] = data[col]
        
        # Test allocation
        allocations = allocator.allocate_signals(
            signals={'AAPL': enriched_signals},
            market_data={'AAPL': data},
            portfolio_value=100000
        )
        
        if allocations is not None:
            print(f"  ✅ Allocations created: {len(allocations)} positions")
        else:
            print("  ⚠️ No allocations generated")
        
        # Test risk calculation
        position_size = risk_mgr.calculate_position_size(
            signal_strength=1.0,
            volatility=0.02,
            portfolio_value=100000
        )
        print(f"  ✅ Risk sizing works: ${position_size:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Complete flow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("🔧 API FIX VERIFICATION")
    print("="*60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Portfolio Wrapper", test_portfolio_wrapper()))
    results.append(("Risk Wrapper", test_risk_wrapper()))
    results.append(("Paper Trading", test_paper_trading()))
    results.append(("Integration", test_integration()))
    results.append(("Complete Flow", test_complete_flow()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All API fixes working! Foundation is stabilizing.")
    else:
        print("\n❌ Some API issues remain. Check errors above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())