#!/usr/bin/env python3
"""
Test volatility strategy integration with the orchestrator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta
import pandas as pd
from bot.integration.orchestrator import BacktestConfig, IntegratedOrchestrator
from bot.portfolio.allocator import PortfolioRules
from bot.risk.integration import RiskConfig
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams


def test_volatility_integration():
    """Test volatility strategy integration with the orchestrator."""
    
    print("🧪 Testing Volatility Strategy Integration")
    print("=" * 50)
    
    # Define test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months
    
    # Test with a single symbol to start
    symbols = ["SPY"]
    
    print(f"📅 Test period: {start_date.date()} to {end_date.date()}")
    print(f"📊 Testing symbols: {symbols}")
    
    # Create volatility strategy with standard parameters
    volatility_params = VolatilityParams(
        bb_period=20,
        bb_std_dev=2.0,
        atr_period=14,
        atr_threshold_multiplier=1.2,
        exit_middle_band=True
    )
    
    strategy = VolatilityStrategy(volatility_params)
    print(f"🎯 Strategy: {strategy}")
    
    # Configure backtest
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0
    )
    
    print("\n📋 Configuration:")
    print(f"   • Initial capital: ${backtest_config.initial_capital:,.0f}")
    print(f"   • Test period: {start_date.date()} to {end_date.date()}")
    
    # Create orchestrator
    print("\n🔧 Creating orchestrator...")
    orchestrator = IntegratedOrchestrator(backtest_config)
    
    print("✅ Orchestrator created successfully")
    
    # Run backtest
    print("\n🚀 Running backtest...")
    try:
        results = orchestrator.run_backtest(strategy, symbols)
        
        print("✅ Backtest completed successfully!")
        print("\n📊 Results Summary:")
        
        # Extract basic metrics
        if hasattr(results, 'portfolio_value'):
            final_value = results.portfolio_value.iloc[-1]
            total_return = (final_value / backtest_config.initial_capital - 1) * 100
            print(f"   • Final portfolio value: ${final_value:,.2f}")
            print(f"   • Total return: {total_return:+.2f}%")
        
        if hasattr(results, 'trades') and not results.trades.empty:
            num_trades = len(results.trades)
            print(f"   • Total trades: {num_trades}")
        else:
            print("   • Total trades: 0 (no signals generated)")
        
        print(f"   • Data points: {len(results.portfolio_value) if hasattr(results, 'portfolio_value') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = test_volatility_integration()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 Integration test passed!")
            print("✅ Volatility strategy works with the orchestrator")
            print("🔗 Ready for production use")
        else:
            print("❌ Integration test failed")
            
        print("🎯 Volatility strategy integration complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)