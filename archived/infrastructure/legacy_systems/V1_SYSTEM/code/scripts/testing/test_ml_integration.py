#!/usr/bin/env python3
"""
Test ML Integration

Simple test script to validate ML-Strategy bridge integration.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.integration.orchestrator import IntegratedOrchestrator, BacktestConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_ml_integration():
    """Test ML strategy integration with a minimal example."""
    print("🧪 Testing ML Strategy Integration")
    print("=" * 40)
    
    # Minimal config for testing
    config = BacktestConfig(
        start_date=datetime(2023, 6, 1),
        end_date=datetime(2023, 6, 30),  # Just 1 month
        initial_capital=100_000.0,
        use_cache=True,
        show_progress=False,
        quiet_mode=True,  # Suppress detailed logging
        save_trades=False,
        save_portfolio=False,
        save_metrics=False,
        generate_plot=False,
    )
    
    # Simple strategy configurations
    strategy_configs = {
        'demo_ma': {
            'fast': 10,
            'slow': 20,
            'atr_period': 14
        },
        'trend_breakout': {
            # Uses default params
        }
    }
    
    # Test with a few symbols
    symbols = ['AAPL', 'MSFT']
    
    print(f"📊 Testing with {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"📈 Strategies: {', '.join(strategy_configs.keys())}")
    print(f"📅 Period: {config.start_date.date()} to {config.end_date.date()}")
    print()
    
    # Create orchestrator
    orchestrator = IntegratedOrchestrator(config)
    
    try:
        print("🚀 Running ML backtest...")
        
        # Run ML backtest
        results = orchestrator.run_ml_backtest(
            strategy_configs=strategy_configs,
            symbols=symbols,
            use_ml_selection=True
        )
        
        print("✅ ML backtest completed!")
        print()
        
        # Check for errors
        if results.errors:
            print("❌ Errors occurred:")
            for error in results.errors:
                print(f"  • {error}")
            return False
        
        # Display basic results
        print("📊 RESULTS SUMMARY")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Total Trades: {results.total_trades}")
        print(f"Duration: {results.duration_days} days")
        print(f"Execution Time: {results.execution_time_seconds:.2f}s")
        
        # ML-specific results
        if hasattr(results, 'ml_stats'):
            stats = results.ml_stats
            print(f"Final Strategy: {stats.get('current_strategy', 'unknown')}")
            print(f"Strategy Switches: {stats.get('total_switches', 0)}")
        
        if results.warnings:
            print(f"\n⚠️  Warnings: {len(results.warnings)}")
        
        print("\n🎯 ML Integration test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ ML Integration test FAILED: {e}")
        return False


def test_feature_engineering():
    """Test feature engineering separately."""
    print("\n🔬 Testing Feature Engineering")
    print("=" * 40)
    
    try:
        from bot.ml.models.simple_strategy_selector import SimpleStrategySelector
        import pandas as pd
        import numpy as np
        
        # Create realistic sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # AAPL-like data
        price_base = 150
        returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
        prices = price_base * (1 + returns).cumprod()
        
        sample_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, 100)),
            'High': prices * (1 + np.random.uniform(0.002, 0.01, 100)),
            'Low': prices * (1 - np.random.uniform(0.002, 0.01, 100)),
            'Close': prices,
            'Volume': np.random.lognormal(15, 0.5, 100).astype(int)
        }, index=dates)
        
        market_data = {'AAPL': sample_data, 'MSFT': sample_data * 0.8}
        
        # Test feature engineering
        features = SimpleStrategySelector.engineer_strategy_features(market_data)
        print(f"✅ Features engineered: {features.shape}")
        print(f"Feature columns: {list(features.columns)}")
        
        # Test strategy selection
        selector = SimpleStrategySelector()
        strategy, confidence, probs = selector.select_strategy_with_confidence(features)
        
        print(f"✅ Strategy selected: {strategy} (confidence: {confidence:.2f})")
        print(f"All probabilities: {probs}")
        
        print("🎯 Feature engineering test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}", exc_info=True)
        print(f"❌ Feature engineering test FAILED: {e}")
        return False


if __name__ == "__main__":
    print("🤖 ML-Strategy Bridge Integration Tests")
    print("=" * 50)
    
    # Test feature engineering first
    fe_success = test_feature_engineering()
    
    # Test full integration
    integration_success = test_ml_integration()
    
    print("\n" + "=" * 50)
    if fe_success and integration_success:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED!")
        sys.exit(1)