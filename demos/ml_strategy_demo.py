#!/usr/bin/env python3
"""
ML Strategy Bridge Demo

This demo shows how to use the ML strategy bridge to dynamically select
strategies based on market conditions.
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


def main():
    """Run ML strategy bridge demo."""
    print("🤖 ML Strategy Bridge Demo")
    print("=" * 50)
    
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 30),  # 6 months for faster demo
        initial_capital=1_000_000.0,
        use_cache=True,
        show_progress=True,
        quiet_mode=False,
        save_trades=False,
        save_portfolio=False,
        save_metrics=False,
        generate_plot=False,
    )
    
    # Define strategy configurations
    strategy_configs = {
        'demo_ma': {
            'fast': 10,
            'slow': 20,
            'atr_period': 14
        },
        'trend_breakout': {
            # Uses default TrendBreakoutParams
        },
        'mean_reversion': {
            # Uses default MeanReversionParams  
        },
        'momentum': {
            # Uses default MomentumParams
        },
        'volatility': {
            # Uses default VolatilityParams
        }
    }
    
    # Test symbols - mix of different market conditions
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    print(f"📊 Testing with {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"📈 Strategies available: {', '.join(strategy_configs.keys())}")
    print(f"📅 Period: {config.start_date.date()} to {config.end_date.date()}")
    print()
    
    # Create orchestrator
    orchestrator = IntegratedOrchestrator(config)
    
    try:
        print("🚀 Starting ML-enhanced backtest...")
        print()
        
        # Run ML backtest
        results = orchestrator.run_ml_backtest(
            strategy_configs=strategy_configs,
            symbols=symbols,
            use_ml_selection=True  # Enable ML strategy selection
        )
        
        print()
        print("✅ Backtest completed!")
        print("=" * 50)
        
        # Display results
        print("📊 PERFORMANCE SUMMARY")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"CAGR: {results.cagr:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Volatility: {results.volatility:.2%}")
        print(f"Total Trades: {results.total_trades}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print()
        
        # ML-specific results
        if hasattr(results, 'ml_stats') and results.ml_stats:
            print("🤖 ML STRATEGY SELECTION")
            stats = results.ml_stats
            print(f"Final Strategy: {stats.get('current_strategy', 'unknown')}")
            print(f"Total Strategy Switches: {stats.get('total_switches', 0)}")
            print(f"ML Model Trained: {stats.get('model_trained', False)}")
            
            # Strategy performance breakdown
            if 'strategy_performance' in stats:
                print("\n📈 STRATEGY USAGE:")
                for strategy, perf in stats['strategy_performance'].items():
                    usage = perf.get('usage_count', 0)
                    positions = perf.get('total_positions', 0)
                    print(f"  {strategy}: {usage} days, {positions} total positions")
            
            print()
        
        # Strategy switches
        if hasattr(results, 'strategy_switches') and results.strategy_switches:
            print("🔄 STRATEGY SWITCHES:")
            for switch in results.strategy_switches[-5:]:  # Show last 5
                date = switch['date'].strftime('%Y-%m-%d')
                print(f"  {date}: {switch['from']} → {switch['to']}")
            
            if len(results.strategy_switches) > 5:
                print(f"  ... and {len(results.strategy_switches) - 5} more")
            print()
        
        # Execution info
        print("⏱️  EXECUTION INFO")
        print(f"Duration: {results.duration_days} trading days")
        print(f"Execution Time: {results.execution_time_seconds:.2f} seconds")
        print(f"Symbols Traded: {len(results.symbols_traded)}")
        
        # Warnings and errors
        if results.warnings:
            print(f"\n⚠️  Warnings: {len(results.warnings)}")
            for warning in results.warnings[:3]:  # Show first 3
                print(f"  • {warning}")
            
        if results.errors:
            print(f"\n❌ Errors: {len(results.errors)}")
            for error in results.errors[:3]:  # Show first 3
                print(f"  • {error}")
        
        print()
        print("=" * 50)
        print("🎯 Demo completed successfully!")
        
        # Also run a comparison with single strategy
        print("\n🔄 Running comparison with single strategy (demo_ma)...")
        
        from bot.strategy.demo_ma import DemoMAStrategy
        single_strategy = DemoMAStrategy(fast=10, slow=20, atr_period=14)
        
        single_results = orchestrator.run_backtest(
            strategy=single_strategy,
            symbols=symbols
        )
        
        print("\n📊 COMPARISON RESULTS")
        print(f"ML Strategy Return: {results.total_return:.2%}")
        print(f"Single Strategy Return: {single_results.total_return:.2%}")
        print(f"Improvement: {(results.total_return - single_results.total_return) * 100:.2f}%")
        print(f"ML Strategy Sharpe: {results.sharpe_ratio:.2f}")
        print(f"Single Strategy Sharpe: {single_results.sharpe_ratio:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        return None


if __name__ == "__main__":
    results = main()
    if results is None:
        sys.exit(1)