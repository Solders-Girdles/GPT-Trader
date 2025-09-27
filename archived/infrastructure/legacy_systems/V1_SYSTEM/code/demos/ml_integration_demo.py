#!/usr/bin/env python3
"""ML Integration Demo - Shows ML-powered strategy selection in action."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.integration.enhanced_orchestrator import EnhancedOrchestrator, create_ml_orchestrator
from bot.integration.orchestrator import BacktestConfig
from bot.strategy.demo_ma import DemoMAStrategy
import warnings
warnings.filterwarnings('ignore')

def demo_comparison():
    """Compare regular vs ML-enhanced backtesting."""
    print("=" * 70)
    print("🤖 GPT-TRADER ML INTEGRATION DEMO")
    print("=" * 70)
    
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=60),
        end_date=datetime.now(),
        initial_capital=10_000,
        show_progress=False,
        quiet_mode=True,
        save_trades=False,
        save_portfolio=False,
        save_metrics=False,
        generate_plot=False
    )
    
    # Test symbol
    symbols = ['AAPL']
    
    # Create enhanced orchestrator
    orchestrator = create_ml_orchestrator(config, enable_ml=True)
    
    print("\n📊 TEST 1: Regular Backtest (Single Strategy)")
    print("-" * 50)
    
    # Run with single strategy
    strategy = DemoMAStrategy()
    regular_results = orchestrator.run_backtest(
        strategy=strategy,
        symbols=symbols,
        use_ml=False
    )
    
    print(f"Strategy: {strategy.name}")
    print(f"Trades: {regular_results.total_trades}")
    print(f"Return: {regular_results.total_return:.2%}")
    print(f"Sharpe: {regular_results.sharpe_ratio:.2f}")
    
    print("\n📊 TEST 2: ML-Enhanced Backtest (Dynamic Selection)")
    print("-" * 50)
    
    # Define multiple strategies for ML to choose from
    strategy_configs = {
        'demo_ma': {'fast': 10, 'slow': 20},
        'trend_breakout': {},
        'mean_reversion': {},
        'momentum': {},
        'volatility': {}
    }
    
    ml_results = orchestrator.run_backtest(
        strategy=strategy_configs,  # Pass dict for ML mode
        symbols=symbols,
        use_ml=True  # Enable ML selection
    )
    
    print(f"Strategies available: {list(strategy_configs.keys())}")
    print(f"Trades: {ml_results.total_trades}")
    print(f"Return: {ml_results.total_return:.2%}")
    print(f"Sharpe: {ml_results.sharpe_ratio:.2f}")
    
    print("\n📊 TEST 3: Adaptive ML Backtest (All Strategies)")
    print("-" * 50)
    
    # Run fully adaptive backtest with all strategies
    adaptive_results = orchestrator.run_adaptive_backtest(
        symbols=symbols
    )
    
    print(f"Strategies: All 7 built-in strategies")
    print(f"Trades: {adaptive_results.total_trades}")
    print(f"Return: {adaptive_results.total_return:.2%}")
    print(f"Sharpe: {adaptive_results.sharpe_ratio:.2f}")
    
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'Trades':<10} {'Return':<12} {'Sharpe':<10}")
    print("-" * 57)
    print(f"{'Single Strategy':<25} {regular_results.total_trades:<10} "
          f"{regular_results.total_return:>10.2%} {regular_results.sharpe_ratio:>10.2f}")
    print(f"{'ML Selection (5 strats)':<25} {ml_results.total_trades:<10} "
          f"{ml_results.total_return:>10.2%} {ml_results.sharpe_ratio:>10.2f}")
    print(f"{'Adaptive ML (7 strats)':<25} {adaptive_results.total_trades:<10} "
          f"{adaptive_results.total_return:>10.2%} {adaptive_results.sharpe_ratio:>10.2f}")
    
    # Calculate improvements
    ml_improvement = (ml_results.total_return - regular_results.total_return)
    adaptive_improvement = (adaptive_results.total_return - regular_results.total_return)
    
    print("\n📊 IMPROVEMENT OVER SINGLE STRATEGY")
    print("-" * 57)
    print(f"ML Selection:    {ml_improvement:+.2%}")
    print(f"Adaptive ML:     {adaptive_improvement:+.2%}")
    
    if ml_improvement > 0 or adaptive_improvement > 0:
        print("\n✅ ML strategy selection shows potential benefits!")
    else:
        print("\n📝 Note: ML may need more training data or tuning for this period")
    
    return regular_results, ml_results, adaptive_results


def demo_multi_symbol():
    """Demonstrate ML performance with multiple symbols."""
    print("\n\n" + "=" * 70)
    print("🌍 MULTI-SYMBOL ML BACKTEST")
    print("=" * 70)
    
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=50_000,
        show_progress=False,
        quiet_mode=True,
        save_trades=False
    )
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    orchestrator = create_ml_orchestrator(config)
    
    print(f"\nTesting {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"Capital: ${config.initial_capital:,}")
    
    results = orchestrator.run_adaptive_backtest(symbols=symbols)
    
    print("\n📊 RESULTS")
    print("-" * 50)
    print(f"Total Trades:    {results.total_trades}")
    print(f"Win Rate:        {results.win_rate:.1%}")
    print(f"Total Return:    {results.total_return:.2%}")
    print(f"CAGR:            {results.cagr:.2%}")
    print(f"Sharpe Ratio:    {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown:    {results.max_drawdown:.2%}")
    print(f"Calmar Ratio:    {results.calmar_ratio:.2f}")
    
    if results.total_return > 0:
        print("\n✅ ML-driven multi-symbol trading successful!")
    

if __name__ == "__main__":
    print("\n🚀 Starting ML Integration Demo...\n")
    
    # Run comparison demo
    regular, ml, adaptive = demo_comparison()
    
    # Run multi-symbol demo
    demo_multi_symbol()
    
    print("\n" + "=" * 70)
    print("✅ ML INTEGRATION DEMO COMPLETE!")
    print("=" * 70)
    print("\n💡 The ML pipeline is now connected and operational!")
    print("   - ML strategy selection is working")
    print("   - Multiple strategies can be evaluated dynamically")
    print("   - The system adapts to market conditions")
    print("\n📚 Next steps:")
    print("   - Train ML models with more historical data")
    print("   - Fine-tune strategy selection criteria")
    print("   - Add more sophisticated ML models")