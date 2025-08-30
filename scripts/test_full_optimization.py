#!/usr/bin/env python3
"""
Test the complete fixed optimization system with all components enabled.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from bot.integration.unified_optimizer import UnifiedOptimizer, UnifiedOptimizationConfig
from bot.strategy.mean_reversion import MeanReversionStrategy
import yfinance as yf


def test_full_optimization():
    """Test FULL optimization with all components."""
    
    print("="*80)
    print("TESTING COMPLETE OPTIMIZATION SYSTEM")
    print("="*80)
    
    # Test FULL optimization with all components
    config = UnifiedOptimizationConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        initial_capital=10000,
        quiet_mode=False,
        auto_apply_optimal_params=True,
        apply_signal_filters=True,
        use_regime_detection=True,
        use_trailing_stops=True,
        use_realistic_costs=True,
        spread_bps=5.0,
        slippage_bps=3.0,
        market_impact_bps=2.0,
        log_optimization_actions=True
    )
    
    optimizer = UnifiedOptimizer(config)
    strategy = MeanReversionStrategy()
    
    print('\nRunning FULL OPTIMIZATION...')
    result = optimizer.run_backtest(strategy, ['AAPL'])
    
    print('\n' + '='*60)
    print('FULL OPTIMIZATION RESULTS')
    print('='*60)
    print(f'Total Return: {result["metrics"]["total_return"]:.2f}%')
    print(f'Sharpe Ratio: {result["metrics"]["sharpe_ratio"]:.2f}')
    print(f'Max Drawdown: {result["metrics"]["max_drawdown"]:.2f}%')
    print(f'Total Trades: {result["metrics"]["total_trades"]}')
    print(f'Win Rate: {result["metrics"]["win_rate"]:.1f}%')
    print(f'Final Equity: ${result["metrics"]["final_equity"]:,.2f}')
    
    print(f'\nOptimization Statistics:')
    print(f'  Signals Generated: {result["optimization"]["signals_generated"]}')
    print(f'  Signals Filtered: {result["optimization"]["signals_filtered"]}')
    print(f'  Regime Changes: {result["optimization"]["regime_changes"]}')
    print(f'  Trailing Stops Hit: {result["optimization"]["trailing_stops_hit"]}')
    print(f'  Profit Targets Hit: {result["optimization"]["profit_targets_hit"]}')
    print(f'  Transaction Costs: ${result["optimization"]["total_transaction_costs"]:.2f}')
    
    # Compare to buy-and-hold
    print('\n' + '='*60)
    print('BENCHMARK COMPARISON')
    print('='*60)
    
    data = yf.download('AAPL', start='2024-01-01', end='2024-06-30', progress=False)
    # Get close prices as a simple float
    start_price = float(data['Close'].iloc[0])
    end_price = float(data['Close'].iloc[-1])
    bh_return = ((end_price - start_price) / start_price) * 100
    
    print(f'Buy-and-Hold Return: {bh_return:.2f}%')
    print(f'Optimized Strategy Return: {result["metrics"]["total_return"]:.2f}%')
    print(f'Outperformance: {result["metrics"]["total_return"] - bh_return:+.2f}%')
    
    # Success verdict
    print('\n' + '='*60)
    if result["metrics"]["total_return"] > bh_return:
        print(f'✅ SUCCESS: Strategy beats buy-and-hold by {result["metrics"]["total_return"] - bh_return:.2f}%')
    else:
        print(f'⚠️ UNDERPERFORMED: Strategy lags buy-and-hold by {bh_return - result["metrics"]["total_return"]:.2f}%')
    
    print('='*60)
    
    return result


if __name__ == "__main__":
    test_full_optimization()