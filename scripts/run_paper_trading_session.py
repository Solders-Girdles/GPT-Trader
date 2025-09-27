#!/usr/bin/env python3
"""
Run Extended Paper Trading Session
Runs paper trading with automatic strategy rotation to test all strategies.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
env_file = Path(__file__).parent.parent / '.env.production'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"')
                if key == 'COINBASE_CDP_PRIVATE_KEY':
                    private_key_lines = [value] if value else []
                    for next_line in f:
                        next_line = next_line.strip()
                        private_key_lines.append(next_line)
                        if 'END EC PRIVATE KEY' in next_line:
                            break
                    value = '\n'.join(private_key_lines)
                os.environ[key] = value

from scripts.paper_trade_strategies_coinbase import AdvancedCoinbasePaperTrader


def run_rotating_strategies(duration_per_strategy_minutes: int = 5, symbols: list = None):
    """Run paper trading with rotating strategies."""
    
    if symbols is None:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'LINK-USD', 'MATIC-USD']
    
    strategies = ['momentum', 'mean_reversion', 'breakout', 'ma_crossover', 'volatility']
    
    print("=" * 70)
    print("EXTENDED PAPER TRADING SESSION")
    print("=" * 70)
    print(f"Duration per strategy: {duration_per_strategy_minutes} minutes")
    print(f"Total duration: {duration_per_strategy_minutes * len(strategies)} minutes")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Strategies: {', '.join(strategies)}")
    print("=" * 70)
    
    overall_results = {}
    session_start = datetime.now()
    
    for strategy_name in strategies:
        print(f"\n\n{'=' * 70}")
        print(f"TESTING STRATEGY: {strategy_name.upper()}")
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70)
        
        # Create trader for this strategy
        trader = AdvancedCoinbasePaperTrader(initial_capital=10000)
        
        # Connect to Coinbase
        if not trader.connect():
            print(f"Failed to connect for {strategy_name}")
            continue
        
        print("✅ Connected to Coinbase")
        
        # Collect initial price history
        print("📊 Collecting 30s of price history...")
        trader.collect_price_history(symbols, duration_seconds=30)
        
        # Select strategy
        trader.select_strategy(strategy_name)
        
        # Run strategy for specified duration
        strategy_end = datetime.now() + timedelta(minutes=duration_per_strategy_minutes)
        iteration = 0
        
        try:
            while datetime.now() < strategy_end:
                iteration += 1
                
                # Run strategy signals
                trader.run_strategy_signals(symbols)
                
                # Check stops
                trader.check_stops()
                
                # Display status every 30 seconds
                if iteration % 3 == 0:
                    trader.display_status()
                
                # Log performance
                trader.performance_log.append({
                    'timestamp': datetime.now(),
                    'equity': trader.get_equity(),
                    'num_positions': len(trader.positions),
                    'cash': trader.cash
                })
                
                # Wait before next check
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            print(f"\n⚠️ {strategy_name} interrupted by user")
        except Exception as e:
            print(f"❌ Error in {strategy_name}: {e}")
        
        # Close all positions at end of strategy test
        print(f"\n📊 Closing all positions for {strategy_name}...")
        for symbol in list(trader.positions.keys()):
            trader.execute_trade(symbol, 'sell', f'{strategy_name} session end')
        
        # Final status
        trader.display_status()
        
        # Calculate results
        final_equity = trader.get_equity()
        returns = (final_equity - 10000) / 10000 * 100
        winning_trades = [t for t in trader.trades if t.get('pnl', 0) > 0]
        
        overall_results[strategy_name] = {
            'final_equity': final_equity,
            'return': returns,
            'total_trades': len(trader.trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(trader.trades) * 100 if trader.trades else 0,
            'max_positions': max(log['num_positions'] for log in trader.performance_log) if trader.performance_log else 0
        }
        
        # Disconnect
        trader.broker.disconnect()
        print(f"✅ Completed {strategy_name} testing")
        
        # Short break between strategies
        if strategy_name != strategies[-1]:
            print(f"\n⏸️ 10 second break before next strategy...")
            time.sleep(10)
    
    # Display overall results
    print("\n\n" + "=" * 70)
    print("SESSION COMPLETE - OVERALL RESULTS")
    print("=" * 70)
    print(f"Total Duration: {str(datetime.now() - session_start).split('.')[0]}")
    
    print("\n📊 STRATEGY PERFORMANCE RANKING:")
    print("-" * 70)
    print(f"{'Strategy':<15} {'Return':<10} {'Trades':<10} {'Win Rate':<12} {'Final Equity':<15}")
    print("-" * 70)
    
    # Sort by return
    sorted_results = sorted(overall_results.items(), key=lambda x: x[1]['return'], reverse=True)
    
    for strategy, metrics in sorted_results:
        print(f"{strategy:<15} {metrics['return']:>+7.2f}%   {metrics['total_trades']:<10} {metrics['win_rate']:>6.1f}%     ${metrics['final_equity']:>10.2f}")
    
    print("-" * 70)
    
    # Best and worst
    if sorted_results:
        best = sorted_results[0]
        worst = sorted_results[-1]
        print(f"\n🏆 Best Strategy: {best[0]} ({best[1]['return']:+.2f}%)")
        print(f"📉 Worst Strategy: {worst[0]} ({worst[1]['return']:+.2f}%)")
    
    # Save results
    import json
    results_file = Path(__file__).parent.parent / 'results' / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'session_start': session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'duration_minutes': duration_per_strategy_minutes * len(strategies),
            'symbols': symbols,
            'strategies': strategies,
            'results': overall_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    best_return = sorted_results[0][1]['return'] if sorted_results else 0
    if best_return > 0:
        print(f"  ✅ {sorted_results[0][0]} showing positive returns")
    else:
        print("  ⚠️ All strategies showing losses - market conditions may be unfavorable")
    
    total_trades = sum(m['total_trades'] for m in overall_results.values())
    if total_trades < 50:
        print(f"  ⚠️ Only {total_trades} total trades - need more data for confidence")
    else:
        print(f"  ✅ {total_trades} trades executed - good sample size")
    
    avg_win_rate = sum(m['win_rate'] for m in overall_results.values()) / len(overall_results) if overall_results else 0
    if avg_win_rate > 50:
        print(f"  ✅ Average win rate {avg_win_rate:.1f}% - strategies performing well")
    else:
        print(f"  ⚠️ Average win rate {avg_win_rate:.1f}% - consider parameter tuning")
    
    print("\n" + "=" * 70)
    print("PAPER TRADING SESSION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run extended paper trading session")
    parser.add_argument('--duration', type=int, default=5, 
                      help='Duration per strategy in minutes (default: 5)')
    parser.add_argument('--quick', action='store_true',
                      help='Quick test mode - 1 minute per strategy')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Running quick test (1 minute per strategy)...")
        run_rotating_strategies(duration_per_strategy_minutes=1)
    else:
        print(f"🚀 Running full session ({args.duration} minutes per strategy)...")
        run_rotating_strategies(duration_per_strategy_minutes=args.duration)