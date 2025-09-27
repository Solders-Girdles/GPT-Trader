#!/usr/bin/env python3
"""
Parallel Paper Trading System
Run multiple strategies simultaneously for faster data collection.
"""

import os
import sys
import json
import asyncio
import threading
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# Add project root to path
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
from src.bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from src.bot_v2.features.brokerages.coinbase.models import APIConfig


class ParallelTrader:
    """Manages parallel trading sessions for multiple strategies."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        
    def run_strategy_thread(self, strategy_name: str, symbols: List[str], duration_seconds: int) -> Dict:
        """Run a single strategy in a thread."""
        print(f"🚀 [{strategy_name}] Starting thread...")
        
        try:
            # Create trader instance
            trader = AdvancedCoinbasePaperTrader(initial_capital=self.initial_capital)
            
            # Connect to Coinbase
            if not trader.connect():
                return {'error': 'Failed to connect', 'strategy': strategy_name}
            
            # Collect price history
            trader.collect_price_history(symbols, duration_seconds=20)
            
            # Select strategy
            trader.select_strategy(strategy_name)
            
            # Run for specified duration
            end_time = datetime.now() + timedelta(seconds=duration_seconds)
            trades_executed = 0
            
            while datetime.now() < end_time:
                # Run strategy signals
                trader.run_strategy_signals(symbols)
                trader.check_stops()
                
                # Track trades
                current_trades = len(trader.trades)
                if current_trades > trades_executed:
                    trades_executed = current_trades
                    print(f"📊 [{strategy_name}] Trade #{trades_executed} executed")
                
                time.sleep(5)  # Check every 5 seconds
            
            # Close positions
            for symbol in list(trader.positions.keys()):
                trader.execute_trade(symbol, 'sell', 'Session end')
            
            # Calculate results
            final_equity = trader.get_equity()
            returns = (final_equity - self.initial_capital) / self.initial_capital * 100
            
            result = {
                'strategy': strategy_name,
                'final_equity': final_equity,
                'returns': returns,
                'total_trades': len(trader.trades),
                'duration_seconds': duration_seconds,
                'timestamp': datetime.now().isoformat()
            }
            
            # Disconnect
            trader.broker.disconnect()
            
            print(f"✅ [{strategy_name}] Complete: {returns:+.2f}% return, {len(trader.trades)} trades")
            return result
            
        except Exception as e:
            print(f"❌ [{strategy_name}] Error: {e}")
            return {'error': str(e), 'strategy': strategy_name}
    
    def run_parallel_strategies(self, strategies: List[str], symbols: List[str], duration_seconds: int):
        """Run multiple strategies in parallel using threads."""
        print(f"\n{'='*70}")
        print(f"PARALLEL PAPER TRADING - {len(strategies)} STRATEGIES")
        print(f"{'='*70}")
        print(f"Strategies: {', '.join(strategies)}")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"{'='*70}\n")
        
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            # Submit all strategies
            futures = {
                executor.submit(self.run_strategy_thread, strategy, symbols, duration_seconds): strategy
                for strategy in strategies
            }
            
            # Collect results
            for future in futures:
                strategy = futures[future]
                try:
                    result = future.result(timeout=duration_seconds + 60)
                    with self.lock:
                        self.results[strategy] = result
                except Exception as e:
                    print(f"❌ [{strategy}] Failed: {e}")
                    with self.lock:
                        self.results[strategy] = {'error': str(e), 'strategy': strategy}
        
        return self.results


class RapidBacktester:
    """Rapid backtesting using historical data for quick strategy validation."""
    
    def __init__(self):
        self.results = {}
        
    async def backtest_strategy_async(self, strategy_name: str, symbols: List[str], days: int = 30) -> Dict:
        """Async backtesting for a single strategy."""
        print(f"📈 Backtesting {strategy_name} for {days} days...")
        
        # Import backtest components
        from src.bot_v2.features.backtest.backtest import Backtester
        from src.bot_v2.features.backtest.strategies import create_strategy
        
        try:
            # Create strategy
            strategy = create_strategy(strategy_name)
            
            # Run backtest for each symbol
            symbol_results = {}
            for symbol in symbols:
                backtester = Backtester(
                    symbol=symbol,
                    strategy=strategy,
                    start_date=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    initial_capital=10000
                )
                
                # Run backtest
                result = backtester.run()
                symbol_results[symbol] = {
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'num_trades': result.num_trades
                }
            
            # Aggregate results
            avg_return = sum(r['total_return'] for r in symbol_results.values()) / len(symbol_results)
            
            return {
                'strategy': strategy_name,
                'avg_return': avg_return,
                'symbol_results': symbol_results,
                'days_tested': days
            }
            
        except Exception as e:
            return {'strategy': strategy_name, 'error': str(e)}
    
    async def run_parallel_backtests(self, strategies: List[str], symbols: List[str], days: int = 30):
        """Run backtests for all strategies in parallel."""
        tasks = [
            self.backtest_strategy_async(strategy, symbols, days)
            for strategy in strategies
        ]
        
        results = await asyncio.gather(*tasks)
        return {r['strategy']: r for r in results}


class MultiAccountSimulator:
    """Simulate multiple trading accounts to test strategies simultaneously."""
    
    def __init__(self, num_accounts: int = 5):
        self.num_accounts = num_accounts
        self.accounts = []
        
    def create_virtual_account(self, account_id: int, strategy: str, initial_capital: float = 10000):
        """Create a virtual trading account."""
        return {
            'id': account_id,
            'strategy': strategy,
            'capital': initial_capital,
            'positions': {},
            'trades': [],
            'equity_history': [initial_capital]
        }
    
    def simulate_parallel_accounts(self, strategies: List[str], duration_minutes: int = 10):
        """Simulate multiple accounts trading different strategies."""
        print(f"\n{'='*70}")
        print(f"MULTI-ACCOUNT SIMULATION - {len(strategies)} ACCOUNTS")
        print(f"{'='*70}\n")
        
        # Create accounts
        for i, strategy in enumerate(strategies):
            account = self.create_virtual_account(i, strategy)
            self.accounts.append(account)
            print(f"📊 Account #{i}: {strategy} strategy - $10,000")
        
        # Simulate trading
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        iteration = 0
        
        while datetime.now() < end_time:
            iteration += 1
            
            # Update each account
            for account in self.accounts:
                # Simulate trades (simplified)
                import random
                if random.random() < 0.1:  # 10% chance of trade
                    trade_return = random.gauss(0, 0.02)  # Mean 0%, std 2%
                    account['capital'] *= (1 + trade_return)
                    account['trades'].append({
                        'time': datetime.now().isoformat(),
                        'return': trade_return
                    })
                    account['equity_history'].append(account['capital'])
            
            if iteration % 12 == 0:  # Every minute
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Account Status:")
                for account in self.accounts:
                    returns = (account['capital'] - 10000) / 10000 * 100
                    print(f"  Account #{account['id']} ({account['strategy']}): "
                          f"${account['capital']:.2f} ({returns:+.2f}%)")
            
            time.sleep(5)
        
        # Final results
        print(f"\n{'='*70}")
        print("SIMULATION COMPLETE")
        print(f"{'='*70}")
        
        results = []
        for account in self.accounts:
            final_return = (account['capital'] - 10000) / 10000 * 100
            results.append({
                'strategy': account['strategy'],
                'final_capital': account['capital'],
                'return': final_return,
                'num_trades': len(account['trades'])
            })
            print(f"{account['strategy']}: ${account['capital']:.2f} ({final_return:+.2f}%)")
        
        return results


def run_speed_test():
    """Run a comprehensive speed test comparing different approaches."""
    
    print("\n" + "="*70)
    print("PAPER TRADING SPEED TEST")
    print("="*70)
    
    strategies = ['momentum', 'mean_reversion', 'breakout', 'ma_crossover', 'volatility']
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    results = {}
    
    # Test 1: Sequential (baseline)
    print("\n1. SEQUENTIAL EXECUTION (Baseline)")
    print("-" * 40)
    start = time.time()
    
    sequential_results = []
    for strategy in strategies[:2]:  # Just 2 for speed
        trader = ParallelTrader()
        result = trader.run_strategy_thread(strategy, symbols, duration_seconds=30)
        sequential_results.append(result)
    
    sequential_time = time.time() - start
    results['sequential'] = {
        'time': sequential_time,
        'strategies': len(sequential_results),
        'time_per_strategy': sequential_time / len(sequential_results)
    }
    print(f"⏱️ Sequential Time: {sequential_time:.1f}s ({sequential_time/len(sequential_results):.1f}s per strategy)")
    
    # Test 2: Parallel Threading
    print("\n2. PARALLEL THREADING")
    print("-" * 40)
    start = time.time()
    
    trader = ParallelTrader()
    parallel_results = trader.run_parallel_strategies(strategies[:2], symbols, duration_seconds=30)
    
    parallel_time = time.time() - start
    results['parallel'] = {
        'time': parallel_time,
        'strategies': len(parallel_results),
        'time_per_strategy': parallel_time / len(parallel_results)
    }
    print(f"⏱️ Parallel Time: {parallel_time:.1f}s ({parallel_time/len(parallel_results):.1f}s per strategy)")
    
    # Test 3: Multi-Account Simulation
    print("\n3. MULTI-ACCOUNT SIMULATION")
    print("-" * 40)
    start = time.time()
    
    simulator = MultiAccountSimulator()
    sim_results = simulator.simulate_parallel_accounts(strategies, duration_minutes=0.5)
    
    sim_time = time.time() - start
    results['simulation'] = {
        'time': sim_time,
        'strategies': len(sim_results),
        'time_per_strategy': sim_time / len(sim_results)
    }
    print(f"⏱️ Simulation Time: {sim_time:.1f}s ({sim_time/len(sim_results):.1f}s per strategy)")
    
    # Summary
    print("\n" + "="*70)
    print("SPEED TEST SUMMARY")
    print("="*70)
    
    print(f"\n{'Method':<20} {'Time (s)':<15} {'Speedup':<15}")
    print("-" * 50)
    
    baseline = results['sequential']['time']
    for method, data in results.items():
        speedup = baseline / data['time'] if data['time'] > 0 else 0
        print(f"{method:<20} {data['time']:<15.1f} {speedup:<15.2f}x")
    
    # Recommendations
    print("\n📊 RECOMMENDATIONS:")
    if results['parallel']['time'] < results['sequential']['time']:
        speedup = results['sequential']['time'] / results['parallel']['time']
        print(f"✅ Parallel execution is {speedup:.1f}x faster - USE FOR TESTING")
    
    print(f"✅ Simulation is fastest for rough estimates")
    print(f"✅ Use parallel for real market data testing")
    
    return results


def main():
    """Main entry point for parallel paper trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel Paper Trading System")
    parser.add_argument('--mode', choices=['parallel', 'speed-test', 'simulate', 'backtest'],
                       default='parallel', help='Execution mode')
    parser.add_argument('--strategies', type=str,
                       default='momentum,mean_reversion,breakout',
                       help='Comma-separated strategy names')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in seconds for each strategy')
    parser.add_argument('--symbols', type=str,
                       default='BTC-USD,ETH-USD,SOL-USD',
                       help='Comma-separated symbols')
    
    args = parser.parse_args()
    
    strategies = args.strategies.split(',')
    symbols = args.symbols.split(',')
    
    if args.mode == 'speed-test':
        run_speed_test()
        
    elif args.mode == 'parallel':
        print(f"Running {len(strategies)} strategies in parallel for {args.duration} seconds each...")
        trader = ParallelTrader()
        results = trader.run_parallel_strategies(strategies, symbols, args.duration)
        
        # Display results
        print("\n" + "="*70)
        print("PARALLEL EXECUTION COMPLETE")
        print("="*70)
        
        for strategy, result in results.items():
            if 'error' not in result:
                print(f"{strategy}: {result.get('returns', 0):+.2f}% return, "
                      f"{result.get('total_trades', 0)} trades")
            else:
                print(f"{strategy}: ERROR - {result['error']}")
        
        # Save results
        results_file = Path(__file__).parent.parent / 'results' / f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {results_file.name}")
        
    elif args.mode == 'simulate':
        simulator = MultiAccountSimulator(num_accounts=len(strategies))
        results = simulator.simulate_parallel_accounts(strategies, duration_minutes=args.duration//60)
        
    elif args.mode == 'backtest':
        print("Running parallel backtests...")
        backtester = RapidBacktester()
        results = asyncio.run(backtester.run_parallel_backtests(strategies, symbols, days=30))
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        
        for strategy, result in results.items():
            if 'error' not in result:
                print(f"{strategy}: {result.get('avg_return', 0):+.2f}% average return")
            else:
                print(f"{strategy}: ERROR - {result['error']}")


if __name__ == "__main__":
    main()