#!/usr/bin/env python3
"""
Rapid Coinbase Testing System
Tests strategies using Coinbase historical data for quick validation.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

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

from src.bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from src.bot_v2.features.brokerages.coinbase.models import APIConfig


class CoinbaseRapidTester:
    """Rapid testing using Coinbase historical data."""
    
    def __init__(self):
        self.broker = None
        self.results = []
        
    def connect(self) -> bool:
        """Connect to Coinbase API."""
        try:
            # Use same config as working paper trading
            config = APIConfig(
                api_key="",
                api_secret="",
                passphrase=None,
                base_url="https://api.coinbase.com",
                cdp_api_key=os.environ.get('COINBASE_CDP_API_KEY_NAME'),
                cdp_private_key=os.environ.get('COINBASE_CDP_PRIVATE_KEY')
            )
            self.broker = CoinbaseBrokerage(config)
            return self.broker.connect()
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def fetch_candles(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Fetch historical candles from Coinbase."""
        if not self.broker:
            return []
        
        try:
            # Get candles (300 second granularity for more data points)
            product_id = symbol
            end = int(datetime.now().timestamp())
            start = end - (hours * 3600)
            
            # Coinbase API: GET /api/v3/brokerage/products/{product_id}/candles
            endpoint = f"/api/v3/brokerage/products/{product_id}/candles"
            params = {
                'start': str(start),
                'end': str(end),
                'granularity': 'FIVE_MINUTE'
            }
            
            response = self.broker._authenticated_request('GET', endpoint, params=params)
            
            if response and 'candles' in response:
                return response['candles']
            
            return []
            
        except Exception as e:
            print(f"⚠️ Failed to fetch candles for {symbol}: {e}")
            return []
    
    def simulate_strategy(self, strategy: str, symbol: str, candles: List[Dict]) -> Dict:
        """Simulate a strategy on historical candles."""
        if not candles:
            return {'error': 'No data'}
        
        capital = 10000
        position = 0
        trades = []
        
        # Convert candles to usable format
        prices = []
        for candle in candles:
            try:
                prices.append({
                    'time': candle['start'],
                    'close': float(candle['close']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'volume': float(candle['volume'])
                })
            except:
                continue
        
        if len(prices) < 20:
            return {'error': 'Insufficient data'}
        
        # Calculate indicators
        closes = [p['close'] for p in prices]
        
        # Simple moving averages
        sma_10 = []
        sma_20 = []
        for i in range(len(closes)):
            if i >= 9:
                sma_10.append(sum(closes[i-9:i+1]) / 10)
            else:
                sma_10.append(closes[i])
            
            if i >= 19:
                sma_20.append(sum(closes[i-19:i+1]) / 20)
            else:
                sma_20.append(closes[i])
        
        # Apply strategy
        for i in range(20, len(prices)):
            price = prices[i]['close']
            
            if strategy == 'momentum':
                # Buy if price rising fast
                if i >= 25:
                    momentum = (prices[i]['close'] - prices[i-5]['close']) / prices[i-5]['close']
                    if momentum > 0.02 and position == 0:
                        # Buy
                        position = capital * 0.95 / price
                        capital *= 0.05
                        trades.append({'type': 'buy', 'price': price})
                    elif momentum < -0.02 and position > 0:
                        # Sell
                        capital += position * price * 0.994
                        position = 0
                        trades.append({'type': 'sell', 'price': price})
            
            elif strategy == 'mean_reversion':
                # Buy oversold, sell overbought
                if sma_10[i] < sma_20[i] * 0.98 and position == 0:
                    # Buy
                    position = capital * 0.95 / price
                    capital *= 0.05
                    trades.append({'type': 'buy', 'price': price})
                elif sma_10[i] > sma_20[i] * 1.02 and position > 0:
                    # Sell
                    capital += position * price * 0.994
                    position = 0
                    trades.append({'type': 'sell', 'price': price})
            
            elif strategy == 'breakout':
                # Buy on breakout
                if i >= 25:
                    recent_high = max(p['high'] for p in prices[i-20:i])
                    if price > recent_high * 1.01 and position == 0:
                        # Buy
                        position = capital * 0.95 / price
                        capital *= 0.05
                        trades.append({'type': 'buy', 'price': price})
                    
                    recent_low = min(p['low'] for p in prices[i-20:i])
                    if price < recent_low * 0.99 and position > 0:
                        # Sell
                        capital += position * price * 0.994
                        position = 0
                        trades.append({'type': 'sell', 'price': price})
        
        # Close final position
        if position > 0:
            final_price = prices[-1]['close']
            capital += position * final_price * 0.994
            position = 0
        
        # Calculate metrics
        final_equity = capital
        returns = (final_equity - 10000) / 10000 * 100
        
        return {
            'strategy': strategy,
            'symbol': symbol,
            'returns': returns,
            'num_trades': len([t for t in trades if t['type'] == 'buy']),
            'final_equity': final_equity
        }
    
    def test_strategy_worker(self, args: tuple) -> Dict:
        """Worker function for parallel testing."""
        strategy, symbol = args
        
        # Fetch candles
        candles = self.fetch_candles(symbol, hours=24)
        
        # Simulate strategy
        result = self.simulate_strategy(strategy, symbol, candles)
        
        return result
    
    def run_parallel_tests(self, strategies: List[str], symbols: List[str]):
        """Run tests in parallel."""
        print("=" * 70)
        print("RAPID COINBASE STRATEGY TESTING")
        print("=" * 70)
        print(f"Strategies: {', '.join(strategies)}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Data: Last 24 hours from Coinbase")
        print("=" * 70)
        
        # Connect to Coinbase
        if not self.connect():
            print("❌ Failed to connect to Coinbase")
            return
        
        print("\n📊 Fetching Coinbase data and testing strategies...")
        start_time = time.time()
        
        # Create work items
        work_items = [(strategy, symbol) for strategy in strategies for symbol in symbols]
        
        # Run in parallel
        with ThreadPoolExecutor(max_workers=min(len(work_items), 10)) as executor:
            results = list(executor.map(self.test_strategy_worker, work_items))
        
        elapsed = time.time() - start_time
        
        # Display results
        print(f"\n✅ Completed {len(results)} tests in {elapsed:.1f} seconds")
        print(f"   ({len(results)/elapsed:.1f} tests per second)")
        
        print("\n" + "=" * 70)
        print("RESULTS BY STRATEGY")
        print("=" * 70)
        
        # Aggregate by strategy
        strategy_performance = {}
        for result in results:
            if 'error' not in result:
                strategy = result['strategy']
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(result['returns'])
        
        print(f"\n{'Strategy':<15} {'Avg Return':<12} {'Best':<12} {'Worst':<12}")
        print("-" * 50)
        
        for strategy, returns in strategy_performance.items():
            if returns:
                avg_return = sum(returns) / len(returns)
                best = max(returns)
                worst = min(returns)
                print(f"{strategy:<15} {avg_return:>+11.2f}% {best:>+11.2f}% {worst:>+11.2f}%")
        
        # Best combinations
        print("\n" + "=" * 70)
        print("TOP STRATEGY-SYMBOL COMBINATIONS")
        print("=" * 70)
        
        valid_results = [r for r in results if 'error' not in r]
        sorted_results = sorted(valid_results, key=lambda x: x['returns'], reverse=True)[:5]
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result['strategy']}-{result['symbol']}: "
                  f"{result['returns']:+.2f}% return, "
                  f"{result['num_trades']} trades")
        
        # Disconnect
        self.broker.disconnect()
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rapid Coinbase Testing")
    parser.add_argument('--strategies', type=str,
                       default='momentum,mean_reversion,breakout',
                       help='Comma-separated strategies')
    parser.add_argument('--symbols', type=str,
                       default='BTC-USD,ETH-USD,SOL-USD',
                       help='Comma-separated symbols')
    
    args = parser.parse_args()
    
    strategies = args.strategies.split(',')
    symbols = args.symbols.split(',')
    
    tester = CoinbaseRapidTester()
    tester.run_parallel_tests(strategies, symbols)


if __name__ == "__main__":
    main()