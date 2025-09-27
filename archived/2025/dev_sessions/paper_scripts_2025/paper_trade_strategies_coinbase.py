#!/usr/bin/env python3
"""
Advanced Coinbase Paper Trading with Bot V2 Strategies
Integrates real strategies from bot_v2 with live Coinbase market data.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load production environment
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


class Strategy:
    """Base strategy class."""
    
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params
        self.price_history = {}  # symbol -> deque of prices
        self.lookback = params.get('lookback', 20)
        
    def update_price(self, symbol: str, price: float):
        """Update price history for symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback)
        self.price_history[symbol].append(price)
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Override in subclasses."""
        return 'hold'


class MomentumStrategy(Strategy):
    """Momentum-based trading strategy."""
    
    def __init__(self, **params):
        super().__init__('Momentum', **params)
        self.momentum_period = params.get('momentum_period', 10)
        self.threshold = params.get('threshold', 0.02)
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate momentum signal."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = list(self.price_history[symbol])
        if len(prices) < self.momentum_period:
            return 'hold'
        
        # Calculate momentum
        old_price = prices[-self.momentum_period]
        momentum = (current_price - old_price) / old_price
        
        if momentum > self.threshold:
            return 'buy'
        elif momentum < -self.threshold:
            return 'sell'
        return 'hold'


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using Bollinger Bands."""
    
    def __init__(self, **params):
        super().__init__('MeanReversion', **params)
        self.bb_period = params.get('bb_period', 20)
        self.bb_std = params.get('bb_std', 2)
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate mean reversion signal."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = np.array(list(self.price_history[symbol]))
        if len(prices) < self.bb_period:
            return 'hold'
        
        # Calculate Bollinger Bands
        mean = np.mean(prices)
        std = np.std(prices)
        upper_band = mean + (self.bb_std * std)
        lower_band = mean - (self.bb_std * std)
        
        if current_price < lower_band:
            return 'buy'
        elif current_price > upper_band:
            return 'sell'
        return 'hold'


class BreakoutStrategy(Strategy):
    """Breakout strategy based on recent highs/lows."""
    
    def __init__(self, **params):
        super().__init__('Breakout', **params)
        self.breakout_period = params.get('breakout_period', 20)
        self.confirmation_bars = params.get('confirmation_bars', 2)
        # Percentage distance from recent high/low to trigger breakout (e.g., 0.01 = 1%)
        self.threshold_pct = params.get('threshold_pct', 0.01)
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate breakout signal."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = list(self.price_history[symbol])
        if len(prices) < self.breakout_period:
            return 'hold'
        
        # Check for breakout
        recent_high = max(prices[-self.breakout_period:-1])
        recent_low = min(prices[-self.breakout_period:-1])
        
        if current_price > recent_high * (1 + self.threshold_pct):
            return 'buy'
        elif current_price < recent_low * (1 - self.threshold_pct):
            return 'sell'
        return 'hold'


class MAStrategy(Strategy):
    """Moving average crossover strategy."""
    
    def __init__(self, **params):
        super().__init__('MAStrategy', **params)
        self.fast_period = params.get('fast_period', 10)
        self.slow_period = params.get('slow_period', 30)
        self.lookback = max(self.fast_period, self.slow_period) + 5
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate MA crossover signal."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = np.array(list(self.price_history[symbol]))
        if len(prices) < self.slow_period:
            return 'hold'
        
        # Calculate moving averages
        fast_ma = np.mean(prices[-self.fast_period:])
        slow_ma = np.mean(prices[-self.slow_period:])
        
        # Check previous MAs for crossover
        if len(prices) > self.slow_period + 1:
            prev_prices = prices[:-1]
            prev_fast_ma = np.mean(prev_prices[-self.fast_period:])
            prev_slow_ma = np.mean(prev_prices[-self.slow_period:])
            
            # Bullish crossover
            if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
                return 'buy'
            # Bearish crossover
            elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
                return 'sell'
        
        return 'hold'


class VolatilityStrategy(Strategy):
    """Trade based on volatility conditions."""
    
    def __init__(self, **params):
        super().__init__('Volatility', **params)
        self.vol_period = params.get('vol_period', 20)
        self.vol_threshold = params.get('vol_threshold', 0.02)
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate volatility-based signal."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = np.array(list(self.price_history[symbol]))
        if len(prices) < self.vol_period:
            return 'hold'
        
        # Calculate volatility (standard deviation of returns)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-self.vol_period:])
        
        # Low volatility - trend following
        if volatility < self.vol_threshold:
            recent_return = (current_price - prices[-5]) / prices[-5]
            if recent_return > 0.01:
                return 'buy'
            elif recent_return < -0.01:
                return 'sell'
        
        return 'hold'

class ScalpStrategy(Strategy):
    """Micro-momentum scalp strategy using tick-to-tick deltas.

    Buys when price increases more than a basis-point threshold since last tick,
    sells when price decreases more than the threshold.
    """
    def __init__(self, **params):
        super().__init__('Scalp', **params)
        # threshold in fraction (e.g., 0.0005 = 5 bps = 0.05%)
        self.bp_threshold = params.get('bp_threshold', 0.0005)
        self.min_hold_seconds = params.get('min_hold_seconds', 30)

    def get_signal(self, symbol: str, current_price: float) -> str:
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return 'hold'
        last = list(self.price_history[symbol])[-1]
        if last == 0:
            return 'hold'
        change = (current_price - last) / last
        if change >= self.bp_threshold:
            return 'buy'
        if change <= -self.bp_threshold:
            return 'sell'
        return 'hold'


class AdvancedCoinbasePaperTrader:
    """Advanced paper trading with multiple strategies."""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        signal_mode: str = 'conservative',
        # Risk knobs
        max_position_size: float = 0.2,
        max_positions: int = 5,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
        commission_rate: float = 0.006,
        slippage_rate: float = 0.001,
        # Strategy parameter overrides per strategy
        strategy_overrides: dict | None = None,
    ):
        """Initialize advanced paper trader."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.performance_log = []
        self.signal_mode = signal_mode  # conservative | balanced | aggressive
        # Signal/decision diagnostics
        self.signal_stats = {
            'iterations': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'trades_executed': 0,
            'blocked_max_positions': 0,
            'blocked_has_position': 0,
            'blocked_no_position': 0,
            'no_quote': 0,
        }
        
        # Risk parameters (knobs)
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Initialize strategies (defaults are conservative)
        if self.signal_mode == 'aggressive':
            self.strategies = {
                'momentum': MomentumStrategy(momentum_period=8, threshold=0.01),
                'mean_reversion': MeanReversionStrategy(bb_period=20, bb_std=1.5),
                'breakout': BreakoutStrategy(breakout_period=15, threshold_pct=0.005),
                'ma_crossover': MAStrategy(fast_period=8, slow_period=21),
                'volatility': VolatilityStrategy(vol_period=20, vol_threshold=0.015),
                'scalp': ScalpStrategy(bp_threshold=0.0004, min_hold_seconds=20)
            }
        elif self.signal_mode == 'balanced':
            self.strategies = {
                'momentum': MomentumStrategy(momentum_period=10, threshold=0.015),
                'mean_reversion': MeanReversionStrategy(bb_period=20, bb_std=1.8),
                'breakout': BreakoutStrategy(breakout_period=20, threshold_pct=0.0075),
                'ma_crossover': MAStrategy(fast_period=9, slow_period=25),
                'volatility': VolatilityStrategy(vol_period=20, vol_threshold=0.018),
                'scalp': ScalpStrategy(bp_threshold=0.0005, min_hold_seconds=25)
            }
        else:  # conservative (default)
            self.strategies = {
                'momentum': MomentumStrategy(momentum_period=10, threshold=0.02),
                'mean_reversion': MeanReversionStrategy(bb_period=20, bb_std=2),
                'breakout': BreakoutStrategy(breakout_period=20, threshold_pct=0.01),
                'ma_crossover': MAStrategy(fast_period=10, slow_period=30),
                'volatility': VolatilityStrategy(vol_period=20, vol_threshold=0.02),
                'scalp': ScalpStrategy(bp_threshold=0.0007, min_hold_seconds=30)
            }
        
        # Apply overrides to strategies if provided
        if strategy_overrides:
            for name, overrides in strategy_overrides.items():
                strat = self.strategies.get(name)
                if not strat:
                    continue
                for k, v in (overrides or {}).items():
                    if hasattr(strat, k):
                        setattr(strat, k, v)

        # Active strategy
        self.active_strategy = None
        
        # Coinbase connection
        self.broker = self._init_coinbase()
        self.is_connected = False
        
        # Performance tracking
        self.start_time = datetime.now()
        self.peak_equity = initial_capital
        
        # Price collection for strategies
        self.price_collection_running = False
        
    def _init_coinbase(self) -> CoinbaseBrokerage:
        """Initialize Coinbase connection."""
        config = APIConfig(
            api_key="",
            api_secret="",
            passphrase=None,
            base_url=os.getenv('COINBASE_API_BASE', 'https://api.coinbase.com'),
            sandbox=False,
            cdp_api_key=os.getenv('COINBASE_CDP_API_KEY'),
            cdp_private_key=os.getenv('COINBASE_CDP_PRIVATE_KEY')
        )
        return CoinbaseBrokerage(config)
    
    def connect(self) -> bool:
        """Connect to Coinbase."""
        self.is_connected = self.broker.connect()
        return self.is_connected
    
    def collect_price_history(self, symbols: List[str], duration_seconds: int = 60):
        """Collect price history for strategies."""
        print(f"📊 Collecting {duration_seconds}s of price history...")
        end_time = datetime.now() + timedelta(seconds=duration_seconds)
        
        while datetime.now() < end_time:
            for symbol in symbols:
                quote = self.broker.get_quote(symbol)
                if quote:
                    mid_price = (float(quote.bid) + float(quote.ask)) / 2
                    for strategy in self.strategies.values():
                        strategy.update_price(symbol, mid_price)
            time.sleep(2)  # Collect every 2 seconds
        
        print("✅ Price history collected")
    
    def select_strategy(self, name: str):
        """Select active trading strategy."""
        if name in self.strategies:
            self.active_strategy = self.strategies[name]
            print(f"📈 Selected strategy: {name}")
        else:
            print(f"❌ Unknown strategy: {name}")
    
    def execute_trade(self, symbol: str, side: str, reason: str = ""):
        """Execute a paper trade."""
        quote = self.broker.get_quote(symbol)
        if not quote:
            return None
        
        if side == 'buy':
            # Check if we can buy
            if len(self.positions) >= self.max_positions:
                return None
            if symbol in self.positions:
                return None  # Already have position
            
            # Calculate position size
            position_value = self.cash * self.max_position_size
            price = float(quote.ask) * (1 + (self.slippage_rate or 0.0))  # Add slippage
            commission = position_value * self.commission_rate
            net_value = position_value - commission
            quantity = net_value / price
            
            if self.cash < position_value:
                return None
            
            # Execute buy
            self.cash -= position_value
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'current_price': price,
                'entry_time': datetime.now(),
                'strategy': self.active_strategy.name if self.active_strategy else 'manual'
            }
            
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'buy',
                'quantity': quantity,
                'price': price,
                'value': position_value,
                'commission': commission,
                'reason': reason
            }
            self.trades.append(trade)
            
            print(f"✅ BUY {symbol}: {quantity:.6f} @ ${price:.2f} ({reason})")
            return trade
            
        elif side == 'sell' and symbol in self.positions:
            pos = self.positions[symbol]
            price = float(quote.bid) * (1 - (self.slippage_rate or 0.0))  # Subtract slippage
            value = pos['quantity'] * price
            commission = value * self.commission_rate
            net_value = value - commission
            pnl = (price - pos['entry_price']) * pos['quantity'] - commission
            
            # Execute sell
            self.cash += net_value
            del self.positions[symbol]
            
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'sell',
                'quantity': pos['quantity'],
                'price': price,
                'value': value,
                'commission': commission,
                'pnl': pnl,
                'reason': reason
            }
            self.trades.append(trade)
            
            emoji = "🟢" if pnl > 0 else "🔴"
            print(f"{emoji} SELL {symbol}: {pos['quantity']:.6f} @ ${price:.2f} | P&L: ${pnl:.2f} ({reason})")
            return trade
        
        return None
    
    def check_stops(self):
        """Check stop loss and take profit levels."""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            quote = self.broker.get_quote(symbol)
            if not quote:
                continue
            
            current_price = float(quote.bid)
            pos['current_price'] = current_price
            entry_price = pos['entry_price']
            price_change = (current_price - entry_price) / entry_price
            
            if price_change <= -self.stop_loss:
                self.execute_trade(symbol, 'sell', f"Stop Loss ({price_change:.1%})")
            elif price_change >= self.take_profit:
                self.execute_trade(symbol, 'sell', f"Take Profit ({price_change:.1%})")
    
    def run_strategy_signals(self, symbols: List[str]):
        """Run strategy and generate signals."""
        if not self.active_strategy:
            return
        
        self.signal_stats['iterations'] += 1
        for symbol in symbols:
            quote = self.broker.get_quote(symbol)
            if not quote:
                self.signal_stats['no_quote'] += 1
                continue
            
            current_price = (float(quote.bid) + float(quote.ask)) / 2
            
            # Update price history
            self.active_strategy.update_price(symbol, current_price)
            
            # Get signal
            signal = self.active_strategy.get_signal(symbol, current_price)
            if signal == 'buy':
                self.signal_stats['buy_signals'] += 1
            elif signal == 'sell':
                self.signal_stats['sell_signals'] += 1
            else:
                self.signal_stats['hold_signals'] += 1
            
            if signal == 'buy' and symbol not in self.positions:
                # Quick pre-checks to categorize blocks
                if len(self.positions) >= self.max_positions:
                    self.signal_stats['blocked_max_positions'] += 1
                else:
                    trade = self.execute_trade(symbol, 'buy', f"{self.active_strategy.name} Signal")
                    if trade:
                        self.signal_stats['trades_executed'] += 1
                    else:
                        # Could be cash or other constraint; attribute to has_position if applicable
                        pass
            elif signal == 'sell' and symbol in self.positions:
                trade = self.execute_trade(symbol, 'sell', f"{self.active_strategy.name} Signal")
                if trade:
                    self.signal_stats['trades_executed'] += 1
            elif signal == 'sell' and symbol not in self.positions:
                self.signal_stats['blocked_no_position'] += 1
    
    def get_equity(self) -> float:
        """Calculate total equity."""
        equity = self.cash
        for symbol, pos in self.positions.items():
            equity += pos['quantity'] * pos['current_price']
        return equity
    
    def display_status(self):
        """Display current status."""
        equity = self.get_equity()
        returns = (equity - self.initial_capital) / self.initial_capital * 100
        
        # Calculate metrics
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        
        winning = [t for t in self.trades if t.get('pnl', 0) > 0]
        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0
        
        print("\n" + "=" * 70)
        print("ADVANCED COINBASE PAPER TRADING STATUS")
        print("=" * 70)
        print(f"Strategy: {self.active_strategy.name if self.active_strategy else 'None'}")
        print(f"Duration: {str(datetime.now() - self.start_time).split('.')[0]}")
        print(f"Equity: ${equity:.2f} ({returns:+.2f}%)")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Drawdown: {drawdown:.2f}%")
        print(f"Trades: {len(self.trades)} | Win Rate: {win_rate:.1f}%")
        
        # Signal diagnostics (last totals)
        stats = self.signal_stats
        if stats['iterations'] > 0:
            print(f"Signals: B {stats['buy_signals']} | S {stats['sell_signals']} | H {stats['hold_signals']} | Executed: {stats['trades_executed']}")
            blocked_total = stats['blocked_max_positions'] + stats['blocked_has_position'] + stats['blocked_no_position']
            if blocked_total or stats['no_quote']:
                print(f"Blocked: max_pos {stats['blocked_max_positions']}, no_pos {stats['blocked_no_position']}, no_quote {stats['no_quote']}")
        
        if self.positions:
            print("\n📊 Open Positions:")
            for symbol, pos in self.positions.items():
                pnl = (pos['current_price'] - pos['entry_price']) * pos['quantity']
                pnl_pct = (pos['current_price'] - pos['entry_price']) / pos['entry_price'] * 100
                print(f"  {symbol}: ${pos['current_price']:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
        
        print("=" * 70)


def run_strategy_comparison(duration_minutes: int = 10):
    """Run all strategies and compare performance."""
    
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON - COINBASE PAPER TRADING")
    print("=" * 70)
    
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'MATIC-USD', 'LINK-USD']
    results = {}
    
    for strategy_name in ['momentum', 'mean_reversion', 'breakout', 'ma_crossover', 'volatility']:
        print(f"\n\n🚀 Testing {strategy_name.upper()} Strategy")
        print("-" * 40)
        
        trader = AdvancedCoinbasePaperTrader(initial_capital=10000)
        
        if not trader.connect():
            print("Failed to connect")
            continue
        
        # Collect initial price history
        trader.collect_price_history(symbols, duration_seconds=30)
        
        # Select strategy
        trader.select_strategy(strategy_name)
        
        # Run for specified duration
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            trader.run_strategy_signals(symbols)
            trader.check_stops()
            time.sleep(10)  # Check every 10 seconds
        
        # Close all positions
        for symbol in list(trader.positions.keys()):
            trader.execute_trade(symbol, 'sell', 'Session End')
        
        # Record results
        final_equity = trader.get_equity()
        results[strategy_name] = {
            'final_equity': final_equity,
            'return': (final_equity - 10000) / 10000 * 100,
            'trades': len(trader.trades),
            'win_rate': len([t for t in trader.trades if t.get('pnl', 0) > 0]) / len(trader.trades) * 100 if trader.trades else 0
        }
        
        trader.display_status()
        trader.broker.disconnect()
    
    # Display comparison
    print("\n\n" + "=" * 70)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 70)
    
    for name, res in sorted(results.items(), key=lambda x: x[1]['return'], reverse=True):
        print(f"\n{name.upper()}:")
        print(f"  Return: {res['return']:+.2f}%")
        print(f"  Trades: {res['trades']}")
        print(f"  Win Rate: {res['win_rate']:.1f}%")
    
    # Save results
    results_file = Path(__file__).parent.parent / 'results' / f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Coinbase Paper Trading")
    parser.add_argument('--strategy', type=str, default='momentum', 
                      help='Strategy to use (momentum, mean_reversion, breakout, ma_crossover, volatility)')
    parser.add_argument('--duration', type=int, default=5, help='Duration in minutes')
    parser.add_argument('--compare', action='store_true', help='Compare all strategies')
    
    args = parser.parse_args()
    
    if args.compare:
        run_strategy_comparison(args.duration)
    else:
        # Run single strategy
        print(f"Running {args.strategy} strategy for {args.duration} minutes...")
        
        trader = AdvancedCoinbasePaperTrader()
        if trader.connect():
            symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'MATIC-USD']
            
            # Collect price history
            trader.collect_price_history(symbols, 30)
            
            # Select and run strategy
            trader.select_strategy(args.strategy)
            
            end_time = datetime.now() + timedelta(minutes=args.duration)
            
            try:
                while datetime.now() < end_time:
                    trader.run_strategy_signals(symbols)
                    trader.check_stops()
                    trader.display_status()
                    time.sleep(15)
            except KeyboardInterrupt:
                print("\nStopping...")
            
            # Close positions
            for symbol in list(trader.positions.keys()):
                trader.execute_trade(symbol, 'sell', 'Session End')
            
            trader.display_status()
            trader.broker.disconnect()
