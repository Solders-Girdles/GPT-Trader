#!/usr/bin/env python3
"""
Optimized Paper Trading Strategies for Coinbase
With adjusted parameters for more frequent trading and better performance.
"""

import os
import sys
import time
import json
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

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
                    # Handle multi-line private key
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


class OptimizedStrategy:
    """Base strategy class with optimized parameters."""
    
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params
        self.price_history = {}
        self.indicators = {}
        self.last_signal = {}
        self.last_trade_time = {}
        
        # Optimized lookback periods (reduced for faster signals)
        self.lookback = params.get('lookback', 50)
        
    def update_price(self, symbol: str, price: float):
        """Update price history."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback)
        self.price_history[symbol].append(price)
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Override in subclasses."""
        return 'hold'
    
    def should_force_trade(self, symbol: str) -> bool:
        """Force a trade if no trades in last hour."""
        if symbol not in self.last_trade_time:
            return True
        
        time_since_trade = (datetime.now() - self.last_trade_time[symbol]).seconds
        return time_since_trade > 3600  # Force trade after 1 hour


class OptimizedMomentumStrategy(OptimizedStrategy):
    """Momentum strategy with aggressive parameters."""
    
    def __init__(self, **params):
        super().__init__('OptimizedMomentum', **params)
        # REDUCED thresholds for more frequent trading
        self.momentum_period = params.get('momentum_period', 5)  # Reduced from 10
        self.threshold = params.get('threshold', 0.01)  # Reduced from 0.02
        self.volume_multiplier = params.get('volume_multiplier', 1.2)
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate momentum signal with adaptive thresholds."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = list(self.price_history[symbol])
        if len(prices) < self.momentum_period:
            return 'hold'
        
        # Calculate momentum
        old_price = prices[-self.momentum_period]
        momentum = (current_price - old_price) / old_price
        
        # Adaptive threshold based on recent volatility
        recent_returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(max(1, len(prices)-20), len(prices))]
        if recent_returns:
            volatility = np.std(recent_returns)
            adaptive_threshold = max(self.threshold * 0.5, min(self.threshold * 2, volatility * 2))
        else:
            adaptive_threshold = self.threshold
        
        # Check if we should force a trade
        if self.should_force_trade(symbol):
            if momentum > 0:
                self.last_trade_time[symbol] = datetime.now()
                return 'buy'
            else:
                self.last_trade_time[symbol] = datetime.now()
                return 'sell'
        
        # Normal momentum signals with adaptive threshold
        if momentum > adaptive_threshold:
            self.last_trade_time[symbol] = datetime.now()
            return 'buy'
        elif momentum < -adaptive_threshold:
            self.last_trade_time[symbol] = datetime.now()
            return 'sell'
        return 'hold'


class OptimizedMeanReversionStrategy(OptimizedStrategy):
    """Mean reversion with tighter bands."""
    
    def __init__(self, **params):
        super().__init__('OptimizedMeanReversion', **params)
        self.bb_period = params.get('bb_period', 10)  # Reduced from 20
        self.bb_std = params.get('bb_std', 1.5)  # Reduced from 2
        self.rsi_period = params.get('rsi_period', 7)  # Added RSI
        self.rsi_oversold = params.get('rsi_oversold', 35)  # More aggressive
        self.rsi_overbought = params.get('rsi_overbought', 65)  # More aggressive
    
    def calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI indicator."""
        if len(prices) < self.rsi_period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate mean reversion signal with RSI confirmation."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = np.array(list(self.price_history[symbol]))
        if len(prices) < self.bb_period:
            return 'hold'
        
        # Calculate Bollinger Bands with tighter bands
        mean = np.mean(prices[-self.bb_period:])
        std = np.std(prices[-self.bb_period:])
        upper_band = mean + (self.bb_std * std)
        lower_band = mean - (self.bb_std * std)
        
        # Calculate RSI
        rsi = self.calculate_rsi(prices)
        
        # Combined signals
        if current_price < lower_band and rsi < self.rsi_oversold:
            self.last_trade_time[symbol] = datetime.now()
            return 'buy'
        elif current_price > upper_band and rsi > self.rsi_overbought:
            self.last_trade_time[symbol] = datetime.now()
            return 'sell'
        
        # Relaxed conditions if no recent trades
        if self.should_force_trade(symbol):
            if current_price < mean and rsi < 50:
                self.last_trade_time[symbol] = datetime.now()
                return 'buy'
            elif current_price > mean and rsi > 50:
                self.last_trade_time[symbol] = datetime.now()
                return 'sell'
        
        return 'hold'


class OptimizedBreakoutStrategy(OptimizedStrategy):
    """Breakout strategy with shorter periods."""
    
    def __init__(self, **params):
        super().__init__('OptimizedBreakout', **params)
        self.breakout_period = params.get('breakout_period', 10)  # Reduced from 20
        self.breakout_threshold = params.get('breakout_threshold', 0.005)  # Reduced from 0.01
        self.volume_confirmation = params.get('volume_confirmation', False)
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate breakout signal with lower thresholds."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = list(self.price_history[symbol])
        if len(prices) < self.breakout_period:
            return 'hold'
        
        # Use more recent data for faster signals
        recent_prices = prices[-self.breakout_period:]
        recent_high = max(recent_prices[:-1])  # Exclude current price
        recent_low = min(recent_prices[:-1])
        
        # Calculate range
        price_range = recent_high - recent_low
        if price_range == 0:
            return 'hold'
        
        # Dynamic threshold based on volatility
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        dynamic_threshold = max(self.breakout_threshold * 0.5, min(self.breakout_threshold * 2, volatility))
        
        # Breakout signals with dynamic threshold
        if current_price > recent_high * (1 + dynamic_threshold):
            self.last_trade_time[symbol] = datetime.now()
            return 'buy'
        elif current_price < recent_low * (1 - dynamic_threshold):
            self.last_trade_time[symbol] = datetime.now()
            return 'sell'
        
        # Force trade if needed
        if self.should_force_trade(symbol):
            position_in_range = (current_price - recent_low) / price_range
            if position_in_range > 0.7:  # Near top of range
                self.last_trade_time[symbol] = datetime.now()
                return 'sell'
            elif position_in_range < 0.3:  # Near bottom of range
                self.last_trade_time[symbol] = datetime.now()
                return 'buy'
        
        return 'hold'


class OptimizedMAStrategy(OptimizedStrategy):
    """MA crossover with faster periods."""
    
    def __init__(self, **params):
        super().__init__('OptimizedMAStrategy', **params)
        self.fast_period = params.get('fast_period', 5)  # Reduced from 10
        self.slow_period = params.get('slow_period', 15)  # Reduced from 30
        self.signal_smoothing = params.get('signal_smoothing', 3)
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate MA crossover signal with momentum confirmation."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = np.array(list(self.price_history[symbol]))
        if len(prices) < self.slow_period + 1:
            return 'hold'
        
        # Calculate moving averages
        fast_ma = np.mean(prices[-self.fast_period:])
        slow_ma = np.mean(prices[-self.slow_period:])
        
        # Calculate previous MAs
        prev_fast_ma = np.mean(prices[-self.fast_period-1:-1])
        prev_slow_ma = np.mean(prices[-self.slow_period-1:-1])
        
        # Add momentum filter
        momentum = (current_price - prices[-5]) / prices[-5] if len(prices) > 5 else 0
        
        # Crossover signals
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma and momentum > 0:
            self.last_trade_time[symbol] = datetime.now()
            return 'buy'
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma and momentum < 0:
            self.last_trade_time[symbol] = datetime.now()
            return 'sell'
        
        # Trend following if no crossover
        if self.should_force_trade(symbol):
            if fast_ma > slow_ma * 1.005:  # 0.5% above
                self.last_trade_time[symbol] = datetime.now()
                return 'buy'
            elif fast_ma < slow_ma * 0.995:  # 0.5% below
                self.last_trade_time[symbol] = datetime.now()
                return 'sell'
        
        return 'hold'


class OptimizedVolatilityStrategy(OptimizedStrategy):
    """Volatility-based strategy with adaptive positioning."""
    
    def __init__(self, **params):
        super().__init__('OptimizedVolatility', **params)
        self.atr_period = params.get('atr_period', 7)  # Reduced from 14
        self.vol_threshold = params.get('vol_threshold', 0.3)  # Percentile threshold
        self.trend_period = params.get('trend_period', 10)
    
    def calculate_atr(self, prices: np.ndarray) -> float:
        """Calculate Average True Range."""
        if len(prices) < 2:
            return 0.0
        
        ranges = []
        for i in range(1, min(len(prices), self.atr_period + 1)):
            high_low = abs(prices[i] - prices[i-1])
            ranges.append(high_low)
        
        return np.mean(ranges) if ranges else 0.0
    
    def get_signal(self, symbol: str, current_price: float) -> str:
        """Generate volatility-based signal."""
        if symbol not in self.price_history:
            return 'hold'
        
        prices = np.array(list(self.price_history[symbol]))
        if len(prices) < self.atr_period:
            return 'hold'
        
        # Calculate ATR
        atr = self.calculate_atr(prices)
        price_mean = np.mean(prices[-self.atr_period:])
        volatility = atr / price_mean if price_mean > 0 else 0
        
        # Calculate trend
        if len(prices) >= self.trend_period:
            old_price = prices[-self.trend_period]
            trend = (current_price - old_price) / old_price
        else:
            trend = 0
        
        # Low volatility - trend following
        if volatility < 0.01:  # Low volatility threshold
            if trend > 0.005:
                self.last_trade_time[symbol] = datetime.now()
                return 'buy'
            elif trend < -0.005:
                self.last_trade_time[symbol] = datetime.now()
                return 'sell'
        
        # High volatility - mean reversion
        elif volatility > 0.02:  # High volatility threshold
            mean = np.mean(prices[-self.atr_period:])
            if current_price < mean * 0.98:
                self.last_trade_time[symbol] = datetime.now()
                return 'buy'
            elif current_price > mean * 1.02:
                self.last_trade_time[symbol] = datetime.now()
                return 'sell'
        
        # Force trade if needed
        if self.should_force_trade(symbol):
            if trend > 0:
                self.last_trade_time[symbol] = datetime.now()
                return 'buy'
            else:
                self.last_trade_time[symbol] = datetime.now()
                return 'sell'
        
        return 'hold'


class OptimizedCoinbasePaperTrader:
    """Enhanced paper trader with optimized strategies."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.strategies = {
            'momentum': OptimizedMomentumStrategy(),
            'mean_reversion': OptimizedMeanReversionStrategy(),
            'breakout': OptimizedBreakoutStrategy(),
            'ma_crossover': OptimizedMAStrategy(),
            'volatility': OptimizedVolatilityStrategy()
        }
        self.active_strategy = None
        self.broker = None
        
        # Risk parameters (more aggressive)
        self.max_positions = 3
        self.max_position_size = 0.4  # Increased from 0.3
        self.stop_loss = 0.03  # Reduced from 0.05
        self.take_profit = 0.05  # Reduced from 0.10
        self.commission_rate = 0.006
        
        # Performance tracking
        self.price_history = {}
        self.start_time = datetime.now()
    
    def connect(self) -> bool:
        """Connect to Coinbase."""
        try:
            config = APIConfig(
                api_key="",
                api_secret="",
                passphrase=None,
                base_url="https://api.coinbase.com",
                cdp_api_key=os.environ.get('COINBASE_CDP_API_KEY_NAME'),
                cdp_private_key=os.environ.get('COINBASE_CDP_PRIVATE_KEY')
            )
            self.broker = CoinbaseBrokerage(config)
            
            if self.broker.connect():
                print("✅ Connected to Coinbase")
                return True
            else:
                print("❌ Failed to connect to Coinbase")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def collect_price_history(self, symbols: List[str], duration_seconds: int = 20):
        """Collect initial price history."""
        print(f"📊 Collecting {duration_seconds}s of price history...")
        end_time = datetime.now() + timedelta(seconds=duration_seconds)
        
        while datetime.now() < end_time:
            for symbol in symbols:
                quote = self.broker.get_quote(symbol)
                if quote:
                    price = float(quote.mid)
                    
                    # Update all strategies
                    for strategy in self.strategies.values():
                        strategy.update_price(symbol, price)
                    
                    # Store for our own tracking
                    if symbol not in self.price_history:
                        self.price_history[symbol] = deque(maxlen=100)
                    self.price_history[symbol].append(price)
            
            time.sleep(1)
        
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
            price = float(quote.ask) * (1 + 0.001)  # Add slippage
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
            price = float(quote.bid) * (1 - 0.001)  # Subtract slippage
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
        
        for symbol in symbols:
            # Get current price
            quote = self.broker.get_quote(symbol)
            if not quote:
                continue
            
            current_price = float(quote.mid)
            
            # Update strategy price history
            self.active_strategy.update_price(symbol, current_price)
            
            # Get signal
            signal = self.active_strategy.get_signal(symbol, current_price)
            
            # Execute based on signal
            if signal == 'buy' and symbol not in self.positions:
                self.execute_trade(symbol, 'buy', f"{self.active_strategy.name} Signal")
            elif signal == 'sell' and symbol in self.positions:
                self.execute_trade(symbol, 'sell', f"{self.active_strategy.name} Signal")
    
    def get_equity(self) -> float:
        """Calculate total equity."""
        equity = self.cash
        
        for symbol, pos in self.positions.items():
            quote = self.broker.get_quote(symbol)
            if quote:
                current_price = float(quote.bid)
                equity += pos['quantity'] * current_price
        
        return equity
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        current_equity = self.get_equity()
        total_return = (current_equity - self.initial_capital) / self.initial_capital * 100
        
        # Win rate
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        total_closed = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_closed * 100 if total_closed > 0 else 0
        
        # Average trade
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Trading frequency
        duration_hours = (datetime.now() - self.start_time).seconds / 3600
        trades_per_hour = len(self.trades) / duration_hours if duration_hours > 0 else 0
        
        return {
            'equity': current_equity,
            'total_return': total_return,
            'num_trades': len([t for t in self.trades if t['side'] == 'buy']),
            'num_positions': len(self.positions),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades_per_hour': trades_per_hour,
            'cash': self.cash,
            'positions_value': current_equity - self.cash
        }


def run_optimized_strategy(strategy_name: str, symbols: List[str], duration_minutes: int = 30):
    """Run a single optimized strategy."""
    print(f"\n{'='*70}")
    print(f"OPTIMIZED PAPER TRADING - {strategy_name.upper()}")
    print(f"{'='*70}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    trader = OptimizedCoinbasePaperTrader()
    
    # Connect to Coinbase
    if not trader.connect():
        return None
    
    # Collect initial price history
    trader.collect_price_history(symbols, duration_seconds=20)
    
    # Select strategy
    trader.select_strategy(strategy_name)
    
    # Run for specified duration
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    update_interval = 5  # seconds
    display_interval = 60  # seconds
    last_display = datetime.now()
    
    print(f"\n📊 Starting {strategy_name} strategy...\n")
    
    while datetime.now() < end_time:
        # Run strategy signals
        trader.run_strategy_signals(symbols)
        
        # Check stops
        trader.check_stops()
        
        # Display metrics periodically
        if (datetime.now() - last_display).seconds >= display_interval:
            metrics = trader.get_performance_metrics()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Performance Update:")
            print(f"  Equity: ${metrics['equity']:.2f} ({metrics['total_return']:+.2f}%)")
            print(f"  Trades: {metrics['num_trades']} | Positions: {metrics['num_positions']}")
            print(f"  Win Rate: {metrics['win_rate']:.1f}% | Trades/Hour: {metrics['trades_per_hour']:.1f}")
            last_display = datetime.now()
        
        time.sleep(update_interval)
    
    # Close all positions
    print("\n📊 Closing all positions...")
    for symbol in list(trader.positions.keys()):
        trader.execute_trade(symbol, 'sell', 'Session End')
    
    # Final metrics
    metrics = trader.get_performance_metrics()
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {strategy_name.upper()}")
    print(f"{'='*70}")
    print(f"Initial Capital: ${trader.initial_capital:.2f}")
    print(f"Final Equity: ${metrics['equity']:.2f}")
    print(f"Total Return: {metrics['total_return']:+.2f}%")
    print(f"Total Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Average Win: ${metrics['avg_win']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"Trades per Hour: {metrics['trades_per_hour']:.1f}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    result_data = {
        'strategy': strategy_name,
        'symbols': symbols,
        'duration_minutes': duration_minutes,
        'metrics': metrics,
        'trades': [
            {
                'timestamp': t['timestamp'].isoformat(),
                'symbol': t['symbol'],
                'side': t['side'],
                'price': t['price'],
                'quantity': t['quantity'],
                'pnl': t.get('pnl', 0),
                'reason': t['reason']
            }
            for t in trader.trades
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"optimized_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_dir / filename, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")
    
    # Disconnect
    trader.broker.disconnect()
    
    return metrics


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Paper Trading with Coinbase")
    parser.add_argument('--strategy', type=str, default='momentum',
                       choices=['momentum', 'mean_reversion', 'breakout', 'ma_crossover', 'volatility'],
                       help='Trading strategy to use')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration in minutes')
    parser.add_argument('--symbols', type=str, default='BTC-USD,ETH-USD,SOL-USD',
                       help='Comma-separated symbols to trade')
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(',')
    
    # Run the optimized strategy
    run_optimized_strategy(args.strategy, symbols, args.duration)


if __name__ == "__main__":
    main()