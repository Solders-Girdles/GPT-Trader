#!/usr/bin/env python3
"""
Extended Paper Trading Simulation Session
Runs optimized strategies for several hours with realistic market simulation.
"""

import numpy as np
import random
import time
import json
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import threading
import signal
import sys

class RealisticMarketSimulator:
    """Simulates realistic crypto market behavior."""
    
    def __init__(self, symbol: str, initial_price: float):
        self.symbol = symbol
        self.price = initial_price
        self.time = datetime.now()
        
        # Market parameters
        self.base_volatility = 0.001  # 0.1% base volatility
        self.trend_strength = 0.0
        self.trend_duration = 0
        self.trend_counter = 0
        
        # Intraday patterns
        self.daily_pattern = self._generate_daily_pattern()
        
    def _generate_daily_pattern(self):
        """Generate realistic intraday volatility pattern."""
        # Higher volatility during US trading hours
        hours = np.arange(24)
        pattern = np.ones(24)
        
        # US market hours (9 AM - 4 PM ET = 14:00 - 21:00 UTC)
        pattern[14:21] = 1.5  # Higher volatility
        # European hours (8 AM - 4 PM GMT = 8:00 - 16:00 UTC)  
        pattern[8:16] = 1.2   # Medium volatility
        # Asian hours
        pattern[0:6] = 0.8    # Lower volatility
        pattern[22:24] = 0.8  # Lower volatility
        
        return pattern
    
    def get_current_volatility(self):
        """Get current volatility based on time of day."""
        hour = datetime.now().hour
        time_multiplier = self.daily_pattern[hour]
        return self.base_volatility * time_multiplier
    
    def update_trend(self):
        """Update market trend periodically."""
        if self.trend_counter <= 0:
            # Start new trend
            self.trend_duration = random.randint(30, 180)  # 30-180 minutes
            self.trend_strength = random.gauss(0, 0.0002)  # Small trend
            self.trend_counter = self.trend_duration
        else:
            self.trend_counter -= 1
            
        # Decay trend over time
        if self.trend_counter < self.trend_duration * 0.3:
            self.trend_strength *= 0.98
    
    def get_next_price(self):
        """Generate next realistic price."""
        self.update_trend()
        
        # Get time-adjusted volatility
        volatility = self.get_current_volatility()
        
        # Random walk with trend and occasional spikes
        base_change = np.random.normal(self.trend_strength, volatility)
        
        # Add occasional larger moves (news events, etc.)
        if random.random() < 0.02:  # 2% chance
            spike = np.random.normal(0, volatility * 5)
            base_change += spike
        
        # Apply change
        self.price *= (1 + base_change)
        
        # Add some mean reversion to prevent extreme prices
        if abs((self.price / 65000) - 1) > 0.1:  # If price deviates >10% from base
            mean_revert = -0.1 * ((self.price / 65000) - 1) * 0.01
            self.price *= (1 + mean_revert)
        
        return self.price


class ExtendedTradingSession:
    """Manages extended trading session with multiple strategies."""
    
    def __init__(self, duration_hours: int = 4):
        self.duration_hours = duration_hours
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=duration_hours)
        
        # Market simulators
        self.markets = {
            'BTC-USD': RealisticMarketSimulator('BTC-USD', 65000),
            'ETH-USD': RealisticMarketSimulator('ETH-USD', 3500),
            'SOL-USD': RealisticMarketSimulator('SOL-USD', 150)
        }
        
        # Trading results
        self.session_results = {}
        self.is_running = True
        
        # Live monitoring
        self.last_update = datetime.now()
        self.update_interval = 60  # seconds
        
    def run_strategy_session(self, strategy_name: str, symbols: list, duration_minutes: int):
        """Run a single strategy for specified duration."""
        print(f"\n{'='*70}")
        print(f"🚀 STARTING {strategy_name.upper()} STRATEGY SESSION")
        print(f"{'='*70}")
        print(f"Duration: {duration_minutes} minutes ({duration_minutes/60:.1f} hours)")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # Initialize strategy
        if strategy_name == 'momentum':
            strategy = self._create_momentum_strategy()
        elif strategy_name == 'mean_reversion':
            strategy = self._create_mean_reversion_strategy()
        elif strategy_name == 'breakout':
            strategy = self._create_breakout_strategy()
        else:
            print(f"❌ Unknown strategy: {strategy_name}")
            return
        
        # Trading state
        positions = {}
        trades = []
        cash = 10000
        initial_capital = 10000
        
        # Price history for each symbol
        price_history = {symbol: deque(maxlen=100) for symbol in symbols}
        
        # Session tracking
        session_start = datetime.now()
        session_end = session_start + timedelta(minutes=duration_minutes)
        last_status_update = session_start
        trade_count = 0
        
        print(f"\n📊 Session started at {session_start.strftime('%H:%M:%S')}")
        print(f"📊 Will end at {session_end.strftime('%H:%M:%S')}")
        print(f"📊 Collecting initial price history...")
        
        # Main trading loop
        iteration = 0
        while datetime.now() < session_end and self.is_running:
            iteration += 1
            current_time = datetime.now()
            
            # Update all market prices
            for symbol in symbols:
                price = self.markets[symbol].get_next_price()
                price_history[symbol].append(price)
                
                # Need minimum history for signals
                if len(price_history[symbol]) < strategy['min_history']:
                    continue
                
                # Generate signal
                signal = strategy['get_signal'](symbol, price, price_history[symbol])
                
                # Execute trades based on signal
                if signal == 'buy' and symbol not in positions and len(positions) < 3:
                    # Buy signal
                    position_value = cash * 0.3  # 30% of capital
                    if cash >= position_value:
                        commission = position_value * 0.006
                        net_value = position_value - commission
                        quantity = net_value / price
                        
                        positions[symbol] = {
                            'quantity': quantity,
                            'entry_price': price,
                            'entry_time': current_time
                        }
                        cash -= position_value
                        
                        trade = {
                            'timestamp': current_time,
                            'symbol': symbol,
                            'side': 'buy',
                            'price': price,
                            'quantity': quantity,
                            'value': position_value,
                            'commission': commission
                        }
                        trades.append(trade)
                        trade_count += 1
                        
                        print(f"[{current_time.strftime('%H:%M:%S')}] ✅ BUY {symbol}: {quantity:.6f} @ ${price:.2f}")
                
                elif signal == 'sell' and symbol in positions:
                    # Sell signal
                    pos = positions[symbol]
                    value = pos['quantity'] * price
                    commission = value * 0.006
                    net_value = value - commission
                    pnl = net_value - (pos['entry_price'] * pos['quantity'])
                    pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
                    
                    cash += net_value
                    del positions[symbol]
                    
                    trade = {
                        'timestamp': current_time,
                        'symbol': symbol,
                        'side': 'sell',
                        'price': price,
                        'quantity': pos['quantity'],
                        'value': value,
                        'commission': commission,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    }
                    trades.append(trade)
                    trade_count += 1
                    
                    emoji = "🟢" if pnl > 0 else "🔴"
                    print(f"[{current_time.strftime('%H:%M:%S')}] {emoji} SELL {symbol}: {pos['quantity']:.6f} @ ${price:.2f} | P&L: {pnl_pct:+.2f}%")
            
            # Periodic status updates
            if (current_time - last_status_update).seconds >= 600:  # Every 10 minutes
                elapsed_minutes = (current_time - session_start).seconds / 60
                remaining_minutes = (session_end - current_time).seconds / 60
                
                # Calculate current equity
                equity = cash
                for symbol, pos in positions.items():
                    current_price = self.markets[symbol].price
                    equity += pos['quantity'] * current_price
                
                total_return = (equity - initial_capital) / initial_capital * 100
                trades_per_hour = trade_count / (elapsed_minutes / 60) if elapsed_minutes > 0 else 0
                
                print(f"\n📊 [{current_time.strftime('%H:%M:%S')}] STATUS UPDATE:")
                print(f"    Elapsed: {elapsed_minutes:.0f}m | Remaining: {remaining_minutes:.0f}m")
                print(f"    Equity: ${equity:.2f} ({total_return:+.2f}%)")
                print(f"    Trades: {trade_count} ({trades_per_hour:.1f}/hour)")
                print(f"    Positions: {len(positions)}")
                
                last_status_update = current_time
            
            # Sleep to simulate real-time trading
            time.sleep(2)  # 2-second intervals
        
        # Close all positions at end
        print(f"\n📊 Session ending - closing all positions...")
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            price = self.markets[symbol].price
            value = pos['quantity'] * price
            commission = value * 0.006
            net_value = value - commission
            pnl = net_value - (pos['entry_price'] * pos['quantity'])
            pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
            
            cash += net_value
            del positions[symbol]
            
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'sell',
                'price': price,
                'quantity': pos['quantity'],
                'value': value,
                'commission': commission,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reason': 'session_end'
            }
            trades.append(trade)
            
            emoji = "🟢" if pnl > 0 else "🔴"
            print(f"[SESSION END] {emoji} SELL {symbol}: {pos['quantity']:.6f} @ ${price:.2f} | P&L: {pnl_pct:+.2f}%")
        
        # Final calculations
        final_equity = cash
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Trade statistics
        buy_trades = [t for t in trades if t['side'] == 'buy']
        sell_trades = [t for t in trades if t['side'] == 'sell' and 'pnl' in t]
        winning_trades = [t for t in sell_trades if t['pnl'] > 0]
        
        win_rate = len(winning_trades) / len(sell_trades) * 100 if sell_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in sell_trades if t['pnl'] < 0]) if sell_trades else 0
        
        # Session duration
        actual_duration = (datetime.now() - session_start).seconds / 60
        trades_per_hour = len(buy_trades) / (actual_duration / 60) if actual_duration > 0 else 0
        
        # Final report
        print(f"\n{'='*70}")
        print(f"📊 {strategy_name.upper()} STRATEGY SESSION COMPLETE")
        print(f"{'='*70}")
        print(f"Duration: {actual_duration:.1f} minutes")
        print(f"Initial Capital: ${initial_capital:.2f}")
        print(f"Final Equity: ${final_equity:.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {len(buy_trades)} (buy signals)")
        print(f"Trades per Hour: {trades_per_hour:.1f}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"{'='*70}")
        
        # Store results
        result = {
            'strategy': strategy_name,
            'symbols': symbols,
            'duration_minutes': actual_duration,
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_trades': len(buy_trades),
            'trades_per_hour': trades_per_hour,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': trades,
            'session_start': session_start.isoformat(),
            'session_end': datetime.now().isoformat()
        }
        
        self.session_results[strategy_name] = result
        return result
    
    def _create_momentum_strategy(self):
        """Create optimized momentum strategy."""
        def get_signal(symbol, price, price_history):
            if len(price_history) < 5:
                return 'hold'
            
            prices = list(price_history)
            old_price = prices[-5]
            momentum = (price - old_price) / old_price
            
            # Adaptive threshold
            recent_returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(max(1, len(prices)-20), len(prices))]
            if recent_returns:
                volatility = np.std(recent_returns)
                threshold = max(0.005, min(0.02, volatility * 2))  # 0.5% to 2%
            else:
                threshold = 0.01
            
            if momentum > threshold:
                return 'buy'
            elif momentum < -threshold:
                return 'sell'
            return 'hold'
        
        return {
            'get_signal': get_signal,
            'min_history': 5
        }
    
    def _create_mean_reversion_strategy(self):
        """Create optimized mean reversion strategy."""
        def get_signal(symbol, price, price_history):
            if len(price_history) < 10:
                return 'hold'
            
            prices = np.array(list(price_history))
            mean = np.mean(prices[-10:])
            std = np.std(prices[-10:])
            
            upper_band = mean + (1.5 * std)
            lower_band = mean - (1.5 * std)
            
            if price < lower_band:
                return 'buy'
            elif price > upper_band:
                return 'sell'
            return 'hold'
        
        return {
            'get_signal': get_signal,
            'min_history': 10
        }
    
    def _create_breakout_strategy(self):
        """Create optimized breakout strategy."""
        def get_signal(symbol, price, price_history):
            if len(price_history) < 10:
                return 'hold'
            
            prices = list(price_history)
            recent_prices = prices[-10:]
            recent_high = max(recent_prices[:-1])
            recent_low = min(recent_prices[:-1])
            
            # Dynamic threshold
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            threshold = max(0.003, min(0.01, volatility))
            
            if price > recent_high * (1 + threshold):
                return 'buy'
            elif price < recent_low * (1 - threshold):
                return 'sell'
            return 'hold'
        
        return {
            'get_signal': get_signal,
            'min_history': 10
        }
    
    def run_sequential_sessions(self):
        """Run multiple strategy sessions sequentially."""
        strategies = [
            ('momentum', ['BTC-USD', 'ETH-USD'], 120),      # 2 hours
            ('mean_reversion', ['BTC-USD', 'ETH-USD'], 120), # 2 hours  
            ('breakout', ['BTC-USD', 'ETH-USD', 'SOL-USD'], 90) # 1.5 hours
        ]
        
        total_duration = sum(duration for _, _, duration in strategies) / 60
        
        print(f"\n🚀 STARTING EXTENDED PAPER TRADING SESSION")
        print(f"{'='*70}")
        print(f"Total Duration: {total_duration:.1f} hours")
        print(f"Strategies: {len(strategies)}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        for strategy_name, symbols, duration_minutes in strategies:
            if not self.is_running:
                break
                
            result = self.run_strategy_session(strategy_name, symbols, duration_minutes)
            
            # Short break between strategies
            if self.is_running:
                print(f"\n⏸️ 5-minute break before next strategy...")
                time.sleep(300)  # 5 minutes
        
        # Final summary
        self.print_final_summary()
        self.save_session_results()
    
    def print_final_summary(self):
        """Print final summary of all sessions."""
        if not self.session_results:
            return
        
        print(f"\n{'='*70}")
        print(f"📊 EXTENDED SESSION FINAL SUMMARY")
        print(f"{'='*70}")
        
        total_trades = 0
        total_return = 0
        
        print(f"{'Strategy':<15} {'Return':<10} {'Trades':<8} {'Rate/Hr':<8} {'Win%':<8}")
        print("-" * 55)
        
        for strategy_name, result in self.session_results.items():
            total_trades += result['total_trades']
            total_return += result['total_return']
            
            print(f"{strategy_name:<15} {result['total_return']:>+7.2f}% {result['total_trades']:>7} "
                  f"{result['trades_per_hour']:>7.1f} {result['win_rate']:>6.1f}%")
        
        avg_return = total_return / len(self.session_results)
        
        print("-" * 55)
        print(f"{'AVERAGE':<15} {avg_return:>+7.2f}% {total_trades:>7} {'':>7} {'':>6}")
        print(f"\n✅ Extended session complete!")
        print(f"✅ Total trades generated: {total_trades}")
        print(f"✅ Average return per strategy: {avg_return:+.2f}%")
    
    def save_session_results(self):
        """Save session results to file."""
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        session_data = {
            'session_type': 'extended_simulation',
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_hours': self.duration_hours,
            'strategies': self.session_results
        }
        
        filename = f"extended_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"\n💾 Session results saved to {filename}")
        return filepath
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print(f"\n\n🛑 Received interrupt signal. Gracefully stopping session...")
        self.is_running = False


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("🚀 EXTENDED PAPER TRADING SESSION")
    print("="*70)
    print("This will run optimized strategies for several hours")
    print("with realistic market simulation.")
    print("="*70)
    
    # Setup signal handler for graceful shutdown
    session = ExtendedTradingSession(duration_hours=5.5)
    signal.signal(signal.SIGINT, session.signal_handler)
    
    try:
        session.run_sequential_sessions()
    except KeyboardInterrupt:
        print(f"\n🛑 Session interrupted by user")
        session.print_final_summary()
        session.save_session_results()
    
    print(f"\n👋 Session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()