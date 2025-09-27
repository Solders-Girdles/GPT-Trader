#!/usr/bin/env python3
"""
Live Paper Trading with Real-Time Updates
Runs paper trading with continuous file updates for live monitoring.
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

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


class LivePaperTrader(AdvancedCoinbasePaperTrader):
    """Extended paper trader with live monitoring support."""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        session_name: str | None = None,
        signal_mode: str = 'conservative',
        # Risk knobs
        max_position_size: float = 0.2,
        max_positions: int = 5,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
        commission_rate: float = 0.006,
        slippage_rate: float = 0.001,
        # Strategy overrides
        strategy_overrides: dict | None = None,
    ):
        super().__init__(
            initial_capital,
            signal_mode=signal_mode,
            max_position_size=max_position_size,
            max_positions=max_positions,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            strategy_overrides=strategy_overrides,
        )
        
        # Session tracking
        self.session_name = session_name or f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_file = Path(__file__).parent.parent / 'results' / f"{self.session_name}.json"
        self.session_file.parent.mkdir(exist_ok=True)
        
        # Update frequency
        self.last_update = datetime.now()
        self.update_interval = 2  # seconds
        
        # Session data
        self.session_data = {
            'session_name': self.session_name,
            'start_time': datetime.now().isoformat(),
            'initial_capital': initial_capital,
            'strategy': None,
            'symbols': [],
            'metrics': {},
            'signal_stats': {},
            'trades': [],
            'positions': {},
            'equity_history': [],
            'config': {
                'signal_mode': signal_mode,
                'max_position_size': max_position_size,
                'max_positions': max_positions,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'commission_rate': commission_rate,
                'slippage_rate': slippage_rate,
                'strategy_overrides': strategy_overrides or {},
            }
        }
        
        # Save initial state
        self.save_session()
    
    def save_session(self):
        """Save current session data to file."""
        try:
            # Update metrics
            self.session_data['metrics'] = self.get_current_metrics()
            # Update signal diagnostics if available
            if hasattr(self, 'signal_stats'):
                self.session_data['signal_stats'] = dict(self.signal_stats)
            
            # Update positions with current prices
            self.session_data['positions'] = {}
            for symbol, pos in self.positions.items():
                # Get current price
                quote = self.broker.get_quote(symbol)
                if quote:
                    current_price = float(quote.bid)
                else:
                    current_price = pos.get('current_price', pos['entry_price'])
                
                self.session_data['positions'][symbol] = {
                    'quantity': pos['quantity'],
                    'entry_price': pos['entry_price'],
                    'current_price': current_price,
                    'value': pos['quantity'] * current_price,
                    'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else pos['entry_time'],
                    'strategy': pos.get('strategy', 'unknown')
                }
            
            # Update trades (keep last 50)
            self.session_data['trades'] = []
            for trade in self.trades[-50:]:
                trade_data = trade.copy()
                if 'timestamp' in trade_data and isinstance(trade_data['timestamp'], datetime):
                    trade_data['timestamp'] = trade_data['timestamp'].isoformat()
                self.session_data['trades'].append(trade_data)
            
            # Add equity point
            current_equity = self.get_equity()
            self.session_data['equity_history'].append({
                'timestamp': datetime.now().isoformat(),
                'equity': current_equity
            })
            
            # Keep only last 100 equity points
            if len(self.session_data['equity_history']) > 100:
                self.session_data['equity_history'] = self.session_data['equity_history'][-100:]
            
            # Write to file
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving session: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics."""
        equity = self.get_equity()
        returns = (equity - self.initial_capital) / self.initial_capital * 100
        
        # Calculate drawdown
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        
        # Calculate win rate
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        # Average P&L
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        avg_pnl = total_pnl / len(self.trades) if self.trades else 0
        
        # Rates
        elapsed_hours = max((datetime.now() - self.start_time).total_seconds() / 3600.0, 1e-6)
        signals_total = self.signal_stats.get('buy_signals', 0) + self.signal_stats.get('sell_signals', 0)
        trades_per_hour = len(self.trades) / elapsed_hours
        signals_per_hour = signals_total / elapsed_hours
        exec_rate = (self.signal_stats.get('trades_executed', 0) / signals_total * 100.0) if signals_total else 0.0

        return {
            'equity': equity,
            'cash': self.cash,
            'positions_value': equity - self.cash,
            'total_return': returns,
            'drawdown': drawdown,
            'num_trades': len(self.trades),
            'num_positions': len(self.positions),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'duration': str(datetime.now() - self.start_time).split('.')[0],
            'trades_per_hour': trades_per_hour,
            'signals_per_hour': signals_per_hour,
            'execution_rate': exec_rate
        }
    
    def execute_trade(self, symbol: str, side: str, reason: str = ""):
        """Override to save after each trade."""
        result = super().execute_trade(symbol, side, reason)
        
        # Save session after trade
        if result:
            self.save_session()
        
        return result
    
    def check_and_update(self):
        """Check if it's time to update the session file."""
        now = datetime.now()
        if (now - self.last_update).total_seconds() >= self.update_interval:
            self.save_session()
            self.last_update = now
    
    def run_live_session(self, strategy_name: str, symbols: List[str], duration_minutes: int = 60, collect_seconds: int = 30, loop_sleep: int = 10):
        """Run a live paper trading session with monitoring."""
        
        print("=" * 70)
        print("LIVE PAPER TRADING SESSION")
        print("=" * 70)
        print(f"Session: {self.session_name}")
        print(f"Strategy: {strategy_name}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Monitor file: {self.session_file.name}")
        print("=" * 70)
        print("\n📊 Open the monitor in another terminal:")
        print("   python scripts/live_monitor.py")
        print("\n" + "=" * 70)
        
        # Update session data
        self.session_data['strategy'] = strategy_name
        self.session_data['symbols'] = symbols
        
        # Connect
        if not self.connect():
            print("Failed to connect to Coinbase")
            return
        
        print("✅ Connected to Coinbase")
        
        # Collect initial price history
        print("📊 Collecting initial price history...")
        self.collect_price_history(symbols, duration_seconds=collect_seconds)
        
        # Select strategy
        self.select_strategy(strategy_name)
        
        # Main trading loop
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        iteration = 0
        
        print(f"\n🚀 Starting live session until {end_time.strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop early\n")
        
        try:
            while datetime.now() < end_time:
                iteration += 1
                
                # Run strategy
                self.run_strategy_signals(symbols)
                
                # Check stops
                self.check_stops()
                
                # Update session file
                self.check_and_update()
                
                # Display brief status every 30 seconds
                if iteration % 3 == 0:
                    metrics = self.get_current_metrics()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Equity: ${metrics['equity']:.2f} ({metrics['total_return']:+.2f}%) | "
                          f"Positions: {metrics['num_positions']} | "
                          f"Trades: {metrics['num_trades']}")
                
                # Wait before next iteration
                time.sleep(loop_sleep)
                
        except KeyboardInterrupt:
            print("\n⚠️ Session interrupted by user")
        
        # Close all positions
        print("\n📊 Closing all positions...")
        for symbol in list(self.positions.keys()):
            self.execute_trade(symbol, 'sell', 'Session end')
        
        # Final update
        self.session_data['end_time'] = datetime.now().isoformat()
        self.save_session()
        
        # Display final results
        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        self.display_status()
        
        # Disconnect
        self.broker.disconnect()
        print("\n✅ Session complete. Monitor file: " + self.session_file.name)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Paper Trading with Monitoring")
    parser.add_argument('--strategy', type=str, default='momentum',
                      choices=['momentum', 'mean_reversion', 'breakout', 'ma_crossover', 'volatility', 'scalp'],
                      help='Strategy to use')
    parser.add_argument('--duration', type=int, default=30,
                      help='Duration in minutes (default: 30)')
    parser.add_argument('--capital', type=float, default=10000,
                      help='Initial capital (default: 10000)')
    parser.add_argument('--symbols', type=str, default='BTC-USD,ETH-USD,SOL-USD',
                      help='Comma-separated symbols (default: BTC-USD,ETH-USD,SOL-USD)')
    parser.add_argument('--mode', type=str, default='conservative', choices=['conservative','balanced','aggressive'],
                      help='Signal aggressiveness preset (default: conservative)')
    # Risk knobs
    parser.add_argument('--max-positions', type=int, default=5, help='Max concurrent positions')
    parser.add_argument('--max-position-size', type=float, default=0.2, help='Max fraction of equity per position (0-1)')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='Stop loss threshold (fraction)')
    parser.add_argument('--take-profit', type=float, default=0.10, help='Take profit threshold (fraction)')
    parser.add_argument('--commission', type=float, default=0.006, help='Commission rate (fraction)')
    parser.add_argument('--slippage', type=float, default=0.001, help='Slippage rate (fraction)')
    # Strategy overrides
    parser.add_argument('--momentum-threshold', type=float, default=None, help='Momentum threshold (fraction)')
    parser.add_argument('--ma-fast', type=int, default=None, help='MA fast period')
    parser.add_argument('--ma-slow', type=int, default=None, help='MA slow period')
    parser.add_argument('--bb-std', type=float, default=None, help='Bollinger Bands std dev')
    parser.add_argument('--breakout-threshold', type=float, default=None, help='Breakout threshold percent (fraction)')
    parser.add_argument('--vol-threshold', type=float, default=None, help='Volatility threshold for signals')
    # Loop timing
    parser.add_argument('--collect-seconds', type=int, default=30, help='Initial price history collection seconds')
    parser.add_argument('--loop-sleep', type=int, default=10, help='Seconds between iterations')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Strategy override map
    overrides = {
        'momentum': {},
        'ma_crossover': {},
        'mean_reversion': {},
        'breakout': {},
        'volatility': {},
    }
    if args.momentum_threshold is not None:
        overrides['momentum']['threshold'] = args.momentum_threshold
    if args.ma_fast is not None:
        overrides['ma_crossover']['fast_period'] = args.ma_fast
    if args.ma_slow is not None:
        overrides['ma_crossover']['slow_period'] = args.ma_slow
    if args.bb_std is not None:
        overrides['mean_reversion']['bb_std'] = args.bb_std
    if args.breakout_threshold is not None:
        overrides['breakout']['threshold_pct'] = args.breakout_threshold
    if args.vol_threshold is not None:
        overrides['volatility']['vol_threshold'] = args.vol_threshold

    # Remove empty override entries
    overrides = {k: v for k, v in overrides.items() if v}

    # Create trader
    trader = LivePaperTrader(
        initial_capital=args.capital,
        signal_mode=args.mode,
        max_position_size=args.max_position_size,
        max_positions=args.max_positions,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        strategy_overrides=overrides or None,
    )
    
    # Run session
    trader.run_live_session(
        strategy_name=args.strategy,
        symbols=symbols,
        duration_minutes=args.duration,
        collect_seconds=args.collect_seconds,
        loop_sleep=args.loop_sleep,
    )


if __name__ == "__main__":
    main()
