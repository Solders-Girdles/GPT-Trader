#!/usr/bin/env python3
"""
Coinbase Paper Trading with Real Market Data
Uses actual Coinbase quotes and spreads for realistic paper trading simulation.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

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
from src.bot_v2.features.brokerages.core.interfaces import OrderType, OrderSide


class CoinbasePaperTrader:
    """Paper trading system using real Coinbase market data."""
    
    def __init__(self, initial_capital: float = 10000.0):
        """Initialize paper trader with Coinbase connection."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {quantity, avg_price, value}
        self.trades: List[Dict] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        # Trading parameters
        self.commission_rate = 0.006  # 0.6% Coinbase fee
        self.slippage_rate = 0.001  # 0.1% slippage
        
        # Risk management
        self.max_position_size = 0.25  # Max 25% per position
        self.max_positions = 4  # Max 4 concurrent positions
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
        
        # Initialize Coinbase connection
        self.broker = self._init_coinbase()
        self.is_connected = False
        
        # Track performance
        self.start_time = datetime.now()
        self.peak_equity = initial_capital
        
    def _init_coinbase(self) -> CoinbaseBrokerage:
        """Initialize Coinbase brokerage connection."""
        config = APIConfig(
            api_key="",
            api_secret="",
            passphrase=None,
            base_url=os.getenv('COINBASE_API_BASE', 'https://api.coinbase.com'),
            sandbox=os.getenv('COINBASE_SANDBOX', '0') == '1',
            ws_url=os.getenv('COINBASE_WS_URL', 'wss://advanced-trade-ws.coinbase.com'),
            cdp_api_key=os.getenv('COINBASE_CDP_API_KEY'),
            cdp_private_key=os.getenv('COINBASE_CDP_PRIVATE_KEY'),
            api_version=os.getenv('COINBASE_API_VERSION', '2024-10-24')
        )
        return CoinbaseBrokerage(config)
    
    def connect(self) -> bool:
        """Connect to Coinbase."""
        self.is_connected = self.broker.connect()
        if self.is_connected:
            print("✅ Connected to Coinbase")
        else:
            print("❌ Failed to connect to Coinbase")
        return self.is_connected
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from Coinbase."""
        try:
            quote = self.broker.get_quote(symbol)
            if quote:
                return {
                    'bid': float(quote.bid),
                    'ask': float(quote.ask),
                    'last': float(quote.last) if quote.last else (float(quote.bid) + float(quote.ask)) / 2,
                    'spread': float(quote.ask) - float(quote.bid),
                    'spread_pct': (float(quote.ask) - float(quote.bid)) / float(quote.bid) * 100
                }
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
        return None
    
    def calculate_position_size(self, symbol: str, signal_strength: float = 1.0) -> float:
        """Calculate position size based on available capital and risk limits."""
        equity = self.get_equity()
        max_position_value = equity * self.max_position_size * signal_strength
        
        # Check position limits
        if len(self.positions) >= self.max_positions:
            return 0.0
        
        # Use available cash
        available_cash = min(self.cash * 0.95, max_position_value)  # Keep 5% cash reserve
        return available_cash
    
    def place_order(self, symbol: str, side: str, amount_usd: float) -> Dict:
        """Simulate order placement with real spreads and fees."""
        quote = self.get_quote(symbol)
        if not quote:
            return {'status': 'failed', 'reason': 'No quote available'}
        
        # Use ask for buys, bid for sells
        if side == 'buy':
            price = quote['ask'] * (1 + self.slippage_rate)  # Add slippage
            commission = amount_usd * self.commission_rate
            net_amount = amount_usd - commission
            quantity = net_amount / price
            
            if self.cash < amount_usd:
                return {'status': 'failed', 'reason': 'Insufficient funds'}
            
            # Execute buy
            self.cash -= amount_usd
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0, 'avg_price': 0, 'value': 0}
            
            # Update position (weighted average)
            pos = self.positions[symbol]
            total_quantity = pos['quantity'] + quantity
            if total_quantity > 0:
                pos['avg_price'] = (pos['quantity'] * pos['avg_price'] + quantity * price) / total_quantity
            pos['quantity'] = total_quantity
            pos['value'] = total_quantity * price
            
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'buy',
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'value': amount_usd
            }
            
        else:  # sell
            if symbol not in self.positions or self.positions[symbol]['quantity'] <= 0:
                return {'status': 'failed', 'reason': 'No position to sell'}
            
            pos = self.positions[symbol]
            price = quote['bid'] * (1 - self.slippage_rate)  # Subtract slippage
            
            # Sell entire position or specified amount
            quantity = min(pos['quantity'], amount_usd / price)
            proceeds = quantity * price
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission
            
            # Execute sell
            self.cash += net_proceeds
            pos['quantity'] -= quantity
            pos['value'] = pos['quantity'] * price
            
            # Remove position if fully sold
            if pos['quantity'] <= 0.0001:
                del self.positions[symbol]
            
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'sell',
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'value': proceeds,
                'pnl': (price - pos['avg_price']) * quantity - commission
            }
        
        self.trades.append(trade)
        return {'status': 'success', 'trade': trade}
    
    def get_equity(self) -> float:
        """Calculate total equity (cash + positions)."""
        equity = self.cash
        for symbol, pos in self.positions.items():
            if pos['quantity'] > 0:
                quote = self.get_quote(symbol)
                if quote:
                    # Use bid price for conservative valuation
                    current_price = quote['bid']
                    equity += pos['quantity'] * current_price
        return equity
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        current_equity = self.get_equity()
        total_return = (current_equity - self.initial_capital) / self.initial_capital * 100
        
        # Calculate drawdown
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
        
        # Win rate
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        # Average win/loss
        avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Trading duration
        duration = datetime.now() - self.start_time
        
        return {
            'equity': current_equity,
            'cash': self.cash,
            'positions_value': current_equity - self.cash,
            'total_return': total_return,
            'drawdown': drawdown,
            'num_trades': len(self.trades),
            'num_positions': len(self.positions),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'duration': str(duration).split('.')[0]
        }
    
    def apply_strategy(self, strategy_name: str = 'momentum'):
        """Apply a simple trading strategy using real market data."""
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'MATIC-USD']
        
        print(f"\n📊 Running {strategy_name} strategy on {', '.join(symbols)}")
        
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if not quote:
                continue
            
            # Simple momentum strategy
            if strategy_name == 'momentum':
                # Random signal for demo (replace with real strategy)
                import random
                signal = random.choice(['buy', 'sell', 'hold', 'hold'])
                
                if signal == 'buy' and symbol not in self.positions:
                    amount = self.calculate_position_size(symbol)
                    if amount > 100:  # Minimum $100 position
                        result = self.place_order(symbol, 'buy', amount)
                        if result['status'] == 'success':
                            print(f"✅ Bought {symbol} for ${amount:.2f}")
                
                elif signal == 'sell' and symbol in self.positions:
                    pos = self.positions[symbol]
                    current_value = pos['quantity'] * quote['bid']
                    result = self.place_order(symbol, 'sell', current_value)
                    if result['status'] == 'success':
                        pnl = result['trade'].get('pnl', 0)
                        print(f"✅ Sold {symbol} - P&L: ${pnl:.2f}")
            
            # Check stop loss and take profit
            if symbol in self.positions:
                pos = self.positions[symbol]
                current_price = quote['bid']
                price_change = (current_price - pos['avg_price']) / pos['avg_price']
                
                if price_change <= -self.stop_loss_pct:
                    print(f"🛑 Stop loss triggered for {symbol}")
                    current_value = pos['quantity'] * current_price
                    self.place_order(symbol, 'sell', current_value)
                
                elif price_change >= self.take_profit_pct:
                    print(f"🎯 Take profit triggered for {symbol}")
                    current_value = pos['quantity'] * current_price
                    self.place_order(symbol, 'sell', current_value)
    
    def display_status(self):
        """Display current trading status."""
        metrics = self.get_performance_metrics()
        
        print("\n" + "=" * 60)
        print("COINBASE PAPER TRADING STATUS")
        print("=" * 60)
        print(f"Duration: {metrics['duration']}")
        print(f"Equity: ${metrics['equity']:.2f}")
        print(f"Cash: ${metrics['cash']:.2f}")
        print(f"Positions Value: ${metrics['positions_value']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Max Drawdown: {metrics['drawdown']:.2f}%")
        print(f"Total Trades: {metrics['num_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        
        if self.positions:
            print("\n📈 Open Positions:")
            for symbol, pos in self.positions.items():
                quote = self.get_quote(symbol)
                if quote:
                    current_price = quote['bid']
                    pnl = (current_price - pos['avg_price']) * pos['quantity']
                    pnl_pct = (current_price - pos['avg_price']) / pos['avg_price'] * 100
                    print(f"  {symbol}: {pos['quantity']:.6f} @ ${pos['avg_price']:.2f}")
                    print(f"    Current: ${current_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        
        print("=" * 60)
    
    def save_results(self, filename: str = "paper_trading_results.json"):
        """Save trading results to file."""
        results = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'metrics': self.get_performance_metrics(),
            'trades': [
                {
                    'timestamp': t['timestamp'].isoformat(),
                    'symbol': t['symbol'],
                    'side': t['side'],
                    'quantity': t['quantity'],
                    'price': t['price'],
                    'commission': t['commission'],
                    'value': t['value'],
                    'pnl': t.get('pnl', 0)
                }
                for t in self.trades
            ],
            'positions': {
                symbol: {
                    'quantity': pos['quantity'],
                    'avg_price': pos['avg_price'],
                    'value': pos['value']
                }
                for symbol, pos in self.positions.items()
            }
        }
        
        filepath = Path(__file__).parent.parent / 'results' / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {filepath}")


def run_paper_trading_session(duration_minutes: int = 5):
    """Run a paper trading session for specified duration."""
    
    print("=" * 60)
    print("COINBASE PAPER TRADING WITH REAL MARKET DATA")
    print("=" * 60)
    
    # Initialize trader
    trader = CoinbasePaperTrader(initial_capital=10000)
    
    # Connect to Coinbase
    if not trader.connect():
        print("Failed to connect to Coinbase. Exiting.")
        return
    
    # Display initial status
    trader.display_status()
    
    # Run trading loop
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    iteration = 0
    
    print(f"\n⏱️  Running for {duration_minutes} minutes...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        while datetime.now() < end_time:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Apply strategy
            trader.apply_strategy('momentum')
            
            # Display status
            trader.display_status()
            
            # Record equity
            trader.equity_history.append((datetime.now(), trader.get_equity()))
            
            # Wait before next iteration
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n⚠️  Trading interrupted by user")
    
    # Final status
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    trader.display_status()
    
    # Save results
    trader.save_results(f"coinbase_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Disconnect
    trader.broker.disconnect()
    print("\n✅ Disconnected from Coinbase")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Coinbase Paper Trading with Real Market Data")
    parser.add_argument('--duration', type=int, default=5, help='Duration in minutes (default: 5)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital (default: 10000)')
    
    args = parser.parse_args()
    
    if args.duration > 0:
        run_paper_trading_session(args.duration)
    else:
        # Quick test mode
        print("Running quick test...")
        trader = CoinbasePaperTrader(args.capital)
        if trader.connect():
            # Test getting quotes
            for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
                quote = trader.get_quote(symbol)
                if quote:
                    print(f"{symbol}: Bid=${quote['bid']:.2f}, Ask=${quote['ask']:.2f}, Spread={quote['spread_pct']:.3f}%")
            
            # Test a trade
            trader.apply_strategy('momentum')
            trader.display_status()
            trader.broker.disconnect()