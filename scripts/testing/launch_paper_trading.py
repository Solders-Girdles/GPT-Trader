#!/usr/bin/env python3
"""
Launch paper trading with Coinbase integration.
Uses V2 architecture with real market data.
"""

import os
import sys
import asyncio
import signal
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import json
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
env_file = Path(__file__).parent.parent / '.env.production'
if not env_file.exists():
    env_file = Path(__file__).parent.parent / '.env'

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce
from bot_v2.features.paper_trade.paper_trade import run_paper_trading
from bot_v2.features.analyze.strategies import MomentumStrategy, MeanReversionStrategy

class PaperTradingSystem:
    """Paper trading system with Coinbase integration."""
    
    def __init__(self):
        self.broker = None
        self.running = False
        self.positions = {}
        self.cash_balance = Decimal("1000")  # Start with simulated $1000
        self.config = None
        
    def initialize(self):
        """Initialize the trading system."""
        logger.info("Initializing Paper Trading System...")
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "paper_trading_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
                self.cash_balance = Decimal(str(self.config.get("initial_capital", 1000)))
                logger.info(f"Loaded configuration from {config_path}")
        else:
            # Default configuration
            self.config = {
                "mode": "paper",
                "initial_capital": 1000.0,
                "max_position_size": 200.0,
                "max_portfolio_risk": 100.0,
                "symbols": ["BTC-USD", "ETH-USD"],
                "strategies": ["momentum", "mean_reversion"],
                "risk_management": {
                    "stop_loss": 0.02,
                    "take_profit": 0.05,
                    "max_positions": 3,
                    "position_sizing": "fixed"
                }
            }
            logger.info("Using default configuration")
        
        # Create Coinbase configuration
        api_config = APIConfig(
            api_key="",
            api_secret="",
            passphrase=None,
            base_url=os.getenv('COINBASE_API_BASE', 'https://api.coinbase.com'),
            sandbox=False,
            ws_url=os.getenv('COINBASE_WS_URL', 'wss://advanced-trade-ws.coinbase.com'),
            cdp_api_key=os.getenv('COINBASE_CDP_API_KEY'),
            cdp_private_key=os.getenv('COINBASE_CDP_PRIVATE_KEY'),
            api_version=os.getenv('COINBASE_API_VERSION', '2024-10-24')
        )
        
        # Create brokerage adapter
        self.broker = CoinbaseBrokerage(api_config)
        
        # Connect to Coinbase
        if self.broker.connect():
            logger.info("✅ Connected to Coinbase")
            
            # Try to get real balance
            try:
                balances = self.broker.list_balances()
                for bal in balances:
                    if bal.asset == "USD" and bal.available > 0:
                        self.cash_balance = bal.available
                        logger.info(f"💵 Using real USD balance: ${self.cash_balance:.2f}")
                        break
            except Exception as e:
                logger.warning(f"Could not get real balance, using simulated: {e}")
        else:
            logger.warning("⚠️  Could not connect to Coinbase, using simulation mode")
            self.broker = None
    
    async def get_market_data(self, symbol: str):
        """Get current market data for a symbol."""
        if self.broker:
            try:
                quote = self.broker.get_quote(symbol)
                return {
                    "symbol": symbol,
                    "bid": float(quote.bid),
                    "ask": float(quote.ask),
                    "last": float(quote.last),
                    "timestamp": datetime.now()
                }
            except Exception as e:
                logger.error(f"Failed to get quote for {symbol}: {e}")
        
        # Return dummy data if no broker
        return {
            "symbol": symbol,
            "bid": 100000.0 if "BTC" in symbol else 3000.0,
            "ask": 100100.0 if "BTC" in symbol else 3010.0,
            "last": 100050.0 if "BTC" in symbol else 3005.0,
            "timestamp": datetime.now()
        }
    
    async def execute_signal(self, symbol: str, signal: str, strategy: str):
        """Execute a trading signal."""
        if signal == "hold":
            return
        
        market_data = await self.get_market_data(symbol)
        current_price = Decimal(str(market_data["last"]))
        
        # Calculate position size
        max_position = Decimal(str(self.config["risk_management"].get("max_position_size", 200)))
        position_size = min(max_position, self.cash_balance * Decimal("0.1"))  # 10% of balance
        
        if signal == "buy" and symbol not in self.positions:
            # Check if we have enough cash
            if position_size > self.cash_balance:
                logger.warning(f"Insufficient funds for {symbol} buy signal")
                return
            
            # Calculate quantity
            qty = position_size / current_price
            
            logger.info(f"📈 {strategy} BUY signal for {symbol}")
            logger.info(f"   Price: ${current_price:.2f}")
            logger.info(f"   Quantity: {qty:.8f}")
            logger.info(f"   Value: ${position_size:.2f}")
            
            # Simulate order (in real trading, would use self.broker.place_order)
            self.positions[symbol] = {
                "qty": qty,
                "entry_price": current_price,
                "entry_time": datetime.now(),
                "strategy": strategy,
                "stop_loss": current_price * Decimal("0.98"),  # 2% stop loss
                "take_profit": current_price * Decimal("1.05")  # 5% take profit
            }
            self.cash_balance -= position_size
            
        elif signal == "sell" and symbol in self.positions:
            position = self.positions[symbol]
            qty = position["qty"]
            entry_price = position["entry_price"]
            
            # Calculate P&L
            exit_value = qty * current_price
            entry_value = qty * entry_price
            pnl = exit_value - entry_value
            pnl_pct = (pnl / entry_value) * 100
            
            logger.info(f"📉 {strategy} SELL signal for {symbol}")
            logger.info(f"   Entry: ${entry_price:.2f}")
            logger.info(f"   Exit: ${current_price:.2f}")
            logger.info(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            
            # Update balance and remove position
            self.cash_balance += exit_value
            del self.positions[symbol]
    
    async def check_stop_loss_take_profit(self):
        """Check and execute stop loss/take profit orders."""
        for symbol, position in list(self.positions.items()):
            market_data = await self.get_market_data(symbol)
            current_price = Decimal(str(market_data["last"]))
            
            # Check stop loss
            if current_price <= position["stop_loss"]:
                logger.warning(f"⛔ Stop loss triggered for {symbol} at ${current_price:.2f}")
                await self.execute_signal(symbol, "sell", "stop_loss")
            
            # Check take profit
            elif current_price >= position["take_profit"]:
                logger.info(f"✅ Take profit triggered for {symbol} at ${current_price:.2f}")
                await self.execute_signal(symbol, "sell", "take_profit")
    
    async def run_trading_loop(self):
        """Main trading loop."""
        self.running = True
        strategies = {
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy()
        }
        
        logger.info("Starting paper trading loop...")
        logger.info(f"Trading symbols: {self.config['symbols']}")
        logger.info(f"Active strategies: {self.config['strategies']}")
        logger.info(f"Initial capital: ${self.cash_balance:.2f}")
        
        while self.running:
            try:
                # Process each symbol
                for symbol in self.config["symbols"]:
                    # Get market data
                    market_data = await self.get_market_data(symbol)
                    
                    # Run each strategy
                    for strategy_name in self.config["strategies"]:
                        if strategy_name in strategies:
                            strategy = strategies[strategy_name]
                            
                            # Generate signal (simplified - normally would use historical data)
                            # For demo, use simple price-based signals
                            price = market_data["last"]
                            if strategy_name == "momentum":
                                # Buy if price is rising (simplified)
                                signal = "buy" if price % 100 < 50 else "sell"
                            else:  # mean_reversion
                                # Buy if price is low (simplified)
                                signal = "sell" if price % 100 < 50 else "buy"
                            
                            # Execute signal
                            await self.execute_signal(symbol, signal, strategy_name)
                
                # Check stop loss/take profit
                await self.check_stop_loss_take_profit()
                
                # Display status
                self.display_status()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    def display_status(self):
        """Display current trading status."""
        total_value = self.cash_balance
        
        logger.info("\n" + "=" * 60)
        logger.info("PAPER TRADING STATUS")
        logger.info("=" * 60)
        logger.info(f"Cash: ${self.cash_balance:.2f}")
        
        if self.positions:
            logger.info("\nOpen Positions:")
            for symbol, position in self.positions.items():
                current_value = position["qty"] * position["entry_price"]  # Simplified
                total_value += current_value
                logger.info(f"  {symbol}: {position['qty']:.8f} @ ${position['entry_price']:.2f}")
                logger.info(f"    Strategy: {position['strategy']}")
                logger.info(f"    Stop: ${position['stop_loss']:.2f}, Target: ${position['take_profit']:.2f}")
        else:
            logger.info("No open positions")
        
        logger.info(f"\nTotal Portfolio Value: ${total_value:.2f}")
        initial = Decimal(str(self.config["initial_capital"]))
        pnl = total_value - initial
        pnl_pct = (pnl / initial) * 100 if initial > 0 else 0
        logger.info(f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        logger.info("=" * 60 + "\n")
    
    def stop(self):
        """Stop the trading system."""
        logger.info("Stopping paper trading...")
        self.running = False
        
        if self.broker:
            self.broker.disconnect()
        
        # Final status
        self.display_status()
        logger.info("Paper trading stopped.")

async def main():
    """Main entry point."""
    system = PaperTradingSystem()
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info("\nReceived interrupt signal, shutting down...")
        system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        system.initialize()
        
        # Run trading loop
        await system.run_trading_loop()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        system.stop()
        raise

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GPT-TRADER V2 - PAPER TRADING SYSTEM")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(main())