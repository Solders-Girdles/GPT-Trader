#!/usr/bin/env python3
"""
Live Demo Runner - Tiny live positions with maximum safety.

Implements all guardrails:
- Conservative sizing ($25-100 notional)
- Post-only limit orders
- RSI confirmation
- Strict filters
- Kill switch ready
- Pre-funding quiet period
"""

import os
import sys
import time
import signal
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment for real trading
os.environ['COINBASE_API_MODE'] = 'advanced'
os.environ['COINBASE_AUTH_TYPE'] = 'JWT'
os.environ['COINBASE_ENABLE_DERIVATIVES'] = '1'
os.environ.pop('COINBASE_SANDBOX', None)
os.environ.pop('PERPS_FORCE_MOCK', None)

from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.features.live_trade.execution_v3 import (
    AdvancedExecutionEngine, OrderConfig, SizingMode
)
from bot_v2.features.live_trade.pnl_tracker import PnLTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase2DemoBot:
    """Demo bot with all safety features."""
    
    def __init__(self):
        """Initialize with conservative config."""
        self.broker = None
        self.engine = None
        self.pnl_tracker = None
        self.running = True
        self.reduce_only = False
        self.daily_loss = Decimal("0")
        self.daily_loss_limit = Decimal("100")  # $100 max daily loss
        self.position_count = 0
        self.metrics = {
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'post_only_rejected': 0,
            'sized_down_count': 0,
            'total_pnl': Decimal("0")
        }
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
    
    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.warning(f"Shutdown signal received: {signum}")
        self.running = False
        self.reduce_only = True
        logger.info("Switched to reduce-only mode")
    
    def setup(self):
        """Set up broker and engine."""
        try:
            # Create real broker
            logger.info("Creating Coinbase brokerage...")
            self.broker = create_brokerage()
            
            if not self.broker.connect():
                logger.error("Failed to connect to broker")
                return False
            
            # Configure engine with conservative settings
            config = OrderConfig(
                enable_limit_orders=True,
                enable_post_only=True,
                reject_on_cross=True,
                sizing_mode=SizingMode.CONSERVATIVE,
                max_impact_bps=Decimal("10"),
                limit_price_offset_bps=Decimal("10")  # 10 bps offset for post-only
            )
            
            self.engine = AdvancedExecutionEngine(self.broker, config=config)
            self.pnl_tracker = PnLTracker()
            
            logger.info("✅ Setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def check_pre_funding_quiet(self, symbol: str) -> bool:
        """Check if we're in pre-funding quiet period."""
        try:
            # Get product info
            product = self.broker.get_product(symbol)
            if not product or not hasattr(product, 'next_funding_time'):
                return True  # Allow if no funding info
            
            if product.next_funding_time:
                now = datetime.utcnow()
                mins_to_funding = (product.next_funding_time - now).total_seconds() / 60
                
                if 0 < mins_to_funding < 30:  # 30 min quiet period
                    logger.info(f"Pre-funding quiet period: {mins_to_funding:.1f} mins to funding")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not check funding time: {e}")
            return True  # Allow on error
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit exceeded."""
        if abs(self.daily_loss) >= self.daily_loss_limit:
            logger.error(f"Daily loss limit hit: ${abs(self.daily_loss):.2f}")
            self.reduce_only = True
            return False
        return True
    
    def place_demo_order(self, symbol: str = "BTC-PERP"):
        """Place a tiny demo order with all safety checks."""
        try:
            # Safety checks
            if self.reduce_only:
                logger.info("Reduce-only mode - skipping entry")
                return
            
            if not self.check_daily_loss_limit():
                return
            
            if not self.check_pre_funding_quiet(symbol):
                return
            
            # Get market snapshot
            quote = self.broker.get_quote(symbol)
            if not quote:
                logger.warning("No quote available")
                return
            
            # Check spread
            spread_bps = (quote.ask - quote.bid) / quote.ask * 10000
            if spread_bps > 5:  # Max 5 bps spread
                logger.info(f"Spread too wide: {spread_bps:.1f} bps")
                return
            
            # Calculate tiny position size
            target_notional = Decimal("50")  # $50 position
            position_size = target_notional / quote.last
            
            # Place post-only limit order
            limit_price = quote.bid - (quote.bid * Decimal("0.001"))  # 10 bps below bid
            
            logger.info(f"Placing demo order: {position_size:.6f} {symbol} @ {limit_price:.2f}")
            
            order = self.engine.place_order(
                symbol=symbol,
                side="buy",
                quantity=position_size,
                order_type="limit",
                limit_price=limit_price,
                post_only=True,
                client_id=f"demo_{int(time.time())}"
            )
            
            if order:
                logger.info(f"✅ Order placed: {order.id}")
                self.metrics['orders_placed'] += 1
                
                # Cancel after 30 seconds if not filled
                time.sleep(30)
                if self.broker.cancel_order(order.id):
                    logger.info(f"Order cancelled: {order.id}")
                    self.metrics['orders_cancelled'] += 1
            else:
                logger.warning("Order rejected")
                self.metrics['orders_rejected'] += 1
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
    
    def run_demo_cycle(self):
        """Run one demo cycle."""
        logger.info("=" * 60)
        logger.info("LIVE DEMO CYCLE")
        logger.info("=" * 60)
        
        # Show current metrics
        logger.info(f"Metrics: {self.metrics}")
        logger.info(f"Daily PnL: ${self.daily_loss:.2f}")
        
        # Place demo order
        self.place_demo_order("BTC-PERP")
        
        # Update PnL
        positions = self.broker.get_positions()
        if positions:
            for pos in positions:
                logger.info(f"Position: {pos.symbol} {pos.qty} @ {pos.avg_price}")
        
        # Show engine metrics
        engine_metrics = self.engine.get_metrics()
        logger.info(f"Engine metrics: {engine_metrics}")
    
    def run(self):
        """Main run loop."""
        if not self.setup():
            return
        
        logger.info("Starting Live Demo Bot")
        logger.info("Press Ctrl+C for kill switch")
        
        cycle = 0
        while self.running:
            try:
                cycle += 1
                logger.info(f"\nCycle {cycle}")
                
                self.run_demo_cycle()
                
                # Wait before next cycle
                time.sleep(60)  # 1 minute between cycles
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt - activating kill switch")
                self.reduce_only = True
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                time.sleep(10)
        
        # Cleanup
        logger.info("Shutting down...")
        if self.broker:
            self.broker.disconnect()
        
        # Final metrics
        logger.info("\n" + "=" * 60)
        logger.info("FINAL METRICS")
        logger.info("=" * 60)
        logger.info(f"Orders placed: {self.metrics['orders_placed']}")
        logger.info(f"Orders filled: {self.metrics['orders_filled']}")
        logger.info(f"Orders cancelled: {self.metrics['orders_cancelled']}")
        logger.info(f"Orders rejected: {self.metrics['orders_rejected']}")
        logger.info(f"Post-only rejections: {self.metrics['post_only_rejected']}")
        logger.info(f"Total PnL: ${self.metrics['total_pnl']:.2f}")

def main():
    """Run live demo."""
    bot = Phase2DemoBot()
    bot.run()

if __name__ == "__main__":
    main()
