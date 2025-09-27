#!/usr/bin/env python3
"""
Week 2 Enhanced Perpetuals Trading Bot Runner

Includes:
- Market condition filters (spread, depth, volume)
- RSI confirmation for signals
- Liquidation distance and slippage guards
- WebSocket market snapshot integration
- Comprehensive metrics tracking

Usage:
    # Development with filters
    python scripts/run_perps_bot_v2.py --profile dev --max-spread-bps 10 --min-vol-1m 100000
    
    # Demo with risk guards
    python scripts/run_perps_bot_v2.py --profile demo --liq-buffer-pct 20 --max-slippage-bps 15
    
    # Production with full filters
    python scripts/run_perps_bot_v2.py --profile prod --rsi-confirm --min-depth-l10 200000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_v2.features.live_trade.strategies.perps_baseline_v2 import (
    EnhancedPerpsStrategy, StrategyConfig, StrategyFiltersConfig, Action
)
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.brokerages.core.interfaces import Product, MarketType
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.persistence.event_store import EventStore
from bot_v2.features.live_trade.execution_v3 import (
    AdvancedExecutionEngine, OrderConfig, SizingMode
)
from bot_v2.features.live_trade.pnl_tracker import PnLTracker
from bot_v2.orchestration.mock_broker import MockBroker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Profile(Enum):
    """Configuration profiles."""
    DEV = "dev"      # Development with mocks
    DEMO = "demo"    # Demo with tiny positions  
    PROD = "prod"    # Production trading


@dataclass
class BotConfig:
    """Enhanced bot configuration with Week 2 features."""
    profile: Profile
    dry_run: bool = False
    symbols: List[str] = None
    update_interval: int = 5  # seconds
    
    # Strategy settings
    short_ma: int = 5
    long_ma: int = 20
    target_leverage: int = 2
    trailing_stop_pct: float = 0.01
    enable_shorts: bool = False
    
    # Market condition filters
    max_spread_bps: Optional[Decimal] = Decimal('10')
    min_depth_l1: Optional[Decimal] = Decimal('50000')
    min_depth_l10: Optional[Decimal] = Decimal('200000')
    min_volume_1m: Optional[Decimal] = Decimal('100000')
    require_rsi_confirmation: bool = False
    
    # Risk guards
    min_liquidation_buffer_pct: Optional[Decimal] = Decimal('20')
    max_slippage_impact_bps: Optional[Decimal] = Decimal('15')
    
    # Risk settings
    max_position_size: Decimal = Decimal("1000")  # USD notional
    max_leverage: int = 3
    reduce_only_mode: bool = False
    
    # Execution settings
    mock_broker: bool = False
    mock_fills: bool = False
    
    # Monitoring
    enable_metrics: bool = True
    metrics_interval: int = 60  # seconds
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC-PERP", "ETH-PERP"]
    
    @classmethod
    def from_profile(cls, profile: str, **overrides) -> BotConfig:
        """Create config from profile name with overrides."""
        profile_enum = Profile(profile)
        
        if profile_enum == Profile.DEV:
            config = cls(
                profile=profile_enum,
                mock_broker=True,
                mock_fills=True,
                max_position_size=Decimal("100"),
                # Conservative filters for dev
                max_spread_bps=Decimal('20'),
                min_depth_l1=Decimal('10000'),
                min_depth_l10=Decimal('50000'),
                min_volume_1m=Decimal('10000')
            )
        elif profile_enum == Profile.DEMO:
            config = cls(
                profile=profile_enum,
                max_position_size=Decimal("500"),
                reduce_only_mode=False,
                # Moderate filters for demo
                max_spread_bps=Decimal('15'),
                min_depth_l1=Decimal('25000'),
                min_depth_l10=Decimal('100000'),
                min_volume_1m=Decimal('50000')
            )
        else:  # PROD
            config = cls(
                profile=profile_enum,
                max_position_size=Decimal("5000"),
                # Strict filters for prod
                max_spread_bps=Decimal('10'),
                min_depth_l1=Decimal('50000'),
                min_depth_l10=Decimal('200000'),
                min_volume_1m=Decimal('100000'),
                require_rsi_confirmation=True
            )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


class EnhancedPerpsBot:
    """Enhanced perpetuals trading bot with Week 2 features."""
    
    def __init__(self, config: BotConfig):
        """Initialize enhanced bot."""
        self.config = config
        self.running = False
        
        # Initialize components
        self._init_storage()
        self._init_broker()
        self._init_risk_manager()
        self._init_strategy()
        self._init_execution()
        
        # State tracking
        self.mark_windows: Dict[str, List[Decimal]] = {
            symbol: [] for symbol in config.symbols
        }
        self.last_decisions: Dict[str, Any] = {}
        self.market_snapshots: Dict[str, Dict[str, Any]] = {}
        self._product_map: Dict[str, Product] = {}
        
        # Metrics tracking
        self.last_metrics_time = datetime.now()
        
    def _init_storage(self):
        """Initialize event store."""
        # Respect EVENT_STORE_ROOT if set
        if 'EVENT_STORE_ROOT' in os.environ:
            storage_dir = Path(os.environ['EVENT_STORE_ROOT']) / f"perps_bot_v2/{self.config.profile.value}"
        else:
            storage_dir = Path(f"data/perps_bot_v2/{self.config.profile.value}")
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.event_store = EventStore(root=storage_dir)
        
    def _init_risk_manager(self):
        """Initialize risk manager."""
        risk_config = RiskConfig(
            leverage_max_global=self.config.max_leverage,
            leverage_max_per_symbol={},
            max_daily_loss_pct=0.05,
            max_exposure_pct=0.8,
            max_position_pct_per_symbol=0.2,
            reduce_only_mode=self.config.reduce_only_mode
        )
        self.risk_manager = LiveRiskManager(
            config=risk_config,
            event_store=self.event_store
        )
        
    def _init_strategy(self):
        """Initialize enhanced trading strategy."""
        # Create filters config
        filters_config = StrategyFiltersConfig(
            max_spread_bps=self.config.max_spread_bps,
            min_depth_l1=self.config.min_depth_l1,
            min_depth_l10=self.config.min_depth_l10,
            min_volume_1m=self.config.min_volume_1m,
            require_rsi_confirmation=self.config.require_rsi_confirmation,
            min_liquidation_buffer_pct=self.config.min_liquidation_buffer_pct,
            max_slippage_impact_bps=self.config.max_slippage_impact_bps
        )
        
        strategy_config = StrategyConfig(
            short_ma_period=self.config.short_ma,
            long_ma_period=self.config.long_ma,
            target_leverage=self.config.target_leverage,
            trailing_stop_pct=self.config.trailing_stop_pct,
            enable_shorts=self.config.enable_shorts,
            filters_config=filters_config
        )
        
        self.strategy = EnhancedPerpsStrategy(
            config=strategy_config,
            risk_manager=self.risk_manager,
            event_store=self.event_store,
            bot_id='enhanced_perps_bot'
        )
        
        logger.info(f"Enhanced strategy initialized with filters: {filters_config}")
        
    def _init_execution(self):
        """Initialize execution engine."""
        self.exec_engine = LiveExecutionEngine(
            broker=self.broker,
            risk_manager=self.risk_manager,
            event_store=self.event_store,
            bot_id='enhanced_perps_bot'
        )
        
    def _init_broker(self):
        """Initialize broker connection."""
        if self.config.mock_broker or os.environ.get('PERPS_FORCE_MOCK') == '1':
            self.broker = MockBroker()
            logger.info("Using mock broker")
        else:
            from bot_v2.orchestration.broker_factory import create_brokerage
            self.broker = create_brokerage()
            
            if not self.broker.connect():
                raise RuntimeError("Failed to connect to broker")
            
            # Cache products
            products = self.broker.list_products()
            for product in products:
                if hasattr(product, 'symbol'):
                    self._product_map[product.symbol] = product
            
            # Start WebSocket market data for configured symbols
            if hasattr(self.broker, 'start_market_data'):
                self.broker.start_market_data(self.config.symbols)
                logger.info(f"Started WebSocket market data for {self.config.symbols}")
    
    async def update_marks(self):
        """Update mark prices for all symbols."""
        for symbol in self.config.symbols:
            try:
                quote = self.broker.get_quote(symbol)
                if quote and hasattr(quote, 'last'):
                    mark = quote.last
                    
                    if symbol not in self.mark_windows:
                        self.mark_windows[symbol] = []
                    
                    self.mark_windows[symbol].append(mark)
                    self.mark_windows[symbol] = self.mark_windows[symbol][-50:]
                    
            except Exception as e:
                logger.error(f"Failed to update mark for {symbol}: {e}")
    
    async def update_market_snapshots(self):
        """Update market snapshots from WebSocket data."""
        for symbol in self.config.symbols:
            try:
                if hasattr(self.broker, 'get_market_snapshot'):
                    snapshot = self.broker.get_market_snapshot(symbol)
                    self.market_snapshots[symbol] = snapshot
                    
                    # Log snapshot stats periodically
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"{symbol} snapshot: "
                            f"spread={snapshot.get('spread_bps', 'N/A')} bps, "
                            f"l1_depth=${snapshot.get('depth_l1', 'N/A')}, "
                            f"vol_1m=${snapshot.get('vol_1m', 'N/A')}"
                        )
            except Exception as e:
                logger.error(f"Failed to update market snapshot for {symbol}: {e}")
    
    async def process_symbol(self, symbol: str):
        """Process trading logic for a symbol with Week 2 enhancements."""
        try:
            # Get current state
            account = self.broker.get_account()
            if not account:
                logger.error(f"No account info for {symbol}")
                return
            
            positions = self.broker.get_positions()
            position_map = {p.symbol: p for p in positions if hasattr(p, 'symbol')}
            
            # Get position state
            position_state = None
            if symbol in position_map:
                pos = position_map[symbol]
                position_state = {
                    'qty': pos.qty,
                    'side': 'long' if pos.qty > 0 else 'short',
                    'entry': pos.avg_price if hasattr(pos, 'avg_price') else None
                }
            
            # Get marks
            marks = self.mark_windows.get(symbol, [])
            if not marks:
                logger.warning(f"No marks for {symbol}")
                return
            
            current_mark = marks[-1]
            recent_marks = marks[:-1] if len(marks) > 1 else []
            
            # Get product
            product = self._product_map.get(symbol)
            
            # Get market snapshot
            market_snapshot = self.market_snapshots.get(symbol, {})
            
            # Check staleness
            is_stale = False
            if hasattr(self.broker, 'is_stale'):
                is_stale = self.broker.is_stale(symbol, threshold_seconds=10)
            
            # Generate enhanced decision
            decision = self.strategy.decide(
                symbol=symbol,
                current_mark=current_mark,
                position_state=position_state,
                recent_marks=recent_marks,
                equity=Decimal(str(account.equity)),
                product=product,
                market_snapshot=market_snapshot,
                is_stale=is_stale
            )
            
            self.last_decisions[symbol] = decision
            
            # Log decision with rejection tracking
            if decision.filter_rejected or decision.guard_rejected:
                logger.warning(
                    f"{symbol} REJECTED: {decision.action.value} - {decision.reason} "
                    f"(type: {decision.rejection_type})"
                )
            else:
                logger.info(f"{symbol} Decision: {decision.action.value} - {decision.reason}")
            
            # Execute if needed
            if decision.action in {Action.BUY, Action.SELL, Action.CLOSE}:
                await self.execute_decision(symbol, decision, current_mark, product)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
    
    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        current_mark: Decimal,
        product: Optional[Product]
    ):
        """Execute trading decision."""
        try:
            # All LiveExecutionEngine methods are synchronous
            if decision.action == Action.CLOSE:
                # Close position with reduce-only market order
                # Get current position to determine side
                positions = self.broker.get_positions()
                position = next((p for p in positions if p.symbol == symbol), None)
                
                if position:
                    # Place reduce-only market order to close
                    close_side = 'sell' if position.qty > 0 else 'buy'
                    result = self.exec_engine.place_order(
                        symbol=symbol,
                        side=close_side,
                        quantity=abs(position.qty),
                        order_type='market',
                        reduce_only=True
                    )
                    logger.info(f"Closed position for {symbol}: {result}")
                else:
                    logger.warning(f"No position to close for {symbol}")
                    return
            else:
                # Open position
                side = 'buy' if decision.action == Action.BUY else 'sell'
                
                # Calculate quantity from notional
                if decision.target_notional and current_mark > 0:
                    qty = decision.target_notional / current_mark
                    
                    # Quantize to product requirements
                    if product and hasattr(product, 'step_size'):
                        qty = (qty // product.step_size) * product.step_size
                else:
                    qty = Decimal('0.001')  # Minimum fallback
                
                # Place order synchronously
                result = self.exec_engine.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type='market',
                    reduce_only=decision.reduce_only
                )
                logger.info(f"Order placed for {symbol}: {result}")
            
        except Exception as e:
            logger.error(f"Execution failed for {symbol}: {e}")
    
    async def log_metrics(self):
        """Log strategy metrics periodically."""
        if not self.config.enable_metrics:
            return
        
        now = datetime.now()
        if (now - self.last_metrics_time).total_seconds() < self.config.metrics_interval:
            return
        
        self.last_metrics_time = now
        
        # Get strategy metrics
        metrics = self.strategy.get_metrics()
        
        # Log metrics
        logger.info("=" * 50)
        logger.info("STRATEGY METRICS")
        logger.info(f"Entries accepted: {metrics['entries_accepted']}")
        logger.info(f"Total rejections: {metrics['total_rejections']}")
        logger.info(f"Acceptance rate: {metrics['acceptance_rate']:.1%}")
        
        logger.info("\nRejection breakdown:")
        for key, count in metrics['rejection_counts'].items():
            if key != 'entries_accepted' and count > 0:
                logger.info(f"  {key}: {count}")
        
        # Log market snapshots
        logger.info("\nMarket conditions:")
        for symbol, snapshot in self.market_snapshots.items():
            if snapshot:
                logger.info(
                    f"  {symbol}: spread={snapshot.get('spread_bps', 'N/A')} bps, "
                    f"l1=${snapshot.get('depth_l1', 'N/A')}, "
                    f"vol_1m=${snapshot.get('vol_1m', 'N/A')}"
                )
        logger.info("=" * 50)
    
    async def run(self):
        """Main bot loop."""
        self.running = True
        logger.info(f"Starting enhanced bot with profile: {self.config.profile.value}")
        
        while self.running:
            try:
                # Update marks and market data
                await self.update_marks()
                await self.update_market_snapshots()
                
                # Process each symbol
                for symbol in self.config.symbols:
                    await self.process_symbol(symbol)
                
                # Log metrics
                await self.log_metrics()
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                self.running = False
            except Exception as e:
                logger.error(f"Bot error: {e}", exc_info=True)
                await asyncio.sleep(self.config.update_interval)
    
    def shutdown(self):
        """Shutdown bot cleanly."""
        logger.info("Shutting down enhanced bot...")
        self.running = False
        
        # Log final metrics
        metrics = self.strategy.get_metrics()
        logger.info(f"Final metrics: {json.dumps(metrics, indent=2)}")
        
        # Disconnect broker
        if hasattr(self.broker, 'disconnect'):
            self.broker.disconnect()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Perpetuals Trading Bot")
    
    # Profile and modes
    parser.add_argument('--profile', choices=['dev', 'demo', 'prod'], default='dev',
                      help='Configuration profile')
    parser.add_argument('--dry-run', action='store_true',
                      help='Dry run mode (no real trades)')
    parser.add_argument('--reduce-only', action='store_true',
                      help='Reduce-only mode')
    
    # Symbols
    parser.add_argument('--symbols', nargs='+', default=None,
                      help='Symbols to trade (e.g., BTC-PERP ETH-PERP)')
    
    # Strategy parameters
    parser.add_argument('--short-ma', type=int, default=5,
                      help='Short MA period')
    parser.add_argument('--long-ma', type=int, default=20,
                      help='Long MA period')
    parser.add_argument('--leverage', type=int, default=2,
                      help='Target leverage')
    parser.add_argument('--enable-shorts', action='store_true',
                      help='Enable short positions')
    
    # Market condition filters
    parser.add_argument('--max-spread-bps', type=float, default=None,
                      help='Maximum spread in basis points')
    parser.add_argument('--min-depth-l1', type=float, default=None,
                      help='Minimum L1 depth in USD')
    parser.add_argument('--min-depth-l10', type=float, default=None,
                      help='Minimum L10 depth in USD')
    parser.add_argument('--min-vol-1m', type=float, default=None,
                      help='Minimum 1-minute volume in USD')
    parser.add_argument('--rsi-confirm', action='store_true',
                      help='Require RSI confirmation for signals')
    
    # Risk guards
    parser.add_argument('--liq-buffer-pct', type=float, default=None,
                      help='Minimum liquidation buffer percentage')
    parser.add_argument('--max-slippage-bps', type=float, default=None,
                      help='Maximum slippage in basis points')
    
    # Execution
    parser.add_argument('--update-interval', type=int, default=5,
                      help='Update interval in seconds')
    parser.add_argument('--metrics-interval', type=int, default=60,
                      help='Metrics logging interval in seconds')
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Build config
    config_overrides = {}
    
    # Apply CLI overrides
    if args.dry_run:
        config_overrides['dry_run'] = True
    if args.reduce_only:
        config_overrides['reduce_only_mode'] = True
    if args.symbols:
        config_overrides['symbols'] = args.symbols
    
    # Strategy parameters
    config_overrides['short_ma'] = args.short_ma
    config_overrides['long_ma'] = args.long_ma
    config_overrides['target_leverage'] = args.leverage
    config_overrides['enable_shorts'] = args.enable_shorts
    
    # Filters
    if args.max_spread_bps is not None:
        config_overrides['max_spread_bps'] = Decimal(str(args.max_spread_bps))
    if args.min_depth_l1 is not None:
        config_overrides['min_depth_l1'] = Decimal(str(args.min_depth_l1))
    if args.min_depth_l10 is not None:
        config_overrides['min_depth_l10'] = Decimal(str(args.min_depth_l10))
    if args.min_vol_1m is not None:
        config_overrides['min_volume_1m'] = Decimal(str(args.min_vol_1m))
    config_overrides['require_rsi_confirmation'] = args.rsi_confirm
    
    # Guards
    if args.liq_buffer_pct is not None:
        config_overrides['min_liquidation_buffer_pct'] = Decimal(str(args.liq_buffer_pct))
    if args.max_slippage_bps is not None:
        config_overrides['max_slippage_impact_bps'] = Decimal(str(args.max_slippage_bps))
    
    # Intervals
    config_overrides['update_interval'] = args.update_interval
    config_overrides['metrics_interval'] = args.metrics_interval
    
    # Create config
    config = BotConfig.from_profile(args.profile, **config_overrides)
    
    # Log configuration
    logger.info("=" * 50)
    logger.info("ENHANCED BOT CONFIGURATION")
    logger.info(f"Profile: {config.profile.value}")
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Filters: spread<{config.max_spread_bps}bps, "
               f"l1>${config.min_depth_l1}, l10>${config.min_depth_l10}, "
               f"vol>${config.min_volume_1m}")
    logger.info(f"Guards: liq_buffer>{config.min_liquidation_buffer_pct}%, "
               f"slippage<{config.max_slippage_impact_bps}bps")
    logger.info(f"RSI confirmation: {config.require_rsi_confirmation}")
    logger.info("=" * 50)
    
    # Create and run bot
    bot = EnhancedPerpsBot(config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        bot.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run bot
    try:
        await bot.run()
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())