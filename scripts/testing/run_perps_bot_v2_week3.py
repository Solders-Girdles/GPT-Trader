#!/usr/bin/env python3
"""
Week 3 Advanced Perpetuals Trading Bot Runner

Includes:
- Market condition filters (spread, depth, volume)
- RSI confirmation for signals
- Liquidation distance and slippage guards
- WebSocket market snapshot integration
- Advanced order types (limit, stop, stop-limit)
- Impact-aware position sizing
- PnL and funding tracking
- Comprehensive metrics tracking

Usage:
    # Development with filters
    python scripts/run_perps_bot_v2_week3.py --profile dev --order-type limit --post-only
    
    # Demo with stop orders
    python scripts/run_perps_bot_v2_week3.py --profile demo --order-type stop --stop-pct 2
    
    # Production with full features
    python scripts/run_perps_bot_v2_week3.py --profile prod --sizing-mode conservative --max-impact-bps 10
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
    """Enhanced bot configuration with Week 3 features."""
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
    
    # Week 3: Advanced orders
    order_type: str = "market"
    enable_limit_orders: bool = False
    enable_stop_orders: bool = False
    limit_offset_bps: Decimal = Decimal('5')
    stop_pct: Decimal = Decimal('2')
    post_only: bool = False
    sizing_mode: str = "conservative"
    max_impact_bps: Decimal = Decimal('10')
    time_in_force: str = "GTC"
    
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


class Week3PerpsBot:
    """Week 3 perpetuals trading bot with advanced features."""
    
    def __init__(self, config: BotConfig):
        """Initialize Week 3 bot."""
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
        
        # Week 3: PnL tracking
        self.pnl_tracker = PnLTracker()
        
        # Metrics tracking
        self.last_metrics_time = datetime.now()
        
    def _init_storage(self):
        """Initialize event store."""
        # Respect EVENT_STORE_ROOT if set
        if 'EVENT_STORE_ROOT' in os.environ:
            storage_dir = Path(os.environ['EVENT_STORE_ROOT']) / f"perps_bot_v3/{self.config.profile.value}"
        else:
            storage_dir = Path(f"data/perps_bot_v3/{self.config.profile.value}")
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
            bot_id='week3_perps_bot'
        )
        
        logger.info(f"Enhanced strategy initialized with filters: {filters_config}")
        
    def _init_execution(self):
        """Initialize advanced execution engine with Week 3 features."""
        # Map sizing mode string to enum
        sizing_mode_map = {
            'conservative': SizingMode.CONSERVATIVE,
            'strict': SizingMode.STRICT,
            'aggressive': SizingMode.AGGRESSIVE
        }
        
        # Build order config from settings
        order_config = OrderConfig(
            enable_limit_orders=self.config.enable_limit_orders,
            enable_stop_orders=self.config.enable_stop_orders,
            enable_stop_limit=self.config.enable_stop_orders,
            enable_post_only=self.config.post_only,
            sizing_mode=sizing_mode_map.get(self.config.sizing_mode, SizingMode.CONSERVATIVE),
            max_impact_bps=self.config.max_impact_bps,
            reject_on_cross=self.config.post_only  # Reject crossing orders if post-only
        )
        
        self.exec_engine = AdvancedExecutionEngine(
            broker=self.broker,
            config=order_config
        )
        
        logger.info(f"Advanced execution engine initialized: order_type={self.config.order_type}, "
                   f"sizing_mode={self.config.sizing_mode}, max_impact={self.config.max_impact_bps}bps")
        
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
        """Process trading logic for a symbol with Week 3 enhancements."""
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
        """Execute trading decision with Week 3 advanced features."""
        try:
            market_snapshot = self.market_snapshots.get(symbol, {})
            is_exit = decision.action == Action.CLOSE
            
            if is_exit:
                # Close position with reduce-only order
                positions = self.broker.get_positions()
                position = next((p for p in positions if p.symbol == symbol), None)
                
                if not position:
                    logger.warning(f"No position to close for {symbol}")
                    return
                
                close_side = 'sell' if position.qty > 0 else 'buy'
                qty = abs(position.qty)
            else:
                # Open/add to position
                side = 'buy' if decision.action == Action.BUY else 'sell'
                
                # Use impact-aware sizing for target notional
                if decision.target_notional and current_mark > 0:
                    # Calculate impact-aware size
                    adjusted_notional, impact = self.exec_engine.calculate_impact_aware_size(
                        target_notional=decision.target_notional,
                        market_snapshot=market_snapshot
                    )
                    
                    if adjusted_notional == 0:
                        logger.warning(f"{symbol}: Impact too high, order rejected")
                        return
                    
                    qty = adjusted_notional / current_mark
                    
                    # Log sizing adjustment if any
                    if adjusted_notional < decision.target_notional:
                        logger.info(f"{symbol}: Sized down from ${decision.target_notional} to "
                                  f"${adjusted_notional:.0f} (impact: {impact:.1f}bps)")
                    
                    # Quantize to product requirements
                    if product and hasattr(product, 'step_size'):
                        qty = (qty // product.step_size) * product.step_size
                else:
                    qty = Decimal('0.001')  # Minimum fallback
            
            # Prepare order parameters based on type
            order_params = {
                'symbol': symbol,
                'side': close_side if is_exit else side,
                'quantity': qty,
                'order_type': self.config.order_type,
                'time_in_force': self.config.time_in_force,
                'reduce_only': is_exit or decision.reduce_only,
                'client_id': f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            }
            
            # Add type-specific parameters
            if self.config.order_type == 'limit':
                # Calculate limit price with offset from bid/ask
                quote = self.broker.get_quote(symbol)
                if quote:
                    if (is_exit and close_side == 'sell') or (not is_exit and side == 'buy'):
                        # Buy side: offset below bid
                        offset = current_mark * (self.config.limit_offset_bps / Decimal('10000'))
                        order_params['limit_price'] = quote.bid - offset
                    else:
                        # Sell side: offset above ask
                        offset = current_mark * (self.config.limit_offset_bps / Decimal('10000'))
                        order_params['limit_price'] = quote.ask + offset
                    
                    order_params['post_only'] = self.config.post_only
            
            elif self.config.order_type == 'stop':
                # Calculate stop price
                stop_offset = current_mark * (self.config.stop_pct / Decimal('100'))
                if side == 'sell' or is_exit:
                    order_params['stop_price'] = current_mark - stop_offset  # Below for sell
                else:
                    order_params['stop_price'] = current_mark + stop_offset  # Above for buy
            
            elif self.config.order_type == 'stop_limit':
                # Calculate both stop and limit prices
                stop_offset = current_mark * (self.config.stop_pct / Decimal('100'))
                limit_offset = current_mark * (self.config.limit_offset_bps / Decimal('10000'))
                
                if side == 'sell' or is_exit:
                    order_params['stop_price'] = current_mark - stop_offset
                    order_params['limit_price'] = current_mark - stop_offset - limit_offset
                else:
                    order_params['stop_price'] = current_mark + stop_offset
                    order_params['limit_price'] = current_mark + stop_offset + limit_offset
            
            # Place order through advanced engine
            result = self.exec_engine.place_order(**order_params)
            
            if result:
                logger.info(f"Order placed for {symbol}: {self.config.order_type} {side if not is_exit else 'close'} "
                          f"{qty:.6f} @ {result.price if hasattr(result, 'price') else 'market'}")
                
                # Update PnL tracker if filled
                if hasattr(result, 'status') and result.status == 'filled':
                    fill_price = result.price if hasattr(result, 'price') else current_mark
                    pnl_result = self.pnl_tracker.update_position(
                        symbol=symbol,
                        side=side if not is_exit else close_side,
                        quantity=qty,
                        price=fill_price,
                        is_reduce=is_exit
                    )
                    
                    if pnl_result['realized_pnl'] != 0:
                        logger.info(f"{symbol} Realized PnL: ${pnl_result['realized_pnl']:.2f}")
            else:
                logger.warning(f"Order failed for {symbol}")
            
        except Exception as e:
            logger.error(f"Execution failed for {symbol}: {e}", exc_info=True)
    
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
        
        # Get execution metrics
        exec_metrics = self.exec_engine.get_metrics()
        
        # Get PnL metrics
        account = self.broker.get_account()
        current_equity = Decimal(str(account.equity)) if account else Decimal('10000')
        pnl_metrics = self.pnl_tracker.generate_daily_metrics(current_equity)
        
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
        
        # Log execution metrics
        logger.info("\nExecution metrics:")
        logger.info(f"  Orders placed: {exec_metrics['orders']['placed']}")
        logger.info(f"  Orders filled: {exec_metrics['orders']['filled']}")
        logger.info(f"  Orders cancelled: {exec_metrics['orders']['cancelled']}")
        logger.info(f"  Orders rejected: {exec_metrics['orders']['rejected']}")
        if exec_metrics['orders'].get('post_only_rejected'):
            logger.info(f"  Post-only rejected: {exec_metrics['orders']['post_only_rejected']}")
        if exec_metrics.get('stop_triggers'):
            logger.info(f"  Stop triggers: {exec_metrics['stop_triggers']}")
        
        # Log PnL metrics
        logger.info("\nPnL metrics:")
        logger.info(f"  Total PnL: ${pnl_metrics['total_pnl']:.2f}")
        logger.info(f"  Realized: ${pnl_metrics['realized_pnl']:.2f}")
        logger.info(f"  Unrealized: ${pnl_metrics['unrealized_pnl']:.2f}")
        logger.info(f"  Funding paid: ${pnl_metrics['funding_paid']:.2f}")
        logger.info(f"  Win rate: {pnl_metrics['win_rate']:.1%}")
        logger.info(f"  Daily return: {pnl_metrics['daily_return']:.2%}")
        logger.info(f"  Sharpe: {pnl_metrics['sharpe']:.2f}")
        
        # Append metrics to event store
        self.event_store.append_metric('week3_perps_bot', {
            'timestamp': datetime.now().isoformat(),
            'strategy': metrics,
            'execution': exec_metrics,
            'pnl': pnl_metrics,
            'market_snapshots': self.market_snapshots
        })
        
        logger.info("=" * 50)
    
    async def run(self):
        """Main bot loop."""
        self.running = True
        logger.info(f"Starting Week 3 bot with profile: {self.config.profile.value}")
        
        while self.running:
            try:
                # Update marks and market data
                await self.update_marks()
                await self.update_market_snapshots()
                
                # Process each symbol
                for symbol in self.config.symbols:
                    await self.process_symbol(symbol)
                
                # Check stop triggers
                current_prices = {}
                for symbol in self.config.symbols:
                    if symbol in self.mark_windows and self.mark_windows[symbol]:
                        current_prices[symbol] = self.mark_windows[symbol][-1]
                
                if current_prices:
                    triggered = self.exec_engine.check_stop_triggers(current_prices)
                    if triggered:
                        logger.info(f"Stop triggers activated: {triggered}")
                
                # Update PnL with current marks
                mark_prices = {}
                for symbol in self.config.symbols:
                    if symbol in self.mark_windows and self.mark_windows[symbol]:
                        mark_prices[symbol] = self.mark_windows[symbol][-1]
                
                if mark_prices:
                    self.pnl_tracker.update_marks(mark_prices)
                    
                    # Check for funding accruals
                    for symbol in self.config.symbols:
                        if symbol in mark_prices:
                            # Get funding rate from market snapshot or broker
                            funding_rate = Decimal('0.0001')  # Default 0.01%
                            snapshot = self.market_snapshots.get(symbol, {})
                            if 'funding_rate' in snapshot:
                                funding_rate = Decimal(str(snapshot['funding_rate']))
                            
                            # Accrue if due
                            funding = self.pnl_tracker.accrue_funding(
                                symbol=symbol,
                                mark_price=mark_prices[symbol],
                                funding_rate=funding_rate
                            )
                            
                            if funding:
                                logger.info(f"{symbol} Funding accrued: ${funding:+.4f}")
                
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
        logger.info("Shutting down Week 3 bot...")
        self.running = False
        
        # Log final metrics
        metrics = self.strategy.get_metrics()
        exec_metrics = self.exec_engine.get_metrics()
        pnl_total = self.pnl_tracker.get_total_pnl()
        
        logger.info("Final metrics:")
        logger.info(f"  Strategy: {json.dumps(metrics, indent=2)}")
        logger.info(f"  Execution: {json.dumps(exec_metrics, indent=2)}")
        logger.info(f"  PnL Total: ${pnl_total['total']:.2f} "
                   f"(R: ${pnl_total['realized']:.2f}, U: ${pnl_total['unrealized']:.2f})")
        
        # Disconnect broker
        if hasattr(self.broker, 'disconnect'):
            self.broker.disconnect()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Week 3 Advanced Perpetuals Trading Bot")
    
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
    
    # Week 3: Advanced orders
    parser.add_argument('--order-type', choices=['market', 'limit', 'stop', 'stop_limit'],
                      default='market', help='Order type to use')
    parser.add_argument('--limit-offset-bps', type=float, default=5,
                      help='Limit order offset from bid/ask in bps')
    parser.add_argument('--stop-pct', type=float, default=2,
                      help='Stop price offset percentage')
    parser.add_argument('--post-only', action='store_true',
                      help='Enable post-only for limit orders')
    parser.add_argument('--sizing-mode', choices=['conservative', 'strict', 'aggressive'],
                      default='conservative', help='Impact-aware sizing mode')
    parser.add_argument('--max-impact-bps', type=float, default=10,
                      help='Maximum market impact in basis points')
    parser.add_argument('--tif', choices=['GTC', 'IOC', 'FOK'], default='GTC',
                      help='Time-in-force for orders')
    
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
    
    # Week 3: Advanced orders
    config_overrides['order_type'] = args.order_type
    config_overrides['enable_limit_orders'] = args.order_type in ['limit', 'stop_limit']
    config_overrides['enable_stop_orders'] = args.order_type in ['stop', 'stop_limit']
    config_overrides['limit_offset_bps'] = Decimal(str(args.limit_offset_bps))
    config_overrides['stop_pct'] = Decimal(str(args.stop_pct))
    config_overrides['post_only'] = args.post_only
    config_overrides['sizing_mode'] = args.sizing_mode
    config_overrides['max_impact_bps'] = Decimal(str(args.max_impact_bps))
    config_overrides['time_in_force'] = args.tif
    
    # Intervals
    config_overrides['update_interval'] = args.update_interval
    config_overrides['metrics_interval'] = args.metrics_interval
    
    # Create config
    config = BotConfig.from_profile(args.profile, **config_overrides)
    
    # Log configuration
    logger.info("=" * 50)
    logger.info("WEEK 3 ADVANCED BOT CONFIGURATION")
    logger.info(f"Profile: {config.profile.value}")
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Order type: {config.order_type} (TIF: {config.time_in_force})")
    if config.order_type == 'limit':
        logger.info(f"  Limit offset: {config.limit_offset_bps}bps, Post-only: {config.post_only}")
    if 'stop' in config.order_type:
        logger.info(f"  Stop offset: {config.stop_pct}%")
    logger.info(f"Sizing: {config.sizing_mode} (max impact: {config.max_impact_bps}bps)")
    logger.info(f"Filters: spread<{config.max_spread_bps}bps, "
               f"l1>${config.min_depth_l1}, l10>${config.min_depth_l10}, "
               f"vol>${config.min_volume_1m}")
    logger.info(f"Guards: liq_buffer>{config.min_liquidation_buffer_pct}%, "
               f"slippage<{config.max_slippage_impact_bps}bps")
    logger.info(f"RSI confirmation: {config.require_rsi_confirmation}")
    logger.info("=" * 50)
    
    # Create and run bot
    bot = Week3PerpsBot(config)
    
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