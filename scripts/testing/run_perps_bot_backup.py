#!/usr/bin/env python3
"""
Phase 7: End-to-End Perpetuals Trading Bot Runner

This script provides a complete E2E runner for the perpetuals trading system
with multiple configuration profiles and safety features.

Usage:
    python scripts/run_perps_bot.py --profile dev    # Development mode with mocks
    python scripts/run_perps_bot.py --profile demo   # Demo mode with tiny positions
    python scripts/run_perps_bot.py --profile prod   # Production mode (requires auth)
    python scripts/run_perps_bot.py --dry-run        # Dry run mode (no trades)
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

from bot_v2.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy, StrategyConfig, Action
)
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.brokerages.core.interfaces import Product, MarketType, OrderStatus, IBrokerage, Order, OrderSide, OrderType
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore
from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.orchestration.mock_broker import MockBroker
from bot_v2.errors import ExecutionError, ValidationError
from typing import Tuple

# Define ExecutionConfig locally if needed
@dataclass 
class ExecutionConfig:
    """Execution configuration."""
    dry_run: bool = False
    mock_fills: bool = False

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
    """Bot configuration."""
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
    
    # Risk settings
    max_position_size: Decimal = Decimal("1000")  # USD notional
    max_leverage: int = 3
    reduce_only_mode: bool = False
    
    # Execution settings
    mock_broker: bool = False
    mock_fills: bool = False
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC-PERP", "ETH-PERP"]
    
    @classmethod
    def from_profile(cls, profile: str, **overrides) -> BotConfig:
        """Create config from profile name."""
        profile_enum = Profile(profile)
        
        if profile_enum == Profile.DEV:
            config = cls(
                profile=profile_enum,
                mock_broker=True,
                mock_fills=True,
                max_position_size=Decimal("10000"),
                dry_run=True
            )
        elif profile_enum == Profile.DEMO:
            config = cls(
                profile=profile_enum,
                max_position_size=Decimal("100"),  # Tiny positions
                max_leverage=1,
                enable_shorts=False
            )
        else:  # PROD
            config = cls(
                profile=profile_enum,
                max_position_size=Decimal("50000"),
                max_leverage=3,
                enable_shorts=True
            )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


class MockBroker:
    """Mock broker for development/testing."""
    
    def __init__(self):
        self.positions = {}
        self.orders = []
        self.equity = Decimal("100000")
        self.marks = {
            "BTC-PERP": Decimal("50000"),
            "ETH-PERP": Decimal("3000")
        }
    
    def get_account(self):
        """Get mock account."""
        from types import SimpleNamespace
        return SimpleNamespace(equity=self.equity)
    
    def get_positions(self):
        """Get mock positions."""
        return list(self.positions.values())
    
    def get_quote(self, symbol: str):
        """Get mock quote with bid/ask spreads."""
        from types import SimpleNamespace
        price = self.marks.get(symbol, Decimal("1000"))
        # Add small bid/ask spread for realism
        spread = price * Decimal("0.0001")  # 0.01% spread
        return SimpleNamespace(
            last_price=price,
            bid=price - spread,
            ask=price + spread
        )
    
    def place_order(self, symbol: str, side: str, qty: Decimal, order_type: str = "market"):
        """Place mock order."""
        order = {
            "id": f"mock_{len(self.orders)}",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "type": order_type,
            "status": "filled" if order_type == "market" else "open",
            "timestamp": datetime.now()
        }
        self.orders.append(order)
        logger.info(f"Mock order placed: {order}")
        return order
    
    def cancel_order(self, order_id: str):
        """Cancel mock order."""
        for order in self.orders:
            if order["id"] == order_id:
                order["status"] = "cancelled"
                logger.info(f"Mock order cancelled: {order_id}")
                return True
        return False


class PerpsBot:
    """Main perpetuals trading bot."""
    
    def __init__(self, config: BotConfig):
        """Initialize bot with configuration."""
        self.config = config
        self.running = False
        
        # State management and persistence
        storage_root = os.environ.get('EVENT_STORE_ROOT', 'data')
        storage_dir = Path(storage_root) / f"perps_bot/{self.config.profile.value}"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.event_store = EventStore(root=storage_dir)
        self.orders_store = OrdersStore(storage_path=storage_dir)
        
        # Initialize components (broker must come before execution)
        self._init_broker()
        self._init_risk_manager()
        self._init_strategy()
        self._init_execution()
        
        # State tracking
        self.mark_windows: Dict[str, List[Decimal]] = {
            symbol: [] for symbol in config.symbols
        }
        self.last_decisions: Dict[str, Any] = {}
        self._product_map: Dict[str, Product] = {}  # Cache for real products

    async def _reconcile_state_on_startup(self):
        """Reconcile local order state with the exchange on startup."""
        logger.info("Reconciling state with exchange...")
        try:
            # Get open orders from both sources
            local_open_orders = {o.order_id: o for o in self.orders_store.get_open_orders()}
            all_exchange_orders = self.broker.list_orders()
            exchange_open_orders = {o.id: o for o in all_exchange_orders if o.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]}

            logger.info(f"Found {len(local_open_orders)} open orders locally and {len(exchange_open_orders)} on the exchange.")

            # Reconcile missing orders (locally open, but not on exchange)
            for order_id, local_order in local_open_orders.items():
                if order_id not in exchange_open_orders:
                    logger.warning(f"Order {order_id} is OPEN locally but not on exchange. Fetching final status...")
                    # Fetch the final order status from the exchange
                    final_order_status = self.broker.get_order(order_id)
                    if final_order_status:
                        self.orders_store.upsert(final_order_status)
                        logger.info(f"Updated order {order_id} to status: {final_order_status.status.value}")
                    else:
                        logger.error(f"Could not retrieve final status for order {order_id}.")
            
            # Add any orders that are open on the exchange but not tracked locally
            for order_id, exchange_order in exchange_open_orders.items():
                if order_id not in local_open_orders:
                    logger.warning(f"Found untracked OPEN order on exchange: {order_id}. Adding to store.")
                    self.orders_store.upsert(exchange_order)

            logger.info("State reconciliation complete.")
        except Exception as e:
            logger.error(f"Failed to reconcile state on startup: {e}", exc_info=True)
            # Depending on the severity, you might want to halt startup
            raise
        
    def _init_storage(self):
        """Initialize event store."""
        # Allow override for tests
        if 'EVENT_STORE_ROOT' in os.environ:
            storage_dir = Path(os.environ['EVENT_STORE_ROOT']) / f"perps_bot/{self.config.profile.value}"
        else:
            storage_dir = Path(f"data/perps_bot/{self.config.profile.value}")
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.event_store = EventStore(root=storage_dir)
        
    def _init_risk_manager(self):
        """Initialize risk manager."""
        risk_config = RiskConfig(
            leverage_max_global=self.config.max_leverage,
            leverage_max_per_symbol={},  # Can set per-symbol if needed
            max_daily_loss_pct=0.05,  # 5% daily loss limit
            max_exposure_pct=0.8,  # 80% max exposure
            max_position_pct_per_symbol=0.2,  # 20% per symbol
            reduce_only_mode=self.config.reduce_only_mode
        )
        self.risk_manager = LiveRiskManager(
            config=risk_config,
            event_store=self.event_store
        )
        
    def _init_strategy(self):
        """Initialize trading strategy."""
        strategy_config = StrategyConfig(
            short_ma_period=self.config.short_ma,
            long_ma_period=self.config.long_ma,
            target_leverage=self.config.target_leverage,
            trailing_stop_pct=self.config.trailing_stop_pct,
            enable_shorts=self.config.enable_shorts
        )
        self.strategy = BaselinePerpsStrategy(
            config=strategy_config,
            risk_manager=self.risk_manager
        )
        
    def _init_execution(self):
        """Initialize execution engine with risk integration."""
        # Use LiveExecutionEngine for consistent risk checks and logging
        self.exec_engine = LiveExecutionEngine(
            broker=self.broker,
            risk_manager=self.risk_manager,
            event_store=self.event_store,
            bot_id='perps_bot'
        )
        logger.info("Initialized LiveExecutionEngine with risk integration")
        
    def _init_broker(self):
        """Initialize broker connection based on profile."""
        # Check for CI testing override first
        if os.environ.get('PERPS_FORCE_MOCK', '').lower() in ('1', 'true', 'yes'):
            self.broker = MockBroker()
            logger.info("Using mock broker (forced by PERPS_FORCE_MOCK)")
        elif self.config.profile == Profile.DEV:
            # Always use mock for dev
            self.broker = MockBroker()
            logger.info("Using mock broker (dev profile)")
        else:
            # Demo/Prod: Use real broker with safety checks
            try:
                self._validate_broker_environment()
                
                from bot_v2.orchestration.broker_factory import create_brokerage
                logger.info(f"Creating real broker for {self.config.profile.value} profile...")
                
                self.broker = create_brokerage()
                
                # Test connection
                if not self.broker.connect():
                    raise RuntimeError("Failed to connect to broker")
                
                # Verify we can list products and cache them
                products = self.broker.list_products()
                logger.info(f"Connected to broker, found {len(products)} products")
                
                # Cache products for accurate metadata
                for product in products:
                    if hasattr(product, 'symbol'):
                        self._product_map[product.symbol] = product
                        logger.debug(f"Cached product: {product.symbol}")
                
            except Exception as e:
                logger.error(f"Failed to initialize real broker: {e}")
                logger.error("Next steps:")
                if self.config.profile == Profile.DEMO:
                    logger.error("  - Set COINBASE_SANDBOX=1 for demo profile")
                    logger.error("  - Set COINBASE_API_KEY, COINBASE_API_SECRET")
                else:  # PROD
                    logger.error("  - Set COINBASE_API_MODE (advanced or exchange)")
                    logger.error("  - Set authentication credentials based on mode")
                    logger.error("  - Ensure COINBASE_ENABLE_DERIVATIVES=1 for perps")
                sys.exit(1)
    
    def _validate_broker_environment(self):
        """Validate environment for broker connection."""
        if self.config.profile == Profile.DEMO:
            # Demo requires sandbox
            if os.environ.get('COINBASE_SANDBOX') != '1':
                raise ValueError(
                    "Demo profile requires COINBASE_SANDBOX=1\n"
                    "This ensures you're using the sandbox environment"
                )
            logger.info("Demo profile: Using sandbox environment")
            
            # Optionally default to reduce-only for demo
            if not self.config.reduce_only_mode:
                logger.warning("Demo profile: Consider using --reduce-only for safety")
                
        elif self.config.profile == Profile.PROD:
            # Prod requires credentials and refuses sandbox
            if os.environ.get('COINBASE_SANDBOX') == '1':
                raise ValueError(
                    "Production profile cannot use sandbox!\n"
                    "Unset COINBASE_SANDBOX or use demo profile"
                )
            
            # Check for required credentials based on API mode
            api_mode = os.environ.get('COINBASE_API_MODE', 'advanced').lower()
            auth_type = os.environ.get('COINBASE_AUTH_TYPE', 'JWT' if api_mode == 'advanced' else 'HMAC').upper()
            
            missing = []
            if api_mode == 'exchange' and auth_type == 'HMAC':
                # Exchange mode with HMAC
                if not os.environ.get('COINBASE_API_KEY'):
                    missing.append('COINBASE_API_KEY')
                if not os.environ.get('COINBASE_API_SECRET'):
                    missing.append('COINBASE_API_SECRET')
                if not os.environ.get('COINBASE_API_PASSPHRASE'):
                    missing.append('COINBASE_API_PASSPHRASE')
            elif api_mode == 'advanced' and auth_type == 'JWT':
                # Advanced mode with JWT
                if not os.environ.get('COINBASE_CDP_API_KEY'):
                    missing.append('COINBASE_CDP_API_KEY')
                if not os.environ.get('COINBASE_CDP_PRIVATE_KEY'):
                    missing.append('COINBASE_CDP_PRIVATE_KEY')
            else:
                raise ValueError(
                    f"Unknown API mode/auth combination: {api_mode}/{auth_type}\n"
                    "Set COINBASE_API_MODE=advanced and COINBASE_AUTH_TYPE=JWT for perps"
                )
            
            if missing:
                raise ValueError(
                    f"Production profile missing required credentials:\n"
                    f"  Missing: {', '.join(missing)}\n"
                    f"  Mode: {api_mode}, Auth: {auth_type}"
                )
            
            # Warn about derivatives
            if not os.environ.get('COINBASE_ENABLE_DERIVATIVES'):
                logger.warning(
                    "COINBASE_ENABLE_DERIVATIVES not set - perps may not work!\n"
                    "Set COINBASE_ENABLE_DERIVATIVES=1 to enable perpetuals"
                )
    
    def get_product(self, symbol: str) -> Product:
        """Get product metadata from cache or create default."""
        # Try cached real product first (for demo/prod)
        if symbol in self._product_map:
            return self._product_map[symbol]
        
        # Fallback to standard perp product (for dev/mock)
        logger.debug(f"Using default product metadata for {symbol}")
        return Product(
            symbol=symbol,
            base_asset=symbol.split("-")[0],
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            step_size=Decimal("0.001"),
            min_size=Decimal("0.001"),
            price_increment=Decimal("0.01"),
            min_notional=Decimal("10")
        )
    
    async def update_marks(self):
        """Update mark prices for all symbols."""
        for symbol in self.config.symbols:
            try:
                quote = self.broker.get_quote(symbol)
                if quote:
                    mark = Decimal(str(quote.last_price))
                    
                    # Update window
                    if symbol not in self.mark_windows:
                        self.mark_windows[symbol] = []
                    
                    self.mark_windows[symbol].append(mark)
                    
                    # Keep window size reasonable
                    max_size = max(self.config.short_ma, self.config.long_ma) + 5
                    if len(self.mark_windows[symbol]) > max_size:
                        self.mark_windows[symbol] = self.mark_windows[symbol][-max_size:]
                    
                    logger.debug(f"{symbol} mark: {mark}")
                    
            except Exception as e:
                logger.error(f"Error updating mark for {symbol}: {e}")
    
    async def process_symbol(self, symbol: str):
        """Process trading logic for a symbol."""
        try:
            # Get current state
            balances = self.broker.list_balances()
            usd_balance = next((b for b in balances if b.asset == 'USD'), None)
            equity = usd_balance.available if usd_balance else Decimal('0')
            
            if equity == Decimal('0'):
                logger.error(f"No equity info for {symbol}")
                return

            positions = self.broker.list_positions()
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
            product = self.get_product(symbol)
            
            # Generate decision
            decision = self.strategy.decide(
                symbol=symbol,
                current_mark=current_mark,
                position_state=position_state,
                recent_marks=recent_marks,
                equity=equity,
                product=product
            )
            
            self.last_decisions[symbol] = decision
            
            # Log decision
            logger.info(f"{symbol} Decision: {decision.action.value} - {decision.reason}")
            
            # Execute if needed - check Enum values
            if decision.action in {Action.BUY, Action.SELL, Action.CLOSE}:
                await self.execute_decision(symbol, decision, current_mark, product, position_state)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: Optional[Dict[str, Any]]
    ):
        """Execute trading decision using ExecutionEngine.
        
        Args:
            symbol: Trading symbol
            decision: Strategy decision
            mark: Current mark price
            product: Product metadata
            position_state: Current position info (for CLOSE actions)
        """
        try:
            if self.config.dry_run:
                logger.info(f"DRY RUN: Would execute {decision.action.value} for {symbol}")
                logger.info(f"  Target notional: {decision.target_notional}")
                logger.info(f"  Leverage: {decision.leverage}")
                return
            
            # Determine quantity
            if decision.action == Action.CLOSE:
                # For close, use position qty
                if not position_state or position_state.get('qty', 0) == 0:
                    logger.warning(f"No position to close for {symbol}")
                    return
                qty = abs(Decimal(str(position_state['qty'])))
            elif decision.target_notional:
                # Calculate qty from notional
                qty = decision.target_notional / mark
            elif decision.qty:
                qty = decision.qty
            else:
                logger.warning(f"No qty or notional in decision for {symbol}")
                return
            
            # Determine side based on action
            if decision.action == Action.BUY:
                side = OrderSide.BUY
            elif decision.action == Action.SELL:
                side = OrderSide.SELL
            elif decision.action == Action.CLOSE:
                # For close, opposite side of position
                if position_state:
                    pos_side = position_state.get('side', '').lower()
                    side = OrderSide.SELL if pos_side == 'long' else OrderSide.BUY
                else:
                    logger.error(f"Cannot determine close side without position_state")
                    return
            else:
                logger.warning(f"Unknown action: {decision.action}")
                return
            
            # Determine if reduce-only
            reduce_only = decision.reduce_only or self.config.reduce_only_mode
            if decision.action == Action.CLOSE:
                reduce_only = True  # Always reduce-only for closes
            
            # Use ExecutionEngine which handles risk validation and placement
            order = await self._place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=OrderType.MARKET, # Baseline strategy uses market orders
                product=product,
                price=None,
                reduce_only=reduce_only,
                leverage=decision.leverage
            )
            
            if order:
                logger.info(f"Order placed successfully via execution engine: {order.id}")
            else:
                logger.warning(f"Order rejected or failed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error executing decision for {symbol}: {e}")

    async def _place_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: Decimal,
        order_type: OrderType,
        product: Product,
        price: Optional[Decimal] = None,
        reduce_only: bool = False,
        leverage: Optional[int] = None
    ) -> Optional[Order]:
        """Centralized order placement and storage."""
        try:
            order_id = self.exec_engine.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                qty=qty,
                price=price,
                reduce_only=reduce_only,
                leverage=leverage,
                product=product
            )
            
            if order_id:
                # Fetch the full order details to store them
                order = self.broker.get_order(order_id)
                if order:
                    self.orders_store.upsert(order)
                    logger.info(f"Successfully placed and stored order {order.id}")
                    return order
                else:
                    logger.error(f"Could not fetch details for placed order {order_id}")
            return None
        except ValidationError as e:
            logger.warning(f"Order validation failed for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}", exc_info=True)
            return None

    async def run_cycle(self):
        """Run one update cycle."""
        logger.debug("Running update cycle")
        
        # Update marks
        await self.update_marks()
        
        # Process each symbol
        tasks = []
        for symbol in self.config.symbols:
            tasks.append(self.process_symbol(symbol))
        
        await asyncio.gather(*tasks)
        
        # Log status
        self.log_status()
    
    def log_status(self):
        """Log current status."""
        positions = self.broker.list_positions()
        balances = self.broker.list_balances()
        usd_balance = next((b for b in balances if b.asset == 'USD'), None)
        equity = usd_balance.available if usd_balance else 'N/A'
        
        logger.info("=" * 60)
        logger.info(f"Bot Status - {datetime.now()}")
        logger.info(f"Profile: {self.config.profile.value}")
        logger.info(f"Equity: ${equity}")
        logger.info(f"Positions: {len(positions)}")
        
        for symbol, decision in self.last_decisions.items():
            logger.info(f"  {symbol}: {decision.action.value} ({decision.reason})")
        
        logger.info("=" * 60)
    
    async def run(self, single_cycle: bool = False):
        """Main bot loop. 
        
        Args:
            single_cycle: If True, run one cycle and exit (for testing)
        """
        logger.info(f"Starting Perps Bot - Profile: {self.config.profile.value}")
        logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"Update interval: {self.config.update_interval}s")
        
        if self.config.dry_run:
            logger.info("🏃 DRY RUN MODE - No real trades")
        
        if single_cycle:
            logger.info("🚀 DEV-FAST MODE - Single cycle only")
        
        self.running = True
        
        try:
            # Reconcile state before starting the main loop
            await self._reconcile_state_on_startup()

            # Create a background task for runtime guards
            async def runtime_guards_task():
                while self.running:
                    try:
                        self.exec_engine.run_runtime_guards()
                    except Exception as e:
                        logger.error(f"Error in runtime guards task: {e}", exc_info=True)
                    await asyncio.sleep(60) # Run guards every 60 seconds

            guards_task = asyncio.create_task(runtime_guards_task())

            # Run at least one cycle
            await self.run_cycle()
            self.write_health_status(ok=True)
            
            # Continue if not single cycle mode
            if not single_cycle:
                while self.running:
                    await asyncio.sleep(self.config.update_interval)
                    if self.running:  # Check again after sleep
                        await self.run_cycle()
                        self.write_health_status(ok=True)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.write_health_status(ok=True, message="Shutdown by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            import traceback
            traceback.print_exc()
            self.write_health_status(ok=False, error=str(e))
        finally:
            self.running = False
            if 'guards_task' in locals() and not guards_task.done():
                guards_task.cancel()
            await self.shutdown()
    
    def write_health_status(self, ok: bool, message: str = "", error: str = ""):
        """Write health status file."""
        status = {
            "ok": ok,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "error": error
        }
        status_file = Path(f"data/perps_bot/{self.config.profile.value}/health.json")
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down bot...")
        self.running = False
        # Add any cleanup here
        """Initialize risk manager."""
        risk_config = RiskConfig(
            leverage_max_global=self.config.max_leverage,
            leverage_max_per_symbol={},  # Can set per-symbol if needed
            max_daily_loss_pct=0.05,  # 5% daily loss limit
            max_exposure_pct=0.8,  # 80% max exposure
            max_position_pct_per_symbol=0.2,  # 20% per symbol
            reduce_only_mode=self.config.reduce_only_mode
        )
        self.risk_manager = LiveRiskManager(
            config=risk_config,
            event_store=self.event_store
        )
        
    def _init_strategy(self):
        """Initialize trading strategy."""
        strategy_config = StrategyConfig(
            short_ma_period=self.config.short_ma,
            long_ma_period=self.config.long_ma,
            target_leverage=self.config.target_leverage,
            trailing_stop_pct=self.config.trailing_stop_pct,
            enable_shorts=self.config.enable_shorts
        )
        self.strategy = BaselinePerpsStrategy(
            config=strategy_config,
            risk_manager=self.risk_manager
        )
        
    def _init_execution(self):
        """Initialize execution engine with risk integration."""
        # Use LiveExecutionEngine for consistent risk checks and logging
        self.exec_engine = LiveExecutionEngine(
            broker=self.broker,
            risk_manager=self.risk_manager,
            event_store=self.event_store,
            bot_id='perps_bot'
        )
        logger.info("Initialized LiveExecutionEngine with risk integration")
        
    def _init_broker(self):
        """Initialize broker connection based on profile."""
        # Check for CI testing override first
        if os.environ.get('PERPS_FORCE_MOCK', '').lower() in ('1', 'true', 'yes'):
            self.broker = MockBroker()
            logger.info("Using mock broker (forced by PERPS_FORCE_MOCK)")
        elif self.config.profile == Profile.DEV:
            # Always use mock for dev
            self.broker = MockBroker()
            logger.info("Using mock broker (dev profile)")
        else:
            # Demo/Prod: Use real broker with safety checks
            try:
                self._validate_broker_environment()
                
                from bot_v2.orchestration.broker_factory import create_brokerage
                logger.info(f"Creating real broker for {self.config.profile.value} profile...")
                
                self.broker = create_brokerage()
                
                # Test connection
                if not self.broker.connect():
                    raise RuntimeError("Failed to connect to broker")
                
                # Verify we can list products and cache them
                products = self.broker.list_products()
                logger.info(f"Connected to broker, found {len(products)} products")
                
                # Cache products for accurate metadata
                for product in products:
                    if hasattr(product, 'symbol'):
                        self._product_map[product.symbol] = product
                        logger.debug(f"Cached product: {product.symbol}")
                
            except Exception as e:
                logger.error(f"Failed to initialize real broker: {e}")
                logger.error("Next steps:")
                if self.config.profile == Profile.DEMO:
                    logger.error("  - Set COINBASE_SANDBOX=1 for demo profile")
                    logger.error("  - Set COINBASE_API_KEY, COINBASE_API_SECRET")
                else:  # PROD
                    logger.error("  - Set COINBASE_API_MODE (advanced or exchange)")
                    logger.error("  - Set authentication credentials based on mode")
                    logger.error("  - Ensure COINBASE_ENABLE_DERIVATIVES=1 for perps")
                sys.exit(1)
    
    def _validate_broker_environment(self):
        """Validate environment for broker connection."""
        if self.config.profile == Profile.DEMO:
            # Demo requires sandbox
            if os.environ.get('COINBASE_SANDBOX') != '1':
                raise ValueError(
                    "Demo profile requires COINBASE_SANDBOX=1\n"
                    "This ensures you're using the sandbox environment"
                )
            logger.info("Demo profile: Using sandbox environment")
            
            # Optionally default to reduce-only for demo
            if not self.config.reduce_only_mode:
                logger.warning("Demo profile: Consider using --reduce-only for safety")
                
        elif self.config.profile == Profile.PROD:
            # Prod requires credentials and refuses sandbox
            if os.environ.get('COINBASE_SANDBOX') == '1':
                raise ValueError(
                    "Production profile cannot use sandbox!\n"
                    "Unset COINBASE_SANDBOX or use demo profile"
                )
            
            # Check for required credentials based on API mode
            api_mode = os.environ.get('COINBASE_API_MODE', 'advanced').lower()
            auth_type = os.environ.get('COINBASE_AUTH_TYPE', 'JWT' if api_mode == 'advanced' else 'HMAC').upper()
            
            missing = []
            if api_mode == 'exchange' and auth_type == 'HMAC':
                # Exchange mode with HMAC
                if not os.environ.get('COINBASE_API_KEY'):
                    missing.append('COINBASE_API_KEY')
                if not os.environ.get('COINBASE_API_SECRET'):
                    missing.append('COINBASE_API_SECRET')
                if not os.environ.get('COINBASE_API_PASSPHRASE'):
                    missing.append('COINBASE_API_PASSPHRASE')
            elif api_mode == 'advanced' and auth_type == 'JWT':
                # Advanced mode with JWT
                if not os.environ.get('COINBASE_CDP_API_KEY'):
                    missing.append('COINBASE_CDP_API_KEY')
                if not os.environ.get('COINBASE_CDP_PRIVATE_KEY'):
                    missing.append('COINBASE_CDP_PRIVATE_KEY')
            else:
                raise ValueError(
                    f"Unknown API mode/auth combination: {api_mode}/{auth_type}\n"
                    "Set COINBASE_API_MODE=advanced and COINBASE_AUTH_TYPE=JWT for perps"
                )
            
            if missing:
                raise ValueError(
                    f"Production profile missing required credentials:\n"
                    f"  Missing: {', '.join(missing)}\n"
                    f"  Mode: {api_mode}, Auth: {auth_type}"
                )
            
            # Warn about derivatives
            if not os.environ.get('COINBASE_ENABLE_DERIVATIVES'):
                logger.warning(
                    "COINBASE_ENABLE_DERIVATIVES not set - perps may not work!\n"
                    "Set COINBASE_ENABLE_DERIVATIVES=1 to enable perpetuals"
                )
    
    def get_product(self, symbol: str) -> Product:
        """Get product metadata from cache or create default."""
        # Try cached real product first (for demo/prod)
        if symbol in self._product_map:
            return self._product_map[symbol]
        
        # Fallback to standard perp product (for dev/mock)
        logger.debug(f"Using default product metadata for {symbol}")
        return Product(
            symbol=symbol,
            base_asset=symbol.split("-")[0],
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            step_size=Decimal("0.001"),
            min_size=Decimal("0.001"),
            price_increment=Decimal("0.01"),
            min_notional=Decimal("10")
        )
    
    async def update_marks(self):
        """Update mark prices for all symbols."""
        for symbol in self.config.symbols:
            try:
                quote = self.broker.get_quote(symbol)
                if quote:
                    mark = Decimal(str(quote.last_price))
                    
                    # Update window
                    if symbol not in self.mark_windows:
                        self.mark_windows[symbol] = []
                    
                    self.mark_windows[symbol].append(mark)
                    
                    # Keep window size reasonable
                    max_size = max(self.config.short_ma, self.config.long_ma) + 5
                    if len(self.mark_windows[symbol]) > max_size:
                        self.mark_windows[symbol] = self.mark_windows[symbol][-max_size:]
                    
                    logger.debug(f"{symbol} mark: {mark}")
                    
            except Exception as e:
                logger.error(f"Error updating mark for {symbol}: {e}")
    
    async def process_symbol(self, symbol: str):
        """Process trading logic for a symbol."""
        try:
            # Get current state
            account = self.broker.get_account()
            if not account:
                logger.error(f"No account info for {symbol}")
                return
            
            positions = self.broker.list_positions()
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
            product = self.get_product(symbol)
            
            # Generate decision
            decision = self.strategy.decide(
                symbol=symbol,
                current_mark=current_mark,
                position_state=position_state,
                recent_marks=recent_marks,
                equity=Decimal(str(account.equity)),
                product=product
            )
            
            self.last_decisions[symbol] = decision
            
            # Log decision
            logger.info(f"{symbol} Decision: {decision.action.value} - {decision.reason}")
            
            # Execute if needed - check Enum values
            if decision.action in {Action.BUY, Action.SELL, Action.CLOSE}:
                await self.execute_decision(symbol, decision, current_mark, product, position_state)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    async def execute_decision(
        self, 
        symbol: str, 
        decision: Any, 
        mark: Decimal, 
        product: Product,
        position_state: Optional[Dict[str, Any]]
    ):
        """Execute trading decision using ExecutionEngine.
        
        Args:
            symbol: Trading symbol
            decision: Strategy decision
            mark: Current mark price
            product: Product metadata
            position_state: Current position info (for CLOSE actions)
        """
        try:
            if self.config.dry_run:
                logger.info(f"DRY RUN: Would execute {decision.action.value} for {symbol}")
                logger.info(f"  Target notional: {decision.target_notional}")
                logger.info(f"  Leverage: {decision.leverage}")
                return
            
            # Determine quantity
            if decision.action == Action.CLOSE:
                # For close, use position qty
                if not position_state or position_state.get('qty', 0) == 0:
                    logger.warning(f"No position to close for {symbol}")
                    return
                qty = abs(Decimal(str(position_state['qty'])))
            elif decision.target_notional:
                # Calculate qty from notional
                qty = decision.target_notional / mark
            elif decision.qty:
                qty = decision.qty
            else:
                logger.warning(f"No qty or notional in decision for {symbol}")
                return
            
            # Determine side based on action
            if decision.action == Action.BUY:
                side = OrderSide.BUY
            elif decision.action == Action.SELL:
                side = OrderSide.SELL
            elif decision.action == Action.CLOSE:
                # For close, opposite side of position
                if position_state:
                    pos_side = position_state.get('side', '').lower()
                    side = OrderSide.SELL if pos_side == 'long' else OrderSide.BUY
                else:
                    logger.error(f"Cannot determine close side without position_state")
                    return
            else:
                logger.warning(f"Unknown action: {decision.action}")
                return
            
            # Determine if reduce-only
            reduce_only = decision.reduce_only or self.config.reduce_only_mode
            if decision.action == Action.CLOSE:
                reduce_only = True  # Always reduce-only for closes
            
            # Use ExecutionEngine which handles risk validation and placement
            order = await self._place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=OrderType.MARKET, # Baseline strategy uses market orders
                product=product,
                price=None,
                reduce_only=reduce_only,
                leverage=decision.leverage
            )
            
            if order:
                logger.info(f"Order placed successfully via execution engine: {order.id}")
            else:
                logger.warning(f"Order rejected or failed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error executing decision for {symbol}: {e}")

    async def _place_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: Decimal,
        order_type: OrderType,
        product: Product,
        price: Optional[Decimal] = None,
        reduce_only: bool = False,
        leverage: Optional[int] = None
    ) -> Optional[Order]:
        """Centralized order placement and storage."""
        try:
            order_id = self.exec_engine.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                qty=qty,
                price=price,
                reduce_only=reduce_only,
                leverage=leverage,
                product=product
            )
            
            if order_id:
                # Fetch the full order details to store them
                order = self.broker.get_order(order_id)
                if order:
                    self.orders_store.upsert(order)
                    logger.info(f"Successfully placed and stored order {order.id}")
                    return order
                else:
                    logger.error(f"Could not fetch details for placed order {order_id}")
            return None
        except ValidationError as e:
            logger.warning(f"Order validation failed for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}", exc_info=True)
            return None

    async def run_cycle(self):
        """Run one update cycle."""
        logger.debug("Running update cycle")
        
        # Update marks
        await self.update_marks()
        
        # Process each symbol
        tasks = []
        for symbol in self.config.symbols:
            tasks.append(self.process_symbol(symbol))
        
        await asyncio.gather(*tasks)
        
        # Log status
        self.log_status()
    
    def log_status(self):
        """Log current status."""
        positions = self.broker.list_positions()
        balances = self.broker.list_balances()
        usd_balance = next((b for b in balances if b.asset == 'USD'), None)
        equity = usd_balance.available if usd_balance else 'N/A'
        
        logger.info("=" * 60)
        logger.info(f"Bot Status - {datetime.now()}")
        logger.info(f"Profile: {self.config.profile.value}")
        logger.info(f"Equity: ${equity}")
        logger.info(f"Positions: {len(positions)}")
        
        for symbol, decision in self.last_decisions.items():
            logger.info(f"  {symbol}: {decision.action.value} ({decision.reason})")
        
        logger.info("=" * 60)
    
    async def run(self, single_cycle: bool = False):
        """Main bot loop.
        
        Args:
            single_cycle: If True, run one cycle and exit (for testing)
        """
        logger.info(f"Starting Perps Bot - Profile: {self.config.profile.value}")
        logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"Update interval: {self.config.update_interval}s")
        
        if self.config.dry_run:
            logger.info("🏃 DRY RUN MODE - No real trades")
        
        if single_cycle:
            logger.info("🚀 DEV-FAST MODE - Single cycle only")
        
        self.running = True
        
        try:
            # Reconcile state before starting the main loop
            await self._reconcile_state_on_startup()

            # Create a background task for runtime guards
            async def runtime_guards_task():
                while self.running:
                    try:
                        self.exec_engine.run_runtime_guards()
                    except Exception as e:
                        logger.error(f"Error in runtime guards task: {e}", exc_info=True)
                    await asyncio.sleep(60) # Run guards every 60 seconds

            guards_task = asyncio.create_task(runtime_guards_task())

            # Run at least one cycle
            await self.run_cycle()
            self.write_health_status(ok=True)
            
            # Continue if not single cycle mode
            if not single_cycle:
                while self.running:
                    await asyncio.sleep(self.config.update_interval)
                    if self.running:  # Check again after sleep
                        await self.run_cycle()
                        self.write_health_status(ok=True)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.write_health_status(ok=True, message="Shutdown by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            import traceback
            traceback.print_exc()
            self.write_health_status(ok=False, error=str(e))
        finally:
            self.running = False
            if 'guards_task' in locals() and not guards_task.done():
                guards_task.cancel()
            await self.shutdown()
    
    def write_health_status(self, ok: bool, message: str = "", error: str = ""):
        """Write health status file for monitoring."""
        try:
            # Use EVENT_STORE_ROOT if set, otherwise default
            if 'EVENT_STORE_ROOT' in os.environ:
                health_dir = Path(os.environ['EVENT_STORE_ROOT']) / f"perps_bot/{self.config.profile.value}"
            else:
                health_dir = Path(f"data/perps_bot/{self.config.profile.value}")
            
            health_file = health_dir / "health.json"
            health_dir.mkdir(parents=True, exist_ok=True)
            
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "ok": ok,
                "profile": self.config.profile.value,
                "symbols": self.config.symbols,
                "last_decisions": {k: v.action.value if hasattr(v, 'action') else str(v) 
                                 for k, v in self.last_decisions.items()},
                "message": message,
                "error": error
            }
            
            with open(health_file, 'w') as f:
                json.dump(health_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.debug(f"Failed to write health status: {e}")
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down bot...")
        self.running = False
        
        # Save state
        state = {
            "profile": self.config.profile.value,
            "symbols": self.config.symbols,
            "last_decisions": {k: v.action.value if hasattr(v, 'action') else str(v) 
                             for k, v in self.last_decisions.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        # Use EVENT_STORE_ROOT if set, otherwise default
        if 'EVENT_STORE_ROOT' in os.environ:
            state_dir = Path(os.environ['EVENT_STORE_ROOT']) / f"perps_bot/{self.config.profile.value}"
        else:
            state_dir = Path(f"data/perps_bot/{self.config.profile.value}")
        
        state_file = state_dir / "last_state.json"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"State saved to {state_file}")
        logger.info("Bot shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Perpetuals Trading Bot")
    
    parser.add_argument(
        "--profile",
        type=str,
        default="dev",
        choices=["dev", "demo", "prod"],
        help="Configuration profile"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without placing real orders"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to trade (e.g., BTC-PERP ETH-PERP)"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Update interval in seconds"
    )
    
    parser.add_argument(
        "--leverage",
        type=int,
        default=2,
        help="Target leverage"
    )
    
    parser.add_argument(
        "--reduce-only",
        action="store_true",
        help="Enable reduce-only mode"
    )
    
    parser.add_argument(
        "--dev-fast",
        action="store_true",
        help="Run single cycle and exit (for smoke tests)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config_overrides = {}
    if args.symbols:
        config_overrides["symbols"] = args.symbols
    if args.interval:
        config_overrides["update_interval"] = args.interval
    if args.leverage:
        config_overrides["target_leverage"] = args.leverage
    if args.reduce_only:
        config_overrides["reduce_only_mode"] = True
    if args.dry_run:
        config_overrides["dry_run"] = True
    
    config = BotConfig.from_profile(args.profile, **config_overrides)
    
    # Create and run bot
    bot = PerpsBot(config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        bot.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run with dev-fast if specified
    single_cycle = args.dev_fast
    asyncio.run(bot.run(single_cycle=single_cycle))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())