"""
Order Management System (OMS) for GPT-Trader Production Trading

Comprehensive order lifecycle management providing:
- Order validation and pre-trade checks
- Smart order routing and execution algorithms
- Real-time order status tracking
- Execution cost analysis and reporting
- Trade settlement and confirmation
- Regulatory compliance and audit trails

This is the core order processing system for production live trading.
"""

import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from bot.dataflow.streaming_data import StreamingDataManager

# Live trading imports
from bot.live.trading_engine_v2 import Order, OrderSide, OrderStatus, OrderType
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class OrderValidationResult(Enum):
    """Order validation results"""

    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"
    CONDITIONAL_APPROVAL = "conditional_approval"


class ExecutionAlgorithm(Enum):
    """Order execution algorithms"""

    MARKET = "market"  # Immediate market execution
    LIMIT = "limit"  # Limit price execution
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    IMPLEMENTATION_SHORTFALL = "is"  # Implementation Shortfall
    ARRIVAL_PRICE = "arrival_price"  # Arrival Price algorithm
    PERCENTAGE_OF_VOLUME = "pov"  # Percentage of Volume


class RoutingVenue(Enum):
    """Order routing venues"""

    SMART = "smart"  # Smart order routing
    NYSE = "nyse"  # New York Stock Exchange
    NASDAQ = "nasdaq"  # NASDAQ
    BATS = "bats"  # BATS Exchange
    IEX = "iex"  # Investors Exchange
    DARK_POOL = "dark_pool"  # Dark pool execution


@dataclass
class OrderValidation:
    """Order validation result"""

    order_id: str
    validation_result: OrderValidationResult
    validation_checks: dict[str, bool]
    risk_score: float
    estimated_cost: Decimal
    warnings: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionInstruction:
    """Detailed execution instructions"""

    algorithm: ExecutionAlgorithm
    routing_venue: RoutingVenue
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    execution_params: dict[str, Any] = field(default_factory=dict)

    # Algorithm-specific parameters
    max_participation_rate: float | None = None  # For POV, TWAP
    target_completion_time: datetime | None = None  # For TWAP, IS
    price_improvement_tolerance: Decimal = Decimal("0.001")  # 0.1 cent

    # Risk controls
    max_slice_size: int | None = None
    min_slice_size: int | None = None
    randomize_timing: bool = True


@dataclass
class OrderSlice:
    """Individual order slice for execution algorithms"""

    slice_id: str
    parent_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    slice_type: OrderType
    limit_price: Decimal | None = None

    # Execution tracking
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    filled_quantity: int = 0
    average_fill_price: Decimal | None = None

    # Venue information
    routing_venue: RoutingVenue | None = None
    broker_slice_id: str | None = None


@dataclass
class ExecutionSummary:
    """Summary of order execution"""

    order_id: str
    symbol: str
    total_quantity: int
    filled_quantity: int
    remaining_quantity: int

    # Pricing
    average_fill_price: Decimal
    arrival_price: Decimal
    benchmark_price: Decimal

    # Costs
    total_commission: Decimal
    market_impact_bps: float
    timing_cost_bps: float
    total_cost_bps: float

    # Performance
    slices_executed: int
    venues_used: list[str]
    execution_time_seconds: float
    fill_rate: float

    # Quality metrics
    price_improvement: Decimal
    adverse_selection: Decimal
    execution_quality_score: float

    timestamp: datetime = field(default_factory=datetime.now)


class OrderManagementSystem:
    """Comprehensive Order Management System"""

    def __init__(
        self, oms_dir: str = "data/order_management", streaming_manager: StreamingDataManager = None
    ) -> None:

        self.oms_dir = Path(oms_dir)
        self.oms_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.oms_dir / "orders").mkdir(exist_ok=True)
        (self.oms_dir / "executions").mkdir(exist_ok=True)
        (self.oms_dir / "slices").mkdir(exist_ok=True)
        (self.oms_dir / "reports").mkdir(exist_ok=True)

        # Initialize streaming data manager
        if streaming_manager is None:
            from bot.dataflow.streaming_data import create_streaming_data_manager

            self.streaming_manager = create_streaming_data_manager()
        else:
            self.streaming_manager = streaming_manager

        # Initialize database
        self.db_path = self.oms_dir / "order_management.db"
        self._initialize_database()

        # Order management state
        self.active_orders: dict[str, Order] = {}
        self.order_slices: dict[str, list[OrderSlice]] = {}  # order_id -> slices
        self.execution_summaries: dict[str, ExecutionSummary] = {}

        # Execution queues
        self.validation_queue = queue.Queue()
        self.execution_queue = queue.Queue()
        self.settlement_queue = queue.Queue()

        # Threading
        self.validation_thread = None
        self.execution_thread = None
        self.settlement_thread = None
        self.is_running = False

        # Risk limits (configurable)
        self.max_order_value = Decimal("1000000")  # $1M max order
        self.max_position_concentration = 0.10  # 10% max position
        self.max_daily_volume_participation = 0.10  # 10% max daily volume

        # Execution parameters
        self.default_slice_size = 100  # Default slice size
        self.max_slice_size = 1000  # Maximum slice size
        self.execution_delay_ms = 100  # 100ms between slices

        # Performance tracking
        self.orders_processed = 0
        self.orders_filled = 0
        self.total_execution_cost_bps = 0.0
        self.average_fill_time_seconds = 0.0

        logger.info(f"Order Management System initialized at {self.oms_dir}")

    def _initialize_database(self) -> None:
        """Initialize SQLite database for OMS"""

        with sqlite3.connect(self.db_path) as conn:
            # Order validations table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_validations (
                    order_id TEXT PRIMARY KEY,
                    validation_result TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    estimated_cost TEXT NOT NULL,
                    validation_data TEXT,
                    validation_timestamp TEXT NOT NULL
                )
            """
            )

            # Execution instructions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_instructions (
                    order_id TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    routing_venue TEXT NOT NULL,
                    time_in_force TEXT NOT NULL,
                    execution_params TEXT,
                    created_at TEXT NOT NULL
                )
            """
            )

            # Order slices table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_slices (
                    slice_id TEXT PRIMARY KEY,
                    parent_order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    slice_type TEXT NOT NULL,
                    limit_price TEXT,
                    status TEXT NOT NULL,
                    submitted_at TEXT,
                    filled_at TEXT,
                    filled_quantity INTEGER DEFAULT 0,
                    average_fill_price TEXT,
                    routing_venue TEXT,
                    broker_slice_id TEXT,
                    slice_data TEXT
                )
            """
            )

            # Execution summaries table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_summaries (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    total_quantity INTEGER NOT NULL,
                    filled_quantity INTEGER NOT NULL,
                    average_fill_price TEXT NOT NULL,
                    arrival_price TEXT NOT NULL,
                    total_commission TEXT NOT NULL,
                    market_impact_bps REAL,
                    total_cost_bps REAL,
                    execution_time_seconds REAL,
                    fill_rate REAL,
                    execution_quality_score REAL,
                    timestamp TEXT NOT NULL,
                    summary_data TEXT
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_slices_parent ON order_slices (parent_order_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_slices_status ON order_slices (status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_summaries_symbol ON execution_summaries (symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_summaries_timestamp ON execution_summaries (timestamp)"
            )

            conn.commit()

    def start_order_processing(self) -> None:
        """Start order processing threads"""

        if self.is_running:
            console.print("⚠️  Order processing is already running")
            return

        console.print("🚀 [bold green]Starting Order Management System[/bold green]")

        self.is_running = True

        # Start processing threads
        self.validation_thread = threading.Thread(target=self._validation_loop, daemon=True)
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.settlement_thread = threading.Thread(target=self._settlement_loop, daemon=True)

        self.validation_thread.start()
        self.execution_thread.start()
        self.settlement_thread.start()

        console.print("   ✅ Order validation thread started")
        console.print("   ⚡ Order execution thread started")
        console.print("   📋 Trade settlement thread started")

        logger.info("Order Management System started successfully")

    def stop_order_processing(self) -> None:
        """Stop order processing threads"""

        console.print("⏹️  [bold yellow]Stopping Order Management System[/bold yellow]")

        self.is_running = False

        # Wait for threads to finish
        for thread in [self.validation_thread, self.execution_thread, self.settlement_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)

        console.print("   ✅ Order processing stopped")

        logger.info("Order Management System stopped")

    def submit_order_for_validation(
        self, order: Order, execution_instruction: ExecutionInstruction = None
    ) -> str:
        """Submit order for validation and processing"""

        try:
            # Create default execution instruction if not provided
            if execution_instruction is None:
                execution_instruction = ExecutionInstruction(
                    algorithm=ExecutionAlgorithm.MARKET, routing_venue=RoutingVenue.SMART
                )

            # Queue for validation
            self.validation_queue.put(
                {
                    "order": order,
                    "execution_instruction": execution_instruction,
                    "submitted_at": datetime.now(),
                }
            )

            console.print(
                f"   📋 Order queued for validation: {order.symbol} {order.side.value} {order.quantity}"
            )

            return order.order_id

        except Exception as e:
            logger.error(f"Failed to submit order for validation: {str(e)}")
            raise

    def _validation_loop(self) -> None:
        """Order validation processing loop"""

        while self.is_running:
            try:
                # Get order from validation queue
                order_data = self.validation_queue.get(timeout=1)

                # Validate the order
                self._validate_order(order_data)

                self.validation_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Validation loop error: {str(e)}")
                time.sleep(0.1)

    def _execution_loop(self) -> None:
        """Order execution processing loop"""

        while self.is_running:
            try:
                # Get order from execution queue
                order_data = self.execution_queue.get(timeout=1)

                # Execute the order
                self._execute_order(order_data)

                self.execution_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Execution loop error: {str(e)}")
                time.sleep(0.1)

    def _settlement_loop(self) -> None:
        """Trade settlement processing loop"""

        while self.is_running:
            try:
                # Get trade from settlement queue
                trade_data = self.settlement_queue.get(timeout=1)

                # Process settlement
                self._process_settlement(trade_data)

                self.settlement_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Settlement loop error: {str(e)}")
                time.sleep(0.1)

    def _validate_order(self, order_data: dict[str, Any]) -> None:
        """Comprehensive order validation"""

        order = order_data["order"]
        execution_instruction = order_data["execution_instruction"]

        try:
            console.print(f"   🔍 Validating order: {order.order_id}")

            validation_checks = {}
            warnings = []
            rejections = []
            risk_score = 0.0

            # 1. Basic order validation
            if order.quantity <= 0:
                validation_checks["positive_quantity"] = False
                rejections.append("Order quantity must be positive")
            else:
                validation_checks["positive_quantity"] = True

            # 2. Price validation
            if order.order_type == OrderType.LIMIT and order.limit_price is None:
                validation_checks["limit_price"] = False
                rejections.append("Limit orders must have limit price")
            else:
                validation_checks["limit_price"] = True

            # 3. Market hours validation
            current_time = datetime.now().time()
            market_open = datetime.strptime("09:30", "%H:%M").time()
            market_close = datetime.strptime("16:00", "%H:%M").time()

            if market_open <= current_time <= market_close:
                validation_checks["market_hours"] = True
            else:
                validation_checks["market_hours"] = False
                warnings.append("Order submitted outside market hours")
                risk_score += 0.1

            # 4. Position concentration check
            # Get current market price for value calculation
            latest_quote = self.streaming_manager.get_latest_quote(order.symbol)
            estimated_price = latest_quote.mid_price if latest_quote else Decimal("100")
            order_value = order.quantity * estimated_price

            if order_value > self.max_order_value:
                validation_checks["order_size"] = False
                rejections.append(
                    f"Order value ${order_value} exceeds maximum ${self.max_order_value}"
                )
            else:
                validation_checks["order_size"] = True
                if order_value > self.max_order_value * Decimal("0.5"):
                    warnings.append("Large order size detected")
                    risk_score += 0.2

            # 5. Liquidity check (simplified)
            if latest_quote:
                spread_bps = latest_quote.spread_bps
                if spread_bps > 50:  # 5 bps threshold
                    validation_checks["liquidity"] = False
                    warnings.append(f"Wide spread detected: {spread_bps:.1f} bps")
                    risk_score += 0.3
                else:
                    validation_checks["liquidity"] = True
            else:
                validation_checks["liquidity"] = False
                warnings.append("No market data available for liquidity assessment")
                risk_score += 0.2

            # 6. Symbol validation
            # In production, this would check if symbol is tradeable
            validation_checks["symbol_validity"] = True

            # 7. Strategy validation (if provided)
            if order.strategy_id:
                validation_checks["strategy_validation"] = True
            else:
                validation_checks["strategy_validation"] = False
                warnings.append("No strategy ID provided")

            # Determine validation result
            failed_checks = [k for k, v in validation_checks.items() if not v]
            critical_failures = [
                "positive_quantity",
                "limit_price",
                "order_size",
                "symbol_validity",
            ]

            has_critical_failure = any(check in failed_checks for check in critical_failures)

            if has_critical_failure:
                validation_result = OrderValidationResult.REJECTED
            elif risk_score > 0.5:
                validation_result = OrderValidationResult.PENDING_REVIEW
            elif warnings:
                validation_result = OrderValidationResult.CONDITIONAL_APPROVAL
            else:
                validation_result = OrderValidationResult.APPROVED

            # Create validation record
            validation = OrderValidation(
                order_id=order.order_id,
                validation_result=validation_result,
                validation_checks=validation_checks,
                risk_score=risk_score,
                estimated_cost=order_value * Decimal("0.0001"),  # 1bp estimate
                warnings=warnings,
                rejections=rejections,
            )

            # Store validation
            self._store_order_validation(validation)

            # Process validation result
            if validation_result == OrderValidationResult.APPROVED:
                console.print(f"      ✅ Order approved: {order.order_id}")
                # Queue for execution
                self.execution_queue.put(
                    {
                        "order": order,
                        "execution_instruction": execution_instruction,
                        "validation": validation,
                    }
                )
            elif validation_result == OrderValidationResult.CONDITIONAL_APPROVAL:
                console.print(f"      ⚠️  Order conditionally approved: {order.order_id}")
                # Queue for execution with warnings
                self.execution_queue.put(
                    {
                        "order": order,
                        "execution_instruction": execution_instruction,
                        "validation": validation,
                    }
                )
            else:
                console.print(f"      ❌ Order rejected: {order.order_id}")
                order.status = OrderStatus.REJECTED
                order.notes = "; ".join(rejections)

        except Exception as e:
            logger.error(f"Order validation failed: {str(e)}")
            order.status = OrderStatus.REJECTED
            order.notes = f"Validation error: {str(e)}"

    def _execute_order(self, order_data: dict[str, Any]) -> None:
        """Execute validated order"""

        order = order_data["order"]
        execution_instruction = order_data["execution_instruction"]
        order_data["validation"]

        try:
            start_time = datetime.now()

            console.print(f"   ⚡ Executing order: {order.order_id}")

            # Store execution instruction
            self._store_execution_instruction(order.order_id, execution_instruction)

            # Create execution slices based on algorithm
            slices = self._create_execution_slices(order, execution_instruction)
            self.order_slices[order.order_id] = slices

            # Execute slices
            filled_slices = []
            total_filled_quantity = 0
            total_value = Decimal("0")

            for slice_order in slices:
                slice_result = self._execute_slice(slice_order, execution_instruction)

                if slice_result["success"]:
                    slice_order.status = OrderStatus.FILLED
                    slice_order.filled_quantity = slice_result["filled_quantity"]
                    slice_order.average_fill_price = slice_result["filled_price"]
                    slice_order.filled_at = datetime.now()

                    filled_slices.append(slice_order)
                    total_filled_quantity += slice_result["filled_quantity"]
                    total_value += slice_result["filled_quantity"] * slice_result["filled_price"]
                else:
                    slice_order.status = OrderStatus.REJECTED

                # Store slice
                self._store_order_slice(slice_order)

                # Brief delay between slices
                if len(slices) > 1:
                    time.sleep(self.execution_delay_ms / 1000.0)

            # Update main order
            if total_filled_quantity > 0:
                order.status = (
                    OrderStatus.FILLED
                    if total_filled_quantity == order.quantity
                    else OrderStatus.PARTIALLY_FILLED
                )
                order.filled_quantity = Decimal(str(total_filled_quantity))
                order.average_fill_price = total_value / Decimal(str(total_filled_quantity))
                order.filled_at = datetime.now()
            else:
                order.status = OrderStatus.REJECTED
                order.notes = "All execution slices failed"

            # Create execution summary
            execution_summary = self._create_execution_summary(order, filled_slices, start_time)

            self.execution_summaries[order.order_id] = execution_summary
            self._store_execution_summary(execution_summary)

            # Queue for settlement
            if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                self.settlement_queue.put(
                    {
                        "order": order,
                        "execution_summary": execution_summary,
                        "filled_slices": filled_slices,
                    }
                )

            # Update performance metrics
            self.orders_processed += 1
            if order.status == OrderStatus.FILLED:
                self.orders_filled += 1

            console.print(
                f"      ✅ Execution complete: {total_filled_quantity}/{order.quantity} filled"
            )

        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}")
            order.status = OrderStatus.REJECTED
            order.notes = f"Execution error: {str(e)}"

    def _create_execution_slices(
        self, order: Order, instruction: ExecutionInstruction
    ) -> list[OrderSlice]:
        """Create execution slices based on algorithm"""

        slices = []

        if instruction.algorithm == ExecutionAlgorithm.MARKET:
            # Single slice for market orders
            slice_id = f"{order.order_id}_slice_0"
            slice_order = OrderSlice(
                slice_id=slice_id,
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=int(order.quantity),
                slice_type=OrderType.MARKET,
                routing_venue=instruction.routing_venue,
            )
            slices.append(slice_order)

        elif instruction.algorithm == ExecutionAlgorithm.TWAP:
            # Time-weighted average price - split into time slices
            num_slices = min(10, max(1, int(order.quantity) // 100))  # 1-10 slices
            slice_size = int(order.quantity) // num_slices
            remainder = int(order.quantity) % num_slices

            for i in range(num_slices):
                quantity = slice_size + (1 if i < remainder else 0)
                slice_id = f"{order.order_id}_twap_{i}"

                slice_order = OrderSlice(
                    slice_id=slice_id,
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=quantity,
                    slice_type=OrderType.LIMIT,  # Use limit orders for TWAP
                    limit_price=order.limit_price,
                    routing_venue=instruction.routing_venue,
                )
                slices.append(slice_order)

        else:
            # Default to single slice
            slice_id = f"{order.order_id}_slice_0"
            slice_order = OrderSlice(
                slice_id=slice_id,
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=int(order.quantity),
                slice_type=order.order_type,
                limit_price=order.limit_price,
                routing_venue=instruction.routing_venue,
            )
            slices.append(slice_order)

        return slices

    def _execute_slice(
        self, slice_order: OrderSlice, instruction: ExecutionInstruction
    ) -> dict[str, Any]:
        """Execute individual order slice"""

        try:
            # Simulate slice execution (in production, this would route to actual venues)
            slice_order.submitted_at = datetime.now()
            slice_order.status = OrderStatus.SUBMITTED

            # Get current market data
            latest_quote = self.streaming_manager.get_latest_quote(slice_order.symbol)
            if not latest_quote:
                return {"success": False, "error": "No market data available"}

            # Determine execution price based on order type
            if slice_order.slice_type == OrderType.MARKET:
                if slice_order.side == OrderSide.BUY:
                    fill_price = latest_quote.ask_price
                else:
                    fill_price = latest_quote.bid_price
            elif slice_order.slice_type == OrderType.LIMIT:
                # For limit orders, assume we get filled at limit price
                fill_price = slice_order.limit_price
            else:
                fill_price = latest_quote.mid_price

            # Add realistic slippage
            slippage_bps = np.random.normal(1.0, 0.5)  # Mean 1bp, std 0.5bp
            slippage_factor = Decimal(str(1 + slippage_bps / 10000))

            if slice_order.side == OrderSide.BUY:
                execution_price = fill_price * slippage_factor
            else:
                execution_price = fill_price / slippage_factor

            # Simulate partial fills for large slices
            fill_rate = min(1.0, np.random.beta(8, 2))  # Typically high fill rates
            filled_quantity = int(slice_order.quantity * fill_rate)

            if filled_quantity > 0:
                return {
                    "success": True,
                    "filled_quantity": filled_quantity,
                    "filled_price": execution_price,
                    "venue": instruction.routing_venue.value,
                    "execution_time": datetime.now(),
                }
            else:
                return {"success": False, "error": "No fill obtained"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_execution_summary(
        self, order: Order, filled_slices: list[OrderSlice], start_time: datetime
    ) -> ExecutionSummary:
        """Create comprehensive execution summary"""

        total_filled = sum(slice_order.filled_quantity for slice_order in filled_slices)

        if total_filled == 0:
            # No fills
            return ExecutionSummary(
                order_id=order.order_id,
                symbol=order.symbol,
                total_quantity=int(order.quantity),
                filled_quantity=0,
                remaining_quantity=int(order.quantity),
                average_fill_price=Decimal("0"),
                arrival_price=Decimal("0"),
                benchmark_price=Decimal("0"),
                total_commission=Decimal("0"),
                market_impact_bps=0.0,
                timing_cost_bps=0.0,
                total_cost_bps=0.0,
                slices_executed=len(filled_slices),
                venues_used=[],
                execution_time_seconds=0.0,
                fill_rate=0.0,
                price_improvement=Decimal("0"),
                adverse_selection=Decimal("0"),
                execution_quality_score=0.0,
            )

        # Calculate weighted average fill price
        total_value = sum(
            slice_order.filled_quantity * slice_order.average_fill_price
            for slice_order in filled_slices
        )
        average_fill_price = total_value / Decimal(str(total_filled))

        # Get benchmark prices
        latest_quote = self.streaming_manager.get_latest_quote(order.symbol)
        arrival_price = latest_quote.mid_price if latest_quote else average_fill_price
        benchmark_price = arrival_price  # Simplified benchmark

        # Calculate costs
        total_commission = Decimal(str(total_filled)) * Decimal("0.005")  # $0.005 per share

        # Market impact (simplified calculation)
        if order.side == OrderSide.BUY:
            market_impact = (average_fill_price - arrival_price) / arrival_price * 10000  # bps
        else:
            market_impact = (arrival_price - average_fill_price) / arrival_price * 10000  # bps

        market_impact_bps = float(market_impact) if market_impact > 0 else 0.0

        # Timing cost (difference between arrival and benchmark)
        timing_cost_bps = 0.0  # Simplified for demo

        total_cost_bps = (
            market_impact_bps + timing_cost_bps + float(total_commission / total_value * 10000)
        )

        # Execution metrics
        execution_time_seconds = (datetime.now() - start_time).total_seconds()
        fill_rate = total_filled / int(order.quantity)
        venues_used = list(
            set(
                slice_order.routing_venue.value
                for slice_order in filled_slices
                if slice_order.routing_venue
            )
        )

        # Quality score (simplified)
        execution_quality_score = max(0.0, min(100.0, 100.0 - total_cost_bps * 10))

        return ExecutionSummary(
            order_id=order.order_id,
            symbol=order.symbol,
            total_quantity=int(order.quantity),
            filled_quantity=total_filled,
            remaining_quantity=int(order.quantity) - total_filled,
            average_fill_price=average_fill_price,
            arrival_price=arrival_price,
            benchmark_price=benchmark_price,
            total_commission=total_commission,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            total_cost_bps=total_cost_bps,
            slices_executed=len(filled_slices),
            venues_used=venues_used,
            execution_time_seconds=execution_time_seconds,
            fill_rate=fill_rate,
            price_improvement=Decimal("0"),  # Simplified
            adverse_selection=Decimal("0"),  # Simplified
            execution_quality_score=execution_quality_score,
        )

    def _process_settlement(self, trade_data: dict[str, Any]) -> None:
        """Process trade settlement"""

        order = trade_data["order"]
        execution_summary = trade_data["execution_summary"]

        try:
            console.print(f"   📋 Processing settlement: {order.order_id}")

            # In production, this would:
            # 1. Send trade confirmations
            # 2. Update position records
            # 3. Calculate settlement amounts
            # 4. Generate regulatory reports
            # 5. Update risk metrics

            # For demo, just log the settlement
            console.print(
                f"      ✅ Settlement processed: {execution_summary.filled_quantity} shares @ ${execution_summary.average_fill_price:.2f}"
            )

        except Exception as e:
            logger.error(f"Settlement processing failed: {str(e)}")

    def _store_order_validation(self, validation: OrderValidation) -> None:
        """Store order validation in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO order_validations (
                    order_id, validation_result, risk_score, estimated_cost,
                    validation_data, validation_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    validation.order_id,
                    validation.validation_result.value,
                    validation.risk_score,
                    str(validation.estimated_cost),
                    json.dumps(asdict(validation), default=str),
                    validation.validation_timestamp.isoformat(),
                ),
            )
            conn.commit()

    def _store_execution_instruction(
        self, order_id: str, instruction: ExecutionInstruction
    ) -> None:
        """Store execution instruction in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO execution_instructions (
                    order_id, algorithm, routing_venue, time_in_force,
                    execution_params, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    order_id,
                    instruction.algorithm.value,
                    instruction.routing_venue.value,
                    instruction.time_in_force,
                    json.dumps(asdict(instruction), default=str),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def _store_order_slice(self, slice_order: OrderSlice) -> None:
        """Store order slice in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO order_slices (
                    slice_id, parent_order_id, symbol, side, quantity, slice_type,
                    limit_price, status, submitted_at, filled_at, filled_quantity,
                    average_fill_price, routing_venue, broker_slice_id, slice_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    slice_order.slice_id,
                    slice_order.parent_order_id,
                    slice_order.symbol,
                    slice_order.side.value,
                    slice_order.quantity,
                    slice_order.slice_type.value,
                    str(slice_order.limit_price) if slice_order.limit_price else None,
                    slice_order.status.value,
                    slice_order.submitted_at.isoformat() if slice_order.submitted_at else None,
                    slice_order.filled_at.isoformat() if slice_order.filled_at else None,
                    slice_order.filled_quantity,
                    str(slice_order.average_fill_price) if slice_order.average_fill_price else None,
                    slice_order.routing_venue.value if slice_order.routing_venue else None,
                    slice_order.broker_slice_id,
                    json.dumps(asdict(slice_order), default=str),
                ),
            )
            conn.commit()

    def _store_execution_summary(self, summary: ExecutionSummary) -> None:
        """Store execution summary in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO execution_summaries (
                    order_id, symbol, total_quantity, filled_quantity,
                    average_fill_price, arrival_price, total_commission,
                    market_impact_bps, total_cost_bps, execution_time_seconds,
                    fill_rate, execution_quality_score, timestamp, summary_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    summary.order_id,
                    summary.symbol,
                    summary.total_quantity,
                    summary.filled_quantity,
                    str(summary.average_fill_price),
                    str(summary.arrival_price),
                    str(summary.total_commission),
                    summary.market_impact_bps,
                    summary.total_cost_bps,
                    summary.execution_time_seconds,
                    summary.fill_rate,
                    summary.execution_quality_score,
                    summary.timestamp.isoformat(),
                    json.dumps(asdict(summary), default=str),
                ),
            )
            conn.commit()

    def get_oms_statistics(self) -> dict[str, Any]:
        """Get OMS performance statistics"""

        fill_rate = (
            (self.orders_filled / self.orders_processed) if self.orders_processed > 0 else 0.0
        )

        return {
            "is_running": self.is_running,
            "orders_processed": self.orders_processed,
            "orders_filled": self.orders_filled,
            "fill_rate": fill_rate,
            "active_orders": len(self.active_orders),
            "average_execution_cost_bps": self.total_execution_cost_bps
            / max(1, self.orders_processed),
            "total_execution_summaries": len(self.execution_summaries),
        }

    def display_oms_dashboard(self) -> None:
        """Display comprehensive OMS dashboard"""

        stats = self.get_oms_statistics()

        console.print(
            Panel(
                f"[bold blue]Order Management System Dashboard[/bold blue]\n"
                f"Status: {'🟢 RUNNING' if stats['is_running'] else '🔴 STOPPED'}\n"
                f"Orders Processed: {stats['orders_processed']}\n"
                f"Fill Rate: {stats['fill_rate']:.1%}",
                title="📋 Order Management",
            )
        )

        # Recent execution summaries
        if self.execution_summaries:
            summaries_table = Table(title="📊 Recent Execution Summaries")
            summaries_table.add_column("Order ID", style="cyan")
            summaries_table.add_column("Symbol", style="white")
            summaries_table.add_column("Filled", justify="right", style="green")
            summaries_table.add_column("Avg Price", justify="right", style="yellow")
            summaries_table.add_column("Cost (bps)", justify="right", style="red")
            summaries_table.add_column("Quality", justify="right", style="blue")

            recent_summaries = list(self.execution_summaries.values())[-10:]  # Last 10
            for summary in recent_summaries:
                summaries_table.add_row(
                    summary.order_id[-8:],  # Last 8 chars
                    summary.symbol,
                    f"{summary.filled_quantity}/{summary.total_quantity}",
                    f"${summary.average_fill_price:.2f}",
                    f"{summary.total_cost_bps:.1f}",
                    f"{summary.execution_quality_score:.1f}",
                )

            console.print(summaries_table)


def create_order_management_system(
    oms_dir: str = "data/order_management", streaming_manager: StreamingDataManager = None
) -> OrderManagementSystem:
    """Factory function to create order management system"""
    return OrderManagementSystem(oms_dir=oms_dir, streaming_manager=streaming_manager)


if __name__ == "__main__":
    # Example usage
    oms = create_order_management_system()
    oms.start_order_processing()

    try:
        # Demo order submission
        from decimal import Decimal

        from bot.live.trading_engine_v2 import Order, OrderSide, OrderType

        demo_order = Order(
            order_id="demo_order_001",
            strategy_id="demo_strategy",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
        )

        order_id = oms.submit_order_for_validation(demo_order)

        # Wait for processing
        time.sleep(5)

        # Display dashboard
        oms.display_oms_dashboard()

    finally:
        oms.stop_order_processing()

    print("Order Management System created successfully!")
