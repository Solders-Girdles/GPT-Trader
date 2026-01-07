"""
File-based status reporter for operational monitoring.

This module provides a StatusReporter that periodically writes the bot's
internal state to a JSON file for external monitoring tools or human inspection.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpt_trader.monitoring.metrics_collector import record_gauge
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.monitoring.heartbeat import HeartbeatService

logger = get_logger(__name__, component="status_reporter")


@dataclass
class EngineStatus:
    """Status snapshot of the trading engine."""

    running: bool = False
    uptime_seconds: float = 0.0
    cycle_count: int = 0
    last_cycle_time: float | None = None
    errors_count: int = 0
    last_error: str | None = None
    last_error_time: float | None = None


@dataclass
class MarketStatus:
    """Status snapshot of market data."""

    symbols: list[str] = field(default_factory=list)
    last_prices: dict[str, Decimal] = field(default_factory=dict)
    last_price_update: float | None = None
    price_history: dict[str, list[Decimal]] = field(default_factory=dict)


@dataclass
class PositionStatus:
    """Status snapshot of positions."""

    count: int = 0
    symbols: list[str] = field(default_factory=list)
    total_unrealized_pnl: Decimal = Decimal("0")
    equity: Decimal = Decimal("0")
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class HeartbeatStatus:
    """Status snapshot of heartbeat service."""

    enabled: bool = False
    running: bool = False
    heartbeat_count: int = 0
    last_heartbeat: float | None = None
    is_healthy: bool = False


@dataclass
class OrderStatus:
    """Status snapshot of an active order."""

    order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal | None
    status: str
    order_type: str = "MARKET"
    time_in_force: str = "GTC"
    creation_time: float = 0.0
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None


@dataclass
class TradeStatus:
    """Status snapshot of a recent trade."""

    trade_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    time: str
    order_id: str
    fee: Decimal = Decimal("0")


@dataclass
class BalanceEntry:
    """Single balance entry with Decimal amounts."""

    asset: str
    total: Decimal
    available: Decimal
    hold: Decimal = Decimal("0")


@dataclass
class DecisionEntry:
    """Single strategy decision entry with typed fields."""

    symbol: str
    action: str = "HOLD"
    reason: str = ""
    confidence: float = 0.0
    indicators: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    decision_id: str = ""  # Unique ID for linking to orders/trades
    blocked_by: str = ""  # Guard/reason that blocked execution (empty if executed)
    # Indicator contributions for transparency (list of dicts with name, value, contribution)
    contributions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AccountStatus:
    """Status snapshot of account metrics."""

    volume_30d: Decimal = Decimal("0")
    fees_30d: Decimal = Decimal("0")
    fee_tier: str = ""
    balances: list[BalanceEntry] = field(default_factory=list)


@dataclass
class StrategyStatus:
    """Status snapshot of strategy engine."""

    active_strategies: list[str] = field(default_factory=list)
    last_decisions: list[DecisionEntry] = field(default_factory=list)
    # Live performance metrics (from real trades)
    performance: dict[str, Any] | None = None
    # Historical backtest performance (from backtesting)
    backtest_performance: dict[str, Any] | None = None
    # Strategy indicator parameters (RSI period, MA periods, etc.)
    parameters: dict[str, Any] | None = None


@dataclass
class RiskStatus:
    """Status snapshot of risk management metrics."""

    max_leverage: float = 0.0
    daily_loss_limit_pct: float = 0.0
    current_daily_loss_pct: float = 0.0
    reduce_only_mode: bool = False
    reduce_only_reason: str = ""
    active_guards: list[str] = field(default_factory=list)


@dataclass
class WebSocketStatus:
    """Status snapshot of WebSocket connection health."""

    connected: bool = False
    last_message_ts: float | None = None
    last_heartbeat_ts: float | None = None
    last_close_ts: float | None = None
    last_error_ts: float | None = None
    gap_count: int = 0
    reconnect_count: int = 0
    message_stale: bool = False
    heartbeat_stale: bool = False


@dataclass
class SystemStatus:
    """Status snapshot of system health and brokerage connection."""

    api_latency: float = 0.0
    connection_status: str = "UNKNOWN"
    rate_limit_usage: str = "0%"
    memory_usage: str = "0MB"
    cpu_usage: str = "0%"


@dataclass
class BotStatus:
    """Complete status snapshot of the trading bot."""

    bot_id: str = ""
    timestamp: float = field(default_factory=time.time)
    timestamp_iso: str = ""
    version: str = "1.0.0"

    engine: EngineStatus = field(default_factory=EngineStatus)
    market: MarketStatus = field(default_factory=MarketStatus)
    positions: PositionStatus = field(default_factory=PositionStatus)
    orders: list[OrderStatus] = field(default_factory=list)
    trades: list[TradeStatus] = field(default_factory=list)
    account: AccountStatus = field(default_factory=AccountStatus)
    strategy: StrategyStatus = field(default_factory=StrategyStatus)
    risk: RiskStatus = field(default_factory=RiskStatus)
    system: SystemStatus = field(default_factory=SystemStatus)
    heartbeat: HeartbeatStatus = field(default_factory=HeartbeatStatus)
    websocket: WebSocketStatus = field(default_factory=WebSocketStatus)

    # Overall health
    healthy: bool = True
    health_issues: list[str] = field(default_factory=list)

    # Reporter interval for TUI connection health tracking
    observer_interval: float = 2.0

    def __post_init__(self) -> None:
        if not self.timestamp_iso:
            self.timestamp_iso = datetime.utcfromtimestamp(self.timestamp).isoformat() + "Z"


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


@dataclass
class StatusReporter:
    """
    Periodically writes bot status to a JSON file and notifies observers.

    Features:
    - Atomic file writes (write to temp, then rename)
    - Observer pattern for event-driven updates
    - Dual-interval design: fast observer updates, slow file writes
    - Includes engine, market, position, risk, and heartbeat status
    - Health summary with issue detection

    The reporter uses two intervals:
    - observer_interval (default 2s): How often to update in-memory status and notify observers (TUI)
    - file_write_interval (default 60s): How often to write status to disk

    Usage:
        reporter = StatusReporter(
            status_file="/var/run/gpt-trader/status.json",
            observer_interval=2,      # Fast updates for TUI
            file_write_interval=60,   # Slow disk writes
        )
        reporter.add_observer(my_callback)
        await reporter.start()
        # ... later ...
        reporter.update_engine_status(running=True, cycle_count=42)
        # ... shutdown ...
        await reporter.stop()
    """

    status_file: str = "status.json"
    observer_interval: float = 2.0  # seconds - fast loop for TUI observers
    file_write_interval: float = 60.0  # seconds - slow loop for disk writes
    update_interval: int = 10  # Deprecated: use file_write_interval instead
    bot_id: str = ""
    enabled: bool = True

    # Internal state
    _running: bool = field(default=False, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _last_file_write: float = field(default=0.0, repr=False)
    _status: BotStatus = field(default_factory=BotStatus, repr=False)

    # Observers
    _observers: list[Callable[[BotStatus], None]] = field(default_factory=list, repr=False)

    # Mutable status tracking
    _cycle_count: int = field(default=0, repr=False)
    _errors_count: int = field(default=0, repr=False)
    _last_error: str | None = field(default=None, repr=False)
    _last_error_time: float | None = field(default=None, repr=False)
    _last_prices: dict[str, Decimal] = field(default_factory=dict, repr=False)
    _last_price_update: float | None = field(default=None, repr=False)
    _price_history: dict[str, deque[Decimal]] = field(default_factory=dict, repr=False)
    _positions: dict[str, Any] = field(default_factory=dict, repr=False)
    _equity: Decimal = field(default=Decimal("0"), repr=False)
    _heartbeat_service: HeartbeatService | None = field(default=None, repr=False)

    async def start(self) -> asyncio.Task[None] | None:
        """Start the status reporter background task."""
        if not self.enabled:
            logger.info("Status reporter disabled")
            return None

        if self._running:
            logger.warning("Status reporter already running")
            return self._task

        self._running = True
        self._start_time = time.time()
        self._last_file_write = 0.0  # Force initial write
        self._status = BotStatus(bot_id=self.bot_id, observer_interval=self.observer_interval)

        # Ensure directory exists
        status_path = Path(self.status_file)
        status_path.parent.mkdir(parents=True, exist_ok=True)

        # Write initial status
        await self._write_status()
        self._last_file_write = time.time()

        self._task = asyncio.create_task(self._report_loop())
        logger.info(
            f"Status reporter started (file={self.status_file}, "
            f"observer_interval={self.observer_interval}s, "
            f"file_write_interval={self.file_write_interval}s)"
        )
        return self._task

    async def stop(self) -> None:
        """Stop the status reporter."""
        if not self._running:
            return

        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

        # Write final status
        await self._write_status()
        logger.info("Status reporter stopped")

    def add_observer(self, callback: Callable[[BotStatus], None]) -> None:
        """
        Add an observer callback that receives status updates.

        The callback receives a BotStatus dataclass with the full typed status.
        """
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[BotStatus], None]) -> None:
        """Remove an observer callback."""
        if callback in self._observers:
            self._observers.remove(callback)

    async def _report_loop(self) -> None:
        """Main reporting loop with dual intervals.

        Uses observer_interval for fast in-memory updates and observer notifications.
        Uses file_write_interval for slow disk writes.
        """
        while self._running:
            try:
                # Always update in-memory status
                self._update_status()

                # Check if it's time to write to disk
                now = time.time()
                time_since_last_write = now - self._last_file_write
                should_write_file = time_since_last_write >= self.file_write_interval

                if should_write_file:
                    await self._write_status_to_file()
                    self._last_file_write = now

                # Always notify observers (fast loop)
                for observer in self._observers:
                    try:
                        # Handle both async and sync observer callbacks
                        if asyncio.iscoroutinefunction(observer):
                            await observer(self._status)
                        else:
                            observer(self._status)
                    except Exception as obs_e:
                        logger.error(f"Observer error: {obs_e}")

            except Exception as e:
                logger.error(f"Status report error: {e}")

            # Sleep for the fast observer interval
            await asyncio.sleep(self.observer_interval)

    async def _write_status(self) -> None:
        """Write current status to file atomically (legacy method for backward compat)."""
        self._update_status()
        await self._write_status_to_file()

    async def _write_status_to_file(self) -> None:
        """Write current status to file atomically."""
        status_dict = asdict(self._status)

        # Atomic write: write to temp file, then rename
        status_path = Path(self.status_file)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=status_path.parent,
            prefix=".status_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(status_dict, f, indent=2, cls=DecimalEncoder)
            os.rename(temp_path, self.status_file)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _update_status(self) -> None:
        """Update the status object with current values."""
        now = time.time()
        self._status.timestamp = now
        self._status.timestamp_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"
        self._status.bot_id = self.bot_id
        self._status.observer_interval = self.observer_interval

        # Engine status
        self._status.engine.running = self._running
        self._status.engine.uptime_seconds = now - self._start_time if self._start_time else 0
        self._status.engine.cycle_count = self._cycle_count
        self._status.engine.errors_count = self._errors_count
        self._status.engine.last_error = self._last_error
        self._status.engine.last_error_time = self._last_error_time

        # Market status - keep Decimal types
        self._status.market.last_prices = dict(self._last_prices)
        self._status.market.last_price_update = self._last_price_update
        self._status.market.symbols = list(self._last_prices.keys())
        self._status.market.price_history = {
            symbol: list(prices) for symbol, prices in self._price_history.items()
        }

        # Position status - keep Decimal types
        self._status.positions.count = len(self._positions)
        self._status.positions.symbols = list(self._positions.keys())
        total_pnl = sum(
            (p.get("unrealized_pnl", Decimal("0")) for p in self._positions.values()),
            Decimal("0"),
        )
        self._status.positions.total_unrealized_pnl = total_pnl
        self._status.positions.equity = self._equity
        # Include actual position details (keep Decimal for numeric values)
        self._status.positions.positions = {
            symbol: dict(pos) for symbol, pos in self._positions.items()
        }

        # Heartbeat status
        if self._heartbeat_service:
            hb_status = self._heartbeat_service.get_status()
            self._status.heartbeat.enabled = hb_status.get("enabled", False)
            self._status.heartbeat.running = hb_status.get("running", False)
            self._status.heartbeat.heartbeat_count = hb_status.get("heartbeat_count", 0)
            self._status.heartbeat.last_heartbeat = hb_status.get("last_heartbeat")
            self._status.heartbeat.is_healthy = self._heartbeat_service.is_healthy

        # Health assessment
        self._assess_health()

    def _assess_health(self) -> None:
        """Assess overall health and populate issues list."""
        issues: list[str] = []

        # Check engine running
        if not self._status.engine.running:
            issues.append("Engine not running")

        # Check for recent errors
        if self._errors_count > 0 and self._last_error_time:
            time_since_error = time.time() - self._last_error_time
            if time_since_error < 300:  # Error in last 5 minutes
                issues.append(f"Recent error: {self._last_error}")

        # Check price staleness
        if self._last_price_update:
            time_since_price = time.time() - self._last_price_update
            if time_since_price > 120:  # No price update in 2 minutes
                issues.append(f"Stale prices ({int(time_since_price)}s old)")

        # Check heartbeat health
        if self._heartbeat_service and self._status.heartbeat.enabled:
            if not self._status.heartbeat.is_healthy:
                issues.append("Heartbeat unhealthy")

        # Check WebSocket health
        if self._status.websocket.message_stale:
            issues.append("WebSocket messages stale")
        if self._status.websocket.heartbeat_stale:
            issues.append("WebSocket heartbeat stale")
        if self._status.websocket.gap_count > 0:
            issues.append(f"WebSocket sequence gaps: {self._status.websocket.gap_count}")

        self._status.healthy = len(issues) == 0
        self._status.health_issues = issues

    # --- Update methods called by TradingEngine ---

    def set_heartbeat_service(self, service: HeartbeatService) -> None:
        """Set the heartbeat service reference for status reporting."""
        self._heartbeat_service = service

    def record_cycle(self) -> None:
        """Record a completed trading cycle."""
        self._cycle_count += 1
        self._status.engine.last_cycle_time = time.time()

    def record_error(self, error: str) -> None:
        """Record an error occurrence."""
        self._errors_count += 1
        self._last_error = error
        self._last_error_time = time.time()

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update the last known price for a symbol and maintain price history."""
        self._last_prices[symbol] = price
        self._last_price_update = time.time()

        # Maintain price history (last 100 prices per symbol)
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=100)
        self._price_history[symbol].append(price)

    def update_positions(self, positions: dict[str, Any]) -> None:
        """
        Update the current positions with Decimal coercion.

        Args:
            positions: Dict of symbol -> position data (may contain string numerics)
        """
        # Normalize positions to ensure Decimal types
        normalized_positions = {}
        for symbol, pos_data in positions.items():
            if not isinstance(pos_data, dict):
                continue

            normalized_pos = {}
            for key, value in pos_data.items():
                # Coerce numeric fields to Decimal
                if key in (
                    "quantity",
                    "mark_price",
                    "entry_price",
                    "unrealized_pnl",
                    "realized_pnl",
                ):
                    try:
                        normalized_pos[key] = Decimal(str(value)) if value else Decimal("0")
                    except (ValueError, InvalidOperation):
                        normalized_pos[key] = Decimal("0")
                else:
                    # Keep non-numeric fields as-is (e.g., side)
                    normalized_pos[key] = value

            normalized_positions[symbol] = normalized_pos

        self._positions = normalized_positions

    def update_equity(self, equity: Decimal) -> None:
        """Update the current equity."""
        self._equity = equity
        # Record equity gauge for metrics
        try:
            record_gauge("gpt_trader_equity_dollars", float(equity))
        except Exception:
            pass  # Don't let metrics errors affect operation

    def update_orders(self, orders: list[dict[str, Any]]) -> None:
        """Update the list of active orders."""
        order_statuses = []
        for o in orders:
            # Extract type and TIF from configuration if available
            # Coinbase format: order_configuration: { limit_limit_gtc: {...} }
            config = o.get("order_configuration", {})
            order_type = "MARKET"
            tif = "GTC"

            if "market_market_ioc" in config:
                order_type = "MARKET"
                tif = "IOC"
            elif "limit_limit_gtc" in config:
                order_type = "LIMIT"
                tif = "GTC"
            elif "limit_limit_gtd" in config:
                order_type = "LIMIT"
                tif = "GTD"
            elif "stop_limit_stop_limit_gtc" in config:
                order_type = "STOP_LIMIT"
                tif = "GTC"

            # Parse quantity
            quantity_raw = o.get("size") or o.get("order_configuration", {}).get(
                "market_market_ioc", {}
            ).get("base_size", "0")
            try:
                quantity = Decimal(str(quantity_raw)) if quantity_raw else Decimal("0")
            except Exception:
                quantity = Decimal("0")

            # Parse price
            price_raw = o.get("price")
            try:
                price = Decimal(str(price_raw)) if price_raw else None
            except Exception:
                price = None

            # Parse creation_time from ISO string or float
            creation_time = self._parse_order_timestamp(o.get("created_time"))

            # Parse filled_quantity (Coinbase uses "filled_size")
            filled_raw = o.get("filled_size", "0")
            try:
                filled_quantity = Decimal(str(filled_raw)) if filled_raw else Decimal("0")
            except Exception:
                filled_quantity = Decimal("0")

            # Parse avg_fill_price (Coinbase uses "average_filled_price")
            avg_fill_raw = o.get("average_filled_price")
            try:
                avg_fill_price = Decimal(str(avg_fill_raw)) if avg_fill_raw else None
            except Exception:
                avg_fill_price = None

            order_statuses.append(
                OrderStatus(
                    order_id=o.get("order_id", ""),
                    symbol=o.get("product_id", ""),
                    side=o.get("side", ""),
                    quantity=quantity,
                    price=price,
                    status=o.get("status", "UNKNOWN"),
                    order_type=order_type,
                    time_in_force=tif,
                    creation_time=creation_time,
                    filled_quantity=filled_quantity,
                    avg_fill_price=avg_fill_price,
                )
            )
        self._status.orders = order_statuses

    def _parse_order_timestamp(self, value: Any) -> float:
        """Parse order timestamp from various formats to epoch float.

        Args:
            value: Timestamp as ISO string, float, int, or None.

        Returns:
            Epoch timestamp as float, or current time if parsing fails.
        """
        if value is None:
            return time.time()

        # Already a numeric type
        if isinstance(value, (int, float)):
            return float(value)

        # ISO string (e.g., "2024-01-15T10:30:00Z" or "2024-01-15T10:30:00.123456Z")
        if isinstance(value, str):
            try:
                # Strip trailing Z and parse
                clean = value.rstrip("Z")
                # Handle optional microseconds
                if "." in clean:
                    dt = datetime.fromisoformat(clean)
                else:
                    dt = datetime.fromisoformat(clean)
                return dt.timestamp()
            except (ValueError, TypeError):
                pass

        return time.time()

    def add_trade(self, trade: dict[str, Any]) -> None:
        """Add a recent trade."""
        # Generate unique trade_id
        trade_id = (
            trade.get("trade_id") or f"trade_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        )

        # Convert timestamp to ISO string
        timestamp = trade.get("timestamp", time.time())
        time_str = datetime.utcfromtimestamp(timestamp).isoformat() + "Z"

        # Parse numeric fields to Decimal
        try:
            quantity = Decimal(str(trade.get("quantity", "0")))
        except Exception:
            quantity = Decimal("0")

        try:
            price = Decimal(str(trade.get("price", "0")))
        except Exception:
            price = Decimal("0")

        try:
            fee = Decimal(str(trade.get("fee", "0")))
        except Exception:
            fee = Decimal("0")

        new_trade = TradeStatus(
            trade_id=trade_id,
            symbol=trade.get("symbol", ""),
            side=trade.get("side", ""),
            quantity=quantity,
            price=price,
            time=time_str,
            order_id=str(trade.get("order_id") or ""),
            fee=fee,
        )
        # Prepend and keep last 50
        self._status.trades.insert(0, new_trade)
        if len(self._status.trades) > 50:
            self._status.trades.pop()

    def update_account(self, balances: list[Any], summary: dict[str, Any]) -> None:
        """Update account metrics with Decimal coercion."""
        # Format balances with Decimal types
        bal_list: list[BalanceEntry] = []
        for b in balances:
            # Handle object or dict
            if hasattr(b, "asset"):
                asset = b.asset
                total_val = b.total
                avail_val = b.available
                hold_val = getattr(b, "hold", Decimal("0"))
            else:
                asset = b.get("currency", "")
                total_val = b.get("balance", "0")
                avail_val = b.get("available", "0")
                hold_val = b.get("hold", "0")

            # Convert to Decimal
            try:
                total = Decimal(str(total_val))
            except (ValueError, InvalidOperation):
                total = Decimal("0")

            try:
                avail = Decimal(str(avail_val))
            except (ValueError, InvalidOperation):
                avail = Decimal("0")

            try:
                hold = Decimal(str(hold_val))
            except (ValueError, InvalidOperation):
                hold = Decimal("0")

            if total > 0:
                bal_list.append(BalanceEntry(asset=asset, total=total, available=avail, hold=hold))

        # Parse numeric fields to Decimal
        try:
            volume_30d = Decimal(str(summary.get("total_volume_30d", "0")))
        except Exception:
            volume_30d = Decimal("0")

        try:
            fees_30d = Decimal(str(summary.get("total_fees_30d", "0")))
        except Exception:
            fees_30d = Decimal("0")

        self._status.account = AccountStatus(
            volume_30d=volume_30d,
            fees_30d=fees_30d,
            fee_tier=str(summary.get("fee_tier", {}).get("pricing_tier", "Unknown")),
            balances=bal_list,
        )

    def update_strategy(
        self, active_strategies: list[str], decisions: list[dict[str, Any]]
    ) -> None:
        """Update strategy status with typed DecisionEntry normalization."""
        self._status.strategy.active_strategies = active_strategies

        # Normalize decisions to typed DecisionEntry objects
        # Keep last 50 decisions
        normalized_decisions: list[DecisionEntry] = []
        for d in decisions:
            # Parse confidence to float
            try:
                confidence = float(d.get("confidence", 0.0))
            except (ValueError, TypeError):
                confidence = 0.0

            # Parse timestamp to float
            timestamp = d.get("timestamp", 0.0)
            if isinstance(timestamp, str):
                try:
                    timestamp = float(timestamp)
                except (ValueError, TypeError):
                    timestamp = 0.0
            elif not isinstance(timestamp, (int, float)):
                timestamp = 0.0

            # Generate decision_id if not provided (timestamp-based)
            decision_id = str(d.get("decision_id", ""))
            if not decision_id and timestamp:
                # Generate ID from timestamp + symbol for linkage
                symbol = str(d.get("symbol", ""))
                decision_id = f"{int(timestamp * 1000)}_{symbol}"

            # Create typed entry
            normalized_decisions.append(
                DecisionEntry(
                    symbol=str(d.get("symbol", "")),
                    action=str(d.get("action", "HOLD")),
                    reason=str(d.get("reason", "")),
                    confidence=confidence,
                    indicators=d.get("indicators", {}),
                    timestamp=timestamp,
                    decision_id=decision_id,
                )
            )

        self._status.strategy.last_decisions.extend(normalized_decisions)
        if len(self._status.strategy.last_decisions) > 50:
            self._status.strategy.last_decisions = self._status.strategy.last_decisions[-50:]

    def update_strategy_performance(
        self,
        performance: dict[str, Any] | None = None,
        backtest: dict[str, Any] | None = None,
    ) -> None:
        """Update strategy performance metrics without changing update_strategy().

        Args:
            performance: Live performance dict with keys like win_rate, profit_factor,
                        total_return, max_drawdown, total_trades, etc.
            backtest: Historical backtest performance dict with same structure.
        """
        if performance is not None:
            self._status.strategy.performance = performance
        if backtest is not None:
            self._status.strategy.backtest_performance = backtest

    def update_strategy_parameters(self, parameters: dict[str, Any] | None) -> None:
        """Update strategy indicator parameters for TUI display.

        Args:
            parameters: Dict with indicator config values, e.g.:
                {
                    "rsi_period": 14,
                    "ma_fast_period": 5,
                    "ma_slow_period": 20,
                    "ma_type": "SMA",
                    "zscore_lookback": 20,
                    ...
                }
        """
        self._status.strategy.parameters = parameters

    def update_risk(
        self,
        max_leverage: float,
        daily_loss_limit: float,
        current_daily_loss: float,
        reduce_only: bool,
        reduce_reason: str,
        active_guards: list[str] | None = None,
    ) -> None:
        """Update risk status."""
        self._status.risk.max_leverage = max_leverage
        self._status.risk.daily_loss_limit_pct = daily_loss_limit
        self._status.risk.current_daily_loss_pct = current_daily_loss
        self._status.risk.reduce_only_mode = reduce_only
        self._status.risk.reduce_only_reason = reduce_reason
        if active_guards:
            self._status.risk.active_guards = active_guards

    def update_system(
        self, latency: float, connection: str, rate_limit: str, memory: str, cpu: str
    ) -> None:
        """Update system status."""
        self._status.system.api_latency = latency
        self._status.system.connection_status = connection
        self._status.system.rate_limit_usage = rate_limit
        self._status.system.memory_usage = memory
        self._status.system.cpu_usage = cpu

    def update_ws_health(self, health: dict[str, Any]) -> None:
        """Update WebSocket health status.

        Args:
            health: Dict with WS health metrics from broker.get_ws_health():
                - connected: bool
                - last_message_ts: float | None
                - last_heartbeat_ts: float | None
                - last_close_ts: float | None
                - last_error_ts: float | None
                - gap_count: int
                - reconnect_count: int
        """
        now = time.time()

        # Extract values with defaults
        self._status.websocket.connected = bool(health.get("connected", False))
        self._status.websocket.last_message_ts = health.get("last_message_ts")
        self._status.websocket.last_heartbeat_ts = health.get("last_heartbeat_ts")
        self._status.websocket.last_close_ts = health.get("last_close_ts")
        self._status.websocket.last_error_ts = health.get("last_error_ts")
        self._status.websocket.gap_count = int(health.get("gap_count", 0))
        self._status.websocket.reconnect_count = int(health.get("reconnect_count", 0))

        # Calculate staleness based on thresholds (15s message, 30s heartbeat)
        last_message_ts = self._status.websocket.last_message_ts
        last_heartbeat_ts = self._status.websocket.last_heartbeat_ts

        self._status.websocket.message_stale = (
            last_message_ts is not None and (now - last_message_ts) > 15
        )
        self._status.websocket.heartbeat_stale = (
            last_heartbeat_ts is not None and (now - last_heartbeat_ts) > 30
        )

        # Record WS gap gauge for metrics
        try:
            record_gauge("gpt_trader_ws_gap_count", float(self._status.websocket.gap_count))
        except Exception:
            pass  # Don't let metrics errors affect operation

    def get_status(self) -> BotStatus:
        """
        Get current status as a BotStatus dataclass.

        Returns:
            BotStatus: Typed status snapshot (BotStatusSnapshot contract)
        """
        self._update_status()
        return self._status

    def get_status_dict(self) -> dict[str, Any]:
        """
        Get current status as a dictionary (backward compatibility).

        Deprecated: Use get_status() for typed access.
        This method will be removed after TUI migration is complete.
        """
        self._update_status()
        return asdict(self._status)


__all__ = [
    "StatusReporter",
    "BotStatus",
    "EngineStatus",
    "MarketStatus",
    "PositionStatus",
    "OrderStatus",
    "TradeStatus",
    "AccountStatus",
    "BalanceEntry",
    "DecisionEntry",
    "StrategyStatus",
    "RiskStatus",
    "SystemStatus",
    "HeartbeatStatus",
    "WebSocketStatus",
]
