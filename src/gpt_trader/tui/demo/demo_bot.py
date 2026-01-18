"""
Demo bot for TUI testing.

A lightweight mock bot that simulates trading activity without
connecting to real exchanges or executing real trades.
"""

import asyncio
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BalanceEntry,
    BotStatus,
    DecisionEntry,
    EngineStatus,
    HeartbeatStatus,
    MarketStatus,
    OrderStatus,
    PositionStatus,
    RiskStatus,
    StrategyStatus,
    SystemStatus,
    TradeStatus,
)
from gpt_trader.tui.demo.mock_data import MockDataGenerator
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="demo")


class DemoEngine:
    """Mock trading engine for demo mode."""

    def __init__(self, data_generator: MockDataGenerator | None = None) -> None:
        self.status_reporter = DemoStatusReporter(data_generator=data_generator)
        self.context = DemoContext()


class DemoContext:
    """Mock coordinator context."""

    def __init__(self) -> None:
        self.runtime_state = DemoRuntimeState()


class DemoRuntimeState:
    """Mock runtime state."""

    def __init__(self) -> None:
        import threading

        self.start_time = time.time()

        # Mark data state (required by RuntimeStateProtocol)
        self.mark_lock = threading.Lock()
        self.mark_windows: dict[str, Any] = {}

        # Order book data state (required by RuntimeStateProtocol)
        self.orderbook_lock = threading.Lock()
        self.orderbook_snapshots: dict[str, Any] = {}

        # Trade flow data state (required by RuntimeStateProtocol)
        self.trade_lock = threading.Lock()
        self.trade_aggregators: dict[str, Any] = {}

        # Other protocol fields (minimal defaults)
        self.equity: Any = None
        self.positions: dict[str, Any] = {}
        self.positions_pnl: dict[str, dict[str, Any]] = {}
        self.positions_dict: dict[str, dict[str, Any]] = {}
        self.strategy: Any = None
        self.symbol_strategies: dict[str, Any] = {}

    @property
    def uptime(self) -> float:
        """Calculate current uptime."""
        return time.time() - self.start_time

    def update_equity(self, value: Any) -> None:
        """Update current equity value."""
        self.equity = value


class DemoStatusReporter:
    """Mock status reporter that generates realistic data."""

    def __init__(
        self, update_interval: float = 2.0, data_generator: MockDataGenerator | None = None
    ) -> None:
        self.update_interval = update_interval
        self.data_generator = data_generator or MockDataGenerator()
        self._observers: list = []
        self._running = False
        self._task: asyncio.Task | None = None

    def add_observer(self, callback: Callable[[BotStatus], None]) -> None:
        """Register an observer for status updates."""
        if callback not in self._observers:
            self._observers.append(callback)
            logger.debug("Observer added. Total observers: %s", len(self._observers))

    def remove_observer(self, callback: Callable[[BotStatus], None]) -> None:
        """Unregister an observer."""
        if callback in self._observers:
            self._observers.remove(callback)
            logger.debug("Observer removed. Total observers: %s", len(self._observers))

    def _dict_to_bot_status(self, data: dict[str, Any]) -> BotStatus:
        """Convert dict data from MockDataGenerator to BotStatus dataclass."""
        try:
            # Helper to safely convert to Decimal
            def to_decimal(value: Any) -> Decimal:
                if isinstance(value, Decimal):
                    return value
                try:
                    return Decimal(str(value))
                except Exception:
                    return Decimal("0")

            # Convert EngineStatus
            engine_data = data.get("engine", {})
            engine_status = EngineStatus(
                running=bool(engine_data.get("running", False)),
                uptime_seconds=float(engine_data.get("uptime", 0.0)),
                cycle_count=int(engine_data.get("cycle_count", 0)),
                last_cycle_time=engine_data.get("last_cycle_time"),
                errors_count=len(engine_data.get("errors", [])),
            )

            # Convert MarketStatus
            market_data = data.get("market", {})
            last_prices = {
                symbol: to_decimal(price)
                for symbol, price in market_data.get("last_prices", {}).items()
            }
            price_history = {
                symbol: [to_decimal(p) for p in history]
                for symbol, history in market_data.get("price_history", {}).items()
            }
            market_status = MarketStatus(
                symbols=list(last_prices.keys()),
                last_prices=last_prices,
                last_price_update=market_data.get("last_price_update"),
                price_history=price_history,
            )

            # Convert PositionStatus
            pos_data = data.get("positions", {})
            positions_dict = {}
            for symbol, p in pos_data.get("positions", {}).items():
                positions_dict[symbol] = {
                    "quantity": to_decimal(p.get("quantity", "0")),
                    "entry_price": to_decimal(p.get("entry_price", "0")),
                    "unrealized_pnl": to_decimal(p.get("unrealized_pnl", "0")),
                    "mark_price": to_decimal(p.get("mark_price", "0")),
                    "side": p.get("side", ""),
                }
            position_status = PositionStatus(
                count=len(positions_dict),
                symbols=list(positions_dict.keys()),
                total_unrealized_pnl=to_decimal(pos_data.get("total_unrealized_pnl", "0")),
                equity=to_decimal(pos_data.get("equity", "0")),
                positions=positions_dict,
            )

            # Convert OrderStatus list
            order_statuses = []
            for o in data.get("orders", []):
                order_statuses.append(
                    OrderStatus(
                        order_id=o.get("order_id", ""),
                        symbol=o.get("symbol", ""),
                        side=o.get("side", ""),
                        quantity=to_decimal(o.get("quantity", "0")),
                        price=to_decimal(o.get("price", "0")) if o.get("price") else None,
                        status=o.get("status", ""),
                        order_type=o.get("order_type", "LIMIT"),
                        time_in_force=o.get("time_in_force", "GTC"),
                        creation_time=float(o.get("creation_time", 0)),
                    )
                )

            # Convert TradeStatus list
            trade_statuses = []
            for t in data.get("trades", []):
                trade_statuses.append(
                    TradeStatus(
                        trade_id=t.get("trade_id", ""),
                        symbol=t.get("symbol", ""),
                        side=t.get("side", ""),
                        quantity=to_decimal(t.get("quantity", "0")),
                        price=to_decimal(t.get("price", "0")),
                        time=t.get("time", ""),
                        order_id=t.get("order_id", ""),
                        fee=to_decimal(t.get("fee", "0")),
                    )
                )

            # Convert AccountStatus
            acc_data = data.get("account", {})
            balances = []
            for b in acc_data.get("balances", []):
                balances.append(
                    BalanceEntry(
                        asset=b.get("asset", ""),
                        total=to_decimal(b.get("total", "0")),
                        available=to_decimal(b.get("available", "0")),
                        hold=to_decimal(b.get("hold", "0")),
                    )
                )
            account_status = AccountStatus(
                volume_30d=to_decimal(acc_data.get("volume_30d", "0")),
                fees_30d=to_decimal(acc_data.get("fees_30d", "0")),
                fee_tier=acc_data.get("fee_tier", ""),
                balances=balances,
            )

            # Convert StrategyStatus
            strat_data = data.get("strategy", {})
            decisions = []
            for d in strat_data.get("last_decisions", []):
                timestamp = float(d.get("timestamp", 0.0))
                symbol = d.get("symbol", "")
                # Generate decision_id if not provided
                decision_id = d.get("decision_id", "")
                if not decision_id and timestamp:
                    decision_id = f"{int(timestamp * 1000)}_{symbol}"
                decisions.append(
                    DecisionEntry(
                        symbol=symbol,
                        action=d.get("action", "HOLD"),
                        reason=d.get("reason", ""),
                        confidence=float(d.get("confidence", 0.0)),
                        indicators=d.get("indicators", {}),
                        timestamp=timestamp,
                        decision_id=decision_id,
                        blocked_by=d.get("blocked_by", ""),
                        contributions=d.get("contributions", []),
                    )
                )
            strategy_status = StrategyStatus(
                active_strategies=strat_data.get("active_strategies", []),
                last_decisions=decisions,
                performance=strat_data.get("performance"),
                backtest_performance=strat_data.get("backtest_performance"),
            )

            # Convert RiskStatus
            risk_data = data.get("risk", {})
            risk_status = RiskStatus(
                max_leverage=float(risk_data.get("max_leverage", 0.0)),
                daily_loss_limit_pct=float(risk_data.get("daily_loss_limit_pct", 0.0)),
                current_daily_loss_pct=float(risk_data.get("current_daily_loss_pct", 0.0)),
                reduce_only_mode=bool(risk_data.get("reduce_only_mode", False)),
                reduce_only_reason=risk_data.get("reduce_only_reason", ""),
                guards=risk_data.get("guards", []),
            )

            # Convert SystemStatus
            sys_data = data.get("system", {})
            system_status = SystemStatus(
                api_latency=float(sys_data.get("api_latency", 0.0)),
                connection_status=sys_data.get("connection_status", "UNKNOWN"),
                rate_limit_usage=sys_data.get("rate_limit_usage", "0%"),
                memory_usage=sys_data.get("memory_usage", "0MB"),
                cpu_usage=sys_data.get("cpu_usage", "0%"),
            )

            # Convert HeartbeatStatus
            hb_data = data.get("heartbeat", {})
            heartbeat_status = HeartbeatStatus(
                enabled=bool(hb_data.get("enabled", False)),
                running=bool(hb_data.get("running", False)),
                heartbeat_count=int(hb_data.get("heartbeat_count", 0)),
                last_heartbeat=hb_data.get("last_heartbeat"),
                is_healthy=bool(hb_data.get("is_healthy", True)),
            )

            # Construct BotStatus
            return BotStatus(
                bot_id="demo-bot",
                timestamp=time.time(),
                timestamp_iso="",  # Will be set by __post_init__
                version="demo",
                engine=engine_status,
                market=market_status,
                positions=position_status,
                orders=order_statuses,
                trades=trade_statuses,
                account=account_status,
                strategy=strategy_status,
                risk=risk_status,
                system=system_status,
                heartbeat=heartbeat_status,
                healthy=data.get("healthy", True),
                health_issues=data.get("health_issues", []),
            )

        except Exception as e:
            logger.error(f"Failed to convert dict to BotStatus: {e}", exc_info=True)
            # Return minimal valid BotStatus as fallback
            return BotStatus(
                bot_id="demo-bot-error",
                healthy=False,
                health_issues=[f"Status conversion error: {str(e)}"],
            )

    def get_status(self) -> BotStatus:
        """Get current status snapshot."""
        data = self.data_generator.generate_full_status()
        return self._dict_to_bot_status(data)

    async def start(self) -> asyncio.Task:
        """Start the status reporter loop."""
        self._running = True
        self._task = asyncio.create_task(self._report_loop())
        logger.debug("Demo status reporter started")
        return self._task

    async def stop(self) -> None:
        """Stop the status reporter loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.debug("Demo status reporter stopped")

    async def _report_loop(self) -> None:
        """Main reporting loop that notifies observers."""
        while self._running:
            try:
                # Generate new status
                status = self.get_status()

                # Notify all observers
                for observer in self._observers:
                    try:
                        if asyncio.iscoroutinefunction(observer):
                            await observer(status)
                        else:
                            observer(status)
                    except Exception as e:
                        logger.error(f"Error notifying observer: {e}", exc_info=True)

                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in demo report loop: {e}", exc_info=True)
                await asyncio.sleep(self.update_interval)


class DemoBot:
    """
    Mock bot for TUI demo mode.

    Simulates a trading bot without connecting to real exchanges.
    Perfect for UI development and testing.

    Args:
        config: Optional configuration object.
        data_generator: Optional pre-configured MockDataGenerator.
        seed: Optional random seed for reproducible demo output.
            Only used if data_generator is not provided.
    """

    def __init__(
        self,
        config: Any | None = None,
        data_generator: MockDataGenerator | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or DemoConfig()

        # Create seeded generator if seed provided and no generator given
        if data_generator is None and seed is not None:
            data_generator = MockDataGenerator(seed=seed)

        self.engine = DemoEngine(data_generator=data_generator)
        self.running = False
        self._task: asyncio.Task | None = None

    async def run(self, single_cycle: bool = False) -> None:
        """Start the demo bot."""
        try:
            logger.info("Starting demo bot")
            self.running = True

            # Start status reporter
            await self.engine.status_reporter.start()

            # Keep running until stopped
            if not single_cycle:
                while self.running:
                    await asyncio.sleep(1)

            logger.info("Demo bot run loop exited")

        except asyncio.CancelledError:
            logger.info("Demo bot run cancelled")
            raise
        except Exception as e:
            logger.error(f"Demo bot error: {e}", exc_info=True)
            raise
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop the demo bot."""
        logger.info("Stopping demo bot")
        self.running = False
        await self.engine.status_reporter.stop()
        logger.info("Demo bot stopped")

    async def shutdown(self) -> None:
        """Alias for stop()."""
        await self.stop()

    async def flatten_and_stop(self) -> list[str]:
        """Mock panic sequence."""
        logger.warning("Demo bot: flatten_and_stop called (simulated)")
        messages = [
            "DEMO MODE: Would close all positions",
            "DEMO MODE: Would cancel all orders",
            "DEMO MODE: Bot stopped",
        ]
        await self.stop()
        return messages


class DemoConfig:
    """Mock configuration for demo bot."""

    def __init__(self) -> None:
        self.symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        self.update_interval = 2.0
        self.exchange = "DEMO"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "symbols": self.symbols,
            "update_interval": self.update_interval,
            "exchange": self.exchange,
            "mode": "DEMO",
        }
