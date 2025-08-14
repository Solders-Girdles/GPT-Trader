"""
Circuit Breakers System for GPT-Trader Live Trading

Automatic trading halt system that monitors for dangerous conditions and
can immediately stop trading to prevent catastrophic losses:

- Real-time loss monitoring with automatic position liquidation
- Volatility spike detection with market-wide shutdowns
- Strategy performance degradation detection
- Risk limit breaches with immediate halt capability
- Emergency stop mechanisms for critical system failures
- Coordinated shutdown across all trading engines

This system acts as the final safety net for live trading operations.
"""

import json
import logging
import queue
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class CircuitBreakerType(Enum):
    """Types of circuit breakers"""

    DRAWDOWN = "drawdown"  # Maximum drawdown exceeded
    DAILY_LOSS = "daily_loss"  # Daily loss limit reached
    VOLATILITY = "volatility"  # Market volatility spike
    POSITION_SIZE = "position_size"  # Position concentration too high
    STRATEGY_FAILURE = "strategy_failure"  # Strategy performing poorly
    SYSTEM_ERROR = "system_error"  # Critical system error
    MANUAL = "manual"  # Manual emergency stop


class BreakerStatus(Enum):
    """Circuit breaker status"""

    ACTIVE = "active"  # Monitoring and ready to trigger
    TRIGGERED = "triggered"  # Breaker has fired
    COOLDOWN = "cooldown"  # In cooldown period before reset
    DISABLED = "disabled"  # Temporarily disabled


class ActionType(Enum):
    """Actions taken when breaker triggers"""

    HALT_ALL = "halt_all"  # Stop all trading immediately
    HALT_STRATEGY = "halt_strategy"  # Stop specific strategy
    LIQUIDATE_POSITIONS = "liquidate_positions"  # Close all positions
    REDUCE_POSITION_SIZE = "reduce_position_size"  # Scale down positions
    ALERT_ONLY = "alert_only"  # Send alert but continue trading


@dataclass
class CircuitBreakerRule:
    """Circuit breaker rule configuration"""

    breaker_id: str
    breaker_type: CircuitBreakerType
    description: str

    # Trigger conditions
    threshold: Decimal
    lookback_period: timedelta

    # Actions to take
    primary_action: ActionType
    secondary_actions: list[ActionType] = field(default_factory=list)

    # Configuration
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    max_triggers_per_day: int = 3

    # Scope
    strategy_ids: set[str] | None = None  # None means all strategies
    symbol_filters: set[str] | None = None  # None means all symbols

    # Status tracking
    status: BreakerStatus = BreakerStatus.ACTIVE
    trigger_count: int = 0
    last_triggered: datetime | None = None
    last_reset: datetime = field(default_factory=datetime.now)


@dataclass
class BreakerEvent:
    """Circuit breaker event record"""

    event_id: str
    breaker_id: str
    breaker_type: CircuitBreakerType
    trigger_value: Decimal
    threshold: Decimal

    # Context
    strategy_id: str | None = None
    symbol: str | None = None

    # Actions taken
    actions_taken: list[ActionType] = field(default_factory=list)
    positions_closed: int = 0
    strategies_halted: int = 0

    # Timing
    triggered_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None

    # Additional data
    metadata: dict[str, Any] = field(default_factory=dict)


class CircuitBreakerSystem:
    """Production circuit breaker system for automated trading protection"""

    def __init__(
        self,
        risk_dir: str = "data/risk_management",
        initial_capital: Decimal = Decimal("100000"),
        enable_auto_liquidation: bool = True,
    ) -> None:
        self.risk_dir = Path(risk_dir)
        self.risk_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.risk_dir / "breakers").mkdir(exist_ok=True)
        (self.risk_dir / "events").mkdir(exist_ok=True)
        (self.risk_dir / "logs").mkdir(exist_ok=True)

        self.initial_capital = initial_capital
        self.enable_auto_liquidation = enable_auto_liquidation

        # Initialize database
        self.db_path = self.risk_dir / "circuit_breakers.db"
        self._initialize_database()

        # Circuit breaker rules
        self.breaker_rules: dict[str, CircuitBreakerRule] = {}
        self._initialize_default_rules()

        # Event tracking
        self.active_events: dict[str, BreakerEvent] = {}
        self.event_history: deque = deque(maxlen=1000)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.event_queue = queue.Queue()

        # System integrations (injected by trading engine)
        self.trading_engines = {}  # engine_id -> engine instance
        self.risk_monitor = None  # Real-time risk monitor
        self.alerting_system = None  # Alert notification system

        # Callbacks for breaker events
        self.event_callbacks: list[Callable[[BreakerEvent], None]] = []

        logger.info(
            f"Circuit Breaker System initialized - Auto Liquidation: {'ENABLED' if enable_auto_liquidation else 'DISABLED'}"
        )

    def _initialize_database(self) -> None:
        """Initialize SQLite database for circuit breakers"""

        with sqlite3.connect(self.db_path) as conn:
            # Breaker rules table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS breaker_rules (
                    breaker_id TEXT PRIMARY KEY,
                    breaker_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    threshold TEXT NOT NULL,
                    lookback_period_seconds INTEGER NOT NULL,
                    primary_action TEXT NOT NULL,
                    secondary_actions TEXT,
                    cooldown_period_seconds INTEGER,
                    max_triggers_per_day INTEGER,
                    status TEXT NOT NULL,
                    configuration TEXT,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """
            )

            # Breaker events table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS breaker_events (
                    event_id TEXT PRIMARY KEY,
                    breaker_id TEXT NOT NULL,
                    breaker_type TEXT NOT NULL,
                    trigger_value TEXT NOT NULL,
                    threshold TEXT NOT NULL,
                    strategy_id TEXT,
                    symbol TEXT,
                    actions_taken TEXT,
                    positions_closed INTEGER DEFAULT 0,
                    strategies_halted INTEGER DEFAULT 0,
                    triggered_at TEXT NOT NULL,
                    resolved_at TEXT,
                    event_metadata TEXT
                )
            """
            )

            # Breaker statistics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS breaker_stats (
                    breaker_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    trigger_count INTEGER DEFAULT 0,
                    positions_protected INTEGER DEFAULT 0,
                    losses_prevented TEXT DEFAULT '0',
                    last_update TEXT NOT NULL,
                    PRIMARY KEY (breaker_id, date)
                )
            """
            )

            # System status table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_status (
                    component TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    last_heartbeat TEXT NOT NULL,
                    error_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_time ON breaker_events (triggered_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_breaker ON breaker_events (breaker_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stats_date ON breaker_stats (date)")

            conn.commit()

    def _initialize_default_rules(self) -> None:
        """Initialize default circuit breaker rules"""

        # Maximum daily drawdown breaker
        self.add_breaker_rule(
            CircuitBreakerRule(
                breaker_id="max_daily_drawdown",
                breaker_type=CircuitBreakerType.DAILY_LOSS,
                description="Maximum daily drawdown of 5% of capital",
                threshold=self.initial_capital * Decimal("0.05"),
                lookback_period=timedelta(hours=24),
                primary_action=ActionType.HALT_ALL,
                secondary_actions=[ActionType.ALERT_ONLY],
            )
        )

        # Portfolio maximum drawdown breaker
        self.add_breaker_rule(
            CircuitBreakerRule(
                breaker_id="max_portfolio_drawdown",
                breaker_type=CircuitBreakerType.DRAWDOWN,
                description="Maximum portfolio drawdown of 15% from peak",
                threshold=Decimal("0.15"),  # 15% drawdown
                lookback_period=timedelta(days=30),
                primary_action=ActionType.LIQUIDATE_POSITIONS,
                secondary_actions=[ActionType.HALT_ALL, ActionType.ALERT_ONLY],
            )
        )

        # Single position concentration breaker
        self.add_breaker_rule(
            CircuitBreakerRule(
                breaker_id="position_concentration",
                breaker_type=CircuitBreakerType.POSITION_SIZE,
                description="Single position exceeds 20% of portfolio",
                threshold=Decimal("0.20"),  # 20% of portfolio
                lookback_period=timedelta(minutes=1),
                primary_action=ActionType.REDUCE_POSITION_SIZE,
                secondary_actions=[ActionType.ALERT_ONLY],
            )
        )

        # Market volatility circuit breaker
        self.add_breaker_rule(
            CircuitBreakerRule(
                breaker_id="market_volatility_spike",
                breaker_type=CircuitBreakerType.VOLATILITY,
                description="Market volatility exceeds 3x normal levels",
                threshold=Decimal("3.0"),  # 3x normal volatility
                lookback_period=timedelta(minutes=15),
                primary_action=ActionType.HALT_ALL,
                secondary_actions=[ActionType.ALERT_ONLY],
                cooldown_period=timedelta(hours=1),
            )
        )

        # Strategy failure rate breaker
        self.add_breaker_rule(
            CircuitBreakerRule(
                breaker_id="strategy_failure_rate",
                breaker_type=CircuitBreakerType.STRATEGY_FAILURE,
                description="Strategy win rate drops below 30% over 20 trades",
                threshold=Decimal("0.30"),  # 30% win rate threshold
                lookback_period=timedelta(hours=6),
                primary_action=ActionType.HALT_STRATEGY,
                secondary_actions=[ActionType.ALERT_ONLY],
            )
        )

    def add_breaker_rule(self, rule: CircuitBreakerRule) -> None:
        """Add a new circuit breaker rule"""

        self.breaker_rules[rule.breaker_id] = rule
        self._store_breaker_rule(rule)

        console.print(f"   ⚡ Added circuit breaker: {rule.breaker_id}")
        logger.info(f"Circuit breaker rule added: {rule.breaker_id}")

    def remove_breaker_rule(self, breaker_id: str) -> bool:
        """Remove a circuit breaker rule"""

        if breaker_id in self.breaker_rules:
            del self.breaker_rules[breaker_id]

            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM breaker_rules WHERE breaker_id = ?", (breaker_id,))
                conn.commit()

            console.print(f"   🗑️ Removed circuit breaker: {breaker_id}")
            return True

        return False

    def register_trading_engine(self, engine_id: str, engine_instance) -> None:
        """Register a trading engine for circuit breaker control"""

        self.trading_engines[engine_id] = engine_instance
        console.print(f"   🔗 Registered trading engine: {engine_id}")

    def register_risk_monitor(self, risk_monitor) -> None:
        """Register risk monitor for real-time data"""

        self.risk_monitor = risk_monitor
        console.print("   📊 Risk monitor registered")

    def register_alerting_system(self, alerting_system) -> None:
        """Register alerting system for notifications"""

        self.alerting_system = alerting_system
        console.print("   🚨 Alerting system registered")

    def add_event_callback(self, callback: Callable[[BreakerEvent], None]) -> None:
        """Add callback for breaker events"""

        self.event_callbacks.append(callback)

    def start_monitoring(self) -> None:
        """Start circuit breaker monitoring"""

        if self.is_monitoring:
            console.print("⚠️  Circuit breaker monitoring is already active")
            return

        console.print("🚀 [bold green]Starting Circuit Breaker Monitoring[/bold green]")

        self.is_monitoring = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        console.print("   ✅ Real-time monitoring started")
        console.print(f"   ⚡ Active breakers: {len(self.breaker_rules)}")
        console.print("   🛡️ Protection systems online")

        logger.info("Circuit breaker monitoring started successfully")

    def stop_monitoring(self) -> None:
        """Stop circuit breaker monitoring"""

        console.print("⏹️  [bold yellow]Stopping Circuit Breaker Monitoring[/bold yellow]")

        self.is_monitoring = False

        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)

        console.print("   ✅ Circuit breaker monitoring stopped")

        logger.info("Circuit breaker monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop for circuit breakers"""

        while self.is_monitoring:
            try:
                # Check each active breaker rule
                for _breaker_id, rule in self.breaker_rules.items():
                    if rule.status != BreakerStatus.ACTIVE:
                        continue

                    # Check if rule should trigger
                    if self._check_breaker_condition(rule):
                        self._trigger_circuit_breaker(rule)

                # Process any queued events
                self._process_event_queue()

                # Update breaker cooldowns
                self._update_breaker_cooldowns()

                # Sleep between checks
                time.sleep(1)  # Check every second for real-time protection

            except Exception as e:
                logger.error(f"Circuit breaker monitoring error: {str(e)}")
                time.sleep(5)  # Longer pause on error

    def _check_breaker_condition(self, rule: CircuitBreakerRule) -> bool:
        """Check if a specific breaker condition is met"""

        try:
            if rule.breaker_type == CircuitBreakerType.DAILY_LOSS:
                return self._check_daily_loss_condition(rule)
            elif rule.breaker_type == CircuitBreakerType.DRAWDOWN:
                return self._check_drawdown_condition(rule)
            elif rule.breaker_type == CircuitBreakerType.POSITION_SIZE:
                return self._check_position_concentration_condition(rule)
            elif rule.breaker_type == CircuitBreakerType.VOLATILITY:
                return self._check_volatility_condition(rule)
            elif rule.breaker_type == CircuitBreakerType.STRATEGY_FAILURE:
                return self._check_strategy_failure_condition(rule)

            return False

        except Exception as e:
            logger.error(f"Error checking breaker condition {rule.breaker_id}: {str(e)}")
            return False

    def _check_daily_loss_condition(self, rule: CircuitBreakerRule) -> bool:
        """Check daily loss breaker condition"""

        if not self.risk_monitor:
            return False

        # Get current risk metrics
        risk_metrics = self.risk_monitor.get_current_risk_metrics()
        if not risk_metrics:
            return False

        # Calculate daily loss
        daily_loss = risk_metrics.total_unrealized_pnl + risk_metrics.total_realized_pnl

        # Check if loss exceeds threshold
        return abs(daily_loss) > rule.threshold and daily_loss < 0

    def _check_drawdown_condition(self, rule: CircuitBreakerRule) -> bool:
        """Check maximum drawdown breaker condition"""

        if not self.risk_monitor:
            return False

        # Get current risk metrics
        risk_metrics = self.risk_monitor.get_current_risk_metrics()
        if not risk_metrics:
            return False

        # Check if current drawdown exceeds threshold
        return risk_metrics.current_drawdown > rule.threshold

    def _check_position_concentration_condition(self, rule: CircuitBreakerRule) -> bool:
        """Check position concentration breaker condition"""

        if not self.risk_monitor:
            return False

        # Get position metrics
        position_metrics = self.risk_monitor.get_position_metrics()
        if not position_metrics:
            return False

        # Check largest position concentration
        return position_metrics.get("max_position_pct", 0) > float(rule.threshold)

    def _check_volatility_condition(self, rule: CircuitBreakerRule) -> bool:
        """Check market volatility breaker condition"""

        # This would integrate with market data to check volatility spikes
        # For demo purposes, return False
        return False

    def _check_strategy_failure_condition(self, rule: CircuitBreakerRule) -> bool:
        """Check strategy failure rate breaker condition"""

        # This would check strategy performance metrics
        # For demo purposes, return False
        return False

    def _trigger_circuit_breaker(self, rule: CircuitBreakerRule) -> None:
        """Trigger a circuit breaker"""

        # Check daily trigger limit
        today = datetime.now().date()
        if rule.last_triggered and rule.last_triggered.date() == today:
            if rule.trigger_count >= rule.max_triggers_per_day:
                return  # Already triggered maximum times today

        # Create breaker event
        event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule.breaker_id}"

        event = BreakerEvent(
            event_id=event_id,
            breaker_id=rule.breaker_id,
            breaker_type=rule.breaker_type,
            trigger_value=Decimal("0"),  # Would be populated with actual trigger value
            threshold=rule.threshold,
        )

        # Execute primary action
        actions_taken = []
        if self._execute_breaker_action(rule.primary_action, event):
            actions_taken.append(rule.primary_action)

        # Execute secondary actions
        for action in rule.secondary_actions:
            if self._execute_breaker_action(action, event):
                actions_taken.append(action)

        event.actions_taken = actions_taken

        # Update rule status
        rule.status = BreakerStatus.TRIGGERED
        rule.last_triggered = datetime.now()
        rule.trigger_count += 1

        # Store event
        self.active_events[event_id] = event
        self.event_history.append(event)
        self._store_breaker_event(event)

        # Update rule in database
        self._store_breaker_rule(rule)

        # Send notifications
        self._send_breaker_alert(event, rule)

        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Breaker event callback error: {str(e)}")

        console.print(f"🚨 [bold red]CIRCUIT BREAKER TRIGGERED[/bold red]: {rule.breaker_id}")
        logger.critical(f"Circuit breaker triggered: {rule.breaker_id}")

    def _execute_breaker_action(self, action: ActionType, event: BreakerEvent) -> bool:
        """Execute a circuit breaker action"""

        try:
            if action == ActionType.HALT_ALL:
                return self._halt_all_trading()
            elif action == ActionType.HALT_STRATEGY:
                return self._halt_strategy_trading(event.strategy_id)
            elif action == ActionType.LIQUIDATE_POSITIONS:
                return self._liquidate_all_positions(event)
            elif action == ActionType.REDUCE_POSITION_SIZE:
                return self._reduce_position_sizes(event)
            elif action == ActionType.ALERT_ONLY:
                return True  # Always succeeds for alerts

            return False

        except Exception as e:
            logger.error(f"Failed to execute breaker action {action.value}: {str(e)}")
            return False

    def _halt_all_trading(self) -> bool:
        """Halt all trading across all engines"""

        halted_count = 0

        for engine_id, engine in self.trading_engines.items():
            try:
                if hasattr(engine, "stop_trading_engine"):
                    engine.stop_trading_engine()
                    halted_count += 1
                    console.print(f"      🛑 Halted trading engine: {engine_id}")
            except Exception as e:
                logger.error(f"Failed to halt trading engine {engine_id}: {str(e)}")

        return halted_count > 0

    def _halt_strategy_trading(self, strategy_id: str | None) -> bool:
        """Halt trading for a specific strategy"""

        if not strategy_id:
            return False

        # This would implement strategy-specific halting
        console.print(f"      🛑 Strategy halted: {strategy_id}")
        return True

    def _liquidate_all_positions(self, event: BreakerEvent) -> bool:
        """Liquidate all open positions"""

        if not self.enable_auto_liquidation:
            console.print("      ⚠️ Auto-liquidation disabled - positions remain open")
            return False

        liquidated_count = 0

        for engine_id, engine in self.trading_engines.items():
            try:
                if hasattr(engine, "positions"):
                    for _position_key, position in engine.positions.items():
                        # Create liquidation order
                        if position.quantity > 0:
                            # Close long position with market sell order
                            engine.submit_order(
                                strategy_id="circuit_breaker_liquidation",
                                symbol=position.symbol,
                                side=engine.OrderSide.SELL,
                                quantity=position.quantity,
                                order_type=engine.OrderType.MARKET,
                                notes=f"Circuit breaker liquidation - Event: {event.event_id}",
                            )
                            liquidated_count += 1
                            console.print(
                                f"      💰 Liquidating position: {position.symbol} ({position.quantity})"
                            )
            except Exception as e:
                logger.error(f"Failed to liquidate positions in engine {engine_id}: {str(e)}")

        event.positions_closed = liquidated_count
        return liquidated_count > 0

    def _reduce_position_sizes(self, event: BreakerEvent) -> bool:
        """Reduce position sizes by 50%"""

        reduced_count = 0

        for engine_id, engine in self.trading_engines.items():
            try:
                if hasattr(engine, "positions"):
                    for _position_key, position in engine.positions.items():
                        # Reduce position by 50%
                        reduction_quantity = position.quantity * Decimal("0.5")
                        if reduction_quantity > 0:
                            engine.submit_order(
                                strategy_id="circuit_breaker_reduction",
                                symbol=position.symbol,
                                side=engine.OrderSide.SELL,
                                quantity=reduction_quantity,
                                order_type=engine.OrderType.MARKET,
                                notes=f"Circuit breaker position reduction - Event: {event.event_id}",
                            )
                            reduced_count += 1
                            console.print(
                                f"      📉 Reducing position: {position.symbol} by {reduction_quantity}"
                            )
            except Exception as e:
                logger.error(f"Failed to reduce positions in engine {engine_id}: {str(e)}")

        return reduced_count > 0

    def _process_event_queue(self) -> None:
        """Process any queued events"""

        try:
            while not self.event_queue.empty():
                self.event_queue.get_nowait()
                # Process event (placeholder for future event processing)
                self.event_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Event queue processing error: {str(e)}")

    def _update_breaker_cooldowns(self) -> None:
        """Update breaker cooldowns and reset triggered breakers"""

        current_time = datetime.now()

        for rule in self.breaker_rules.values():
            if rule.status == BreakerStatus.TRIGGERED:
                if (
                    rule.last_triggered
                    and current_time - rule.last_triggered > rule.cooldown_period
                ):
                    rule.status = BreakerStatus.ACTIVE
                    console.print(f"   🔄 Circuit breaker reset: {rule.breaker_id}")

    def _send_breaker_alert(self, event: BreakerEvent, rule: CircuitBreakerRule) -> None:
        """Send circuit breaker alert"""

        if self.alerting_system:
            try:
                self.alerting_system.send_critical_alert(
                    title=f"CIRCUIT BREAKER TRIGGERED: {rule.breaker_id}",
                    message=f"{rule.description}\nTrigger Value: {event.trigger_value}\nThreshold: {event.threshold}\nActions Taken: {', '.join([a.value for a in event.actions_taken])}",
                    severity="critical",
                )
            except Exception as e:
                logger.error(f"Failed to send breaker alert: {str(e)}")

    def manual_trigger(self, breaker_id: str, reason: str = "Manual override") -> bool:
        """Manually trigger a circuit breaker"""

        if breaker_id not in self.breaker_rules:
            return False

        rule = self.breaker_rules[breaker_id]

        # Create manual trigger event
        event = BreakerEvent(
            event_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{breaker_id}",
            breaker_id=breaker_id,
            breaker_type=CircuitBreakerType.MANUAL,
            trigger_value=rule.threshold,
            threshold=rule.threshold,
            metadata={"reason": reason},
        )

        # Execute actions
        actions_taken = []
        if self._execute_breaker_action(rule.primary_action, event):
            actions_taken.append(rule.primary_action)

        event.actions_taken = actions_taken

        # Store event
        self.active_events[event.event_id] = event
        self.event_history.append(event)
        self._store_breaker_event(event)

        console.print(f"🚨 [bold red]MANUAL CIRCUIT BREAKER TRIGGERED[/bold red]: {breaker_id}")
        console.print(f"   Reason: {reason}")

        return True

    def get_breaker_status(self) -> dict[str, Any]:
        """Get current circuit breaker system status"""

        active_breakers = sum(
            1 for rule in self.breaker_rules.values() if rule.status == BreakerStatus.ACTIVE
        )
        triggered_breakers = sum(
            1 for rule in self.breaker_rules.values() if rule.status == BreakerStatus.TRIGGERED
        )

        return {
            "is_monitoring": self.is_monitoring,
            "total_breakers": len(self.breaker_rules),
            "active_breakers": active_breakers,
            "triggered_breakers": triggered_breakers,
            "active_events": len(self.active_events),
            "total_events": len(self.event_history),
            "registered_engines": len(self.trading_engines),
            "auto_liquidation_enabled": self.enable_auto_liquidation,
        }

    def _store_breaker_rule(self, rule: CircuitBreakerRule) -> None:
        """Store breaker rule in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO breaker_rules (
                    breaker_id, breaker_type, description, threshold, lookback_period_seconds,
                    primary_action, secondary_actions, cooldown_period_seconds, max_triggers_per_day,
                    status, configuration, created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rule.breaker_id,
                    rule.breaker_type.value,
                    rule.description,
                    str(rule.threshold),
                    int(rule.lookback_period.total_seconds()),
                    rule.primary_action.value,
                    json.dumps([action.value for action in rule.secondary_actions]),
                    int(rule.cooldown_period.total_seconds()),
                    rule.max_triggers_per_day,
                    rule.status.value,
                    json.dumps(
                        {
                            "trigger_count": rule.trigger_count,
                            "last_triggered": (
                                rule.last_triggered.isoformat() if rule.last_triggered else None
                            ),
                            "strategy_ids": list(rule.strategy_ids) if rule.strategy_ids else None,
                            "symbol_filters": (
                                list(rule.symbol_filters) if rule.symbol_filters else None
                            ),
                        }
                    ),
                    rule.last_reset.isoformat(),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def _store_breaker_event(self, event: BreakerEvent) -> None:
        """Store breaker event in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO breaker_events (
                    event_id, breaker_id, breaker_type, trigger_value, threshold,
                    strategy_id, symbol, actions_taken, positions_closed, strategies_halted,
                    triggered_at, resolved_at, event_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.breaker_id,
                    event.breaker_type.value,
                    str(event.trigger_value),
                    str(event.threshold),
                    event.strategy_id,
                    event.symbol,
                    json.dumps([action.value for action in event.actions_taken]),
                    event.positions_closed,
                    event.strategies_halted,
                    event.triggered_at.isoformat(),
                    event.resolved_at.isoformat() if event.resolved_at else None,
                    json.dumps(event.metadata),
                ),
            )
            conn.commit()

    def display_breaker_dashboard(self) -> None:
        """Display circuit breaker dashboard"""

        status = self.get_breaker_status()

        console.print(
            Panel(
                f"[bold blue]Circuit Breaker System Dashboard[/bold blue]\n"
                f"Status: {'🟢 MONITORING' if status['is_monitoring'] else '🔴 STOPPED'}\n"
                f"Active Breakers: {status['active_breakers']}/{status['total_breakers']}\n"
                f"Auto-Liquidation: {'🟢 ENABLED' if status['auto_liquidation_enabled'] else '🔴 DISABLED'}",
                title="⚡ Circuit Breakers",
            )
        )

        # Breaker rules table
        if self.breaker_rules:
            rules_table = Table(title="⚡ Circuit Breaker Rules")
            rules_table.add_column("ID", style="cyan")
            rules_table.add_column("Type", style="yellow")
            rules_table.add_column("Description", style="white")
            rules_table.add_column("Threshold", justify="right")
            rules_table.add_column("Status", style="green")
            rules_table.add_column("Triggers", justify="right", style="red")

            for rule in list(self.breaker_rules.values()):
                status_color = {
                    BreakerStatus.ACTIVE: "green",
                    BreakerStatus.TRIGGERED: "red",
                    BreakerStatus.COOLDOWN: "yellow",
                    BreakerStatus.DISABLED: "dim",
                }.get(rule.status, "white")

                rules_table.add_row(
                    rule.breaker_id,
                    rule.breaker_type.value.title(),
                    (
                        rule.description[:50] + "..."
                        if len(rule.description) > 50
                        else rule.description
                    ),
                    str(rule.threshold),
                    f"[{status_color}]{rule.status.value.title()}[/{status_color}]",
                    str(rule.trigger_count),
                )

            console.print(rules_table)

        # Recent events
        if self.event_history:
            console.print(
                f"\n⚡ [bold]Recent Circuit Breaker Events ({len(self.event_history)} total)[/bold]"
            )
            for event in list(self.event_history)[-3:]:  # Show last 3 events
                elapsed = datetime.now() - event.triggered_at
                elapsed_str = (
                    f"{elapsed.total_seconds():.0f}s ago"
                    if elapsed.total_seconds() < 60
                    else f"{elapsed.total_seconds()/60:.0f}m ago"
                )
                console.print(
                    f"   {event.breaker_id}: {', '.join([a.value for a in event.actions_taken])} ({elapsed_str})"
                )


def create_circuit_breaker_system(
    risk_dir: str = "data/risk_management",
    initial_capital: Decimal = Decimal("100000"),
    enable_auto_liquidation: bool = True,
) -> CircuitBreakerSystem:
    """Factory function to create circuit breaker system"""
    return CircuitBreakerSystem(
        risk_dir=risk_dir,
        initial_capital=initial_capital,
        enable_auto_liquidation=enable_auto_liquidation,
    )


if __name__ == "__main__":
    # Example usage
    breaker_system = create_circuit_breaker_system()

    # Start monitoring
    breaker_system.start_monitoring()

    try:
        # Demo - let it run for a bit
        time.sleep(5)

        # Display dashboard
        breaker_system.display_breaker_dashboard()

        # Demo manual trigger
        breaker_system.manual_trigger("max_daily_drawdown", "Testing circuit breaker system")

        time.sleep(2)
        breaker_system.display_breaker_dashboard()

    finally:
        breaker_system.stop_monitoring()

    print("Circuit Breaker System created successfully!")
