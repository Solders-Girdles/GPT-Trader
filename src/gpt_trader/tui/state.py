"""
TUI State Management.

Provides reactive state management for the TUI with validation and
delta update capabilities to minimize UI flicker and catch data issues.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from textual.reactive import reactive
from textual.widget import Widget

from gpt_trader.core.account import CFMBalance
from gpt_trader.monitoring.status_reporter import BotStatus
from gpt_trader.tui.events import (
    StateDeltaUpdateApplied,
    StateValidationFailed,
    StateValidationPassed,
)
from gpt_trader.tui.events import (
    ValidationError as ValidationErrorEvent,
)
from gpt_trader.tui.formatting import safe_decimal
from gpt_trader.tui.state_management.delta_updater import StateDeltaUpdater
from gpt_trader.tui.state_management.validators import StateValidator
from gpt_trader.tui.types import (
    AccountBalance,
    AccountSummary,
    ActiveOrders,
    DecisionData,
    ExecutionMetrics,
    IndicatorContribution,
    MarketState,
    Order,
    PortfolioSummary,
    Position,
    RegimeData,
    ResilienceState,
    RiskGuard,
    RiskState,
    StrategyParameters,
    StrategyPerformance,
    StrategyState,
    SystemStatus,
    Trade,
    TradeHistory,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class TuiState(Widget):
    """Reactive state for the TUI.

    Acts as a ViewModel that the App observes. Includes validation layer
    and delta update support to minimize UI flicker.

    Attributes:
        validator: State validation layer for incoming data
        delta_updater: Delta update calculator for efficient updates
        validation_enabled: Whether to validate incoming data
        delta_updates_enabled: Whether to use delta updates (vs full replacement)
        _changed_fields: Tracks which fields changed in the last update
    """

    # Reactive properties that widgets can watch
    running = reactive(False)
    uptime = reactive(0.0)
    cycle_count = reactive(0)

    # Mode and connection tracking
    data_source_mode = reactive("demo")  # demo, paper, read_only, live
    last_update_timestamp = reactive(0.0)
    update_interval = reactive(2.0)
    connection_healthy = reactive(True)

    # Data fetch state (separate from trading state)
    data_fetching = reactive(False)  # True during active data fetch
    data_available = reactive(False)  # True after first successful data received
    last_data_fetch = reactive(0.0)  # Timestamp of last successful fetch

    # Degraded mode tracking (when StatusReporter unavailable)
    degraded_mode = reactive(False)
    degraded_reason = reactive("")

    # Validation state tracking
    validation_error_count = reactive(0)
    validation_warning_count = reactive(0)

    # We use reactive for complex objects too, but need to be careful about mutation
    # For simple updates, replacing the whole object triggers the watcher
    market_data = reactive(MarketState())
    position_data = reactive(PortfolioSummary())
    order_data = reactive(ActiveOrders())
    trade_data = reactive(TradeHistory())
    account_data = reactive(AccountSummary())
    strategy_data = reactive(StrategyState())
    risk_data = reactive(RiskState())
    system_data = reactive(SystemStatus())

    # CFM (Coinbase Financial Markets) futures state
    cfm_balance: reactive[CFMBalance | None] = reactive(None)
    has_cfm_access = reactive(False)

    # API resilience metrics
    resilience_data = reactive(ResilienceState())

    # Strategy performance metrics (from PerformanceSnapshot)
    strategy_performance = reactive(StrategyPerformance())

    # Backtest performance metrics (from historical backtesting)
    backtest_performance: reactive[StrategyPerformance | None] = reactive(None)

    # Market regime data
    regime_data = reactive(RegimeData())

    def __init__(
        self,
        *args: Any,
        validation_enabled: bool = True,
        delta_updates_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize TuiState with validation and delta update support.

        Args:
            validation_enabled: Whether to validate incoming data (default True)
            delta_updates_enabled: Whether to use delta updates (default True)
            *args: Positional arguments passed to Widget
            **kwargs: Keyword arguments passed to Widget
        """
        super().__init__(*args, **kwargs)

        # Validation and delta update components
        self.validator = StateValidator()
        self.delta_updater = StateDeltaUpdater()
        self.validation_enabled = validation_enabled
        self.delta_updates_enabled = delta_updates_enabled

        # Track which fields changed in the last update (for optimized broadcasts)
        self._changed_fields: set[str] = set()

        # Ensure we have fresh instances for each TuiState
        self.market_data = MarketState()
        self.position_data = PortfolioSummary()
        self.order_data = ActiveOrders()
        self.trade_data = TradeHistory()
        self.account_data = AccountSummary()
        self.strategy_data = StrategyState()
        self.risk_data = RiskState()
        self.system_data = SystemStatus()
        self.resilience_data = ResilienceState()
        self.execution_data = ExecutionMetrics()
        self.strategy_performance = StrategyPerformance()
        self.regime_data = RegimeData()

    def update_from_bot_status(
        self,
        status: BotStatus,
        runtime_state: Any | None = None,
        use_delta: bool | None = None,
    ) -> None:
        """Update state from the bot's typed status snapshot.

        Each component update is isolated to prevent cascade failures.
        If one component fails, others still update successfully.

        Args:
            status: Typed BotStatus snapshot from StatusReporter
            runtime_state: Optional runtime state (engine state, uptime, etc.)
            use_delta: Whether to use delta updates. If None, uses instance default.
        """
        # Connection health is based on observer update cadence, not market ticks.
        # Track the time of the last status snapshot received.
        try:
            self.last_update_timestamp = float(getattr(status, "timestamp", 0.0)) or 0.0
        except Exception:
            import time

            self.last_update_timestamp = time.time()

        # Sync update_interval from reporter's observer_interval for connection health
        if hasattr(status, "observer_interval") and status.observer_interval > 0:
            self.update_interval = status.observer_interval

        # Determine if we should use delta updates
        should_use_delta = use_delta if use_delta is not None else self.delta_updates_enabled

        # Run validation if enabled
        if self.validation_enabled:
            validation_result = self.validator.validate_full_state(status)

            # Update validation counts
            self.validation_error_count = len(validation_result.errors)
            self.validation_warning_count = len(validation_result.warnings)

            # Post validation events
            if not validation_result.valid or validation_result.warnings:
                all_issues = validation_result.errors + validation_result.warnings
                event_errors = [
                    ValidationErrorEvent(
                        field=e.field,
                        message=e.message,
                        severity=e.severity,
                        value=e.value,
                    )
                    for e in all_issues
                ]
                self.post_message(
                    StateValidationFailed(errors=event_errors, component="full_state")
                )

                # Log validation issues
                for error in validation_result.errors:
                    logger.warning(f"Validation error: {error.field} - {error.message}")

                # Continue with update despite validation warnings/errors
                # (data may still be partially usable)
            else:
                self.post_message(StateValidationPassed())

        # Clear changed fields tracking for this update cycle
        self._changed_fields.clear()

        # Define update operations with error isolation
        update_operations = [
            ("market", lambda: self._update_market_data(status.market)),
            ("positions", lambda: self._update_position_data(status.positions)),
            ("orders", lambda: self._update_order_data(status.orders)),
            ("trades", lambda: self._update_trade_data(status.trades)),
            ("account", lambda: self._update_account_data(status.account)),
            ("strategy", lambda: self._update_strategy_data(status.strategy)),
            ("risk", lambda: self._update_risk_data(status.risk)),
            ("system", lambda: self._update_system_data(status.system)),
            ("runtime", lambda: self._update_runtime_stats(runtime_state)),
        ]

        failed_updates = []
        successful_updates = []
        for component_name, update_operation in update_operations:
            try:
                update_operation()
                successful_updates.append(component_name)
                # Track which components were successfully updated
                self._changed_fields.add(component_name)
            except Exception as e:
                logger.error(f"Failed to update {component_name} data: {e}", exc_info=True)
                failed_updates.append(component_name)

        # Post delta update event
        if should_use_delta:
            self.post_message(
                StateDeltaUpdateApplied(
                    components_updated=successful_updates,
                    use_full_update=not should_use_delta,
                )
            )

        if failed_updates:
            logger.warning(
                f"State update completed with {len(failed_updates)} failures: "
                f"{', '.join(failed_updates)}"
            )

    def check_connection_health(self) -> bool:
        """
        Check if data connection is healthy based on update interval.

        Returns:
            True if data is fresh, False if stale
        """
        if self.data_source_mode == "demo":
            return True  # Demo always healthy

        # When the bot is intentionally stopped (manual start), we don't expect
        # periodic StatusReporter updates. Treat this as a healthy "stopped" state
        # so the UI doesn't spam stale warnings while waiting for the user to start.
        if not self.running and not self.degraded_mode:
            if not self.connection_healthy:
                self.connection_healthy = True
            return True

        import time

        time_since_update = time.time() - self.last_update_timestamp
        staleness_threshold = self.update_interval * 2.5
        is_healthy = time_since_update < staleness_threshold

        # Update connection_healthy if changed
        if is_healthy != self.connection_healthy:
            self.connection_healthy = is_healthy

        return is_healthy

    @property
    def is_data_stale(self) -> bool:
        """Check if data is older than staleness threshold (30s).

        Used by widgets to show stale data warnings.

        Returns:
            True if data hasn't been updated in over 30 seconds.
        """
        if not self.data_available or self.last_data_fetch == 0:
            return False
        import time

        return (time.time() - self.last_data_fetch) > 30.0

    @staticmethod
    def _iter_key_values(value: Any) -> Any:
        """Iterate key/value pairs from dict-like or attribute-like objects.

        Status snapshots are normally typed dataclasses, but tests and some
        adapters may provide simple objects (e.g., SimpleNamespace).
        """
        if not value:
            return ()
        if isinstance(value, dict):
            return value.items()
        if hasattr(value, "items"):
            try:
                return value.items()  # type: ignore[no-any-return]
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            return getattr(value, "__dict__", {}).items()
        return ()

    def _update_market_data(self, market: Any) -> None:  # MarketStatus from status_reporter
        """Update market data from typed MarketStatus."""
        # Convert prices to Decimal (StatusReporter may provide str or already Decimal)
        prices_decimal = {}
        for symbol, price in self._iter_key_values(getattr(market, "last_prices", None)):
            prices_decimal[str(symbol)] = safe_decimal(price)

        # Convert price history to Decimal
        price_history_converted = {}
        for symbol, history in self._iter_key_values(getattr(market, "price_history", None)):
            if isinstance(history, list):
                price_history_converted[str(symbol)] = [safe_decimal(p) for p in history]
            else:
                price_history_converted[str(symbol)] = history

        self.market_data = MarketState(
            prices=prices_decimal,
            last_update=float(getattr(market, "last_price_update", 0.0) or 0.0),
            price_history=price_history_converted,
        )

    def _update_position_data(self, pos_data: Any) -> None:  # PositionStatus from status_reporter
        """Update position data from typed PositionStatus."""
        positions_map = {}

        # PositionStatus has positions dict that contains position data
        # Each position is a dict with keys: quantity, entry_price, unrealized_pnl, mark_price, side
        for symbol, p_data in self._iter_key_values(getattr(pos_data, "positions", None)):
            symbol_str = str(symbol)
            if isinstance(p_data, dict):
                quantity = safe_decimal(p_data.get("quantity", "0"))
                entry_price = safe_decimal(p_data.get("entry_price", "0"))
                unrealized_pnl = safe_decimal(p_data.get("unrealized_pnl", "0"))
                mark_price = safe_decimal(p_data.get("mark_price", "0"))
                side = str(p_data.get("side", ""))
            else:
                quantity = safe_decimal(getattr(p_data, "quantity", "0"))
                entry_price = safe_decimal(getattr(p_data, "entry_price", "0"))
                unrealized_pnl = safe_decimal(getattr(p_data, "unrealized_pnl", "0"))
                mark_price = safe_decimal(getattr(p_data, "mark_price", "0"))
                side = str(getattr(p_data, "side", ""))

            positions_map[symbol_str] = Position(
                symbol=symbol_str,
                quantity=quantity,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                mark_price=mark_price,
                side=side,
            )

        # Calculate total fees from trade history
        total_fees = Decimal("0")
        try:
            for trade in self.trade_data.trades:
                total_fees += trade.fee
        except Exception:
            pass

        # Get realized P&L from status if available
        total_realized_pnl = safe_decimal(getattr(pos_data, "total_realized_pnl", "0"))

        self.position_data = PortfolioSummary(
            positions=positions_map,
            total_unrealized_pnl=safe_decimal(pos_data.total_unrealized_pnl),
            equity=safe_decimal(pos_data.equity),
            total_realized_pnl=total_realized_pnl,
            total_fees=total_fees,
        )

    def _update_order_data(self, raw_orders: list[Any]) -> None:  # list[OrderStatus]
        """Update order data from typed list of OrderStatus."""
        orders_list = []
        for o in raw_orders:
            # OrderStatus from status_reporter has order_type field (not type)
            # creation_time is now a float epoch timestamp
            creation_time = float(o.creation_time) if o.creation_time else 0.0

            orders_list.append(
                Order(
                    order_id=o.order_id,
                    symbol=o.symbol,
                    side=o.side,
                    quantity=safe_decimal(o.quantity),
                    price=safe_decimal(o.price) if o.price else safe_decimal("0"),
                    status=o.status,
                    type=o.order_type,  # OrderStatus uses order_type field
                    time_in_force=o.time_in_force,
                    creation_time=creation_time,
                    filled_quantity=safe_decimal(o.filled_quantity),
                    avg_fill_price=(safe_decimal(o.avg_fill_price) if o.avg_fill_price else None),
                )
            )
        self.order_data = ActiveOrders(orders=orders_list)

    def _update_trade_data(self, raw_trades: list[Any]) -> None:  # list[TradeStatus]
        """Update trade data from typed list of TradeStatus."""
        trades_list = []
        for t in raw_trades:
            # TradeStatus from status_reporter has all fields typed
            trades_list.append(
                Trade(
                    trade_id=t.trade_id,
                    symbol=t.symbol,
                    side=t.side,
                    quantity=safe_decimal(t.quantity),
                    price=safe_decimal(t.price),
                    order_id=t.order_id,
                    time=t.time,
                    fee=safe_decimal(t.fee),
                )
            )
        self.trade_data = TradeHistory(trades=trades_list)

    def _update_account_data(self, acc: Any) -> None:  # AccountStatus from status_reporter
        """Update account data from typed AccountStatus."""
        balances_list = []
        for b in acc.balances:
            # BalanceEntry from StatusReporter → AccountBalance for TUI
            # Both have identical fields (asset, total, available, hold as Decimal)
            if hasattr(b, "asset"):
                balances_list.append(
                    AccountBalance(
                        asset=b.asset,
                        total=b.total,  # Already Decimal from BalanceEntry
                        available=b.available,
                        hold=b.hold,
                    )
                )

        self.account_data = AccountSummary(
            volume_30d=acc.volume_30d,  # Already Decimal from AccountStatus
            fees_30d=acc.fees_30d,
            fee_tier=acc.fee_tier,
            balances=balances_list,
        )

    def _update_strategy_data(self, strat: Any) -> None:  # StrategyStatus from status_reporter
        """Update strategy data from typed StrategyStatus with DecisionEntry objects."""
        decisions = {}

        # StrategyStatus stores last_decisions as list[DecisionEntry] (already typed)
        for dec in strat.last_decisions:
            # DecisionEntry from StatusReporter has all fields typed
            if not hasattr(dec, "symbol") or not dec.symbol:
                continue

            # Parse indicator contributions if available
            contributions: list[IndicatorContribution] = []
            raw_contributions = getattr(dec, "contributions", []) or []
            for contrib in raw_contributions:
                if isinstance(contrib, dict):
                    contributions.append(
                        IndicatorContribution(
                            name=str(contrib.get("name", "")),
                            value=(
                                float(contrib.get("value", 0))
                                if isinstance(contrib.get("value"), (int, float))
                                else 0.0
                            ),
                            contribution=float(contrib.get("contribution", 0)),
                            weight=float(contrib.get("weight", 1.0)),
                        )
                    )
                elif isinstance(contrib, IndicatorContribution):
                    contributions.append(contrib)

            decisions[dec.symbol] = DecisionData(
                symbol=dec.symbol,
                action=dec.action,
                reason=dec.reason,
                confidence=dec.confidence,  # Already float from DecisionEntry
                indicators=dec.indicators,
                timestamp=dec.timestamp,  # Already float from DecisionEntry
                decision_id=getattr(dec, "decision_id", ""),  # For order linkage
                blocked_by=getattr(dec, "blocked_by", ""),  # Guard that blocked execution
                contributions=contributions,
            )

        # Parse strategy parameters if present
        params: StrategyParameters | None = None
        raw_params = getattr(strat, "parameters", None)
        if raw_params and isinstance(raw_params, dict):
            params = StrategyParameters(
                rsi_period=raw_params.get("rsi_period"),
                rsi_overbought=raw_params.get("rsi_overbought"),
                rsi_oversold=raw_params.get("rsi_oversold"),
                ma_fast_period=raw_params.get("ma_fast_period"),
                ma_slow_period=raw_params.get("ma_slow_period"),
                ma_type=raw_params.get("ma_type"),
                zscore_lookback=raw_params.get("zscore_lookback"),
                zscore_entry_threshold=raw_params.get("zscore_entry_threshold"),
                zscore_exit_threshold=raw_params.get("zscore_exit_threshold"),
                vwap_deviation_threshold=raw_params.get("vwap_deviation_threshold"),
                spread_tight_bps=raw_params.get("spread_tight_bps"),
                spread_normal_bps=raw_params.get("spread_normal_bps"),
                spread_wide_bps=raw_params.get("spread_wide_bps"),
                orderbook_levels=raw_params.get("orderbook_levels"),
                orderbook_imbalance_threshold=raw_params.get("orderbook_imbalance_threshold"),
            )

        self.strategy_data = StrategyState(
            active_strategies=strat.active_strategies,
            last_decisions=decisions,
            parameters=params,
        )

        # Update performance metrics if present
        if hasattr(strat, "performance") and strat.performance is not None:
            self.update_strategy_performance(strat.performance)
        if hasattr(strat, "backtest_performance") and strat.backtest_performance is not None:
            self.update_backtest_performance(strat.backtest_performance)

    def _update_risk_data(self, risk: Any) -> None:  # RiskStatus from status_reporter
        """Update risk data from typed RiskStatus."""
        # Parse enhanced guards if available
        guards: list[RiskGuard] = []
        raw_guards = getattr(risk, "guards", []) or []
        for guard_data in raw_guards:
            if isinstance(guard_data, dict):
                guards.append(
                    RiskGuard(
                        name=str(guard_data.get("name", "")),
                        severity=str(guard_data.get("severity", "MEDIUM")),
                        last_triggered=float(guard_data.get("last_triggered", 0.0) or 0.0),
                        triggered_count=int(guard_data.get("triggered_count", 0) or 0),
                        description=str(guard_data.get("description", "") or ""),
                    )
                )
            elif isinstance(guard_data, RiskGuard):
                guards.append(guard_data)

        self.risk_data = RiskState(
            max_leverage=float(getattr(risk, "max_leverage", 0.0) or 0.0),
            daily_loss_limit_pct=float(getattr(risk, "daily_loss_limit_pct", 0.0) or 0.0),
            current_daily_loss_pct=float(getattr(risk, "current_daily_loss_pct", 0.0) or 0.0),
            reduce_only_mode=bool(getattr(risk, "reduce_only_mode", False)),
            reduce_only_reason=str(getattr(risk, "reduce_only_reason", "") or ""),
            active_guards=list(getattr(risk, "active_guards", []) or []),
            guards=guards,
        )

    def _update_system_data(self, sys: Any) -> None:  # SystemStatus from status_reporter
        """Update system data from typed SystemStatus."""
        self.system_data = SystemStatus(
            api_latency=sys.api_latency,
            connection_status=sys.connection_status,
            rate_limit_usage=sys.rate_limit_usage,
            memory_usage=sys.memory_usage,
            cpu_usage=sys.cpu_usage,
        )

    def _update_runtime_stats(self, runtime_state: Any | None) -> None:
        """Update runtime statistics (uptime, cycle count, etc.)."""
        if runtime_state and hasattr(runtime_state, "uptime"):
            self.uptime = runtime_state.uptime
        if runtime_state and hasattr(runtime_state, "cycle_count"):
            self.cycle_count = runtime_state.cycle_count

    def update_cfm_balance(self, cfm_balance: CFMBalance | None) -> None:
        """Update CFM futures balance state.

        This method is called separately from the main bot status update
        since CFM data comes from a different source (PortfolioService).

        Args:
            cfm_balance: CFMBalance object or None if CFM access unavailable.
        """
        self.cfm_balance = cfm_balance
        self.has_cfm_access = cfm_balance is not None
        self._changed_fields.add("cfm")

        if cfm_balance:
            logger.debug(
                f"[TuiState] CFM balance updated: "
                f"buying_power={cfm_balance.futures_buying_power}, "
                f"margin_utilization={cfm_balance.margin_utilization_pct:.1f}%, "
                f"liq_buffer={cfm_balance.liquidation_buffer_percentage:.1f}%"
            )

    def update_resilience_data(self, resilience_status: dict[str, Any]) -> None:
        """Update API resilience metrics from CoinbaseClient.get_resilience_status().

        This method is called periodically to refresh resilience metrics
        shown in the System tile.

        Args:
            resilience_status: Dict with keys: metrics, cache, circuit_breakers, rate_limit_usage
        """
        import time

        metrics = resilience_status.get("metrics") or {}
        cache = resilience_status.get("cache") or {}
        breakers = resilience_status.get("circuit_breakers") or {}

        # Parse circuit breaker states
        breaker_states: dict[str, str] = {}
        for category, status in breakers.items():
            if isinstance(status, dict):
                breaker_states[category] = status.get("state", "closed")
            else:
                breaker_states[category] = str(status)
        any_open = any(s == "open" for s in breaker_states.values())

        # Parse rate limit usage (comes as "45%" string)
        rate_limit_str = resilience_status.get("rate_limit_usage", "0%")
        try:
            if isinstance(rate_limit_str, str):
                rate_limit_pct = float(rate_limit_str.rstrip("%"))
            else:
                rate_limit_pct = float(rate_limit_str)
        except (ValueError, TypeError):
            rate_limit_pct = 0.0

        self.resilience_data = ResilienceState(
            latency_p50_ms=float(metrics.get("p50_latency_ms", 0)),
            latency_p95_ms=float(metrics.get("p95_latency_ms", 0)),
            avg_latency_ms=float(metrics.get("avg_latency_ms", 0)),
            error_rate=float(metrics.get("error_rate", 0)),
            total_requests=int(metrics.get("total_requests", 0)),
            total_errors=int(metrics.get("total_errors", 0)),
            rate_limit_hits=int(metrics.get("rate_limit_hits", 0)),
            rate_limit_usage_pct=rate_limit_pct,
            cache_hit_rate=float(cache.get("hit_rate", 0)),
            cache_size=int(cache.get("size", 0)),
            cache_enabled=bool(cache.get("enabled", False)),
            circuit_breakers=breaker_states,
            any_circuit_open=any_open,
            last_update=time.time(),
        )
        self._changed_fields.add("resilience")

    def update_validation_metrics(self, metrics: dict[str, Any]) -> None:
        """Update validation failure metrics from the failure tracker.

        Updates the system_data with validation failure counts, allowing
        operators to monitor when validation checks are failing repeatedly.

        Args:
            metrics: Dict with keys:
                - failures: Dict mapping check_type to consecutive failure count
                - escalation_threshold: Threshold for escalation
                - any_escalated: True if any check has reached threshold
        """
        failures = metrics.get("failures", {})
        any_escalated = metrics.get("any_escalated", False)

        # Update system_data with validation metrics
        self.system_data = SystemStatus(
            api_latency=self.system_data.api_latency,
            connection_status=self.system_data.connection_status,
            rate_limit_usage=self.system_data.rate_limit_usage,
            memory_usage=self.system_data.memory_usage,
            cpu_usage=self.system_data.cpu_usage,
            validation_failures=failures,
            validation_escalated=any_escalated,
        )
        self._changed_fields.add("system")

    def update_execution_data(self) -> None:
        """Update execution metrics from the telemetry collector.

        Pulls current metrics from the global ExecutionTelemetryCollector
        if available. Safe to call even if collector hasn't been initialized.
        """
        try:
            from gpt_trader.tui.services.execution_telemetry import get_execution_telemetry

            collector = get_execution_telemetry()
            self.execution_data = collector.get_metrics()
            self._changed_fields.add("execution")
        except Exception:
            # Collector not available or error - leave existing data
            pass

    def get_changed_fields(self) -> set[str]:
        """Get the set of fields that changed in the last update.

        Returns:
            Set of field names that were updated. Useful for optimized
            widget notifications where only affected widgets need updating.
        """
        return self._changed_fields.copy()

    def get_decision_by_id(self, decision_id: str) -> DecisionData | None:
        """Look up a strategy decision by its ID.

        Args:
            decision_id: The unique decision identifier.

        Returns:
            DecisionData if found, None otherwise.
        """
        if not decision_id:
            return None
        for decision in self.strategy_data.last_decisions.values():
            if decision.decision_id == decision_id:
                return decision
        return None

    def update_strategy_performance(self, performance_data: dict[str, Any]) -> None:
        """Update strategy performance metrics.

        Args:
            performance_data: Dict with performance metrics from PerformanceSnapshot.
                Expected keys: win_rate, profit_factor, total_return, daily_return,
                max_drawdown, total_trades, winning_trades, losing_trades,
                sharpe_ratio, sortino_ratio, volatility.
        """
        self.strategy_performance = StrategyPerformance(
            win_rate=float(performance_data.get("win_rate", 0.0)),
            profit_factor=float(performance_data.get("profit_factor", 0.0)),
            total_return_pct=float(performance_data.get("total_return", 0.0)) * 100,
            daily_return_pct=float(performance_data.get("daily_return", 0.0)) * 100,
            max_drawdown_pct=float(performance_data.get("max_drawdown", 0.0)) * 100,
            total_trades=int(performance_data.get("total_trades", 0)),
            winning_trades=int(performance_data.get("winning_trades", 0)),
            losing_trades=int(performance_data.get("losing_trades", 0)),
            sharpe_ratio=float(performance_data.get("sharpe_ratio", 0.0)),
            sortino_ratio=float(performance_data.get("sortino_ratio", 0.0)),
            volatility_pct=float(performance_data.get("volatility", 0.0)) * 100,
        )
        self._changed_fields.add("strategy_performance")

    def update_backtest_performance(self, performance_data: dict[str, Any] | None) -> None:
        """Update backtest performance metrics.

        Args:
            performance_data: Dict with backtest metrics, or None to clear.
                Expected keys: win_rate, profit_factor, total_return, max_drawdown,
                total_trades, winning_trades, losing_trades.
        """
        if performance_data is None:
            self.backtest_performance = None
        else:
            self.backtest_performance = StrategyPerformance(
                win_rate=float(performance_data.get("win_rate", 0.0)),
                profit_factor=float(performance_data.get("profit_factor", 0.0)),
                total_return_pct=float(performance_data.get("total_return", 0.0)) * 100,
                max_drawdown_pct=float(performance_data.get("max_drawdown", 0.0)) * 100,
                total_trades=int(performance_data.get("total_trades", 0)),
                winning_trades=int(performance_data.get("winning_trades", 0)),
                losing_trades=int(performance_data.get("losing_trades", 0)),
            )
        self._changed_fields.add("backtest_performance")

    def update_regime_data(self, regime_state: dict[str, Any]) -> None:
        """Update market regime data.

        Args:
            regime_state: Dict with regime detection results from RegimeState.
                Expected keys: regime, confidence, trend_score, volatility_percentile,
                momentum_score, regime_age_ticks, transition_probability.
        """
        self.regime_data = RegimeData(
            regime=str(regime_state.get("regime", "UNKNOWN")),
            confidence=float(regime_state.get("confidence", 0.0)),
            trend_score=float(regime_state.get("trend_score", 0.0)),
            volatility_pct=float(regime_state.get("volatility_percentile", 0.0)),
            momentum_score=float(regime_state.get("momentum_score", 0.0)),
            regime_age_ticks=int(regime_state.get("regime_age_ticks", 0)),
            transition_probability=float(regime_state.get("transition_probability", 0.0)),
        )
        self._changed_fields.add("regime")
