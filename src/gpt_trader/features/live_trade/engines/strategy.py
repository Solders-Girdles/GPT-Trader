"""
Simplified Strategy Engine.
Replaces the 608-line enterprise coordinator with a simple loop.

State Recovery:
On startup, reads `price_tick` events from EventStore to restore price history.
During operation, persists price ticks to EventStore for crash recovery.
"""

import asyncio
import time
from decimal import Decimal
from typing import Any

from gpt_trader.core import OrderSide, OrderType, Position, Product
from gpt_trader.features.live_trade.engines.base import (
    BaseEngine,
    CoordinatorContext,
    HealthStatus,
)
from gpt_trader.features.live_trade.engines.price_tick_store import (
    EVENT_PRICE_TICK,
    PriceTickStore,
)
from gpt_trader.features.live_trade.engines.system_maintenance import (
    SystemMaintenanceService,
)
from gpt_trader.features.live_trade.factory import create_strategy
from gpt_trader.features.live_trade.risk.manager import ValidationError
from gpt_trader.features.live_trade.strategies.perps_baseline import (
    Action,
    Decision,
)
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.heartbeat import HeartbeatService
from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.orchestration.execution.guard_manager import GuardManager
from gpt_trader.orchestration.execution.order_submission import OrderSubmitter
from gpt_trader.orchestration.execution.state_collection import StateCollector
from gpt_trader.orchestration.execution.validation import (
    OrderValidator,
    get_failure_tracker,
)
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="trading_engine")

# Re-export for backward compatibility
__all__ = ["TradingEngine", "EVENT_PRICE_TICK"]


class TradingEngine(BaseEngine):
    """
    Simple trading loop that fetches data and executes strategy.

    Supports state recovery via EventStore persistence.
    """

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self.running = False
        # Create strategy via factory (supports baseline and mean_reversion)
        self.strategy = create_strategy(self.context.config)
        self._current_positions: dict[str, Position] = {}
        self._rehydrated = False
        self._cycle_count = 0

        # Initialize price tick store for state recovery
        self._price_tick_store = PriceTickStore(
            event_store=context.event_store,
            symbols=list(context.config.symbols),
            bot_id=context.bot_id,
        )

        # Initialize heartbeat service
        self._heartbeat = HeartbeatService(
            event_store=context.event_store,
            ping_url=getattr(context.config, "heartbeat_url", None),
            interval_seconds=getattr(context.config, "heartbeat_interval", 60),
            bot_id=context.bot_id,
            enabled=getattr(context.config, "heartbeat_enabled", True),
        )

        # Initialize status reporter
        self._status_reporter = StatusReporter(
            status_file=getattr(context.config, "status_file", "status.json"),
            update_interval=getattr(context.config, "status_interval", 10),
            bot_id=context.bot_id,
            enabled=getattr(context.config, "status_enabled", True),
        )
        self._status_reporter.set_heartbeat_service(self._heartbeat)

        # Initialize system maintenance service (health reporting + pruning)
        self._system_maintenance = SystemMaintenanceService(
            status_reporter=self._status_reporter,
            event_store=context.event_store,
        )

        # System health tracking
        self._last_latency = 0.0
        self._connection_status = "UNKNOWN"

        # Initialize pre-trade guard stack (Option A: embedded guards)
        self._init_guard_stack()

    def _init_guard_stack(self) -> None:
        """Initialize StateCollector, OrderValidator, OrderSubmitter for pre-trade guards."""
        # Event store fallback
        event_store = self.context.event_store or EventStore()

        # Broker and risk manager must exist
        broker = self.context.broker
        risk_manager = self.context.risk_manager

        # Track open orders
        self._open_orders: list[str] = []

        # StateCollector: needs broker, config
        self._state_collector = StateCollector(
            broker=broker,  # type: ignore[arg-type]
            config=self.context.config,
            integration_mode=False,
        )

        # OrderSubmitter: broker + event store + bot_id + open_orders
        self._order_submitter = OrderSubmitter(
            broker=broker,  # type: ignore[arg-type]
            event_store=event_store,
            bot_id=self.context.bot_id or self.context.config.profile or "live",
            open_orders=self._open_orders,
            integration_mode=False,
        )

        # Failure tracker (global)
        failure_tracker = get_failure_tracker()

        # OrderValidator: broker + risk_manager + preview config + callbacks + tracker
        self._order_validator: OrderValidator | None = None
        if risk_manager is not None:
            self._order_validator = OrderValidator(
                broker=broker,  # type: ignore[arg-type]
                risk_manager=risk_manager,
                enable_order_preview=self.context.config.enable_order_preview,
                record_preview_callback=self._order_submitter.record_preview,
                record_rejection_callback=self._order_submitter.record_rejection,
                failure_tracker=failure_tracker,
            )

        # GuardManager: runtime guards (daily loss, liquidation buffer, volatility)
        self._guard_manager: GuardManager | None = None
        if broker is not None and risk_manager is not None:
            self._guard_manager = GuardManager(
                broker=broker,  # type: ignore[arg-type]
                risk_manager=risk_manager,
                equity_calculator=self._state_collector.calculate_equity_from_balances,
                open_orders=self._open_orders,
                invalidate_cache_callback=lambda: None,
            )

    @property
    def status_reporter(self) -> StatusReporter:
        return self._status_reporter

    @property
    def price_history(self) -> dict[str, list[Decimal]]:
        """Access price history via PriceTickStore."""
        return self._price_tick_store.price_history

    async def _notify(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Send notification if service is available."""
        if self.context.notification_service is None:
            return
        try:
            await self.context.notification_service.notify(
                title=title,
                message=message,
                severity=severity,
                source="TradingEngine",
                context=context,
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    @property
    def name(self) -> str:
        return "strategy"

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        """Start the main trading loop and heartbeat service.

        Before starting, attempts to rehydrate state from EventStore.
        """
        # Rehydrate state from EventStore before starting
        if not self._rehydrated:
            self._rehydrate_from_events()
            self._rehydrated = True

        self.running = True

        tasks: list[asyncio.Task[Any]] = []

        # Start main trading loop
        trading_task = asyncio.create_task(self._run_loop())
        self._register_background_task(trading_task)
        tasks.append(trading_task)

        # Start heartbeat service
        heartbeat_task = await self._heartbeat.start()
        if heartbeat_task:
            self._register_background_task(heartbeat_task)
            tasks.append(heartbeat_task)

        # Start status reporter
        status_task = await self._status_reporter.start()
        if status_task:
            self._register_background_task(status_task)
            tasks.append(status_task)

        # Start database pruning task via system maintenance service
        prune_task = await self._system_maintenance.start_prune_loop()
        self._register_background_task(prune_task)
        tasks.append(prune_task)

        # Start runtime guard sweep (daily loss, liquidation buffer, volatility)
        if self._guard_manager is not None:
            guard_task = asyncio.create_task(
                self._runtime_guard_sweep(), name="runtime_guard_sweep"
            )
            self._register_background_task(guard_task)
            tasks.append(guard_task)

        return tasks

    def _rehydrate_from_events(self) -> int:
        """Restore price history from persisted events.

        Delegates to PriceTickStore for the actual rehydration logic.

        Returns:
            Number of price ticks restored
        """
        # Prepare strategy rehydration callback if strategy supports it
        strategy_callback = None
        if hasattr(self.strategy, "rehydrate"):
            strategy_callback = self.strategy.rehydrate

        return self._price_tick_store.rehydrate(strategy_rehydrate_callback=strategy_callback)

    async def _runtime_guard_sweep(self) -> None:
        """Periodically run runtime guards to check risk limits.

        Runs on a cadence to proactively detect risk breaches (daily loss,
        liquidation buffer, volatility) rather than only at order time.
        """
        interval = getattr(self.context.config, "runtime_guard_interval", 60)
        while self.running:
            try:
                if self._guard_manager is not None:
                    self._guard_manager.safe_run_runtime_guards()
            except Exception:
                logger.exception("Runtime guard sweep failed", operation="runtime_guards")
            await asyncio.sleep(interval)

    async def _run_loop(self) -> None:
        logger.info("Starting strategy loop...")
        while self.running:
            try:
                await self._cycle()
                # Record successful cycle
                self._status_reporter.record_cycle()
            except Exception as e:
                logger.error(f"Error in strategy cycle: {e}", exc_info=True)
                # Record error in status reporter
                self._status_reporter.record_error(str(e))
                await self._notify(
                    title="Strategy Cycle Error",
                    message=f"Error during trading cycle: {e}",
                    severity=AlertSeverity.ERROR,
                    context={"error": str(e)},
                )

            await asyncio.sleep(self.context.config.interval)

    def _report_system_status(self) -> None:
        """Collect and report system health metrics.

        Delegates to SystemMaintenanceService for the actual reporting.
        """
        self._system_maintenance.report_system_status(
            latency_seconds=self._last_latency,
            connection_status=self._connection_status,
        )

    async def _cycle(self) -> None:
        """One trading cycle."""
        assert self.context.broker is not None, "Broker not initialized"
        self._cycle_count += 1

        logger.info(f"=== CYCLE {self._cycle_count} START ===")

        # Report system status at start of cycle
        self._report_system_status()

        # 1. Fetch positions first (needed for equity calculation)
        logger.info("Step 1: Fetching positions...")
        positions = await self._fetch_positions()
        self._current_positions = positions
        logger.info(f"Fetched {len(positions)} positions")

        # Update status reporter with positions (complete data for TUI)
        self._status_reporter.update_positions(self._positions_to_status_format(positions))

        # 1b. Audit open orders (Reconciliation)
        await self._audit_orders()

        # 2. Calculate total equity including unrealized PnL
        logger.info("Step 2: Calculating total equity...")
        equity = await self._fetch_total_equity(positions)
        if equity is None:
            logger.error(
                "Failed to fetch equity - cannot continue cycle. "
                "Check logs above for balance fetch errors."
            )
            # Update status reporter with error state
            self._status_reporter.record_error("Failed to fetch equity")
            return

        logger.info(f"Successfully calculated equity: ${equity}")
        # Update status reporter with equity
        self._status_reporter.update_equity(equity)
        logger.info("Equity updated in status reporter")

        # Track daily PnL for risk management
        if self.context.risk_manager:
            triggered = self.context.risk_manager.track_daily_pnl(equity, {})
            if triggered:
                logger.warning("Daily loss limit triggered! Reduce-only mode activated.")

            # Update status reporter with risk metrics
            rm = self.context.risk_manager
            # Calculate current daily loss pct if possible
            # Assuming rm tracks start_of_day_equity
            daily_loss_pct = 0.0
            start_equity = getattr(rm, "_start_of_day_equity", 0)
            if start_equity and start_equity > 0:
                daily_pnl = equity - start_equity
                daily_loss_pct = float(-daily_pnl / start_equity)

            self._status_reporter.update_risk(
                max_leverage=float(getattr(rm.config, "max_leverage", 0.0) if rm.config else 0.0),
                daily_loss_limit=float(
                    getattr(rm.config, "daily_loss_limit_pct", 0.0) if rm.config else 0.0
                ),
                current_daily_loss=daily_loss_pct,
                reduce_only=getattr(rm, "_reduce_only_mode", False),
                reduce_reason=getattr(rm, "_reduce_only_reason", ""),
            )

        # 2. Process Symbols
        for symbol in self.context.config.symbols:
            # Offload blocking network call
            try:
                start_time = time.time()
                ticker = await asyncio.to_thread(self.context.broker.get_ticker, symbol)

                # Update latency and connection status
                self._last_latency = time.time() - start_time
                self._connection_status = "CONNECTED"
            except Exception as e:
                logger.error(f"Failed to fetch ticker for {symbol}: {e}")
                self._connection_status = "DISCONNECTED"
                continue

            price = Decimal(str(ticker.get("price", 0)))
            logger.info(f"{symbol} price: {price}")

            # Seed mark staleness timestamp from REST fetch (prevents deadlock on startup)
            if self.context.risk_manager is not None:
                self.context.risk_manager.last_mark_update[symbol] = time.time()

            # Update status reporter with price
            self._status_reporter.update_price(symbol, price)

            # Record price tick (updates in-memory history + persists for crash recovery)
            self._record_price_tick(symbol, price)

            position_state = self._build_position_state(symbol, positions)

            # Fetch candles for advanced strategies (e.g. ADX)
            candles = []
            try:
                # Fetch last 50 candles (enough for ADX 14 + smoothing)
                # Assuming granularity defaults to ONE_MINUTE or similar
                candles = await asyncio.to_thread(
                    self.context.broker.get_candles, symbol, granularity="ONE_MINUTE"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch candles for {symbol}: {e}")

            decision = self.strategy.decide(
                symbol=symbol,
                current_mark=price,
                position_state=position_state,
                recent_marks=self.price_history[symbol],
                equity=equity,
                product=None,  # Future: fetch from broker
                candles=candles,
            )

            logger.info(f"Strategy Decision for {symbol}: {decision.action} ({decision.reason})")

            # Report strategy decision
            active_strats = getattr(
                self.strategy, "active_strategies", [self.strategy.__class__.__name__]
            )
            decision_record = {
                "symbol": symbol,
                "action": decision.action.value,
                "reason": decision.reason,
                "confidence": str(decision.confidence),
                "timestamp": time.time(),
            }
            self._status_reporter.update_strategy(active_strats, [decision_record])

            if decision.action in (Action.BUY, Action.SELL):
                logger.info(f"EXECUTING {decision.action} for {symbol}")
                try:
                    await self._validate_and_place_order(
                        symbol=symbol,
                        decision=decision,
                        price=price,
                        equity=equity,
                    )
                except ValidationError as e:
                    logger.warning(f"Risk validation failed for {symbol}: {e}")
                    await self._notify(
                        title="Risk Validation Failed",
                        message=f"Order blocked by risk manager: {e}",
                        severity=AlertSeverity.WARNING,
                        context={
                            "symbol": symbol,
                            "action": decision.action.value,
                            "reason": str(e),
                        },
                    )
                except Exception as e:
                    logger.error(f"Order placement failed: {e}")
                    await self._notify(
                        title="Order Placement Failed",
                        message=f"Failed to execute {decision.action} for {symbol}: {e}",
                        severity=AlertSeverity.ERROR,
                        context={
                            "symbol": symbol,
                            "action": decision.action.value,
                            "error": str(e),
                        },
                    )

            elif decision.action == Action.CLOSE and position_state:
                # Handle CLOSE action separately if needed, or integrate into place_order
                # For now, logging it as per original logic, or we can implement close logic here
                logger.info(f"CLOSE signal for {symbol} - not fully implemented yet")

    async def _fetch_total_equity(self, positions: dict[str, Position]) -> Decimal | None:
        """Fetch total equity = collateral + unrealized PnL."""
        assert self.context.broker is not None
        try:
            balances = await asyncio.to_thread(self.context.broker.list_balances)

            # Log summary of all assets returned (before filtering)
            if balances:
                all_assets = [b.asset for b in balances]
                non_zero_assets = [(b.asset, b.available, b.total) for b in balances if b.total > 0]

                logger.info(f"Fetched {len(balances)} balances from broker")
                logger.info(
                    f"All assets in response: {', '.join(all_assets) if all_assets else 'NONE'}"
                )

                if non_zero_assets:
                    logger.info(f"Assets with non-zero balances: {len(non_zero_assets)}")
                    for asset, avail, total in non_zero_assets:
                        logger.info(f"  {asset}: available={avail}, total={total}")
                else:
                    logger.warning(
                        "All balances are zero - this may indicate an API permission or portfolio scoping issue"
                    )
            else:
                logger.warning("Received empty balance list from broker - check API configuration")

            cash_collateral = Decimal("0")
            converted_collateral = Decimal("0")
            usd_usdc_found = []
            other_assets_found = []
            priced_assets: list[str] = []
            unpriced_assets: list[str] = []

            quote = str(
                getattr(self.context.config, "coinbase_default_quote", None) or "USD"
            ).upper()
            # Treat stable USD quotes as 1:1 for valuation display.
            stable_quotes = {"USD", "USDC"}
            use_total_balance = bool(getattr(self.context.config, "read_only", False))

            valuation_quotes: list[str] = [quote]
            if quote in stable_quotes:
                for stable in ("USD", "USDC"):
                    if stable not in valuation_quotes:
                        valuation_quotes.append(stable)
            else:
                # Fall back to USD/USDC if the configured quote isn't available for a token.
                for stable in ("USD", "USDC"):
                    if stable not in valuation_quotes:
                        valuation_quotes.append(stable)

            for balance in balances:
                logger.debug(
                    f"Balance: {balance.asset} = {balance.available} available, {balance.total} total"
                )
                asset = str(balance.asset or "").upper()
                amount = balance.total if use_total_balance else balance.available

                if asset in ("USD", "USDC"):
                    cash_collateral += amount
                    if amount > 0:
                        usd_usdc_found.append(f"{asset}=${amount}")
                    continue

                if balance.total > 0:
                    other_assets_found.append(f"{asset}={balance.total}")

                if amount <= 0:
                    continue

                if asset == quote:
                    cash_collateral += amount
                    priced_assets.append(f"{asset}=${amount}")
                    continue

                if asset in stable_quotes and quote in stable_quotes:
                    cash_collateral += amount
                    priced_assets.append(f"{asset}≈${amount}")
                    continue

                last_price: Decimal | None = None
                used_pair: str | None = None

                for q in valuation_quotes:
                    product_id = f"{asset}-{q}"
                    history = self.price_history.get(product_id)
                    if history:
                        last_price = history[-1]
                    else:
                        last_price = None

                    try:
                        if last_price is None:
                            ticker = await asyncio.to_thread(
                                self.context.broker.get_ticker, product_id
                            )
                            last_price = Decimal(str(ticker.get("price", 0)))
                        if last_price and last_price > 0:
                            used_pair = product_id
                            break
                    except Exception as exc:
                        logger.debug(
                            "Unable to value %s via %s: %s",
                            asset,
                            product_id,
                            exc,
                        )
                        continue

                if last_price and last_price > 0 and used_pair:
                    usd_value = amount * last_price
                    converted_collateral += usd_value
                    priced_assets.append(
                        f"{asset}={amount} @ {used_pair}≈{usd_value.quantize(Decimal('0.01'))}"
                    )
                else:
                    unpriced_assets.append(asset)

            collateral = cash_collateral + converted_collateral

            # Log USD/USDC result with context
            cash_label = "Total cash holdings" if use_total_balance else "Available cash collateral"
            logger.info(
                "%s (%s): $%s", cash_label, quote, cash_collateral.quantize(Decimal("0.01"))
            )
            if usd_usdc_found:
                logger.info(f"USD/USDC assets counted: {', '.join(usd_usdc_found)}")
            else:
                collateral_scope = "total holdings" if use_total_balance else "available collateral"
                logger.warning(
                    "No USD/USDC balances found in %s; valuing non-USD assets using %s tickers",
                    collateral_scope,
                    quote,
                )
                if other_assets_found:
                    logger.info("Non-USD assets detected: %s", ", ".join(other_assets_found))
            if priced_assets:
                logger.info("Included non-USD assets in equity: %s", "; ".join(priced_assets))
            if unpriced_assets:
                logger.warning(
                    "Could not value these assets in %s: %s",
                    quote,
                    ", ".join(sorted(set(unpriced_assets))),
                )

            # Add unrealized PnL from open positions
            unrealized_pnl = sum(
                (p.unrealized_pnl for p in positions.values()),
                Decimal("0"),
            )
            logger.info(f"Unrealized PnL: ${unrealized_pnl}")

            total_equity = collateral + unrealized_pnl
            logger.info(
                f"Total equity calculated: ${total_equity} (collateral=${collateral} + unrealized_pnl=${unrealized_pnl})"
            )

            # Add diagnostic warning if equity is zero
            if total_equity == 0:
                logger.warning(
                    "Total equity is $0.00. This typically means: "
                    "1) No USD/USDC in account (only crypto assets), "
                    "2) Wrong portfolio selected (check portfolio_uuid), "
                    "3) API permission issue, or "
                    "4) No funds in account"
                )

            return total_equity
        except Exception as e:
            logger.error(
                f"Failed to fetch balances: {e}",
                error_type=type(e).__name__,
                operation="fetch_total_equity",
                exc_info=True,
            )
            logger.error(
                "Unable to calculate equity. Check: "
                "1) Network connectivity, "
                "2) API credentials validity, "
                "3) Broker service health"
            )
            return None

    async def _fetch_positions(self) -> dict[str, Position]:
        """Fetch current positions as a lookup dict."""
        assert self.context.broker is not None
        try:
            positions_list = await asyncio.to_thread(self.context.broker.list_positions)
            return {p.symbol: p for p in positions_list}
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return {}

    def _build_position_state(
        self, symbol: str, positions: dict[str, Position]
    ) -> dict[str, Any] | None:
        """Build position state dict for strategy.decide()."""
        if symbol not in positions:
            return None
        pos = positions[symbol]
        return {
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "side": pos.side,
            # Add other fields if needed by strategy
        }

    def _record_price_tick(self, symbol: str, price: Decimal) -> None:
        """Persist price tick to EventStore for crash recovery.

        Delegates to PriceTickStore which handles both in-memory
        history update and EventStore persistence.
        """
        self._price_tick_store.record_price_tick(symbol, price)

    def _positions_to_risk_format(
        self, positions: dict[str, Position]
    ) -> dict[str, dict[str, Any]]:
        """Convert Position objects to dict format expected by risk manager."""
        return {
            symbol: {
                "quantity": pos.quantity,
                "mark": pos.mark_price,
            }
            for symbol, pos in positions.items()
        }

    def _positions_to_status_format(
        self, positions: dict[str, Position]
    ) -> dict[str, dict[str, Any]]:
        """Convert Position objects to dict format for StatusReporter with complete TUI data."""
        return {
            symbol: {
                "quantity": str(pos.quantity),
                "mark_price": str(pos.mark_price),
                "entry_price": str(pos.entry_price),
                "unrealized_pnl": str(pos.unrealized_pnl),
                "realized_pnl": str(pos.realized_pnl),
                "side": pos.side,
            }
            for symbol, pos in positions.items()
        }

    def _calculate_order_quantity(
        self,
        symbol: str,
        price: Decimal,
        equity: Decimal,
        product: Product | None,
    ) -> Decimal:
        """Calculate order size based on equity and position_fraction."""
        # 1. Determine fraction
        fraction = Decimal("0.1")  # Default
        if hasattr(self.strategy, "config") and self.strategy.config.position_fraction:
            fraction = Decimal(str(self.strategy.config.position_fraction))
        elif (
            hasattr(self.context.config, "perps_position_fraction")
            and self.context.config.perps_position_fraction is not None
        ):
            fraction = Decimal(str(self.context.config.perps_position_fraction))

        # 2. Calculate raw quantity
        if price == 0:
            return Decimal("0")

        target_notional = equity * fraction
        quantity = target_notional / price

        # 3. Apply constraints
        if product and product.min_size:
            if quantity < product.min_size:
                logger.warning(f"Quantity {quantity} below min size {product.min_size}")
                return Decimal("0")

            # Round to step size if needed (simplified)
            # quantity = (quantity // product.step_size) * product.step_size

        return quantity

    async def _validate_and_place_order(
        self,
        symbol: str,
        decision: Decision,
        price: Decimal,
        equity: Decimal,
    ) -> None:
        """Validate order with risk manager before execution.

        Raises:
            ValidationError: If risk validation fails.
        """
        side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL

        # Dynamic position sizing
        quantity = self._calculate_order_quantity(symbol, price, equity, product=None)

        if quantity <= 0:
            logger.warning(f"Calculated quantity is {quantity}, skipping order")
            return

        # Security Validation (Hard Limits)
        from gpt_trader.security.security_validator import get_validator

        security_order = {
            "symbol": symbol,
            "side": side.value,
            "quantity": float(quantity),
            "price": float(price),
            "type": "MARKET",
        }

        # Construct dynamic limits from config
        limits = {}
        if hasattr(self.context.config, "risk"):
            risk = self.context.config.risk
            if risk:
                limits["max_position_size"] = float(getattr(risk, "max_position_pct", 0.05))
                limits["max_leverage"] = float(getattr(risk, "max_leverage", 2.0))
                limits["max_daily_loss"] = float(getattr(risk, "daily_loss_limit_pct", 0.02))
                # Map other fields if available or use defaults

        security_result = get_validator().validate_order_request(
            security_order, account_value=float(equity), limits=limits
        )

        if not security_result.is_valid:
            error_msg = f"Security validation failed: {', '.join(security_result.errors)}"
            logger.error(error_msg)
            await self._notify(
                title="Security Validation Failed",
                message=error_msg,
                severity=AlertSeverity.ERROR,
                context=security_order,
            )
            return

        # Run pre-trade validation if risk manager is available
        if self.context.risk_manager is not None:
            self.context.risk_manager.pre_trade_validate(
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=price,
                product=None,
                equity=equity,
                current_positions=self._positions_to_risk_format(self._current_positions),
            )
            logger.info(f"Risk validation passed for {symbol} {side.value}")

            # Check reduce-only mode enforcement
            # Determine if this order would reduce an existing position
            current_pos = self._current_positions.get(symbol)
            is_reducing = False

            if current_pos is not None:
                # Handle both Position objects and dicts
                if hasattr(current_pos, "side") and hasattr(current_pos, "quantity"):
                    # Position object: use side attribute
                    pos_side = current_pos.side.lower() if current_pos.side else ""
                    pos_qty = current_pos.quantity
                    # Reducing = LONG + SELL or SHORT + BUY
                    is_reducing = (
                        pos_side == "long" and side == OrderSide.SELL and pos_qty > 0
                    ) or (pos_side == "short" and side == OrderSide.BUY and pos_qty > 0)
                elif isinstance(current_pos, dict):
                    # Dict: check for side key, fall back to quantity sign
                    pos_side = str(current_pos.get("side", "")).lower()
                    pos_qty = Decimal(str(current_pos.get("quantity", 0)))
                    if pos_side in ("long", "short"):
                        is_reducing = (
                            pos_side == "long" and side == OrderSide.SELL and pos_qty > 0
                        ) or (pos_side == "short" and side == OrderSide.BUY and pos_qty > 0)
                    else:
                        # Legacy: quantity sign indicates direction
                        is_reducing = (pos_qty > 0 and side == OrderSide.SELL) or (
                            pos_qty < 0 and side == OrderSide.BUY
                        )

            # In reduce-only mode, clamp quantity to prevent position flips
            reduce_only_active = (
                self.context.risk_manager._reduce_only_mode
                or self.context.risk_manager._daily_pnl_triggered
            )
            if reduce_only_active and is_reducing and current_pos is not None:
                # Get current position quantity
                if hasattr(current_pos, "quantity"):
                    current_qty = abs(current_pos.quantity)
                elif isinstance(current_pos, dict):
                    current_qty = abs(Decimal(str(current_pos.get("quantity", 0))))
                else:
                    current_qty = Decimal("0")

                # Clamp order quantity to current position size
                if quantity > current_qty:
                    logger.warning(
                        f"Reduce-only: clamping order from {quantity} to {current_qty} "
                        f"to prevent position flip for {symbol}"
                    )
                    quantity = current_qty

                # If clamped to zero, skip the order
                if quantity <= 0:
                    logger.info(f"Reduce-only: no position to reduce for {symbol}, skipping order")
                    return

            # Create order dict for check_order
            order_for_check = {
                "symbol": symbol,
                "side": side.value,
                "quantity": float(quantity),
                "reduce_only": is_reducing,
            }

            if not self.context.risk_manager.check_order(order_for_check):
                error_msg = (
                    f"Order blocked by risk manager: "
                    f"reduce_only_mode={self.context.risk_manager._reduce_only_mode}, "
                    f"daily_pnl_triggered={self.context.risk_manager._daily_pnl_triggered}"
                )
                logger.warning(error_msg)
                await self._notify(
                    title="Order Blocked - Reduce Only Mode",
                    message=f"Cannot open new {side.value} position for {symbol} while in reduce-only mode",
                    severity=AlertSeverity.WARNING,
                    context=order_for_check,
                )
                return
        else:
            logger.warning("No risk manager configured - skipping validation")

        # Guard: Check mark price staleness before placing order
        if self.context.risk_manager is not None:
            if self.context.risk_manager.check_mark_staleness(symbol):
                logger.warning(f"Order blocked: mark price stale for {symbol}")
                await self._notify(
                    title="Order Blocked - Stale Mark Price",
                    message=f"Cannot place order for {symbol}: mark price data is stale",
                    severity=AlertSeverity.WARNING,
                    context={"symbol": symbol, "side": side.value},
                )
                return

        # Pre-trade guards via OrderValidator (exchange rules, slippage, preview)
        if self._order_validator is not None:
            try:
                # Get product for exchange rules validation
                product = self._state_collector.require_product(symbol, product=None)

                # Resolve effective price via StateCollector
                effective_price = self._state_collector.resolve_effective_price(
                    symbol, side.value.lower(), price, product
                )

                # Exchange rules + quantization
                quantity, _ = self._order_validator.validate_exchange_rules(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    order_quantity=quantity,
                    price=None,
                    effective_price=effective_price,
                    product=product,
                )

                # Slippage guard
                self._order_validator.enforce_slippage_guard(
                    symbol, side, quantity, effective_price
                )

                # Mark staleness via OrderValidator
                self._order_validator.ensure_mark_is_fresh(symbol)

                # Pre-trade validation via OrderValidator (leverage/exposure)
                current_positions_dict = self._state_collector.build_positions_dict(
                    list(self._current_positions.values())
                )
                self._order_validator.run_pre_trade_validation(
                    symbol=symbol,
                    side=side,
                    order_quantity=quantity,
                    effective_price=effective_price,
                    product=product,
                    equity=equity,
                    current_positions=current_positions_dict,
                )

                # Order preview (if enabled)
                self._order_validator.maybe_preview_order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    order_quantity=quantity,
                    effective_price=effective_price,
                    stop_price=None,
                    tif=self.context.config.time_in_force,
                    reduce_only=is_reducing,
                    leverage=None,
                )

                # Finalize reduce-only flag (risk manager may have triggered it)
                is_reducing = self._order_validator.finalize_reduce_only_flag(is_reducing, symbol)

            except ValidationError as exc:
                logger.warning(f"Pre-trade guard rejected order: {exc}")
                self._order_submitter.record_rejection(
                    symbol, side.value, quantity, effective_price, str(exc)
                )
                await self._notify(
                    title="Order Blocked - Guard Rejection",
                    message=f"Cannot place order for {symbol}: {exc}",
                    severity=AlertSeverity.WARNING,
                    context={"symbol": symbol, "side": side.value, "reason": str(exc)},
                )
                return
            except Exception as exc:
                # Non-validation errors: log + record metrics but still block order (fail-closed)
                logger.error(f"Guard check error: {exc}")
                self._order_submitter.record_rejection(
                    symbol, side.value, quantity, price, f"guard_error: {exc}"
                )
                await self._notify(
                    title="Order Blocked - Guard Error",
                    message=f"Cannot place order for {symbol}: guard check failed",
                    severity=AlertSeverity.ERROR,
                    context={"symbol": symbol, "side": side.value, "error": str(exc)},
                )
                return
        else:
            effective_price = price

        # Place order only after validation passes
        assert self.context.broker is not None, "Broker not initialized"
        order_id = await asyncio.to_thread(
            self.context.broker.place_order,
            symbol,
            side,
            OrderType.MARKET,
            quantity,
        )

        # Track order if ID returned
        if order_id and isinstance(order_id, str):
            self._open_orders.append(order_id)

        # Notify on successful order placement
        await self._notify(
            title="Order Executed",
            message=f"{side.value} {quantity} {symbol} at ~{price}",
            severity=AlertSeverity.INFO,
            context={
                "symbol": symbol,
                "side": side.value,
                "quantity": str(quantity),
                "price": str(price),
            },
        )

        # Record trade in status reporter
        self._status_reporter.add_trade(
            {
                "symbol": symbol,
                "side": side.value,
                "quantity": str(quantity),
                "price": str(price),
                "order_id": "N/A",  # We don't get ID back from place_order in this adapter version easily without refactor
            }
        )

    async def shutdown(self) -> None:
        self.running = False
        await self._system_maintenance.stop()
        await self._status_reporter.stop()
        await self._heartbeat.stop()
        await super().shutdown()

    def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self.running, component=self.name)

    async def _audit_orders(self) -> None:
        """Audit open orders for reconciliation."""
        assert self.context.broker is not None
        try:
            # Fetch open orders
            # Note: Coinbase API uses 'order_status' for filtering
            # Use getattr to safely call list_orders (not part of base BrokerProtocol)
            list_orders = getattr(self.context.broker, "list_orders", None)
            if list_orders:
                response = await asyncio.to_thread(list_orders, order_status="OPEN")
                orders = response.get("orders", [])
            else:
                orders = []

            if orders:
                logger.info(f"AUDIT: Found {len(orders)} OPEN orders")
                for order in orders:
                    logger.info(
                        f"  Order {order.get('order_id')}: {order.get('side')} "
                        f"{order.get('product_id')} {order.get('order_configuration')}"
                    )

            # Update status reporter
            self._status_reporter.update_orders(orders)

            # Update Account Metrics (every 60 cycles ~ 1 minute)
            if self._cycle_count % 60 == 0:
                try:
                    balances = self.context.broker.list_balances()
                    # Check if broker supports transaction summary (Coinbase specific)
                    summary = {}
                    if hasattr(self.context.broker, "client") and hasattr(
                        self.context.broker.client, "get_transaction_summary"
                    ):
                        try:
                            summary = self.context.broker.client.get_transaction_summary()
                        except Exception:
                            pass  # Feature might not be available or API mode issue

                    self._status_reporter.update_account(balances, summary)
                except Exception as e:
                    logger.warning(f"Failed to update account metrics: {e}")

        except Exception as e:
            logger.warning(f"Failed to audit orders: {e}")
