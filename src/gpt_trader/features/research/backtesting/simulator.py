"""
Backtest simulator for strategy evaluation.

Simulates strategy execution against historical data with
realistic fill modeling, fees, and slippage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker
from gpt_trader.backtesting.types import FeeTier
from gpt_trader.core.trading import OrderStatus
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.strategies.perps_baseline import Decision
    from gpt_trader.features.research.backtesting.data_loader import HistoricalDataPoint

logger = get_logger(__name__, component="backtest_simulator")


class StrategyProtocol(Protocol):
    """Protocol for strategies compatible with backtesting."""

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: list[Decimal],
        equity: Decimal,
        product: Any | None,
        market_data: Any | None,
    ) -> Decision:
        """Generate a trading decision."""
        ...


@dataclass
class SimulatedTrade:
    """Record of a simulated trade.

    Attributes:
        timestamp: When the trade occurred.
        symbol: Trading pair.
        side: "buy" or "sell".
        quantity: Amount traded in base currency.
        price: Execution price after slippage.
        fee: Trading fee paid.
        reason: Explanation from strategy.
    """

    timestamp: datetime
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    reason: str

    @property
    def value(self) -> Decimal:
        """Total trade value (price * quantity)."""
        return self.price * self.quantity


@dataclass
class Position:
    """Current position state during simulation.

    Attributes:
        symbol: Trading pair.
        side: "long", "short", or "flat".
        quantity: Position size in base currency.
        entry_price: Average entry price.
        entry_time: When position was opened.
        unrealized_pnl: Current unrealized P&L.
    """

    symbol: str
    side: str = "flat"
    quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    entry_price: Decimal | None = None
    entry_time: datetime | None = None
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for strategy consumption."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
        }


@dataclass
class BacktestConfig:
    """Configuration for backtest simulation.

    Attributes:
        initial_equity: Starting account balance.
        fee_rate_bps: Trading fee in basis points (default 10 = 0.1%).
        use_tiered_fees: Use tiered taker fees when True.
        fee_tier: Fee tier for tiered fee calculation.
        slippage_bps: Base slippage in basis points.
        use_spread_slippage: Use orderbook spread for slippage modeling.
        position_size_pct: Default position size as % of equity.
        max_position_pct: Maximum position size as % of equity.
        enable_funding_pnl: Whether to apply funding payments.
        funding_rates_8h: Funding rates keyed by symbol (8h rate).
        funding_accrual_hours: Funding accrual interval in hours.
        funding_settlement_hours: Funding settlement interval in hours.
        order_fill_delay_bars: Bars to delay order fills.
        cancel_pending_on_new_signal: Cancel pending orders on new signal.
    """

    initial_equity: Decimal = field(default_factory=lambda: Decimal("10000"))
    fee_rate_bps: float = 10.0  # 10 bps = 0.1%
    use_tiered_fees: bool = True
    fee_tier: FeeTier = FeeTier.TIER_2
    slippage_bps: float = 5.0  # 5 bps base slippage
    use_spread_slippage: bool = True
    position_size_pct: float = 0.1  # 10% of equity per trade
    max_position_pct: float = 0.5  # Max 50% of equity in position
    enable_funding_pnl: bool = True
    funding_rates_8h: dict[str, Decimal] | None = None
    funding_accrual_hours: int = 1
    funding_settlement_hours: int = 8
    order_fill_delay_bars: int = 0
    cancel_pending_on_new_signal: bool = True


@dataclass
class SimulatedOrder:
    """Record of a simulated order lifecycle."""

    id: str
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    status: OrderStatus
    submitted_at: datetime
    filled_at: datetime | None = None
    fill_price: Decimal | None = None
    reason: str = ""
    intent: str = ""
    cancel_reason: str | None = None
    cancelled_at: datetime | None = None


@dataclass
class BacktestResult:
    """Results of a backtest run.

    Attributes:
        trades: List of all executed trades.
        orders: List of all simulated orders.
        final_equity: Ending account balance.
        final_position: Final position state.
        equity_curve: Equity at each data point.
        start_time: Backtest start time.
        end_time: Backtest end time.
        data_points_processed: Number of data points evaluated.
    """

    trades: list[SimulatedTrade]
    final_equity: Decimal
    final_position: Position
    orders: list[SimulatedOrder] = field(default_factory=list)
    equity_curve: list[tuple[datetime, Decimal]] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    data_points_processed: int = 0

    @property
    def total_return(self) -> float:
        """Total return as a decimal (0.1 = 10%)."""
        if not self.equity_curve:
            return 0.0
        initial = float(self.equity_curve[0][1])
        if initial == 0:
            return 0.0
        return (float(self.final_equity) - initial) / initial

    @property
    def trade_count(self) -> int:
        """Number of trades executed."""
        return len(self.trades)


class BacktestSimulator:
    """Simulate strategy execution against historical data.

    Features:
    - Realistic fill modeling with spread-based slippage
    - Configurable fee structure
    - Position tracking with P&L calculation
    - Equity curve generation

    Example:
        simulator = BacktestSimulator(config)
        result = simulator.run(strategy, data_points)
        print(f"Total return: {result.total_return:.2%}")
        print(f"Trades: {result.trade_count}")
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Initialize simulator.

        Args:
            config: Simulation configuration (uses defaults if None).
        """
        self.config = config or BacktestConfig()
        self._equity = self.config.initial_equity
        self._position: Position | None = None
        self._trades: list[SimulatedTrade] = []
        self._equity_curve: list[tuple[datetime, Decimal]] = []
        self._recent_marks: list[Decimal] = []
        self._orders: list[SimulatedOrder] = []
        self._pending_orders: list[tuple[SimulatedOrder, int]] = []
        self._order_counter = 0
        self._current_index = 0
        self._fee_calculator: FeeCalculator | None = None
        self._funding_tracker: FundingPnLTracker | None = None

    def run(
        self,
        strategy: StrategyProtocol,
        data_points: list[HistoricalDataPoint],
        symbol: str | None = None,
    ) -> BacktestResult:
        """Run backtest simulation.

        Args:
            strategy: Strategy to evaluate.
            data_points: Historical data to replay.
            symbol: Trading symbol (inferred from data if None).

        Returns:
            BacktestResult with trades, equity curve, and statistics.
        """
        if not data_points:
            return BacktestResult(
                trades=[],
                orders=[],
                final_equity=self.config.initial_equity,
                final_position=Position(symbol=symbol or "UNKNOWN"),
            )

        # Infer symbol from data
        if symbol is None:
            symbol = data_points[0].symbol

        # Reset state
        self._equity = self.config.initial_equity
        self._position = Position(symbol=symbol)
        self._trades = []
        self._orders = []
        self._pending_orders = []
        self._order_counter = 0
        self._current_index = 0
        self._equity_curve = []
        self._recent_marks = []
        self._fee_calculator = (
            FeeCalculator(tier=self.config.fee_tier) if self.config.use_tiered_fees else None
        )
        self._funding_tracker = (
            FundingPnLTracker(
                accrual_interval_hours=self.config.funding_accrual_hours,
                settlement_interval_hours=self.config.funding_settlement_hours,
            )
            if self.config.enable_funding_pnl
            else None
        )

        logger.info(
            "Starting backtest",
            symbol=symbol,
            data_points=len(data_points),
            initial_equity=str(self.config.initial_equity),
        )

        # Replay each data point
        for i, point in enumerate(data_points):
            self._current_index = i
            # Update recent marks for strategy
            self._recent_marks.append(point.mark_price)
            if len(self._recent_marks) > 100:
                self._recent_marks = self._recent_marks[-100:]

            self._process_pending_orders(point, i)

            # Update unrealized P&L
            if self._position and self._position.side != "flat":
                self._update_unrealized_pnl(point.mark_price)

            self._process_funding(point)

            # Build market data context
            from gpt_trader.features.live_trade.strategies.base import MarketDataContext

            market_data = MarketDataContext(
                orderbook_snapshot=point.orderbook_snapshot,
                trade_volume_stats=point.trade_flow_stats,
                spread_bps=Decimal(str(point.spread_bps)) if point.spread_bps else None,
            )

            # Get strategy decision
            decision = strategy.decide(
                symbol=point.symbol,
                current_mark=point.mark_price,
                position_state=self._position.to_dict() if self._position else None,
                recent_marks=list(self._recent_marks[:-1]),
                equity=self._equity,
                product=None,
                market_data=market_data,
            )

            # Execute decision
            self._execute_decision(decision, point)

            # Record equity curve
            total_equity = self._equity + (
                self._position.unrealized_pnl if self._position else Decimal("0")
            )
            self._equity_curve.append((point.timestamp, total_equity))

        # Final position mark-to-market
        if self._position and self._position.side != "flat" and data_points:
            final_point = data_points[-1]
            close_side = "sell" if self._position.side == "long" else "buy"
            fill_price = self._calculate_fill_price(final_point, close_side)
            self._apply_close(final_point, fill_price, "end_of_backtest")

        return BacktestResult(
            trades=self._trades,
            orders=self._orders,
            final_equity=self._equity,
            final_position=self._position or Position(symbol=symbol),
            equity_curve=self._equity_curve,
            start_time=data_points[0].timestamp if data_points else None,
            end_time=data_points[-1].timestamp if data_points else None,
            data_points_processed=len(data_points),
        )

    def _execute_decision(
        self,
        decision: Decision,
        point: HistoricalDataPoint,
    ) -> None:
        """Execute a trading decision.

        Args:
            decision: Strategy decision.
            point: Current market data point.
        """
        from gpt_trader.features.live_trade.strategies.perps_baseline import Action

        action = decision.action
        pending = self._has_pending_orders(point.symbol)

        if action == Action.HOLD:
            return

        if pending:
            if action == Action.CLOSE:
                self._cancel_pending_orders(point.symbol, "close_signal", point.timestamp)
                return
            if action in (Action.BUY, Action.SELL):
                if self.config.cancel_pending_on_new_signal:
                    self._cancel_pending_orders(point.symbol, "new_signal", point.timestamp)
                else:
                    return

        if action == Action.CLOSE:
            if self._position and self._position.side != "flat":
                close_side = "sell" if self._position.side == "long" else "buy"
                intent = "close_long" if self._position.side == "long" else "close_short"
                self._submit_order(
                    point=point,
                    intent=intent,
                    side=close_side,
                    quantity=self._position.quantity,
                    reason=decision.reason,
                )
            return

        if action == Action.BUY:
            if self._position and self._position.side == "short":
                # Close short first
                self._submit_order(
                    point=point,
                    intent="close_short",
                    side="buy",
                    quantity=self._position.quantity,
                    reason="close_before_buy",
                )
                if self.config.order_fill_delay_bars != 0:
                    return
            if self._position is None or self._position.side == "flat":
                quantity = self._calculate_order_quantity(point, "long")
                self._submit_order(
                    point=point,
                    intent="open_long",
                    side="buy",
                    quantity=quantity,
                    reason=decision.reason,
                )
            return

        if action == Action.SELL:
            if self._position and self._position.side == "long":
                # Close long first
                self._submit_order(
                    point=point,
                    intent="close_long",
                    side="sell",
                    quantity=self._position.quantity,
                    reason="close_before_sell",
                )
                if self.config.order_fill_delay_bars != 0:
                    return
            if self._position is None or self._position.side == "flat":
                quantity = self._calculate_order_quantity(point, "short")
                self._submit_order(
                    point=point,
                    intent="open_short",
                    side="sell",
                    quantity=quantity,
                    reason=decision.reason,
                )
            return

    def _has_pending_orders(self, symbol: str) -> bool:
        return any(order.symbol == symbol for order, _ in self._pending_orders)

    def _calculate_order_quantity(self, point: HistoricalDataPoint, side: str) -> Decimal:
        position_value = self._equity * Decimal(str(self.config.position_size_pct))
        max_value = self._equity * Decimal(str(self.config.max_position_pct))
        position_value = min(position_value, max_value)
        fill_price = self._calculate_fill_price(point, side)
        return position_value / fill_price

    def _submit_order(
        self,
        point: HistoricalDataPoint,
        intent: str,
        side: str,
        quantity: Decimal,
        reason: str,
    ) -> None:
        self._order_counter += 1
        order = SimulatedOrder(
            id=f"order-{self._order_counter}",
            symbol=point.symbol,
            side=side,
            order_type="market",
            quantity=quantity,
            status=OrderStatus.PENDING,
            submitted_at=point.timestamp,
            reason=reason,
            intent=intent,
        )
        self._orders.append(order)

        if self.config.order_fill_delay_bars <= 0:
            self._fill_order(order, point)
        else:
            fill_index = self._current_index + self.config.order_fill_delay_bars
            self._pending_orders.append((order, fill_index))

    def _process_pending_orders(self, point: HistoricalDataPoint, index: int) -> None:
        if not self._pending_orders:
            return

        remaining: list[tuple[SimulatedOrder, int]] = []
        for order, fill_index in self._pending_orders:
            if order.status != OrderStatus.PENDING:
                continue
            if index >= fill_index:
                self._fill_order(order, point)
            else:
                remaining.append((order, fill_index))

        self._pending_orders = remaining

    def _cancel_pending_orders(self, symbol: str, reason: str, timestamp: datetime) -> None:
        if not self._pending_orders:
            return

        remaining: list[tuple[SimulatedOrder, int]] = []
        for order, fill_index in self._pending_orders:
            if order.symbol != symbol:
                remaining.append((order, fill_index))
                continue
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                order.cancel_reason = reason
                order.cancelled_at = timestamp

        self._pending_orders = remaining

    def _fill_order(self, order: SimulatedOrder, point: HistoricalDataPoint) -> None:
        if order.status == OrderStatus.CANCELLED:
            return

        fill_price = self._calculate_fill_price(point, order.side)
        order.status = OrderStatus.FILLED
        order.filled_at = point.timestamp
        order.fill_price = fill_price

        if order.intent == "open_long":
            self._apply_open(point, "long", order.quantity, fill_price, order.reason)
        elif order.intent == "open_short":
            self._apply_open(point, "short", order.quantity, fill_price, order.reason)
        elif order.intent in ("close_long", "close_short"):
            self._apply_close(point, fill_price, order.reason)

    def _apply_open(
        self,
        point: HistoricalDataPoint,
        side: str,
        quantity: Decimal,
        fill_price: Decimal,
        reason: str,
    ) -> None:
        trade_value = fill_price * quantity
        fee = self.calculate_fee(trade_value)

        # Update equity (deduct fee)
        self._equity -= fee

        # Record trade
        trade = SimulatedTrade(
            timestamp=point.timestamp,
            symbol=point.symbol,
            side="buy" if side == "long" else "sell",
            quantity=quantity,
            price=fill_price,
            fee=fee,
            reason=reason,
        )
        self._trades.append(trade)

        # Update position
        self._position = Position(
            symbol=point.symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=point.timestamp,
        )

        if (
            self.config.enable_funding_pnl
            and self._funding_tracker
            and self.config.funding_rates_8h
        ):
            funding_rate = self.config.funding_rates_8h.get(point.symbol)
            if funding_rate is not None:
                position_size = quantity if side == "long" else -quantity
                self._funding_tracker.accrue(
                    symbol=point.symbol,
                    position_size=position_size,
                    mark_price=point.mark_price,
                    funding_rate_8h=funding_rate,
                    current_time=point.timestamp,
                )

        logger.debug(
            "Opened position",
            side=side,
            quantity=str(quantity),
            price=str(fill_price),
            fee=str(fee),
        )

    def _apply_close(
        self,
        point: HistoricalDataPoint,
        fill_price: Decimal,
        reason: str,
    ) -> None:
        if self._position is None or self._position.side == "flat":
            return

        close_side = "sell" if self._position.side == "long" else "buy"

        # Calculate P&L
        if self._position.entry_price:
            if self._position.side == "long":
                pnl = (fill_price - self._position.entry_price) * self._position.quantity
            else:
                pnl = (self._position.entry_price - fill_price) * self._position.quantity
        else:
            pnl = Decimal("0")

        # Calculate fee
        trade_value = fill_price * self._position.quantity
        fee = self.calculate_fee(trade_value)

        # Update equity
        self._equity += pnl - fee

        # Record trade
        trade = SimulatedTrade(
            timestamp=point.timestamp,
            symbol=point.symbol,
            side=close_side,
            quantity=self._position.quantity,
            price=fill_price,
            fee=fee,
            reason=reason,
        )
        self._trades.append(trade)

        logger.debug(
            "Closed position",
            side=self._position.side,
            pnl=str(pnl),
            fee=str(fee),
        )

        # Reset position
        self._position = Position(symbol=point.symbol)

    def calculate_fee(self, notional: Decimal) -> Decimal:
        """Calculate trading fee for a notional value."""
        if self.config.use_tiered_fees and self._fee_calculator:
            return self._fee_calculator.calculate(notional_usd=notional, is_maker=False)
        return notional * Decimal(str(self.config.fee_rate_bps)) / Decimal("10000")

    def _process_funding(self, point: HistoricalDataPoint) -> None:
        """Accrue and settle funding for open positions."""
        if not self.config.enable_funding_pnl:
            return
        if not self.config.funding_rates_8h:
            return
        if self._position is None or self._position.side == "flat":
            return
        if self._funding_tracker is None:
            return

        funding_rate = self.config.funding_rates_8h.get(point.symbol)
        if funding_rate is None:
            return

        position_size = (
            self._position.quantity if self._position.side == "long" else -self._position.quantity
        )
        self._funding_tracker.accrue(
            symbol=point.symbol,
            position_size=position_size,
            mark_price=point.mark_price,
            funding_rate_8h=funding_rate,
            current_time=point.timestamp,
        )

        if self._funding_tracker.should_settle(point.timestamp, point.symbol):
            settled = self._funding_tracker.settle(point.symbol, point.timestamp)
            self._equity -= settled

    def _calculate_fill_price(
        self,
        point: HistoricalDataPoint,
        side: str,
    ) -> Decimal:
        """Calculate fill price with slippage.

        Uses spread from orderbook if available, otherwise uses base slippage.

        Args:
            point: Current market data.
            side: "buy" or "sell" (or "long"/"short").

        Returns:
            Estimated fill price.
        """
        base_price = point.mark_price
        slippage_bps = Decimal(str(self.config.slippage_bps))

        # Use spread-based slippage if available
        if self.config.use_spread_slippage and point.spread_bps:
            # Half the spread as slippage (we cross the spread)
            slippage_bps = max(slippage_bps, Decimal(str(point.spread_bps)) / 2)

        slippage_pct = slippage_bps / Decimal("10000")

        # Buy/long pays above mark, sell/short receives below mark
        if side in ("buy", "long"):
            return base_price * (1 + slippage_pct)
        else:
            return base_price * (1 - slippage_pct)

    def _update_unrealized_pnl(self, current_price: Decimal) -> None:
        """Update unrealized P&L for current position.

        Args:
            current_price: Current mark price.
        """
        if self._position is None or self._position.entry_price is None:
            return

        if self._position.side == "long":
            self._position.unrealized_pnl = (
                current_price - self._position.entry_price
            ) * self._position.quantity
        elif self._position.side == "short":
            self._position.unrealized_pnl = (
                self._position.entry_price - current_price
            ) * self._position.quantity
