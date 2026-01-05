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
    ) -> "Decision":
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
        slippage_bps: Base slippage in basis points.
        use_spread_slippage: Use orderbook spread for slippage modeling.
        position_size_pct: Default position size as % of equity.
        max_position_pct: Maximum position size as % of equity.
    """

    initial_equity: Decimal = field(default_factory=lambda: Decimal("10000"))
    fee_rate_bps: float = 10.0  # 10 bps = 0.1%
    slippage_bps: float = 5.0  # 5 bps base slippage
    use_spread_slippage: bool = True
    position_size_pct: float = 0.1  # 10% of equity per trade
    max_position_pct: float = 0.5  # Max 50% of equity in position


@dataclass
class BacktestResult:
    """Results of a backtest run.

    Attributes:
        trades: List of all executed trades.
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

    def run(
        self,
        strategy: StrategyProtocol,
        data_points: list["HistoricalDataPoint"],
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
        self._equity_curve = []
        self._recent_marks = []

        logger.info(
            "Starting backtest",
            symbol=symbol,
            data_points=len(data_points),
            initial_equity=str(self.config.initial_equity),
        )

        # Replay each data point
        for i, point in enumerate(data_points):
            # Update recent marks for strategy
            self._recent_marks.append(point.mark_price)
            if len(self._recent_marks) > 100:
                self._recent_marks = self._recent_marks[-100:]

            # Update unrealized P&L
            if self._position and self._position.side != "flat":
                self._update_unrealized_pnl(point.mark_price)

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
            self._close_position(final_point, "end_of_backtest")

        return BacktestResult(
            trades=self._trades,
            final_equity=self._equity,
            final_position=self._position or Position(symbol=symbol),
            equity_curve=self._equity_curve,
            start_time=data_points[0].timestamp if data_points else None,
            end_time=data_points[-1].timestamp if data_points else None,
            data_points_processed=len(data_points),
        )

    def _execute_decision(
        self,
        decision: "Decision",
        point: "HistoricalDataPoint",
    ) -> None:
        """Execute a trading decision.

        Args:
            decision: Strategy decision.
            point: Current market data point.
        """
        from gpt_trader.features.live_trade.strategies.perps_baseline import Action

        action = decision.action

        if action == Action.HOLD:
            return

        if action == Action.CLOSE:
            if self._position and self._position.side != "flat":
                self._close_position(point, decision.reason)
            return

        if action == Action.BUY:
            if self._position and self._position.side == "short":
                # Close short first
                self._close_position(point, "close_before_buy")
            if self._position is None or self._position.side == "flat":
                self._open_position(point, "long", decision.reason)
            return

        if action == Action.SELL:
            if self._position and self._position.side == "long":
                # Close long first
                self._close_position(point, "close_before_sell")
            if self._position is None or self._position.side == "flat":
                self._open_position(point, "short", decision.reason)
            return

    def _open_position(
        self,
        point: "HistoricalDataPoint",
        side: str,
        reason: str,
    ) -> None:
        """Open a new position.

        Args:
            point: Current market data.
            side: "long" or "short".
            reason: Trade reason.
        """
        # Calculate position size
        position_value = self._equity * Decimal(str(self.config.position_size_pct))
        max_value = self._equity * Decimal(str(self.config.max_position_pct))
        position_value = min(position_value, max_value)

        # Calculate fill price with slippage
        fill_price = self._calculate_fill_price(point, side)
        quantity = position_value / fill_price

        # Calculate fee
        fee = position_value * Decimal(str(self.config.fee_rate_bps)) / Decimal("10000")

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

        logger.debug(
            "Opened position",
            side=side,
            quantity=str(quantity),
            price=str(fill_price),
            fee=str(fee),
        )

    def _close_position(
        self,
        point: "HistoricalDataPoint",
        reason: str,
    ) -> None:
        """Close the current position.

        Args:
            point: Current market data.
            reason: Close reason.
        """
        if self._position is None or self._position.side == "flat":
            return

        # Calculate fill price with slippage
        close_side = "sell" if self._position.side == "long" else "buy"
        fill_price = self._calculate_fill_price(point, close_side)

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
        fee = trade_value * Decimal(str(self.config.fee_rate_bps)) / Decimal("10000")

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

    def _calculate_fill_price(
        self,
        point: "HistoricalDataPoint",
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
                (current_price - self._position.entry_price) * self._position.quantity
            )
        elif self._position.side == "short":
            self._position.unrealized_pnl = (
                (self._position.entry_price - current_price) * self._position.quantity
            )
