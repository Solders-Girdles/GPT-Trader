"""
Realistic Backtesting Framework
Phase 2.5 - Day 5

Implements realistic backtesting with transaction costs, slippage, and market impact.
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(Enum):
    """Position side"""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class TransactionCosts:
    """Transaction cost model"""

    commission_rate: float = 0.001  # 0.1% per trade
    commission_minimum: float = 1.0  # $1 minimum
    spread_cost: float = 0.0005  # 0.05% bid-ask spread
    market_impact_factor: float = 0.1  # Linear market impact
    slippage_factor: float = 0.0002  # 0.02% slippage
    borrowing_cost: float = 0.02  # 2% annual for shorts

    def calculate_total_cost(
        self, trade_value: float, shares: int, is_short: bool = False
    ) -> float:
        """Calculate total transaction cost"""
        # Commission
        commission = max(self.commission_minimum, trade_value * self.commission_rate)

        # Spread cost
        spread = trade_value * self.spread_cost

        # Market impact (increases with size)
        impact = trade_value * self.market_impact_factor * np.sqrt(shares / 1000)

        # Slippage
        slippage = trade_value * self.slippage_factor

        # Total
        total = commission + spread + impact + slippage

        # Add borrowing cost for shorts (daily rate)
        if is_short:
            total += trade_value * (self.borrowing_cost / 252)

        return total


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    initial_capital: float = 100000
    position_size: float = 0.1  # 10% per position
    max_positions: int = 10

    # Risk management
    stop_loss: float = 0.02  # 2% stop loss
    take_profit: float = 0.05  # 5% take profit
    max_drawdown: float = 0.20  # 20% maximum drawdown

    # Transaction costs
    costs: TransactionCosts = field(default_factory=TransactionCosts)

    # Execution
    use_limit_orders: bool = False
    limit_order_buffer: float = 0.001  # 0.1% from market price
    fill_probability: float = 0.95  # Probability of limit order fill

    # Data
    lookback_period: int = 252  # Days for indicators
    warmup_period: int = 50  # Warmup period before trading

    # Reporting
    verbose: bool = True
    save_trades: bool = True


@dataclass
class Trade:
    """Individual trade record"""

    timestamp: datetime
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    exit_price: float | None = None
    exit_timestamp: datetime | None = None

    # Costs
    entry_cost: float = 0
    exit_cost: float = 0

    # P&L
    gross_pnl: float = 0
    net_pnl: float = 0
    return_pct: float = 0

    # Metadata
    signal_strength: float | None = None
    holding_period: int | None = None
    max_profit: float = 0
    max_loss: float = 0

    def calculate_pnl(self):
        """Calculate P&L for the trade"""
        if self.exit_price:
            if self.side == PositionSide.LONG:
                self.gross_pnl = (self.exit_price - self.entry_price) * self.quantity
            else:  # SHORT
                self.gross_pnl = (self.entry_price - self.exit_price) * self.quantity

            self.net_pnl = self.gross_pnl - self.entry_cost - self.exit_cost
            self.return_pct = self.net_pnl / (self.entry_price * self.quantity)

            if self.exit_timestamp and self.timestamp:
                self.holding_period = (self.exit_timestamp - self.timestamp).days


@dataclass
class BacktestResults:
    """Backtesting results"""

    # Returns
    total_return: float
    annualized_return: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Costs
    total_commission: float
    total_slippage: float
    total_costs: float

    # Time series
    equity_curve: pd.Series
    returns: pd.Series
    trades: list[Trade]

    # Additional metrics
    best_trade: Trade | None = None
    worst_trade: Trade | None = None
    avg_holding_period: float = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        metrics = {
            "Total Return": f"{self.total_return:.2%}",
            "Annualized Return": f"{self.annualized_return:.2%}",
            "Volatility": f"{self.volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Calmar Ratio": f"{self.calmar_ratio:.2f}",
            "Total Trades": self.total_trades,
            "Win Rate": f"{self.win_rate:.2%}",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Avg Win": f"{self.avg_win:.2%}",
            "Avg Loss": f"{self.avg_loss:.2%}",
            "Total Costs": f"${self.total_costs:,.2f}",
            "Avg Holding Period": f"{self.avg_holding_period:.1f} days",
        }

        return pd.DataFrame(metrics.items(), columns=["Metric", "Value"])


class RealisticBacktester:
    """
    Realistic backtesting engine with transaction costs and market microstructure.

    Features:
    - Transaction costs (commission, spread, slippage, market impact)
    - Position sizing and risk management
    - Multiple order types
    - Realistic execution simulation
    - Comprehensive performance metrics
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

        # Portfolio state
        self.capital = self.config.initial_capital
        self.positions: dict[str, Trade] = {}
        self.trades: list[Trade] = []

        # Performance tracking
        self.equity_curve = []
        self.returns = []

        logger.info(f"RealisticBacktester initialized with ${self.capital:,.0f} capital")

    def run(
        self, data: pd.DataFrame, signals: pd.Series, prices: pd.Series | None = None
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals (-1, 0, 1)
            prices: Optional custom price series for execution

        Returns:
            Backtest results
        """
        if prices is None:
            prices = data["close"]

        # Initialize tracking
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # Warmup period
        start_idx = self.config.warmup_period

        # Main backtest loop
        for i in range(start_idx, len(data)):
            timestamp = data.index[i]
            price = prices.iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0

            # Update existing positions
            self._update_positions(timestamp, price, data.iloc[i])

            # Check risk limits
            if self._check_risk_limits():
                self._close_all_positions(timestamp, price)
                continue

            # Process signal
            if signal != 0:
                self._process_signal(
                    timestamp=timestamp, signal=signal, price=price, data_row=data.iloc[i]
                )

            # Record equity
            equity = self._calculate_equity(price)
            self.equity_curve.append(equity)

            if i > start_idx:
                daily_return = (equity - self.equity_curve[-2]) / self.equity_curve[-2]
                self.returns.append(daily_return)

        # Close remaining positions
        if self.positions:
            last_price = prices.iloc[-1]
            last_timestamp = data.index[-1]
            self._close_all_positions(last_timestamp, last_price)

        # Calculate results
        results = self._calculate_results(data.index[start_idx:])

        return results

    def _process_signal(
        self, timestamp: datetime, signal: float, price: float, data_row: pd.Series
    ):
        """Process trading signal"""
        symbol = "Asset"  # Single asset for now

        # Determine position side
        if signal > 0:
            target_side = PositionSide.LONG
        elif signal < 0:
            target_side = PositionSide.SHORT
        else:
            target_side = PositionSide.FLAT

        # Check current position
        current_position = self.positions.get(symbol)

        if target_side == PositionSide.FLAT:
            # Close position if exists
            if current_position:
                self._close_position(symbol, timestamp, price)

        elif current_position is None:
            # Open new position
            self._open_position(symbol, timestamp, price, target_side, abs(signal))

        elif current_position.side != target_side:
            # Reverse position
            self._close_position(symbol, timestamp, price)
            self._open_position(symbol, timestamp, price, target_side, abs(signal))

    def _open_position(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        side: PositionSide,
        signal_strength: float = 1.0,
    ):
        """Open new position"""
        # Check position limits
        if len(self.positions) >= self.config.max_positions:
            return

        # Calculate position size
        position_value = self.capital * self.config.position_size * signal_strength
        shares = int(position_value / price)

        if shares == 0:
            return

        # Calculate execution price with slippage
        if self.config.use_limit_orders:
            # Limit order simulation
            if np.random.random() > self.config.fill_probability:
                return  # Order didn't fill

            if side == PositionSide.LONG:
                exec_price = price * (1 - self.config.limit_order_buffer)
            else:
                exec_price = price * (1 + self.config.limit_order_buffer)
        else:
            # Market order with slippage
            slippage = self.config.costs.slippage_factor
            if side == PositionSide.LONG:
                exec_price = price * (1 + slippage)
            else:
                exec_price = price * (1 - slippage)

        # Calculate costs
        trade_value = exec_price * shares
        costs = self.config.costs.calculate_total_cost(
            trade_value, shares, is_short=(side == PositionSide.SHORT)
        )

        # Check if we have enough capital
        required_capital = trade_value + costs
        if required_capital > self.capital:
            return

        # Create trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=shares,
            entry_price=exec_price,
            entry_cost=costs,
            signal_strength=signal_strength,
        )

        # Update portfolio
        self.positions[symbol] = trade
        self.capital -= required_capital

        if self.config.verbose:
            logger.debug(f"Opened {side.value} position: {shares} shares @ ${exec_price:.2f}")

    def _close_position(self, symbol: str, timestamp: datetime, price: float):
        """Close existing position"""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]

        # Calculate execution price with slippage
        slippage = self.config.costs.slippage_factor
        if trade.side == PositionSide.LONG:
            exec_price = price * (1 - slippage)
        else:
            exec_price = price * (1 + slippage)

        # Calculate costs
        trade_value = exec_price * trade.quantity
        costs = self.config.costs.calculate_total_cost(
            trade_value, trade.quantity, is_short=(trade.side == PositionSide.SHORT)
        )

        # Update trade
        trade.exit_price = exec_price
        trade.exit_timestamp = timestamp
        trade.exit_cost = costs
        trade.calculate_pnl()

        # Update portfolio
        self.capital += trade_value - costs
        self.trades.append(trade)
        del self.positions[symbol]

        if self.config.verbose:
            logger.debug(f"Closed position: P&L = ${trade.net_pnl:.2f} ({trade.return_pct:.2%})")

    def _update_positions(self, timestamp: datetime, price: float, data_row: pd.Series):
        """Update existing positions (stop loss, take profit, etc.)"""
        positions_to_close = []

        for symbol, trade in self.positions.items():
            # Calculate current P&L
            if trade.side == PositionSide.LONG:
                current_pnl_pct = (price - trade.entry_price) / trade.entry_price
            else:
                current_pnl_pct = (trade.entry_price - price) / trade.entry_price

            # Track max profit/loss
            trade.max_profit = max(trade.max_profit, current_pnl_pct)
            trade.max_loss = min(trade.max_loss, current_pnl_pct)

            # Check stop loss
            if current_pnl_pct <= -self.config.stop_loss:
                positions_to_close.append(symbol)
                if self.config.verbose:
                    logger.debug(f"Stop loss triggered for {symbol}")

            # Check take profit
            elif current_pnl_pct >= self.config.take_profit:
                positions_to_close.append(symbol)
                if self.config.verbose:
                    logger.debug(f"Take profit triggered for {symbol}")

        # Close positions
        for symbol in positions_to_close:
            self._close_position(symbol, timestamp, price)

    def _close_all_positions(self, timestamp: datetime, price: float):
        """Close all open positions"""
        symbols = list(self.positions.keys())
        for symbol in symbols:
            self._close_position(symbol, timestamp, price)

    def _check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        if len(self.equity_curve) < 2:
            return False

        # Calculate current drawdown
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        drawdown = (peak - current) / peak

        return drawdown > self.config.max_drawdown

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity value"""
        equity = self.capital

        for symbol, trade in self.positions.items():
            if trade.side == PositionSide.LONG:
                position_value = current_price * trade.quantity
            else:
                # Short position
                position_value = (
                    2 * trade.entry_price * trade.quantity - current_price * trade.quantity
                )

            equity += position_value

        return equity

    def _calculate_results(self, index: pd.DatetimeIndex) -> BacktestResults:
        """Calculate backtest results"""
        equity_series = pd.Series(self.equity_curve, index=index[: len(self.equity_curve)])
        returns_series = pd.Series(self.returns, index=index[1 : len(self.returns) + 1])

        # Basic returns
        total_return = (
            equity_series.iloc[-1] - self.config.initial_capital
        ) / self.config.initial_capital
        years = (index[-1] - index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Risk metrics
        volatility = returns_series.std() * np.sqrt(252)

        sharpe_ratio = 0
        if volatility > 0:
            sharpe_ratio = (annualized_return - 0.02) / volatility  # Assuming 2% risk-free rate

        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_vol = (
            downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        )
        sortino_ratio = (annualized_return - 0.02) / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Trading statistics
        winning_trades = [t for t in self.trades if t.net_pnl > 0]
        losing_trades = [t for t in self.trades if t.net_pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        avg_win = np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.return_pct for t in losing_trades]) if losing_trades else 0

        total_wins = sum(t.net_pnl for t in winning_trades)
        total_losses = abs(sum(t.net_pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

        # Costs
        total_commission = sum(t.entry_cost + t.exit_cost for t in self.trades)
        total_costs = total_commission  # Could add more cost breakdowns

        # Best/worst trades
        if self.trades:
            best_trade = max(self.trades, key=lambda t: t.net_pnl)
            worst_trade = min(self.trades, key=lambda t: t.net_pnl)
            avg_holding = np.mean([t.holding_period for t in self.trades if t.holding_period])
        else:
            best_trade = worst_trade = None
            avg_holding = 0

        # Consecutive wins/losses
        max_cons_wins = max_cons_losses = current_wins = current_losses = 0
        for trade in self.trades:
            if trade.net_pnl > 0:
                current_wins += 1
                current_losses = 0
                max_cons_wins = max(max_cons_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_cons_losses = max(max_cons_losses, current_losses)

        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_commission=total_commission,
            total_slippage=0,  # Already included in costs
            total_costs=total_costs,
            equity_curve=equity_series,
            returns=returns_series,
            trades=self.trades,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_holding_period=avg_holding,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses,
        )


def create_backtester(config: BacktestConfig | None = None) -> RealisticBacktester:
    """Create backtester instance"""
    return RealisticBacktester(config)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    # Get sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="2y")
    data.columns = [c.lower() for c in data.columns]

    # Generate simple signals (moving average crossover)
    sma_short = data["close"].rolling(20).mean()
    sma_long = data["close"].rolling(50).mean()

    signals = pd.Series(0, index=data.index)
    signals[sma_short > sma_long] = 1
    signals[sma_short < sma_long] = -1

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000, position_size=0.1, stop_loss=0.02, take_profit=0.05
    )

    # Run backtest
    backtester = create_backtester(config)
    results = backtester.run(data, signals)

    # Display results
    print("\nBacktest Results:")
    print(results.to_dataframe())

    print(f"\nTotal trades: {results.total_trades}")
    print(f"Best trade: ${results.best_trade.net_pnl:.2f}" if results.best_trade else "N/A")
    print(f"Worst trade: ${results.worst_trade.net_pnl:.2f}" if results.worst_trade else "N/A")
