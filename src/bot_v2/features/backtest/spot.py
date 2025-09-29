from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Bar:
    """Represents a single time-bar used in backtesting."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class StrategySignal:
    """Simple signal constants to avoid Enum overhead in tight loops."""

    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"


@dataclass
class BacktestMetrics:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    equity_curve: pd.DataFrame
    trades: list[dict]


@dataclass
class SpotBacktestConfig:
    initial_cash: Decimal = Decimal("10000")
    commission_bps: Decimal = Decimal("2.5")  # 0.025%
    risk_free_rate: float = 0.0


class MovingAverageCrossStrategy:
    """Basic long-only moving-average cross strategy for prototyping."""

    def __init__(self, short_window: int = 12, long_window: int = 26) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("MA windows must be positive")
        if short_window >= long_window:
            raise ValueError("Short window must be smaller than long window")
        self.short_window = short_window
        self.long_window = long_window
        self._closes: list[Decimal] = []

    def on_bar(self, bar: Bar, has_position: bool) -> str:
        self._closes.append(bar.close)
        if len(self._closes) < self.long_window:
            return StrategySignal.HOLD

        short_ma = _mean(self._closes[-self.short_window :])
        long_ma = _mean(self._closes[-self.long_window :])

        if not has_position and short_ma > long_ma:
            return StrategySignal.BUY
        if has_position and short_ma < long_ma:
            return StrategySignal.SELL
        return StrategySignal.HOLD


class BollingerMeanReversionStrategy:
    """Simple Bollinger-band mean-reversion strategy (buy lower band, sell upper)."""

    def __init__(self, window: int = 20, num_std: Decimal | float = Decimal("2")) -> None:
        if window <= 1:
            raise ValueError("Window must be greater than 1")
        self.window = window
        self.num_std = Decimal(str(num_std))
        self._closes: list[Decimal] = []

    def on_bar(self, bar: Bar, has_position: bool) -> str:
        self._closes.append(bar.close)
        if len(self._closes) < self.window:
            return StrategySignal.HOLD

        recent = self._closes[-self.window :]
        mean = _mean(recent)
        std = _stddev(recent, mean)
        if std == 0:
            return StrategySignal.HOLD

        upper = mean + self.num_std * std
        lower = mean - self.num_std * std

        if not has_position and bar.close <= lower:
            return StrategySignal.BUY
        if has_position and bar.close >= upper:
            return StrategySignal.SELL
        return StrategySignal.HOLD


class VolatilityFilteredStrategy:
    """Wraps a base strategy and suppresses entries when volatility is out of range."""

    def __init__(
        self,
        base_strategy,
        window: int = 14,
        min_vol: Decimal | float = Decimal("0.001"),
        max_vol: Decimal | float = Decimal("0.05"),
    ) -> None:
        if window <= 1:
            raise ValueError("Volatility window must be greater than 1")
        self.base_strategy = base_strategy
        self.window = window
        self.min_vol = Decimal(str(min_vol))
        self.max_vol = Decimal(str(max_vol))
        self._trs: list[Decimal] = []
        self._prev_close: Decimal | None = None

    def on_bar(self, bar: Bar, has_position: bool) -> str:
        tr = _true_range(bar, self._prev_close)
        self._prev_close = bar.close

        self._trs.append(tr)
        if len(self._trs) > self.window:
            self._trs.pop(0)

        base_signal = self.base_strategy.on_bar(bar, has_position)

        if base_signal == StrategySignal.BUY and not has_position:
            if len(self._trs) < self.window:
                return StrategySignal.HOLD
            atr = _mean(self._trs)
            if atr <= Decimal("0"):
                return StrategySignal.HOLD
            vol_pct = atr / bar.close
            if vol_pct < self.min_vol or vol_pct > self.max_vol:
                return StrategySignal.HOLD

        return base_signal


class VolumeConfirmationStrategy:
    """Require volume to exceed a moving average before passing through a base signal."""

    def __init__(
        self,
        base_strategy,
        window: int = 20,
        multiplier: Decimal | float = Decimal("1.2"),
    ) -> None:
        if window <= 1:
            raise ValueError("Volume window must be greater than 1")
        self.base_strategy = base_strategy
        self.window = window
        self.multiplier = Decimal(str(multiplier))
        self._volumes: list[Decimal] = []

    def on_bar(self, bar: Bar, has_position: bool) -> str:
        self._volumes.append(bar.volume)
        if len(self._volumes) > self.window:
            self._volumes.pop(0)

        base_signal = self.base_strategy.on_bar(bar, has_position)
        if base_signal == StrategySignal.BUY and not has_position:
            if len(self._volumes) < self.window:
                return StrategySignal.HOLD
            avg_vol = _mean(self._volumes)
            if avg_vol <= Decimal("0"):
                return StrategySignal.HOLD
            if bar.volume < avg_vol * self.multiplier:
                return StrategySignal.HOLD
        return base_signal


class MomentumOscillatorStrategy:
    """RSI-based momentum filter that only allows base entries when oscillator aligns."""

    def __init__(
        self,
        base_strategy,
        window: int = 14,
        overbought: Decimal | float = Decimal("70"),
        oversold: Decimal | float = Decimal("30"),
    ) -> None:
        if window <= 1:
            raise ValueError("Momentum window must be greater than 1")
        self.base_strategy = base_strategy
        self.window = window
        self.overbought = Decimal(str(overbought))
        self.oversold = Decimal(str(oversold))
        self._closes: list[Decimal] = []

    def on_bar(self, bar: Bar, has_position: bool) -> str:
        self._closes.append(bar.close)
        base_signal = self.base_strategy.on_bar(bar, has_position)

        if len(self._closes) <= self.window:
            return base_signal if has_position else StrategySignal.HOLD

        rsi = _rsi(self._closes[-(self.window + 1) :])

        if base_signal == StrategySignal.BUY and not has_position:
            if rsi > self.oversold:
                return StrategySignal.HOLD
        if base_signal == StrategySignal.SELL and has_position:
            if rsi < self.overbought:
                return StrategySignal.HOLD
        return base_signal


class TrendStrengthStrategy:
    """Require a minimum slope (based on moving averages) before allowing entries."""

    def __init__(
        self,
        base_strategy,
        window: int = 10,
        min_slope: Decimal | float = Decimal("0.0"),
    ) -> None:
        if window <= 1:
            raise ValueError("Trend window must be greater than 1")
        self.base_strategy = base_strategy
        self.window = window
        self.min_slope = Decimal(str(min_slope))
        self._closes: list[Decimal] = []

    def on_bar(self, bar: Bar, has_position: bool) -> str:
        self._closes.append(bar.close)
        base_signal = self.base_strategy.on_bar(bar, has_position)

        if len(self._closes) < self.window + 1:
            return base_signal if has_position else StrategySignal.HOLD

        current_ma = _mean(self._closes[-self.window :])
        prev_ma = _mean(self._closes[-(self.window + 1) : -1])
        slope = (current_ma - prev_ma) / Decimal(self.window)

        if base_signal == StrategySignal.BUY and not has_position:
            if slope < self.min_slope:
                return StrategySignal.HOLD
        if base_signal == StrategySignal.SELL and has_position:
            if slope > -self.min_slope:
                return StrategySignal.HOLD
        return base_signal


class SpotBacktester:
    def __init__(
        self,
        bars: list[Bar],
        strategy,
        config: SpotBacktestConfig | None = None,
    ) -> None:
        if not bars:
            raise ValueError("Backtester requires at least one bar")
        self.bars = bars
        self.strategy = strategy
        self.config = config or SpotBacktestConfig()

    def run(self) -> BacktestResult:
        position_quantity = Decimal("0")
        trades: list[dict] = []
        equity_records = []
        commission_factor = self.config.commission_bps / Decimal("10000")

        initial_cash = Decimal(str(self.config.initial_cash))
        cash = initial_cash

        for bar in self.bars:
            signal = self.strategy.on_bar(bar, has_position=(position_quantity > 0))
            price = bar.close

            if signal == StrategySignal.BUY and position_quantity == 0:
                quantity = (cash / price).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
                if quantity > 0:
                    cost = quantity * price
                    fee = cost * commission_factor
                    cash -= cost + fee
                    position_quantity = quantity
                    trades.append(
                        {
                            "timestamp": bar.timestamp,
                            "side": "buy",
                            "price": float(price),
                            "quantity": float(quantity),
                            "fee": float(fee),
                        }
                    )

            elif signal == StrategySignal.SELL and position_quantity > 0:
                proceeds = position_quantity * price
                fee = proceeds * commission_factor
                cash += proceeds - fee
                trades.append(
                    {
                        "timestamp": bar.timestamp,
                        "side": "sell",
                        "price": float(price),
                        "quantity": float(position_quantity),
                        "fee": float(fee),
                    }
                )
                position_quantity = Decimal("0")

            equity = cash + position_quantity * price
            equity_records.append(
                {
                    "timestamp": bar.timestamp,
                    "equity": float(equity),
                    "cash": float(cash),
                    "position_quantity": float(position_quantity),
                    "price": float(price),
                }
            )

        if position_quantity > 0:
            final_price = self.bars[-1].close
            proceeds = position_quantity * final_price
            fee = proceeds * commission_factor
            cash += proceeds - fee
            trades.append(
                {
                    "timestamp": self.bars[-1].timestamp,
                    "side": "sell",
                    "price": float(final_price),
                    "quantity": float(position_quantity),
                    "fee": float(fee),
                    "note": "forced_exit",
                }
            )
            position_quantity = Decimal("0")
            equity_records[-1]["equity"] = float(cash)
            equity_records[-1]["cash"] = float(cash)
            equity_records[-1]["position_quantity"] = 0.0

        equity_df = pd.DataFrame(equity_records)
        initial_capital_float = float(initial_cash)
        metrics = _compute_metrics(equity_df, initial_capital_float, self.config.risk_free_rate)
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades,
            initial_capital=initial_capital_float,
        )


def load_candles_from_parquet(parquet_path: Path) -> list[Bar]:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return []
    df = df.sort_values("timestamp")
    bars: list[Bar] = []
    for row in df.itertuples():
        ts = row.timestamp
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        bars.append(
            Bar(
                timestamp=ts,
                open=Decimal(str(row.open)),
                high=Decimal(str(row.high)),
                low=Decimal(str(row.low)),
                close=Decimal(str(row.close)),
                volume=Decimal(str(row.volume)),
            )
        )
    return bars


def _mean(values: Iterable[Decimal]) -> Decimal:
    if not isinstance(values, list):
        values = list(values)
    total = sum(values, Decimal("0"))
    count = len(values)
    if count == 0:
        return Decimal("0")
    return (total / Decimal(count)).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


def _stddev(values: Iterable[Decimal], mean: Decimal) -> Decimal:
    if not isinstance(values, list):
        values = list(values)
    count = len(values)
    if count <= 1:
        return Decimal("0")
    variance = sum((v - mean) ** 2 for v in values) / Decimal(count - 1)
    return variance.sqrt().quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


def _true_range(bar: Bar, prev_close: Decimal | None) -> Decimal:
    high_low = bar.high - bar.low
    if prev_close is None:
        return high_low
    high_close = (bar.high - prev_close).copy_abs()
    low_close = (bar.low - prev_close).copy_abs()
    return max(high_low, high_close, low_close)


def _rsi(closes: list[Decimal]) -> Decimal:
    gains = []
    losses = []
    for prev, curr in zip(closes[:-1], closes[1:], strict=False):
        delta = curr - prev
        if delta > 0:
            gains.append(delta)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(-delta)
    avg_gain = _mean(gains)
    avg_loss = _mean(losses)
    if avg_loss == 0:
        return Decimal("100")
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return Decimal(rsi).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


def _compute_metrics(
    equity_df: pd.DataFrame, initial_equity: float, risk_free_rate: float
) -> BacktestMetrics:
    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity / initial_equity) - 1.0

    start = equity_df["timestamp"].iloc[0]
    end = equity_df["timestamp"].iloc[-1]
    total_days = max((end - start).total_seconds() / 86400.0, 1.0)
    annualized_return = (
        (1 + total_return) ** (365.0 / total_days) - 1 if total_return > -1 else -1.0
    )

    equity_series = equity_df["equity"].values
    peaks = pd.Series(equity_series).cummax()
    drawdowns = (equity_series - peaks) / peaks
    max_drawdown = float(drawdowns.min()) if len(drawdowns) else 0.0

    returns = pd.Series(equity_series).pct_change().dropna()
    if returns.empty:
        sharpe = 0.0
    else:
        excess = returns - (risk_free_rate / 252.0)
        sharpe = float(math.sqrt(252) * excess.mean() / (excess.std() + 1e-9))

    return BacktestMetrics(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        max_drawdown=float(max_drawdown),
        sharpe_ratio=sharpe,
    )
