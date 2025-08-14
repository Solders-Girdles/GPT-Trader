"""
Optimized Moving Average Strategy with Full Vectorization

This strategy provides maximum performance through:
- Complete pandas/numpy vectorization
- Optimized indicator calculations
- Minimal memory allocation
- Efficient signal generation
- Advanced risk management integration
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Strategy


@dataclass
class OptimizedMAParams:
    """Optimized MA strategy parameters"""

    fast: int = 10
    slow: int = 20
    atr_period: int = 14
    volume_filter: bool = True
    volume_ma_period: int = 20
    rsi_filter: bool = False
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Advanced features
    trend_strength_filter: bool = True
    volatility_adjustment: bool = True
    position_sizing: str = "atr"  # "fixed", "atr", "volatility"
    max_risk_per_trade: float = 0.02


class OptimizedMAStrategy(Strategy):
    """
    Highly optimized moving average strategy with vectorized operations.

    Performance optimizations:
    - All operations use pandas vectorization
    - Minimal intermediate DataFrame creation
    - Efficient memory usage
    - Pre-computed indicators for reuse
    """

    name = "optimized_ma"
    supports_short = True

    def __init__(self, params: OptimizedMAParams | None = None) -> None:
        self.params = params or OptimizedMAParams()

        # Validate parameters
        if self.params.fast >= self.params.slow:
            raise ValueError("Fast period must be less than slow period")

        # Cache for expensive calculations
        self._indicator_cache = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with full vectorization"""

        # Pre-allocate result DataFrame for memory efficiency
        len(df)
        result = pd.DataFrame(index=df.index)

        # Calculate core indicators (vectorized)
        close = df["Close"].astype(np.float64)  # Ensure numeric type

        # Moving averages with optimized calculation
        sma_fast = self._calculate_sma_optimized(close, self.params.fast)
        sma_slow = self._calculate_sma_optimized(close, self.params.slow)

        # ATR calculation with vectorization
        atr_values = self._calculate_atr_optimized(df, self.params.atr_period)

        # Base signal generation (vectorized)
        ma_signal = self._generate_ma_signals(sma_fast, sma_slow)

        # Apply additional filters if enabled
        final_signal = ma_signal.copy()

        if self.params.volume_filter:
            volume_filter = self._calculate_volume_filter(df)
            final_signal = final_signal & volume_filter

        if self.params.rsi_filter:
            rsi_filter = self._calculate_rsi_filter(close)
            final_signal = final_signal & rsi_filter

        if self.params.trend_strength_filter:
            trend_filter = self._calculate_trend_strength_filter(sma_fast, sma_slow)
            final_signal = final_signal & trend_filter

        # Convert boolean signals to numeric
        signal_numeric = np.where(final_signal, 1.0, 0.0)

        # Apply volatility adjustment if enabled
        if self.params.volatility_adjustment:
            vol_adjustment = self._calculate_volatility_adjustment(atr_values, close)
            signal_numeric = signal_numeric * vol_adjustment

        # Ensure no look-ahead bias
        min_periods = max(self.params.fast, self.params.slow, self.params.atr_period)
        signal_numeric[:min_periods] = 0.0

        # Build result DataFrame
        result["signal"] = signal_numeric
        result["sma_fast"] = sma_fast
        result["sma_slow"] = sma_slow
        result["atr"] = atr_values

        # Add additional columns for analysis
        result["signal_strength"] = np.abs(sma_fast - sma_slow) / close
        result["volatility_regime"] = self._classify_volatility_regime(atr_values)

        if self.params.volume_filter:
            result["volume_strength"] = (
                df["Volume"] / df["Volume"].rolling(self.params.volume_ma_period).mean()
            )

        return result

    def _calculate_sma_optimized(self, prices: pd.Series, period: int) -> pd.Series:
        """Optimized SMA calculation with caching"""
        cache_key = f"sma_{period}_{hash(tuple(prices))}"

        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]

        # Use pandas rolling with min_periods for efficiency
        sma = prices.rolling(window=period, min_periods=period).mean()

        # Cache result for potential reuse
        self._indicator_cache[cache_key] = sma

        return sma

    def _calculate_atr_optimized(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Optimized ATR calculation"""
        # Vectorized True Range calculation
        high = df["High"].astype(np.float64)
        low = df["Low"].astype(np.float64)
        close = df["Close"].astype(np.float64)
        prev_close = close.shift(1)

        # Calculate all three TR components at once
        tr_components = np.column_stack(
            [
                (high - low).values,
                np.abs((high - prev_close).values),
                np.abs((low - prev_close).values),
            ]
        )

        # Take maximum across columns (vectorized)
        true_range = pd.Series(np.nanmax(tr_components, axis=1), index=df.index)

        # Use Wilder's smoothing (more efficient than simple MA for ATR)
        return true_range.ewm(alpha=1.0 / period, adjust=False).mean()

    def _generate_ma_signals(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """Generate MA crossover signals (vectorized)"""
        # Simple crossover: fast > slow = bullish
        return fast_ma > slow_ma

    def _calculate_volume_filter(self, df: pd.DataFrame) -> pd.Series:
        """Volume confirmation filter (vectorized)"""
        volume = df["Volume"].astype(np.float64)
        volume_ma = volume.rolling(self.params.volume_ma_period).mean()

        # Require above-average volume for signals
        return volume > volume_ma

    def _calculate_rsi_filter(self, prices: pd.Series) -> pd.Series:
        """RSI-based filter (vectorized)"""
        # Vectorized RSI calculation
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use exponential moving average for RSI smoothing
        avg_gain = gain.ewm(alpha=1.0 / self.params.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.params.rsi_period).mean()

        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Filter: avoid extreme overbought/oversold
        return (rsi > self.params.rsi_oversold) & (rsi < self.params.rsi_overbought)

    def _calculate_trend_strength_filter(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """Trend strength confirmation (vectorized)"""
        # Calculate MA spread as % of price
        spread = np.abs(fast_ma - slow_ma) / slow_ma

        # Require minimum trend strength (1% spread)
        min_strength = 0.01
        return spread > min_strength

    def _calculate_volatility_adjustment(self, atr: pd.Series, prices: pd.Series) -> pd.Series:
        """Volatility-based position sizing adjustment"""
        # Calculate volatility as % of price
        volatility_pct = atr / prices

        # Adjust position size inversely with volatility
        # Higher volatility = smaller position
        vol_median = volatility_pct.rolling(50).median()
        vol_adjustment = vol_median / volatility_pct

        # Cap adjustment between 0.5x and 2.0x
        return np.clip(vol_adjustment, 0.5, 2.0)

    def _classify_volatility_regime(self, atr: pd.Series) -> pd.Series:
        """Classify volatility regime (vectorized)"""
        # Calculate rolling volatility percentiles
        vol_20pct = atr.rolling(100).quantile(0.2)
        vol_80pct = atr.rolling(100).quantile(0.8)

        # Classify regimes: 0=low, 1=normal, 2=high
        regime = np.where(atr <= vol_20pct, 0, np.where(atr >= vol_80pct, 2, 1))

        return pd.Series(regime, index=atr.index)

    def get_position_size(self, current_price: float, atr: float, account_value: float) -> float:
        """Calculate optimized position size"""
        if self.params.position_sizing == "fixed":
            return account_value * self.params.max_risk_per_trade / current_price

        elif self.params.position_sizing == "atr":
            # ATR-based position sizing
            risk_amount = account_value * self.params.max_risk_per_trade
            position_size = risk_amount / (atr * 2.0)  # 2x ATR stop

        elif self.params.position_sizing == "volatility":
            # Volatility-adjusted position sizing
            vol_factor = atr / current_price
            base_size = account_value * self.params.max_risk_per_trade / current_price
            position_size = base_size / vol_factor

        else:
            # Default to ATR-based
            risk_amount = account_value * self.params.max_risk_per_trade
            position_size = risk_amount / (atr * 2.0)

        return max(0, position_size)

    def get_stop_loss(self, entry_price: float, atr: float, signal: int) -> float:
        """Calculate dynamic stop loss"""
        atr_multiplier = 2.0

        if signal > 0:  # Long position
            return entry_price - (atr * atr_multiplier)
        elif signal < 0:  # Short position
            return entry_price + (atr * atr_multiplier)
        else:
            return entry_price

    def get_take_profit(self, entry_price: float, atr: float, signal: int) -> float:
        """Calculate dynamic take profit"""
        atr_multiplier = 4.0  # 2:1 reward/risk ratio

        if signal > 0:  # Long position
            return entry_price + (atr * atr_multiplier)
        elif signal < 0:  # Short position
            return entry_price - (atr * atr_multiplier)
        else:
            return entry_price

    def clear_cache(self) -> None:
        """Clear indicator cache to free memory"""
        self._indicator_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring"""
        return {
            "cache_size": len(self._indicator_cache),
            "memory_usage_estimate": sum(
                getattr(val, "memory_usage", lambda: 0)() for val in self._indicator_cache.values()
            ),
        }


def benchmark_optimized_strategy():
    """Benchmark the optimized strategy performance"""
    import time

    # Generate test data
    np.random.seed(42)
    n_days = 5000
    dates = pd.date_range(start="2010-01-01", periods=n_days, freq="D")

    # Generate realistic price data
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)
    highs = prices * np.random.uniform(1.005, 1.02, n_days)
    lows = prices * np.random.uniform(0.98, 0.995, n_days)
    volumes = np.random.lognormal(15, 0.5, n_days).astype(int)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        }
    ).set_index("Date")

    # Test different parameter configurations
    configs = [
        OptimizedMAParams(fast=5, slow=20),
        OptimizedMAParams(fast=10, slow=30, volume_filter=True),
        OptimizedMAParams(fast=20, slow=50, rsi_filter=True, trend_strength_filter=True),
    ]

    results = []

    for i, params in enumerate(configs):
        print(f"\n🔧 Testing configuration {i+1}: MA({params.fast}, {params.slow})")

        strategy = OptimizedMAStrategy(params)

        # Benchmark execution time
        start_time = time.time()
        signals = strategy.generate_signals(df)
        execution_time = time.time() - start_time

        # Analyze results
        signal_count = signals["signal"].sum()
        cache_stats = strategy.get_cache_stats()

        result = {
            "config": i + 1,
            "params": params,
            "execution_time": execution_time,
            "signal_count": signal_count,
            "signal_rate": signal_count / len(df),
            "cache_size": cache_stats["cache_size"],
            "throughput": len(df) / execution_time,
        }

        results.append(result)

        print(f"   ⏱️  Execution time: {execution_time:.4f}s")
        print(f"   📊 Signal count: {signal_count}/{len(df)} ({signal_count/len(df):.1%})")
        print(f"   🚀 Throughput: {len(df)/execution_time:,.0f} rows/second")
        print(f"   💾 Cache usage: {cache_stats['cache_size']} items")

    # Summary
    avg_throughput = sum(r["throughput"] for r in results) / len(results)
    print("\n📈 BENCHMARK SUMMARY:")
    print(f"   Average throughput: {avg_throughput:,.0f} rows/second")
    print(f"   Test dataset: {len(df):,} rows")
    print(f"   Configurations tested: {len(configs)}")

    return results


if __name__ == "__main__":
    print("🚀 Optimized MA Strategy Benchmark")
    print("=" * 50)

    results = benchmark_optimized_strategy()

    print("\n✅ Benchmark complete!")
    print("   Ready for production optimization")
