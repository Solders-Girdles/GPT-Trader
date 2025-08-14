"""
Enhanced trend breakout strategy with expanded parameter space.
Utilizes volume, momentum, volatility, and time-based filters for more sophisticated signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from bot.indicators.enhanced import (
    calculate_all_enhanced_indicators,
    correlation_filter,
    regime_filter,
)
from bot.strategy.base import Strategy


@dataclass
class EnhancedTrendBreakoutParams:
    """Enhanced trend breakout parameters with expanded feature set."""

    # Core trend parameters
    donchian_lookback: int = 55
    atr_period: int = 20
    atr_k: float = 2.0

    # Volume-based features
    volume_ma_period: int = 20
    volume_threshold: float = 1.5
    use_volume_filter: bool = True

    # Momentum features
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    use_rsi_filter: bool = False

    # Volatility features
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    use_bollinger_filter: bool = False

    # Time-based features
    day_of_week_filter: int | None = None  # 0=Monday, 4=Friday
    month_filter: int | None = None  # 1-12
    use_time_filter: bool = False

    # Entry/Exit enhancements
    entry_confirmation_periods: int = 1
    exit_confirmation_periods: int = 1
    cooldown_periods: int = 0

    # Risk management
    max_risk_per_trade: float = 0.02
    position_sizing_method: str = "atr"  # "atr", "fixed", "kelly"

    # Advanced features
    use_regime_filter: bool = False
    regime_lookback: int = 200
    use_correlation_filter: bool = False
    correlation_threshold: float = 0.7
    correlation_lookback: int = 60


class EnhancedTrendBreakoutStrategy(Strategy):
    """Enhanced trend breakout strategy with expanded parameter space."""

    name = "enhanced_trend_breakout"
    supports_short = True  # Now supports short positions

    def __init__(self, params: EnhancedTrendBreakoutParams | None = None) -> None:
        self.params = params or EnhancedTrendBreakoutParams()
        self.last_signal = 0
        self.confirmation_count = 0
        self.cooldown_counter = 0

    def generate_signals(
        self, bars: pd.DataFrame, market_data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Generate enhanced trading signals using multiple filters and indicators."""
        df = bars.copy()

        # Calculate all enhanced indicators
        indicators = calculate_all_enhanced_indicators(df, self.params.__dict__)

        # Initialize signal column
        df["signal"] = 0

        # Core breakout signal
        breakout_signal = indicators["breakout_signal"]

        # Apply volume filter
        if self.params.use_volume_filter:
            volume_filter = indicators["volume_breakout"]
            breakout_signal = breakout_signal & volume_filter

        # Apply RSI filter
        if self.params.use_rsi_filter:
            rsi = indicators["rsi"]
            rsi_filter = (rsi < self.params.rsi_oversold) | (rsi > self.params.rsi_overbought)
            breakout_signal = breakout_signal & rsi_filter

        # Apply Bollinger Bands filter
        if self.params.use_bollinger_filter:
            bb_upper = indicators["bb_upper"]
            bb_lower = indicators["bb_lower"]
            bb_filter = (df["Close"] > bb_upper) | (df["Close"] < bb_lower)
            breakout_signal = breakout_signal & bb_filter

        # Apply time filter
        if self.params.use_time_filter:
            time_filter = indicators["time_filter"]
            breakout_signal = breakout_signal & time_filter

        # Apply regime filter
        if self.params.use_regime_filter:
            regime_ok = regime_filter(df, self.params.regime_lookback)
            breakout_signal = breakout_signal & regime_ok

        # Apply correlation filter
        if self.params.use_correlation_filter and market_data is not None:
            correlation_ok = correlation_filter(
                df, market_data, self.params.correlation_lookback, self.params.correlation_threshold
            )
            breakout_signal = breakout_signal & correlation_ok

        # Multi-timeframe confirmation
        multi_tf_signal = indicators["multi_tf_signal"]
        mtf_confirmation = (breakout_signal == 1) & (multi_tf_signal >= 0)
        mtf_confirmation = mtf_confirmation | ((breakout_signal == -1) & (multi_tf_signal <= 0))

        # Volatility regime adjustment
        vol_regime = indicators["volatility_regime"]
        volatility_adjustment = self._adjust_for_volatility_regime(vol_regime)

        # Support/Resistance levels
        support = indicators["support"]
        resistance = indicators["resistance"]
        sr_confirmation = self._check_support_resistance(df, support, resistance)

        # Combine all signals
        final_signal = self._combine_signals(
            breakout_signal, mtf_confirmation, volatility_adjustment, sr_confirmation
        )

        # Apply confirmation periods
        confirmed_signal = self._apply_confirmation_periods(final_signal)

        # Apply cooldown
        final_signal = self._apply_cooldown(confirmed_signal)

        # Store signals in dataframe
        df["signal"] = final_signal
        df["breakout_signal"] = breakout_signal
        df["volume_filter"] = indicators.get("volume_breakout", pd.Series(True, index=df.index))
        df["rsi"] = indicators.get("rsi", pd.Series(50, index=df.index))
        df["atr"] = indicators["atr"]
        df["volatility_regime"] = indicators["volatility_regime"]
        df["multi_tf_signal"] = indicators["multi_tf_signal"]
        df["support"] = indicators["support"]
        df["resistance"] = indicators["resistance"]

        return df[
            [
                "signal",
                "breakout_signal",
                "volume_filter",
                "rsi",
                "atr",
                "volatility_regime",
                "multi_tf_signal",
                "support",
                "resistance",
            ]
        ]

    def _adjust_for_volatility_regime(self, vol_regime: pd.Series) -> pd.Series:
        """Adjust signals based on volatility regime."""
        adjustment = pd.Series(1.0, index=vol_regime.index)

        # High volatility: reduce signal strength
        adjustment[vol_regime == 2] = 0.7

        # Low volatility: increase signal strength
        adjustment[vol_regime == 0] = 1.3

        return adjustment

    def _check_support_resistance(
        self, df: pd.DataFrame, support: pd.Series, resistance: pd.Series
    ) -> pd.Series:
        """Check if price is near support/resistance levels."""
        sr_confirmation = pd.Series(True, index=df.index)

        # Near resistance: confirm short signals
        near_resistance = df["Close"] > resistance * 0.98
        sr_confirmation = sr_confirmation & (~near_resistance | (df["signal"] <= 0))

        # Near support: confirm long signals
        near_support = df["Close"] < support * 1.02
        sr_confirmation = sr_confirmation & (~near_support | (df["signal"] >= 0))

        return sr_confirmation

    def _combine_signals(
        self,
        breakout_signal: pd.Series,
        mtf_confirmation: pd.Series,
        volatility_adjustment: pd.Series,
        sr_confirmation: pd.Series,
    ) -> pd.Series:
        """Combine multiple signals into final signal."""
        # Start with breakout signal
        combined_signal = breakout_signal.copy()

        # Apply multi-timeframe confirmation
        combined_signal = combined_signal * mtf_confirmation.astype(int)

        # Apply support/resistance confirmation
        combined_signal = combined_signal * sr_confirmation.astype(int)

        # Apply volatility adjustment
        combined_signal = combined_signal * volatility_adjustment

        # Convert to integer signals
        final_signal = pd.Series(0, index=combined_signal.index)
        final_signal[combined_signal > 0.5] = 1
        final_signal[combined_signal < -0.5] = -1

        return final_signal

    def _apply_confirmation_periods(self, signal: pd.Series) -> pd.Series:
        """Apply entry and exit confirmation periods."""
        confirmed_signal = signal.copy()

        # Entry confirmation
        if self.params.entry_confirmation_periods > 1:
            for i in range(len(signal)):
                if i < self.params.entry_confirmation_periods - 1:
                    continue

                # Check if we have enough consecutive signals
                recent_signals = signal.iloc[i - self.params.entry_confirmation_periods + 1 : i + 1]
                if not (recent_signals == 1).all() and not (recent_signals == -1).all():
                    confirmed_signal.iloc[i] = 0

        # Exit confirmation
        if self.params.exit_confirmation_periods > 1:
            for i in range(len(signal)):
                if i < self.params.exit_confirmation_periods - 1:
                    continue

                # Check if we have enough consecutive exit signals
                recent_signals = signal.iloc[i - self.params.exit_confirmation_periods + 1 : i + 1]
                if not (recent_signals == 0).all():
                    # Don't exit yet
                    if confirmed_signal.iloc[i] == 0:
                        confirmed_signal.iloc[i] = confirmed_signal.iloc[i - 1]

        return confirmed_signal

    def _apply_cooldown(self, signal: pd.Series) -> pd.Series:
        """Apply cooldown periods between trades."""
        if self.params.cooldown_periods <= 0:
            return signal

        final_signal = signal.copy()
        cooldown_counter = 0

        for i in range(len(signal)):
            if cooldown_counter > 0:
                final_signal.iloc[i] = 0
                cooldown_counter -= 1
            elif signal.iloc[i] != 0:
                cooldown_counter = self.params.cooldown_periods

        return final_signal

    def get_position_size(self, current_price: float, atr: float, account_value: float) -> float:
        """Calculate position size based on risk management parameters."""
        if self.params.position_sizing_method == "atr":
            # ATR-based position sizing
            risk_amount = account_value * self.params.max_risk_per_trade
            position_size = risk_amount / (atr * self.params.atr_k)

        elif self.params.position_sizing_method == "fixed":
            # Fixed percentage of account
            position_size = account_value * self.params.max_risk_per_trade / current_price

        elif self.params.position_sizing_method == "kelly":
            # Kelly criterion (simplified)
            # This would need historical win rate and average win/loss
            position_size = account_value * self.params.max_risk_per_trade * 0.5 / current_price

        else:
            # Default to ATR-based
            risk_amount = account_value * self.params.max_risk_per_trade
            position_size = risk_amount / (atr * self.params.atr_k)

        return max(0, position_size)

    def get_stop_loss(self, entry_price: float, atr: float, signal: int) -> float:
        """Calculate stop loss based on ATR."""
        if signal > 0:  # Long position
            return entry_price - (atr * self.params.atr_k)
        elif signal < 0:  # Short position
            return entry_price + (atr * self.params.atr_k)
        else:
            return entry_price

    def get_take_profit(self, entry_price: float, atr: float, signal: int) -> float:
        """Calculate take profit based on ATR."""
        if signal > 0:  # Long position
            return entry_price + (atr * self.params.atr_k * 2)  # 2:1 reward/risk
        elif signal < 0:  # Short position
            return entry_price - (atr * self.params.atr_k * 2)
        else:
            return entry_price
