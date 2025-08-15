from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from bot.indicators.atr import atr
from bot.indicators.optimized import OptimizedIndicators

from .base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionParams:
    """Parameters for the Mean Reversion strategy."""

    rsi_period: int = 14
    oversold_threshold: float = 30.0
    overbought_threshold: float = 70.0
    atr_period: int = 14
    exit_rsi_threshold: float = 50.0  # Exit when RSI crosses back toward 50


def _safe_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Graceful fallback if High/Low missing."""
    work = df.copy()
    if "High" not in work.columns or "Low" not in work.columns:
        work["High"] = work["Close"].astype(float)
        work["Low"] = work["Close"].astype(float)
    return atr(work, period=period, method="wilder")


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Trading Strategy using RSI.

    Strategy Logic:
    - Buy (Long) when RSI drops below oversold threshold (default 30)
    - Sell (Exit) when RSI rises above exit threshold (default 50) or hits overbought (default 70)
    - Uses RSI as the primary mean reversion indicator
    - Includes ATR for position sizing and risk management

    The strategy assumes that extreme RSI readings (oversold/overbought) tend to revert
    back toward the mean (RSI 50), providing trading opportunities.
    """

    name = "mean_reversion"
    supports_short = False  # Long-only strategy for mean reversion

    def __init__(self, params: MeanReversionParams | None = None) -> None:
        """
        Initialize the Mean Reversion strategy.

        Args:
            params: Strategy parameters. If None, uses default MeanReversionParams.
        """
        self.params = params or MeanReversionParams()
        logger.info(
            f"Initialized {self.name} strategy with RSI period={self.params.rsi_period}, "
            f"oversold={self.params.oversold_threshold}, overbought={self.params.overbought_threshold}"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI mean reversion.

        Args:
            df: OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']

        Returns:
            DataFrame with columns:
            - signal: 1 for buy/long, 0 for flat/exit
            - rsi: RSI values
            - atr: Average True Range values
            - oversold_signal: 1 when RSI is oversold
            - overbought_signal: 1 when RSI is overbought
        """
        if df.empty or len(df) < self.params.rsi_period + 1:
            logger.warning(
                f"Insufficient data for RSI calculation. Need at least {self.params.rsi_period + 1} bars"
            )
            return pd.DataFrame(
                {
                    "signal": np.zeros(len(df)),
                    "rsi": np.full(len(df), np.nan),
                    "atr": np.full(len(df), np.nan),
                    "oversold_signal": np.zeros(len(df)),
                    "overbought_signal": np.zeros(len(df)),
                },
                index=df.index,
            )

        # Calculate RSI using optimized implementation
        rsi_values = OptimizedIndicators.rsi(df["Close"], period=self.params.rsi_period)

        # Calculate ATR for risk management
        atr_values = _safe_atr(df, self.params.atr_period)

        # Generate mean reversion signals
        signals = np.zeros(len(df))
        oversold_signals = np.zeros(len(df))
        overbought_signals = np.zeros(len(df))

        # Track current position state (0 = flat, 1 = long)
        position = 0

        for i in range(len(df)):
            current_rsi = rsi_values.iloc[i] if not pd.isna(rsi_values.iloc[i]) else np.nan

            if pd.isna(current_rsi):
                signals[i] = 0
                continue

            # Detect oversold condition (potential buy signal)
            if current_rsi <= self.params.oversold_threshold:
                oversold_signals[i] = 1
                if position == 0:  # Enter long position
                    signals[i] = 1
                    position = 1
                    logger.debug(f"RSI oversold signal at {df.index[i]}: RSI={current_rsi:.2f}")
                else:
                    signals[i] = 1  # Stay long

            # Detect overbought condition or exit threshold
            elif (
                current_rsi >= self.params.overbought_threshold
                or current_rsi >= self.params.exit_rsi_threshold
            ):
                if current_rsi >= self.params.overbought_threshold:
                    overbought_signals[i] = 1

                if position == 1:  # Exit long position
                    signals[i] = 0
                    position = 0
                    logger.debug(f"RSI exit signal at {df.index[i]}: RSI={current_rsi:.2f}")
                else:
                    signals[i] = 0  # Stay flat

            else:
                # Maintain current position in neutral zone
                signals[i] = position

        # Create output DataFrame
        result = pd.DataFrame(
            {
                "signal": signals,
                "rsi": rsi_values,
                "atr": atr_values,
                "oversold_signal": oversold_signals,
                "overbought_signal": overbought_signals,
            },
            index=df.index,
        )

        # Zero out signals until indicators are ready
        ready_mask = (~result["rsi"].isna()) & (~result["atr"].isna())
        result.loc[~ready_mask, "signal"] = 0.0

        # Log strategy statistics
        total_signals = int(result["signal"].sum())
        oversold_count = int(result["oversold_signal"].sum())
        overbought_count = int(result["overbought_signal"].sum())

        logger.info(
            f"Generated {total_signals} total signals, "
            f"{oversold_count} oversold conditions, "
            f"{overbought_count} overbought conditions"
        )

        return result

    def __str__(self) -> str:
        """String representation of the strategy."""
        return (
            f"MeanReversionStrategy(rsi_period={self.params.rsi_period}, "
            f"oversold={self.params.oversold_threshold}, "
            f"overbought={self.params.overbought_threshold})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the strategy."""
        return self.__str__()
