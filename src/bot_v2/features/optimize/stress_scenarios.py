"""
Stress test scenario generators for derivatives backtesting.

Generates synthetic market stress conditions to test system resilience:
- Gap moves (overnight gaps, flash crashes)
- High volatility periods
- Funding rate shocks
- Liquidity crises
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="optimize")


class StressScenarioType(Enum):
    """Types of stress scenarios."""

    GAP_MOVE = "gap_move"
    HIGH_VOLATILITY = "high_volatility"
    FUNDING_SHOCK = "funding_shock"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"
    TREND_REVERSAL = "trend_reversal"


class StressScenarioGenerator:
    """Base class for stress scenario generators."""

    def __init__(self, seed: int | None = None):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stress scenario to data.

        Args:
            data: Historical OHLC data

        Returns:
            Modified data with stress scenario applied
        """
        raise NotImplementedError


class GapMoveScenario(StressScenarioGenerator):
    """
    Injects gap moves (overnight gaps or flash crashes).

    Simulates scenarios where price jumps instantly without intermediate prices,
    testing liquidation and stop-loss logic.
    """

    def __init__(
        self,
        *,
        gap_size_pct: float = 0.05,  # 5% gap
        gap_direction: str = "down",  # "up" or "down"
        gap_location: str = "random",  # "random", "overnight", or specific index
        num_gaps: int = 1,
        seed: int | None = None,
    ):
        """
        Initialize gap move scenario.

        Args:
            gap_size_pct: Size of gap as percentage (0.05 = 5%)
            gap_direction: "up", "down", or "random"
            gap_location: Where to inject gaps ("random", "overnight", or int index)
            num_gaps: Number of gaps to inject
            seed: Random seed
        """
        super().__init__(seed)
        self.gap_size_pct = gap_size_pct
        self.gap_direction = gap_direction
        self.gap_location = gap_location
        self.num_gaps = num_gaps

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply gap moves to data."""
        data = data.copy()

        # Determine gap locations
        if self.gap_location == "random":
            gap_indices = np.random.choice(
                range(1, len(data)), size=min(self.num_gaps, len(data) - 1), replace=False
            )
        elif self.gap_location == "overnight":
            # Find overnight periods (assume daily data or gaps between trading sessions)
            gap_indices = self._find_overnight_gaps(data)
            gap_indices = gap_indices[: self.num_gaps]
        elif isinstance(self.gap_location, int):
            gap_indices = [self.gap_location]
        else:
            gap_indices = []

        # Apply gaps
        for idx in gap_indices:
            direction = self.gap_direction
            if direction == "random":
                direction = np.random.choice(["up", "down"])

            multiplier = 1 + self.gap_size_pct if direction == "up" else 1 - self.gap_size_pct

            # Apply gap to OHLC
            data.loc[idx:, "open"] *= multiplier
            data.loc[idx:, "high"] *= multiplier
            data.loc[idx:, "low"] *= multiplier
            data.loc[idx:, "close"] *= multiplier

            logger.info(
                "Gap move injected | idx=%d | direction=%s | size=%.2f%%",
                idx,
                direction,
                self.gap_size_pct * 100,
            )

        return data

    def _find_overnight_gaps(self, data: pd.DataFrame) -> list[int]:
        """Find indices that could represent overnight gaps."""
        if "timestamp" not in data.columns:
            # Assume uniform spacing, pick every ~24 bars if hourly
            return list(range(24, len(data), 24))

        # Find gaps > 12 hours in timestamp
        timestamps = pd.to_datetime(data["timestamp"])
        time_diffs = timestamps.diff()
        overnight_mask = time_diffs > timedelta(hours=12)
        return overnight_mask[overnight_mask].index.tolist()


class HighVolatilityScenario(StressScenarioGenerator):
    """
    Increases volatility to test system behavior in choppy markets.

    Multiplies price movements by a volatility factor while maintaining
    overall trend direction.
    """

    def __init__(
        self,
        *,
        volatility_multiplier: float = 2.0,  # 2x normal volatility
        start_pct: float = 0.25,  # Start at 25% through data
        duration_pct: float = 0.30,  # Last for 30% of data
        seed: int | None = None,
    ):
        """
        Initialize high volatility scenario.

        Args:
            volatility_multiplier: Factor to multiply volatility by
            start_pct: Where to start high vol period (0.0 to 1.0)
            duration_pct: Duration of high vol period (0.0 to 1.0)
            seed: Random seed
        """
        super().__init__(seed)
        self.volatility_multiplier = volatility_multiplier
        self.start_pct = start_pct
        self.duration_pct = duration_pct

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply high volatility to data."""
        data = data.copy()

        # Calculate volatility period
        start_idx = int(len(data) * self.start_pct)
        end_idx = int(start_idx + len(data) * self.duration_pct)

        # Get baseline volatility (standard deviation of returns)
        returns = data["close"].pct_change()
        baseline_vol = returns.std()

        # Apply volatility multiplier to the selected period
        for idx in range(start_idx, min(end_idx, len(data))):
            if idx == 0:
                continue

            # Calculate return
            ret = (data.loc[idx, "close"] - data.loc[idx - 1, "close"]) / data.loc[
                idx - 1, "close"
            ]

            # Amplify return
            amplified_ret = ret * self.volatility_multiplier

            # Apply amplified return
            prev_close = data.loc[idx - 1, "close"]
            new_close = prev_close * (1 + amplified_ret)

            # Adjust OHLC proportionally
            ratio = new_close / data.loc[idx, "close"]
            data.loc[idx, "open"] *= ratio
            data.loc[idx, "high"] *= ratio
            data.loc[idx, "low"] *= ratio
            data.loc[idx, "close"] = new_close

        logger.info(
            "High volatility injected | start_idx=%d | end_idx=%d | multiplier=%.1fx | baseline_vol=%.4f",
            start_idx,
            end_idx,
            self.volatility_multiplier,
            baseline_vol,
        )

        return data


class FundingShockScenario(StressScenarioGenerator):
    """
    Simulates funding rate shocks for perpetual futures.

    Returns a funding rate schedule that can be applied during backtesting.
    """

    def __init__(
        self,
        *,
        baseline_rate: float = 0.0001,  # 0.01% per 8h (normal)
        shock_rate: float = 0.02,  # 2% per 8h (extreme)
        shock_start_pct: float = 0.50,  # Start shock at 50% through backtest
        shock_duration_hours: int = 24,  # Duration of shock
        seed: int | None = None,
    ):
        """
        Initialize funding shock scenario.

        Args:
            baseline_rate: Normal funding rate per interval
            shock_rate: Shocked funding rate per interval
            shock_start_pct: When to start shock (0.0 to 1.0)
            shock_duration_hours: How long shock lasts
            seed: Random seed
        """
        super().__init__(seed)
        self.baseline_rate = Decimal(str(baseline_rate))
        self.shock_rate = Decimal(str(shock_rate))
        self.shock_start_pct = shock_start_pct
        self.shock_duration_hours = shock_duration_hours

    def generate_funding_schedule(
        self, data: pd.DataFrame
    ) -> dict[datetime, Decimal]:
        """
        Generate funding rate schedule.

        Args:
            data: Historical data with timestamps

        Returns:
            Dict mapping timestamp -> funding_rate
        """
        if "timestamp" not in data.columns:
            raise ValueError("Data must have 'timestamp' column for funding schedule")

        timestamps = pd.to_datetime(data["timestamp"])
        start_time = timestamps.iloc[0]
        end_time = timestamps.iloc[-1]

        # Calculate shock period
        duration = end_time - start_time
        shock_start_time = start_time + duration * self.shock_start_pct
        shock_end_time = shock_start_time + timedelta(hours=self.shock_duration_hours)

        # Build schedule (funding every 8 hours at 00:00, 08:00, 16:00 UTC)
        schedule = {}
        current = start_time.replace(hour=(start_time.hour // 8) * 8, minute=0, second=0)

        while current <= end_time:
            if shock_start_time <= current <= shock_end_time:
                schedule[current] = self.shock_rate
            else:
                schedule[current] = self.baseline_rate

            current += timedelta(hours=8)

        logger.info(
            "Funding shock schedule | shock_start=%s | shock_end=%s | baseline=%.4f%% | shock=%.4f%%",
            shock_start_time,
            shock_end_time,
            float(self.baseline_rate) * 100,
            float(self.shock_rate) * 100,
        )

        return schedule


class LiquidityCrisisScenario(StressScenarioGenerator):
    """
    Simulates liquidity crisis with increased slippage and wider spreads.

    Increases slippage and commission to simulate difficult execution conditions.
    """

    def __init__(
        self,
        *,
        slippage_multiplier: float = 5.0,  # 5x normal slippage
        spread_multiplier: float = 3.0,  # 3x normal spread
        crisis_start_pct: float = 0.40,
        crisis_duration_pct: float = 0.20,
        seed: int | None = None,
    ):
        """
        Initialize liquidity crisis scenario.

        Args:
            slippage_multiplier: Factor to multiply slippage by
            spread_multiplier: Factor to multiply bid-ask spread by
            crisis_start_pct: When crisis starts (0.0 to 1.0)
            crisis_duration_pct: Duration of crisis (0.0 to 1.0)
            seed: Random seed
        """
        super().__init__(seed)
        self.slippage_multiplier = slippage_multiplier
        self.spread_multiplier = spread_multiplier
        self.crisis_start_pct = crisis_start_pct
        self.crisis_duration_pct = crisis_duration_pct

    def get_crisis_schedule(
        self, data_length: int
    ) -> dict[str, tuple[int, int, float, float]]:
        """
        Get crisis schedule.

        Args:
            data_length: Length of data

        Returns:
            Dict with crisis parameters
        """
        start_idx = int(data_length * self.crisis_start_pct)
        end_idx = int(start_idx + data_length * self.crisis_duration_pct)

        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "slippage_multiplier": self.slippage_multiplier,
            "spread_multiplier": self.spread_multiplier,
        }


class FlashCrashScenario(StressScenarioGenerator):
    """
    Simulates a flash crash: rapid price drop followed by partial recovery.

    Tests system behavior during extreme, rapid market moves.
    """

    def __init__(
        self,
        *,
        crash_size_pct: float = 0.20,  # 20% drop
        recovery_pct: float = 0.50,  # Recover 50% of drop
        crash_duration_bars: int = 5,  # Crash over 5 bars
        recovery_duration_bars: int = 20,  # Recover over 20 bars
        crash_location_pct: float = 0.60,  # Crash at 60% through data
        seed: int | None = None,
    ):
        """
        Initialize flash crash scenario.

        Args:
            crash_size_pct: Size of price drop (0.20 = 20%)
            recovery_pct: How much to recover (0.50 = 50% of drop)
            crash_duration_bars: Bars for crash to occur
            recovery_duration_bars: Bars for recovery
            crash_location_pct: Where in data to inject crash
            seed: Random seed
        """
        super().__init__(seed)
        self.crash_size_pct = crash_size_pct
        self.recovery_pct = recovery_pct
        self.crash_duration_bars = crash_duration_bars
        self.recovery_duration_bars = recovery_duration_bars
        self.crash_location_pct = crash_location_pct

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply flash crash to data."""
        data = data.copy()

        # Calculate crash location
        crash_start = int(len(data) * self.crash_location_pct)
        crash_end = crash_start + self.crash_duration_bars
        recovery_end = crash_end + self.recovery_duration_bars

        if crash_end >= len(data):
            logger.warning("Crash location too late in data, adjusting")
            crash_start = len(data) - self.crash_duration_bars - self.recovery_duration_bars - 1
            crash_end = crash_start + self.crash_duration_bars
            recovery_end = crash_end + self.recovery_duration_bars

        # Get price before crash
        pre_crash_price = data.loc[crash_start, "close"]

        # Calculate crash and recovery targets
        crash_price = pre_crash_price * (1 - self.crash_size_pct)
        recovery_price = crash_price + (pre_crash_price - crash_price) * self.recovery_pct

        # Apply crash (linear drop)
        for i, idx in enumerate(range(crash_start, min(crash_end, len(data)))):
            progress = (i + 1) / self.crash_duration_bars
            target_price = pre_crash_price - (pre_crash_price - crash_price) * progress

            ratio = target_price / data.loc[idx, "close"]
            data.loc[idx, "open"] *= ratio
            data.loc[idx, "high"] *= ratio
            data.loc[idx, "low"] *= ratio
            data.loc[idx, "close"] = target_price

        # Apply recovery (linear)
        for i, idx in enumerate(range(crash_end, min(recovery_end, len(data)))):
            progress = (i + 1) / self.recovery_duration_bars
            target_price = crash_price + (recovery_price - crash_price) * progress

            ratio = target_price / data.loc[idx, "close"]
            data.loc[idx, "open"] *= ratio
            data.loc[idx, "high"] *= ratio
            data.loc[idx, "low"] *= ratio
            data.loc[idx, "close"] = target_price

        logger.info(
            "Flash crash injected | start=%d | crash_size=%.1f%% | recovery=%.1f%%",
            crash_start,
            self.crash_size_pct * 100,
            self.recovery_pct * 100,
        )

        return data


__all__ = [
    "StressScenarioType",
    "StressScenarioGenerator",
    "GapMoveScenario",
    "HighVolatilityScenario",
    "FundingShockScenario",
    "LiquidityCrisisScenario",
    "FlashCrashScenario",
]
