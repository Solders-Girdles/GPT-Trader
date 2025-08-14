"""
Rebalancing triggers for portfolio management
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


class RebalancingTrigger(ABC):
    """Base class for rebalancing triggers"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_triggered = None

    @abstractmethod
    def check(
        self,
        current_positions: dict[str, float],
        portfolio_value: float,
        last_rebalance_date: datetime | None = None,
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Check if trigger conditions are met

        Args:
            current_positions: Current position values
            portfolio_value: Total portfolio value
            last_rebalance_date: Date of last rebalancing
            market_data: Optional market data

        Returns:
            Tuple of (triggered, urgency, details)
            - triggered: Whether trigger fired
            - urgency: 0-1 score (1 = most urgent)
            - details: Additional information
        """
        pass


class ThresholdTrigger(RebalancingTrigger):
    """Trigger based on weight deviation thresholds"""

    def __init__(self, threshold: float = 0.05, emergency_threshold: float = 0.2):
        """Initialize threshold trigger

        Args:
            threshold: Normal deviation threshold (e.g., 5%)
            emergency_threshold: Emergency deviation threshold (e.g., 20%)
        """
        super().__init__()
        self.threshold = threshold
        self.emergency_threshold = emergency_threshold
        self.target_weights = {}

    def set_target_weights(self, weights: dict[str, float]):
        """Set target portfolio weights

        Args:
            weights: Target weights (sum to 1)
        """
        self.target_weights = weights

    def check(
        self,
        current_positions: dict[str, float],
        portfolio_value: float,
        last_rebalance_date: datetime | None = None,
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Check for weight deviations"""

        if not self.target_weights:
            # If no targets set, assume equal weight
            n_assets = len(current_positions)
            if n_assets > 0:
                self.target_weights = {asset: 1.0 / n_assets for asset in current_positions}
            else:
                return False, 0.0, {}

        # Calculate current weights
        current_weights = {}
        if portfolio_value > 0:
            current_weights = {
                asset: value / portfolio_value for asset, value in current_positions.items()
            }

        # Calculate deviations
        max_deviation = 0.0
        deviations = {}

        for asset in set(current_weights.keys()) | set(self.target_weights.keys()):
            current = current_weights.get(asset, 0.0)
            target = self.target_weights.get(asset, 0.0)
            deviation = abs(current - target)
            deviations[asset] = deviation
            max_deviation = max(max_deviation, deviation)

        # Determine if triggered and urgency
        triggered = False
        urgency = 0.0

        if max_deviation >= self.emergency_threshold:
            triggered = True
            urgency = 1.0  # Maximum urgency
        elif max_deviation >= self.threshold:
            triggered = True
            # Scale urgency based on how close to emergency threshold
            urgency = 0.5 + 0.5 * (
                (max_deviation - self.threshold) / (self.emergency_threshold - self.threshold)
            )

        details = {
            "max_deviation": max_deviation,
            "deviations": deviations,
            "current_weights": current_weights,
            "target_weights": self.target_weights,
        }

        if triggered:
            self.last_triggered = datetime.now()
            self.logger.info(
                f"Threshold trigger fired: max deviation {max_deviation:.1%}, urgency {urgency:.1f}"
            )

        return triggered, urgency, details


class TimeTrigger(RebalancingTrigger):
    """Trigger based on time intervals"""

    def __init__(self, interval_days: int = 30, max_interval_days: int = 90):
        """Initialize time trigger

        Args:
            interval_days: Normal rebalancing interval
            max_interval_days: Maximum time before forced rebalancing
        """
        super().__init__()
        self.interval_days = interval_days
        self.max_interval_days = max_interval_days

    def check(
        self,
        current_positions: dict[str, float],
        portfolio_value: float,
        last_rebalance_date: datetime | None = None,
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Check if time interval exceeded"""

        if last_rebalance_date is None:
            # First rebalancing
            return True, 0.5, {"reason": "Initial rebalancing"}

        days_since = (datetime.now() - last_rebalance_date).days

        triggered = False
        urgency = 0.0

        if days_since >= self.max_interval_days:
            triggered = True
            urgency = 1.0  # Maximum urgency
        elif days_since >= self.interval_days:
            triggered = True
            # Scale urgency based on how overdue
            urgency = 0.3 + 0.7 * (
                (days_since - self.interval_days) / (self.max_interval_days - self.interval_days)
            )

        details = {
            "days_since_last": days_since,
            "interval_days": self.interval_days,
            "max_interval_days": self.max_interval_days,
        }

        if triggered:
            self.last_triggered = datetime.now()
            self.logger.info(
                f"Time trigger fired: {days_since} days since last rebalance, urgency {urgency:.1f}"
            )

        return triggered, urgency, details


class VolatilityTrigger(RebalancingTrigger):
    """Trigger based on market volatility"""

    def __init__(self, high_vol_threshold: float = 0.3, low_vol_threshold: float = 0.1):
        """Initialize volatility trigger

        Args:
            high_vol_threshold: High volatility threshold (annualized)
            low_vol_threshold: Low volatility threshold (annualized)
        """
        super().__init__()
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.last_vol_regime = "normal"

    def check(
        self,
        current_positions: dict[str, float],
        portfolio_value: float,
        last_rebalance_date: datetime | None = None,
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Check for volatility regime changes"""

        if not market_data:
            return False, 0.0, {}

        # Calculate portfolio volatility
        returns_list = []
        weights = []

        for asset, position in current_positions.items():
            if asset in market_data and position > 0:
                df = market_data[asset]
                if "close" in df.columns:
                    returns = df["close"].pct_change().dropna()
                    if len(returns) > 20:  # Need sufficient data
                        returns_list.append(returns.iloc[-20:])  # Last 20 days
                        weights.append(position / portfolio_value)

        if not returns_list:
            return False, 0.0, {}

        # Calculate weighted portfolio returns
        portfolio_returns = sum(w * r for w, r in zip(weights, returns_list, strict=False))
        current_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized

        # Determine volatility regime
        current_regime = "normal"
        if current_vol >= self.high_vol_threshold:
            current_regime = "high"
        elif current_vol <= self.low_vol_threshold:
            current_regime = "low"

        # Check for regime change
        triggered = False
        urgency = 0.0

        if current_regime != self.last_vol_regime:
            triggered = True
            if current_regime == "high":
                urgency = 0.8  # High urgency for high volatility
            elif current_regime == "low":
                urgency = 0.3  # Low urgency for low volatility
            else:
                urgency = 0.5

        details = {
            "current_volatility": current_vol,
            "current_regime": current_regime,
            "previous_regime": self.last_vol_regime,
        }

        if triggered:
            self.last_vol_regime = current_regime
            self.last_triggered = datetime.now()
            self.logger.info(
                f"Volatility trigger fired: regime change from {self.last_vol_regime} to {current_regime}"
            )

        return triggered, urgency, details


class RegimeTrigger(RebalancingTrigger):
    """Trigger based on market regime changes"""

    def __init__(self, regime_detector):
        """Initialize regime trigger

        Args:
            regime_detector: ML regime detector model
        """
        super().__init__()
        self.regime_detector = regime_detector
        self.last_regime = None

    def check(
        self,
        current_positions: dict[str, float],
        portfolio_value: float,
        last_rebalance_date: datetime | None = None,
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Check for market regime changes"""

        if not self.regime_detector or not market_data:
            return False, 0.0, {}

        # Get primary market data (use first asset or SPY if available)
        primary_data = None
        if "SPY" in market_data:
            primary_data = market_data["SPY"]
        elif market_data:
            primary_data = list(market_data.values())[0]

        if primary_data is None or primary_data.empty:
            return False, 0.0, {}

        # Detect current regime
        try:
            # Would need feature engineering here
            from ..ml.features import FeatureEngineeringPipeline

            feature_pipeline = FeatureEngineeringPipeline()
            features = feature_pipeline.generate_features(primary_data, store_features=False)

            if not features.empty:
                current_features = features.iloc[[-1]]
                regime, confidence = self.regime_detector.get_regime_confidence(current_features)
                current_regime = regime[-1] if isinstance(regime, (list, np.ndarray)) else regime
            else:
                return False, 0.0, {}
        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return False, 0.0, {}

        # Check for regime change
        triggered = False
        urgency = 0.0

        if self.last_regime is not None and current_regime != self.last_regime:
            triggered = True

            # Set urgency based on regime transition
            regime_urgency = {
                ("bull_quiet", "bear_volatile"): 1.0,  # Crash protection
                ("bull_quiet", "bear_quiet"): 0.7,
                ("bull_volatile", "bear_volatile"): 0.9,
                ("bear_quiet", "bull_quiet"): 0.6,
                ("bear_volatile", "bull_quiet"): 0.8,
                ("sideways", "bear_volatile"): 0.8,
                ("sideways", "bull_volatile"): 0.5,
            }

            transition = (self.last_regime, current_regime)
            urgency = regime_urgency.get(transition, 0.5)

        details = {
            "current_regime": current_regime,
            "previous_regime": self.last_regime,
            "confidence": float(confidence) if "confidence" in locals() else 0.0,
            "transition": (
                f"{self.last_regime} -> {current_regime}" if self.last_regime else "Initial"
            ),
        }

        if triggered:
            self.logger.info(
                f"Regime trigger fired: {details['transition']}, urgency {urgency:.1f}"
            )
            self.last_triggered = datetime.now()

        self.last_regime = current_regime

        return triggered, urgency, details


class DrawdownTrigger(RebalancingTrigger):
    """Trigger based on portfolio drawdown"""

    def __init__(self, drawdown_threshold: float = 0.1, emergency_drawdown: float = 0.2):
        """Initialize drawdown trigger

        Args:
            drawdown_threshold: Normal drawdown threshold (e.g., 10%)
            emergency_drawdown: Emergency drawdown threshold (e.g., 20%)
        """
        super().__init__()
        self.drawdown_threshold = drawdown_threshold
        self.emergency_drawdown = emergency_drawdown
        self.peak_value = 0.0

    def check(
        self,
        current_positions: dict[str, float],
        portfolio_value: float,
        last_rebalance_date: datetime | None = None,
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Check for portfolio drawdown"""

        # Update peak value
        self.peak_value = max(self.peak_value, portfolio_value)

        # Calculate drawdown
        drawdown = 0.0
        if self.peak_value > 0:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value

        triggered = False
        urgency = 0.0

        if drawdown >= self.emergency_drawdown:
            triggered = True
            urgency = 1.0  # Maximum urgency
        elif drawdown >= self.drawdown_threshold:
            triggered = True
            # Scale urgency
            urgency = 0.5 + 0.5 * (
                (drawdown - self.drawdown_threshold)
                / (self.emergency_drawdown - self.drawdown_threshold)
            )

        details = {
            "current_drawdown": drawdown,
            "peak_value": self.peak_value,
            "current_value": portfolio_value,
            "drawdown_amount": self.peak_value - portfolio_value,
        }

        if triggered:
            self.last_triggered = datetime.now()
            self.logger.info(
                f"Drawdown trigger fired: {drawdown:.1%} drawdown, urgency {urgency:.1f}"
            )

        return triggered, urgency, details


class CompositeTrigger(RebalancingTrigger):
    """Composite trigger combining multiple triggers"""

    def __init__(self, triggers: list, aggregation: str = "max"):
        """Initialize composite trigger

        Args:
            triggers: List of triggers to combine
            aggregation: How to aggregate ('max', 'mean', 'any')
        """
        super().__init__()
        self.triggers = triggers
        self.aggregation = aggregation

    def check(
        self,
        current_positions: dict[str, float],
        portfolio_value: float,
        last_rebalance_date: datetime | None = None,
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Check all triggers and aggregate results"""

        results = []
        all_details = {}

        for i, trigger in enumerate(self.triggers):
            triggered, urgency, details = trigger.check(
                current_positions, portfolio_value, last_rebalance_date, market_data
            )

            if triggered:
                results.append(urgency)
                all_details[trigger.__class__.__name__] = details

        if not results:
            return False, 0.0, {}

        # Aggregate urgencies
        if self.aggregation == "max":
            final_urgency = max(results)
        elif self.aggregation == "mean":
            final_urgency = np.mean(results)
        elif self.aggregation == "any":
            final_urgency = 0.5  # Fixed urgency if any trigger fires
        else:
            final_urgency = max(results)

        details = {
            "triggered_count": len(results),
            "urgencies": results,
            "trigger_details": all_details,
        }

        return True, final_urgency, details
