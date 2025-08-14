"""
ML-enhanced portfolio allocator that combines optimization with ML predictions
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ...core.base import ComponentConfig
from ...portfolio.allocator import Allocator
from ..features import FeatureEngineeringPipeline
from ..models import MarketRegimeDetector, StrategyMetaSelector
from .optimizer import MarkowitzOptimizer, OptimizationConstraints


class MLEnhancedAllocator(Allocator):
    """Portfolio allocator enhanced with ML predictions and optimization"""

    def __init__(
        self,
        config: ComponentConfig | None = None,
        db_manager=None,
        optimizer: MarkowitzOptimizer | None = None,
        regime_detector: MarketRegimeDetector | None = None,
        strategy_selector: StrategyMetaSelector | None = None,
    ):
        """Initialize ML-enhanced allocator

        Args:
            config: Component configuration
            db_manager: Database manager
            optimizer: Portfolio optimizer
            regime_detector: Market regime detector
            strategy_selector: Strategy selector
        """
        if config is None:
            config = ComponentConfig(
                component_id="ml_enhanced_allocator", component_type="allocator"
            )
        super().__init__(config)

        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager

        # ML components
        self.optimizer = optimizer or MarkowitzOptimizer(db_manager=db_manager)
        self.regime_detector = regime_detector
        self.strategy_selector = strategy_selector
        self.feature_pipeline = FeatureEngineeringPipeline(db_manager=db_manager)

        # Allocation settings
        self.regime_constraints = self._define_regime_constraints()
        self.optimization_objectives = self._define_optimization_objectives()

        # Current state
        self.current_regime = "unknown"
        self.current_allocation = {}
        self.allocation_history = []

    def allocate(
        self,
        universe: list[str],
        market_data: dict[str, pd.DataFrame],
        signals: dict[str, float] | None = None,
        capital: float = 100000,
    ) -> dict[str, float]:
        """Allocate capital across assets using ML-enhanced optimization

        Args:
            universe: List of asset symbols
            market_data: Dictionary of OHLCV data per asset
            signals: Optional trading signals per asset
            capital: Total capital to allocate

        Returns:
            Dictionary of position sizes (in dollars)
        """
        self.logger.info(f"Allocating ${capital:,.0f} across {len(universe)} assets")

        # Prepare returns data
        returns = self._prepare_returns(market_data)

        if returns.empty:
            self.logger.warning("No returns data available, using equal weights")
            return self._equal_weight_allocation(universe, capital)

        # Generate features if ML models available
        current_features = None
        if self.regime_detector or self.strategy_selector:
            # Use first asset's data for market features
            primary_asset = universe[0] if universe else "SPY"
            if primary_asset in market_data:
                current_features = self.feature_pipeline.generate_features(
                    market_data[primary_asset], store_features=False
                ).iloc[[-1]]

        # Detect current regime
        if self.regime_detector and current_features is not None:
            regime, confidence = self.regime_detector.get_regime_confidence(current_features)
            self.current_regime = regime[-1] if isinstance(regime, (list, np.ndarray)) else regime
            regime_confidence = (
                confidence[-1] if isinstance(confidence, (list, np.ndarray)) else confidence
            )
            self.logger.info(
                f"Current regime: {self.current_regime} (confidence: {regime_confidence:.1%})"
            )
        else:
            self.current_regime = "unknown"
            regime_confidence = 1.0

        # Get regime-specific constraints and objective
        constraints = self.regime_constraints.get(self.current_regime, OptimizationConstraints())
        objective = self.optimization_objectives.get(self.current_regime, "max_sharpe")

        # Adjust risk based on regime
        risk_aversion = self._get_regime_risk_aversion(self.current_regime)

        # Filter universe based on signals if provided
        if signals:
            active_universe = [
                asset for asset in universe if asset in signals and abs(signals.get(asset, 0)) > 0.1
            ]
            if not active_universe:
                active_universe = universe
        else:
            active_universe = universe

        # Filter returns to active universe
        available_assets = [a for a in active_universe if a in returns.columns]
        if not available_assets:
            self.logger.warning("No assets available for optimization")
            return self._equal_weight_allocation(universe, capital)

        returns_subset = returns[available_assets]

        # Optimize portfolio
        try:
            weights, metrics = self.optimizer.optimize(
                returns_subset,
                constraints=constraints,
                objective=objective,
                risk_aversion=risk_aversion,
            )

            self.logger.info(
                f"Optimization complete - Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                f"Positions: {metrics.get('n_positions', 0)}"
            )

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return self._equal_weight_allocation(active_universe, capital)

        # Apply signal adjustments if provided
        if signals:
            weights = self._adjust_weights_by_signals(weights, signals)

        # Apply regime-based position sizing
        weights = self._apply_regime_sizing(weights, self.current_regime, regime_confidence)

        # Convert weights to position sizes
        positions = self._weights_to_positions(weights, capital)

        # Store allocation
        self.current_allocation = positions
        self._record_allocation(positions, metrics, self.current_regime)

        return positions

    def _prepare_returns(self, market_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare returns matrix from market data

        Args:
            market_data: Dictionary of OHLCV data

        Returns:
            DataFrame of returns
        """
        returns_dict = {}

        for symbol, data in market_data.items():
            if "close" in data.columns:
                returns = data["close"].pct_change().dropna()
                returns_dict[symbol] = returns

        if not returns_dict:
            return pd.DataFrame()

        # Align all returns to same index
        returns_df = pd.DataFrame(returns_dict)

        # Forward fill missing values then drop remaining NaN
        returns_df = returns_df.ffill().dropna()

        # Use last 252 days (1 year) for optimization
        if len(returns_df) > 252:
            returns_df = returns_df.iloc[-252:]

        return returns_df

    def _define_regime_constraints(self) -> dict[str, OptimizationConstraints]:
        """Define optimization constraints for each regime

        Returns:
            Dictionary of constraints per regime
        """
        return {
            "bull_quiet": OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.3,
                long_only=True,
                max_positions=10,
                min_position_size=0.02,
            ),
            "bull_volatile": OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.2,
                long_only=True,
                max_positions=15,
                min_position_size=0.01,
                max_risk=0.25,
            ),
            "bear_quiet": OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.15,
                long_only=True,
                max_positions=5,
                min_position_size=0.05,
                max_risk=0.15,
            ),
            "bear_volatile": OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.1,
                long_only=True,
                max_positions=3,
                min_position_size=0.05,
                max_risk=0.10,
            ),
            "sideways": OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.25,
                long_only=True,
                max_positions=8,
                min_position_size=0.02,
            ),
            "unknown": OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.2,
                long_only=True,
                max_positions=10,
                min_position_size=0.02,
            ),
        }

    def _define_optimization_objectives(self) -> dict[str, str]:
        """Define optimization objectives for each regime

        Returns:
            Dictionary of objectives per regime
        """
        return {
            "bull_quiet": "max_sharpe",  # Focus on risk-adjusted returns
            "bull_volatile": "max_return",  # Aggressive in strong trends
            "bear_quiet": "min_risk",  # Capital preservation
            "bear_volatile": "min_risk",  # Maximum capital preservation
            "sideways": "max_sharpe",  # Balanced approach
            "unknown": "max_sharpe",  # Default to risk-adjusted
        }

    def _get_regime_risk_aversion(self, regime: str) -> float:
        """Get risk aversion parameter for regime

        Args:
            regime: Current market regime

        Returns:
            Risk aversion parameter
        """
        risk_aversion_map = {
            "bull_quiet": 0.5,  # Low risk aversion
            "bull_volatile": 0.3,  # Very low risk aversion
            "bear_quiet": 2.0,  # High risk aversion
            "bear_volatile": 3.0,  # Very high risk aversion
            "sideways": 1.0,  # Neutral
            "unknown": 1.5,  # Slightly risk averse
        }

        return risk_aversion_map.get(regime, 1.0)

    def _adjust_weights_by_signals(
        self, weights: dict[str, float], signals: dict[str, float]
    ) -> dict[str, float]:
        """Adjust optimized weights based on trading signals

        Args:
            weights: Optimized weights
            signals: Trading signals (-1 to 1)

        Returns:
            Adjusted weights
        """
        adjusted_weights = {}

        for asset, weight in weights.items():
            if asset in signals:
                signal = signals[asset]

                # Adjust weight based on signal strength
                if signal > 0:
                    # Positive signal: maintain or increase weight
                    adjusted_weight = weight * (1 + signal * 0.2)
                elif signal < 0:
                    # Negative signal: reduce weight
                    adjusted_weight = weight * max(0, 1 + signal * 0.5)
                else:
                    # No signal: maintain weight
                    adjusted_weight = weight

                adjusted_weights[asset] = adjusted_weight
            else:
                adjusted_weights[asset] = weight

        # Re-normalize to sum to 1
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return adjusted_weights

    def _apply_regime_sizing(
        self, weights: dict[str, float], regime: str, confidence: float
    ) -> dict[str, float]:
        """Apply regime-based position sizing

        Args:
            weights: Portfolio weights
            regime: Current regime
            confidence: Regime confidence

        Returns:
            Sized weights
        """
        # Define regime exposure levels
        regime_exposure = {
            "bull_quiet": 1.0,  # Full exposure
            "bull_volatile": 0.8,  # Slightly reduced
            "bear_quiet": 0.5,  # Half exposure
            "bear_volatile": 0.3,  # Minimal exposure
            "sideways": 0.7,  # Moderate exposure
            "unknown": 0.6,  # Conservative
        }

        exposure = regime_exposure.get(regime, 0.6)

        # Adjust exposure by confidence
        exposure = exposure * (0.5 + 0.5 * confidence)

        # Scale weights
        sized_weights = {k: v * exposure for k, v in weights.items()}

        # The remainder goes to cash (not explicitly modeled here)
        self.logger.info(f"Portfolio exposure: {exposure:.1%} (cash: {1-exposure:.1%})")

        return sized_weights

    def _weights_to_positions(self, weights: dict[str, float], capital: float) -> dict[str, float]:
        """Convert weights to position sizes

        Args:
            weights: Portfolio weights
            capital: Total capital

        Returns:
            Position sizes in dollars
        """
        positions = {}

        for asset, weight in weights.items():
            position_size = capital * weight

            # Apply minimum position size (e.g., $100)
            if position_size > 100:
                positions[asset] = round(position_size, 2)

        return positions

    def _equal_weight_allocation(self, universe: list[str], capital: float) -> dict[str, float]:
        """Create equal weight allocation as fallback

        Args:
            universe: List of assets
            capital: Total capital

        Returns:
            Equal weight positions
        """
        if not universe:
            return {}

        weight = 1.0 / len(universe)
        position_size = capital * weight

        return {asset: round(position_size, 2) for asset in universe}

    def _record_allocation(self, positions: dict[str, float], metrics: dict[str, Any], regime: str):
        """Record allocation in history

        Args:
            positions: Position sizes
            metrics: Optimization metrics
            regime: Current regime
        """
        record = {
            "timestamp": datetime.now(),
            "positions": positions.copy(),
            "metrics": metrics.copy(),
            "regime": regime,
            "n_positions": len(positions),
        }

        self.allocation_history.append(record)

        # Keep only last 100 allocations
        if len(self.allocation_history) > 100:
            self.allocation_history = self.allocation_history[-100:]

        # Store in database if available
        if self.db_manager:
            try:
                import json

                self.db_manager.execute(
                    """INSERT INTO portfolio_allocations
                       (allocation_date, positions, metrics, regime, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        datetime.now(),
                        json.dumps(positions),
                        json.dumps(metrics),
                        regime,
                        datetime.now(),
                    ),
                )
            except Exception as e:
                self.logger.error(f"Error storing allocation: {e}")

    def get_allocation_summary(self) -> dict[str, Any]:
        """Get summary of current allocation

        Returns:
            Dictionary with allocation summary
        """
        if not self.current_allocation:
            return {"status": "No active allocation"}

        total_allocated = sum(self.current_allocation.values())

        summary = {
            "timestamp": datetime.now().isoformat(),
            "regime": self.current_regime,
            "n_positions": len(self.current_allocation),
            "total_allocated": total_allocated,
            "top_positions": sorted(
                self.current_allocation.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "optimization_metrics": self.optimizer.last_metrics if self.optimizer else {},
        }

        return summary

    def analyze_allocation_history(self) -> dict[str, Any]:
        """Analyze historical allocation patterns

        Returns:
            Dictionary with allocation analysis
        """
        if not self.allocation_history:
            return {"status": "No allocation history"}

        # Analyze allocation patterns
        regime_counts = {}
        avg_positions = []
        sharpe_ratios = []

        for record in self.allocation_history:
            regime = record.get("regime", "unknown")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            avg_positions.append(record.get("n_positions", 0))

            metrics = record.get("metrics", {})
            if "sharpe_ratio" in metrics:
                sharpe_ratios.append(metrics["sharpe_ratio"])

        analysis = {
            "total_allocations": len(self.allocation_history),
            "regime_distribution": regime_counts,
            "avg_positions": np.mean(avg_positions) if avg_positions else 0,
            "avg_sharpe": np.mean(sharpe_ratios) if sharpe_ratios else 0,
            "allocation_frequency": self._calculate_allocation_frequency(),
        }

        return analysis

    def _calculate_allocation_frequency(self) -> str:
        """Calculate average allocation frequency

        Returns:
            String describing frequency
        """
        if len(self.allocation_history) < 2:
            return "Insufficient data"

        # Calculate time between allocations
        timestamps = [r["timestamp"] for r in self.allocation_history]
        deltas = []

        for i in range(1, len(timestamps)):
            delta = timestamps[i] - timestamps[i - 1]
            deltas.append(delta.total_seconds() / 86400)  # Convert to days

        avg_days = np.mean(deltas)

        if avg_days < 1:
            return "Intraday"
        elif avg_days < 7:
            return "Daily"
        elif avg_days < 30:
            return "Weekly"
        elif avg_days < 90:
            return "Monthly"
        else:
            return "Quarterly"
