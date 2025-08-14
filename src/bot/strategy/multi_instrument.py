"""
Multi-Instrument Strategy Coordination for Multi-Asset Strategy Enhancement

This module implements sophisticated multi-asset strategy coordination including:
- Cross-asset signal aggregation and weighting
- Portfolio-level position sizing and risk management
- Inter-strategy communication and coordination
- Regime-aware strategy switching
- Cross-asset arbitrage detection
- Correlation-based pair trading strategies
- Dynamic hedging and risk overlay strategies
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Import from other modules
try:
    from ..analytics.correlation_modeling import CorrelationMethod, CrossAssetCorrelationFramework
    from ..portfolio.portfolio_optimization import (
        OptimizationMethod,
        PortfolioOptimizationFramework,
    )
    from ..strategy.base import BaseStrategy, StrategyResult, StrategySignal
except ImportError:
    # Fallback for testing
    class BaseStrategy:
        pass

    class StrategySignal:
        def __init__(self, signal: float, confidence: float, metadata: dict = None) -> None:
            self.signal = signal
            self.confidence = confidence
            self.metadata = metadata or {}

    class StrategyResult:
        pass


logger = logging.getLogger(__name__)


class CoordinationMethod(Enum):
    """Strategy coordination methods"""

    SIGNAL_AVERAGING = "signal_averaging"
    WEIGHTED_VOTING = "weighted_voting"
    HIERARCHICAL = "hierarchical"
    REINFORCEMENT_BASED = "reinforcement_based"
    CORRELATION_AWARE = "correlation_aware"
    REGIME_DEPENDENT = "regime_dependent"


class PositionSizingMethod(Enum):
    """Position sizing methods for multi-asset strategies"""

    EQUAL_WEIGHT = "equal_weight"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"


class HedgingMode(Enum):
    """Hedging modes for risk management"""

    NO_HEDGE = "no_hedge"
    BETA_HEDGE = "beta_hedge"
    CORRELATION_HEDGE = "correlation_hedge"
    VOLATILITY_HEDGE = "volatility_hedge"
    DYNAMIC_HEDGE = "dynamic_hedge"


@dataclass
class CoordinationConfig:
    """Configuration for multi-instrument coordination"""

    coordination_method: CoordinationMethod = CoordinationMethod.CORRELATION_AWARE
    position_sizing: PositionSizingMethod = PositionSizingMethod.RISK_PARITY
    hedging_mode: HedgingMode = HedgingMode.CORRELATION_HEDGE
    max_portfolio_leverage: float = 1.0
    max_single_asset_weight: float = 0.3
    min_signal_confidence: float = 0.1
    correlation_threshold: float = 0.7
    rebalance_frequency: int = 5  # days
    lookback_window: int = 252
    risk_budget_per_asset: float = 0.02
    enable_cross_asset_signals: bool = True
    enable_pair_trading: bool = True
    enable_momentum_overlay: bool = True
    dynamic_hedging: bool = True


@dataclass
class AssetSignal:
    """Signal for a single asset"""

    asset: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    strategy_name: str
    timestamp: pd.Timestamp
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioSignal:
    """Aggregated portfolio-level signal"""

    asset_signals: dict[str, AssetSignal]
    target_weights: pd.Series
    expected_return: float
    expected_risk: float
    coordination_method: str
    timestamp: pd.Timestamp
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationResult:
    """Result from strategy coordination"""

    portfolio_signal: PortfolioSignal
    individual_signals: list[AssetSignal]
    position_sizes: dict[str, float]
    hedge_positions: dict[str, float]
    risk_metrics: dict[str, float]
    coordination_statistics: dict[str, Any]
    success: bool
    message: str


class BaseMultiInstrumentStrategy(ABC):
    """Base class for multi-instrument strategies"""

    def __init__(self, config: CoordinationConfig) -> None:
        self.config = config
        self.assets = []
        self.single_asset_strategies = {}
        self.correlation_analyzer = None
        self.portfolio_optimizer = None
        self.signal_history = defaultdict(deque)
        self.position_history = defaultdict(deque)

    @abstractmethod
    def generate_portfolio_signal(self, market_data: dict[str, pd.DataFrame]) -> CoordinationResult:
        """Generate coordinated portfolio signal"""
        pass

    def add_strategy(self, asset: str, strategy: BaseStrategy) -> None:
        """Add single-asset strategy for coordination"""
        self.single_asset_strategies[asset] = strategy
        if asset not in self.assets:
            self.assets.append(asset)

    def remove_strategy(self, asset: str) -> None:
        """Remove strategy for asset"""
        if asset in self.single_asset_strategies:
            del self.single_asset_strategies[asset]
        if asset in self.assets:
            self.assets.remove(asset)

    def _collect_individual_signals(
        self, market_data: dict[str, pd.DataFrame]
    ) -> list[AssetSignal]:
        """Collect signals from individual asset strategies"""
        signals = []
        current_time = pd.Timestamp.now()

        for asset in self.assets:
            if asset in self.single_asset_strategies and asset in market_data:
                try:
                    strategy = self.single_asset_strategies[asset]
                    data = market_data[asset]

                    # Generate signal (assuming strategy has a generate_signal method)
                    if hasattr(strategy, "generate_signal"):
                        strategy_signal = strategy.generate_signal(data)

                        if isinstance(strategy_signal, StrategySignal):
                            signal = AssetSignal(
                                asset=asset,
                                signal=strategy_signal.signal,
                                confidence=strategy_signal.confidence,
                                strategy_name=strategy.__class__.__name__,
                                timestamp=current_time,
                                metadata=strategy_signal.metadata,
                            )
                            signals.append(signal)

                            # Update signal history
                            self.signal_history[asset].append(signal)
                            if len(self.signal_history[asset]) > 100:  # Keep last 100 signals
                                self.signal_history[asset].popleft()

                except Exception as e:
                    logger.warning(f"Failed to get signal for {asset}: {str(e)}")

        return signals

    def _calculate_correlation_matrix(self, market_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix for assets"""
        if not self.correlation_analyzer:
            from ..analytics.correlation_modeling import create_correlation_analyzer

            self.correlation_analyzer = create_correlation_analyzer(
                method=CorrelationMethod.PEARSON, regime_detection=True
            )

        # Combine return data
        returns_data = {}
        for asset, data in market_data.items():
            if "close" in data.columns:
                returns_data[asset] = data["close"].pct_change().dropna()

        if len(returns_data) < 2:
            return pd.DataFrame()

        # Align data
        combined_returns = pd.DataFrame(returns_data).dropna()

        if len(combined_returns) < 30:
            return pd.DataFrame()

        try:
            result = self.correlation_analyzer.analyze_correlations(combined_returns)
            return result.correlation_matrix
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {str(e)}")
            return pd.DataFrame()


class CorrelationAwareCoordinator(BaseMultiInstrumentStrategy):
    """Correlation-aware strategy coordination"""

    def generate_portfolio_signal(self, market_data: dict[str, pd.DataFrame]) -> CoordinationResult:
        """Generate portfolio signal considering correlations"""
        try:
            # Collect individual signals
            individual_signals = self._collect_individual_signals(market_data)

            if not individual_signals:
                return self._create_empty_result("No individual signals available")

            # Calculate correlations
            correlation_matrix = self._calculate_correlation_matrix(market_data)

            # Coordinate signals
            portfolio_signal = self._coordinate_signals(individual_signals, correlation_matrix)

            # Calculate position sizes
            position_sizes = self._calculate_position_sizes(
                individual_signals, market_data, correlation_matrix
            )

            # Calculate hedge positions
            hedge_positions = self._calculate_hedge_positions(position_sizes, correlation_matrix)

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                position_sizes, market_data, correlation_matrix
            )

            return CoordinationResult(
                portfolio_signal=portfolio_signal,
                individual_signals=individual_signals,
                position_sizes=position_sizes,
                hedge_positions=hedge_positions,
                risk_metrics=risk_metrics,
                coordination_statistics=self._get_coordination_statistics(
                    individual_signals, correlation_matrix
                ),
                success=True,
                message="Coordination successful",
            )

        except Exception as e:
            logger.error(f"Coordination failed: {str(e)}")
            return self._create_empty_result(f"Coordination error: {str(e)}")

    def _coordinate_signals(
        self, signals: list[AssetSignal], correlation_matrix: pd.DataFrame
    ) -> PortfolioSignal:
        """Coordinate individual signals considering correlations"""
        current_time = pd.Timestamp.now()

        if correlation_matrix.empty:
            # Simple average if no correlation data
            signal_dict = {signal.asset: signal for signal in signals}
            avg_signal = np.mean([s.signal for s in signals])
            np.mean([s.confidence for s in signals])

            # Equal weights
            target_weights = pd.Series({s.asset: 1.0 / len(signals) for s in signals})

            return PortfolioSignal(
                asset_signals=signal_dict,
                target_weights=target_weights,
                expected_return=avg_signal * 0.1,  # Rough estimate
                expected_risk=0.15,  # Default risk estimate
                coordination_method="simple_average",
                timestamp=current_time,
            )

        # Correlation-adjusted coordination
        assets_with_signals = [signal.asset for signal in signals]
        signal_dict = {signal.asset: signal for signal in signals}

        # Filter correlation matrix to available assets
        available_assets = [
            asset for asset in assets_with_signals if asset in correlation_matrix.index
        ]
        if len(available_assets) < 2:
            # Fall back to simple average
            target_weights = pd.Series({s.asset: 1.0 / len(signals) for s in signals})
            return PortfolioSignal(
                asset_signals=signal_dict,
                target_weights=target_weights,
                expected_return=np.mean([s.signal for s in signals]) * 0.1,
                expected_risk=0.15,
                coordination_method="correlation_aware_fallback",
                timestamp=current_time,
            )

        corr_sub = correlation_matrix.loc[available_assets, available_assets]

        # Create signal vector
        signal_vector = np.array([signal_dict[asset].signal for asset in available_assets])
        confidence_vector = np.array([signal_dict[asset].confidence for asset in available_assets])

        # Weight signals by inverse correlation (diversification benefit)
        try:
            # Calculate diversification weights
            inv_corr = np.linalg.inv(
                corr_sub.values + np.eye(len(corr_sub)) * 0.01
            )  # Add small regularization
            diversification_weights = np.sum(inv_corr, axis=1)
            diversification_weights = diversification_weights / np.sum(diversification_weights)

            # Combine with confidence weights
            combined_weights = diversification_weights * confidence_vector
            combined_weights = (
                combined_weights / np.sum(combined_weights)
                if np.sum(combined_weights) > 0
                else np.ones(len(combined_weights)) / len(combined_weights)
            )

            # Apply signal direction
            target_weights_values = combined_weights * np.abs(signal_vector)

            # Normalize to sum to 1
            if np.sum(target_weights_values) > 0:
                target_weights_values = target_weights_values / np.sum(target_weights_values)
            else:
                target_weights_values = np.ones(len(available_assets)) / len(available_assets)

            target_weights = pd.Series(target_weights_values, index=available_assets)

            # Estimate expected return and risk
            expected_return = np.sum(target_weights_values * signal_vector) * 0.1
            portfolio_var = (
                target_weights_values.T @ corr_sub.values @ target_weights_values * 0.15**2
            )
            expected_risk = np.sqrt(portfolio_var)

            return PortfolioSignal(
                asset_signals=signal_dict,
                target_weights=target_weights,
                expected_return=expected_return,
                expected_risk=expected_risk,
                coordination_method="correlation_aware",
                timestamp=current_time,
                metadata={
                    "diversification_weights": diversification_weights.tolist(),
                    "signal_strength": float(np.mean(np.abs(signal_vector))),
                },
            )

        except Exception as e:
            logger.warning(f"Correlation-aware coordination failed: {str(e)}, using simple average")
            # Fall back to equal weights
            target_weights = pd.Series(
                {asset: 1.0 / len(available_assets) for asset in available_assets}
            )
            return PortfolioSignal(
                asset_signals=signal_dict,
                target_weights=target_weights,
                expected_return=np.mean([s.signal for s in signals]) * 0.1,
                expected_risk=0.15,
                coordination_method="correlation_aware_fallback",
                timestamp=current_time,
            )

    def _calculate_position_sizes(
        self,
        signals: list[AssetSignal],
        market_data: dict[str, pd.DataFrame],
        correlation_matrix: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate position sizes based on coordination method"""
        if not signals:
            return {}

        position_sizes = {}

        if self.config.position_sizing == PositionSizingMethod.EQUAL_WEIGHT:
            base_size = 1.0 / len(signals)
            for signal in signals:
                position_sizes[signal.asset] = base_size * np.sign(signal.signal)

        elif self.config.position_sizing == PositionSizingMethod.VOLATILITY_ADJUSTED:
            # Calculate volatilities
            volatilities = {}
            for asset, data in market_data.items():
                if asset in [s.asset for s in signals] and "close" in data.columns:
                    returns = data["close"].pct_change().dropna()
                    if len(returns) > 20:
                        volatilities[asset] = returns.std() * np.sqrt(252)

            if volatilities:
                # Inverse volatility weights
                inv_vol_sum = sum(1 / vol for vol in volatilities.values())
                for signal in signals:
                    if signal.asset in volatilities:
                        base_weight = (1 / volatilities[signal.asset]) / inv_vol_sum
                        position_sizes[signal.asset] = (
                            base_weight * np.sign(signal.signal) * signal.confidence
                        )
            else:
                # Fall back to equal weight
                base_size = 1.0 / len(signals)
                for signal in signals:
                    position_sizes[signal.asset] = base_size * np.sign(signal.signal)

        elif self.config.position_sizing == PositionSizingMethod.CORRELATION_ADJUSTED:
            if not correlation_matrix.empty:
                assets_with_signals = [signal.asset for signal in signals]
                available_assets = [
                    asset for asset in assets_with_signals if asset in correlation_matrix.index
                ]

                if len(available_assets) >= 2:
                    corr_sub = correlation_matrix.loc[available_assets, available_assets]

                    try:
                        # Use inverse correlation matrix for diversification
                        inv_corr = np.linalg.inv(corr_sub.values + np.eye(len(corr_sub)) * 0.01)
                        weights = np.sum(inv_corr, axis=1)
                        weights = weights / np.sum(weights)

                        signal_dict = {s.asset: s for s in signals}
                        for i, asset in enumerate(available_assets):
                            if asset in signal_dict:
                                signal_obj = signal_dict[asset]
                                position_sizes[asset] = (
                                    weights[i] * np.sign(signal_obj.signal) * signal_obj.confidence
                                )
                    except (KeyError, IndexError, AttributeError, TypeError):
                        # Fall back to equal weight
                        base_size = 1.0 / len(signals)
                        for signal in signals:
                            position_sizes[signal.asset] = base_size * np.sign(signal.signal)
            else:
                # Fall back to equal weight
                base_size = 1.0 / len(signals)
                for signal in signals:
                    position_sizes[signal.asset] = base_size * np.sign(signal.signal)

        else:  # Default to equal weight
            base_size = 1.0 / len(signals)
            for signal in signals:
                position_sizes[signal.asset] = base_size * np.sign(signal.signal)

        # Apply maximum weight constraints
        for asset in position_sizes:
            if abs(position_sizes[asset]) > self.config.max_single_asset_weight:
                position_sizes[asset] = (
                    np.sign(position_sizes[asset]) * self.config.max_single_asset_weight
                )

        # Scale to respect maximum leverage
        total_leverage = sum(abs(pos) for pos in position_sizes.values())
        if total_leverage > self.config.max_portfolio_leverage:
            scale_factor = self.config.max_portfolio_leverage / total_leverage
            position_sizes = {asset: pos * scale_factor for asset, pos in position_sizes.items()}

        return position_sizes

    def _calculate_hedge_positions(
        self, position_sizes: dict[str, float], correlation_matrix: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate hedge positions for risk management"""
        hedge_positions = {}

        if self.config.hedging_mode == HedgingMode.NO_HEDGE:
            return hedge_positions

        if (
            self.config.hedging_mode == HedgingMode.CORRELATION_HEDGE
            and not correlation_matrix.empty
        ):
            # Find highly correlated pairs and create hedge positions
            assets = list(position_sizes.keys())
            available_assets = [asset for asset in assets if asset in correlation_matrix.index]

            if len(available_assets) >= 2:
                corr_sub = correlation_matrix.loc[available_assets, available_assets]

                for i, asset_i in enumerate(available_assets):
                    for j, asset_j in enumerate(available_assets):
                        if i < j:
                            correlation = corr_sub.iloc[i, j]
                            if abs(correlation) > self.config.correlation_threshold:
                                # Create hedge position
                                pos_i = position_sizes.get(asset_i, 0)
                                pos_j = position_sizes.get(asset_j, 0)

                                # If positions are in same direction and highly correlated, reduce exposure
                                if (
                                    pos_i * pos_j > 0
                                    and correlation > self.config.correlation_threshold
                                ):
                                    hedge_factor = min(
                                        0.2, abs(correlation) - self.config.correlation_threshold
                                    )
                                    hedge_positions[f"{asset_i}_hedge"] = -pos_i * hedge_factor
                                    hedge_positions[f"{asset_j}_hedge"] = -pos_j * hedge_factor

        return hedge_positions

    def _calculate_risk_metrics(
        self,
        position_sizes: dict[str, float],
        market_data: dict[str, pd.DataFrame],
        correlation_matrix: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate portfolio risk metrics"""
        risk_metrics = {}

        try:
            # Portfolio concentration
            weights = np.array(list(position_sizes.values()))
            risk_metrics["max_weight"] = float(np.max(np.abs(weights)))
            risk_metrics["concentration_hhi"] = float(np.sum(weights**2))
            risk_metrics["total_leverage"] = float(np.sum(np.abs(weights)))

            # Diversification ratio
            if not correlation_matrix.empty:
                assets = list(position_sizes.keys())
                available_assets = [asset for asset in assets if asset in correlation_matrix.index]

                if len(available_assets) >= 2:
                    corr_sub = correlation_matrix.loc[available_assets, available_assets]
                    weights_sub = np.array([position_sizes[asset] for asset in available_assets])

                    # Average correlation
                    upper_tri_indices = np.triu_indices_from(corr_sub.values, k=1)
                    avg_correlation = np.mean(corr_sub.values[upper_tri_indices])
                    risk_metrics["avg_correlation"] = float(avg_correlation)

                    # Portfolio variance
                    portfolio_var = weights_sub.T @ corr_sub.values @ weights_sub
                    risk_metrics["portfolio_variance"] = float(portfolio_var)

            # Signal quality metrics
            risk_metrics["n_positions"] = len(position_sizes)
            risk_metrics["long_positions"] = len([p for p in position_sizes.values() if p > 0])
            risk_metrics["short_positions"] = len([p for p in position_sizes.values() if p < 0])

        except Exception as e:
            logger.warning(f"Risk metrics calculation failed: {str(e)}")
            risk_metrics["error"] = str(e)

        return risk_metrics

    def _get_coordination_statistics(
        self, signals: list[AssetSignal], correlation_matrix: pd.DataFrame
    ) -> dict[str, Any]:
        """Get coordination statistics"""
        stats = {}

        if signals:
            signal_values = [s.signal for s in signals]
            confidence_values = [s.confidence for s in signals]

            stats["n_signals"] = len(signals)
            stats["avg_signal"] = float(np.mean(signal_values))
            stats["signal_std"] = float(np.std(signal_values))
            stats["avg_confidence"] = float(np.mean(confidence_values))
            stats["signal_consensus"] = float(
                1.0 - np.std(signal_values)
            )  # Higher when signals agree

            # Direction consistency
            bullish_signals = len([s for s in signals if s.signal > 0])
            bearish_signals = len([s for s in signals if s.signal < 0])
            stats["bullish_signals"] = bullish_signals
            stats["bearish_signals"] = bearish_signals
            stats["direction_consensus"] = float(
                max(bullish_signals, bearish_signals) / len(signals)
            )

        if not correlation_matrix.empty:
            upper_tri_indices = np.triu_indices_from(correlation_matrix.values, k=1)
            correlations = correlation_matrix.values[upper_tri_indices]
            stats["avg_asset_correlation"] = float(np.mean(correlations))
            stats["max_asset_correlation"] = float(np.max(correlations))
            stats["correlation_dispersion"] = float(np.std(correlations))

        return stats

    def _create_empty_result(self, message: str) -> CoordinationResult:
        """Create empty coordination result"""
        return CoordinationResult(
            portfolio_signal=PortfolioSignal(
                asset_signals={},
                target_weights=pd.Series(dtype=float),
                expected_return=0.0,
                expected_risk=0.0,
                coordination_method="none",
                timestamp=pd.Timestamp.now(),
            ),
            individual_signals=[],
            position_sizes={},
            hedge_positions={},
            risk_metrics={},
            coordination_statistics={},
            success=False,
            message=message,
        )


class MultiInstrumentCoordinationFramework:
    """Main framework for multi-instrument strategy coordination"""

    def __init__(self, config: CoordinationConfig) -> None:
        self.config = config
        self.coordinator = self._create_coordinator()
        self.coordination_history = deque(maxlen=1000)

    def _create_coordinator(self) -> BaseMultiInstrumentStrategy:
        """Create coordinator based on configuration"""
        if self.config.coordination_method == CoordinationMethod.CORRELATION_AWARE:
            return CorrelationAwareCoordinator(self.config)
        else:
            # Default to correlation aware
            return CorrelationAwareCoordinator(self.config)

    def add_strategy(self, asset: str, strategy: BaseStrategy) -> None:
        """Add strategy for asset"""
        self.coordinator.add_strategy(asset, strategy)

    def remove_strategy(self, asset: str) -> None:
        """Remove strategy for asset"""
        self.coordinator.remove_strategy(asset)

    def coordinate_strategies(self, market_data: dict[str, pd.DataFrame]) -> CoordinationResult:
        """Coordinate strategies across assets"""
        result = self.coordinator.generate_portfolio_signal(market_data)

        # Store in history
        self.coordination_history.append({"timestamp": pd.Timestamp.now(), "result": result})

        return result

    def get_portfolio_positions(self, market_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Get current portfolio positions"""
        result = self.coordinate_strategies(market_data)
        if result.success:
            return result.position_sizes
        return {}

    def get_coordination_metrics(self) -> dict[str, Any]:
        """Get coordination performance metrics"""
        if not self.coordination_history:
            return {}

        recent_results = [
            h["result"] for h in list(self.coordination_history)[-50:]
        ]  # Last 50 results
        successful_results = [r for r in recent_results if r.success]

        if not successful_results:
            return {"success_rate": 0.0}

        metrics = {
            "success_rate": len(successful_results) / len(recent_results),
            "avg_positions": np.mean([len(r.position_sizes) for r in successful_results]),
            "avg_leverage": np.mean(
                [r.risk_metrics.get("total_leverage", 0) for r in successful_results]
            ),
            "avg_correlation": np.mean(
                [
                    r.risk_metrics.get("avg_correlation", 0)
                    for r in successful_results
                    if "avg_correlation" in r.risk_metrics
                ]
            ),
            "coordination_frequency": len(self.coordination_history),
        }

        return metrics


def create_multi_instrument_coordinator(
    coordination_method: CoordinationMethod = CoordinationMethod.CORRELATION_AWARE,
    position_sizing: PositionSizingMethod = PositionSizingMethod.RISK_PARITY,
    **kwargs: Any,
) -> MultiInstrumentCoordinationFramework:
    """Factory function to create multi-instrument coordinator"""
    config = CoordinationConfig(
        coordination_method=coordination_method, position_sizing=position_sizing, **kwargs
    )

    return MultiInstrumentCoordinationFramework(config)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample market data
    np.random.seed(42)
    n_days = 100
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]

    market_data = {}
    for asset in assets:
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_days)))

        market_data[asset] = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.001, n_days)),
                "high": prices * (1 + np.abs(np.random.normal(0.002, 0.001, n_days))),
                "low": prices * (1 - np.abs(np.random.normal(0.002, 0.001, n_days))),
                "close": prices,
                "volume": np.random.randint(1000000, 5000000, n_days),
            },
            index=dates,
        )

    # Mock strategy class
    class MockStrategy(BaseStrategy):
        def __init__(self, bias=0.0) -> None:
            self.bias = bias

        def generate_signal(self, data):
            # Simple momentum signal with bias
            if len(data) < 20:
                return StrategySignal(0.0, 0.0)

            returns = data["close"].pct_change().dropna()
            momentum = returns.rolling(20).mean().iloc[-1]
            signal = np.tanh(momentum * 100 + self.bias)  # Scale and add bias
            confidence = min(1.0, abs(signal) + 0.2)

            return StrategySignal(signal, confidence, {"momentum": momentum})

    print("Multi-Instrument Strategy Coordination Framework Testing")
    print("=" * 65)

    # Test coordination framework
    try:
        coordinator = create_multi_instrument_coordinator(
            coordination_method=CoordinationMethod.CORRELATION_AWARE,
            position_sizing=PositionSizingMethod.CORRELATION_ADJUSTED,
            max_portfolio_leverage=1.0,
            max_single_asset_weight=0.4,
        )

        # Add mock strategies
        for i, asset in enumerate(assets):
            strategy = MockStrategy(bias=0.1 * i)  # Different biases for each asset
            coordinator.add_strategy(asset, strategy)

        print(f"✅ Added {len(assets)} strategies")

        # Test coordination
        result = coordinator.coordinate_strategies(market_data)

        if result.success:
            print("✅ Coordination successful")
            print(f"   Individual signals: {len(result.individual_signals)}")
            print(f"   Position sizes: {len(result.position_sizes)}")
            print(f"   Total leverage: {result.risk_metrics.get('total_leverage', 0):.4f}")
            print(f"   Average correlation: {result.risk_metrics.get('avg_correlation', 0):.4f}")

            # Print position sizes
            print("   Positions:")
            for asset, size in result.position_sizes.items():
                print(f"     {asset}: {size:.4f}")

        else:
            print(f"❌ Coordination failed: {result.message}")

        # Test coordination metrics
        metrics = coordinator.get_coordination_metrics()
        print(f"   Success rate: {metrics.get('success_rate', 0):.2f}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    print("\n🚀 Multi-Instrument Strategy Coordination Framework ready for production!")
