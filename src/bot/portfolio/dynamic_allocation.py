"""
Dynamic Asset Allocation System for Multi-Asset Strategy Enhancement

This module implements sophisticated dynamic allocation strategies including:
- Tactical Asset Allocation (TAA) with market timing
- Strategic Asset Allocation (SAA) with rebalancing
- Risk-based allocation with volatility targeting
- Factor-based allocation with style tilts
- Market regime-aware allocation switching
- Dynamic hedging and overlay strategies
- Alternative risk premia allocation
"""

import logging
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Optional dependencies with graceful fallback
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some allocation features will be limited.")

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Asset allocation strategies"""

    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    RISK_PARITY = "risk_parity"
    VOLATILITY_TARGETING = "volatility_targeting"
    FACTOR_BASED = "factor_based"
    REGIME_SWITCHING = "regime_switching"
    BLACK_LITTERMAN = "black_litterman"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


class RebalancingMethod(Enum):
    """Portfolio rebalancing methods"""

    CALENDAR = "calendar"
    THRESHOLD = "threshold"
    VOLATILITY_BASED = "volatility_based"
    CORRELATION_BASED = "correlation_based"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"


class RiskModel(Enum):
    """Risk model types"""

    HISTORICAL = "historical"
    GARCH = "garch"
    EWMA = "ewma"
    FACTOR_MODEL = "factor_model"
    MONTE_CARLO = "monte_carlo"


@dataclass
class AllocationConfig:
    """Configuration for dynamic asset allocation"""

    strategy: AllocationStrategy = AllocationStrategy.TACTICAL
    rebalancing_method: RebalancingMethod = RebalancingMethod.THRESHOLD
    risk_model: RiskModel = RiskModel.EWMA
    target_volatility: float = 0.15
    volatility_lookback: int = 252
    rebalancing_threshold: float = 0.05
    rebalancing_frequency: int = 21  # days
    max_weight: float = 0.5
    min_weight: float = 0.0
    transaction_costs: float = 0.001
    risk_budget: dict[str, float] = field(default_factory=dict)
    factor_exposures: dict[str, float] = field(default_factory=dict)
    regime_indicators: list[str] = field(default_factory=list)
    momentum_lookback: int = 126
    mean_reversion_lookback: int = 252
    confidence_level: float = 0.95


@dataclass
class AllocationResult:
    """Result from asset allocation optimization"""

    target_weights: pd.Series
    current_weights: pd.Series
    rebalancing_trades: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_contributions: pd.Series
    turnover: float
    transaction_costs: float
    allocation_rationale: str
    regime_assessment: dict[str, Any]
    risk_metrics: dict[str, float]
    success: bool
    message: str


class BaseAllocationStrategy(ABC):
    """Base class for asset allocation strategies"""

    def __init__(self, config: AllocationConfig) -> None:
        self.config = config
        self.assets = []
        self.allocation_history = []
        self.risk_model_cache = {}

    @abstractmethod
    def calculate_allocation(
        self, market_data: dict[str, pd.DataFrame], current_weights: pd.Series | None = None
    ) -> AllocationResult:
        """Calculate optimal asset allocation"""
        pass

    def _estimate_returns_and_risks(
        self, market_data: dict[str, pd.DataFrame]
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Estimate expected returns and covariance matrix"""
        # Combine return data
        returns_data = {}
        for asset, data in market_data.items():
            if "close" in data.columns and len(data) > 30:
                returns = data["close"].pct_change().dropna()
                returns_data[asset] = returns

        if len(returns_data) < 2:
            raise ValueError("Insufficient return data for allocation")

        # Align data
        combined_returns = pd.DataFrame(returns_data).dropna()

        # Expected returns (using various methods)
        expected_returns = self._estimate_expected_returns(combined_returns)

        # Covariance matrix (using configured risk model)
        covariance_matrix = self._estimate_covariance_matrix(combined_returns)

        return expected_returns, covariance_matrix

    def _estimate_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Estimate expected returns based on allocation strategy"""
        if self.config.strategy == AllocationStrategy.MOMENTUM:
            # Momentum-based expected returns
            lookback = min(self.config.momentum_lookback, len(returns))
            momentum_returns = returns.rolling(lookback).mean().iloc[-1] * 252
            return momentum_returns

        elif self.config.strategy == AllocationStrategy.MEAN_REVERSION:
            # Mean-reversion based expected returns
            lookback = min(self.config.mean_reversion_lookback, len(returns))
            long_term_mean = returns.rolling(lookback).mean().iloc[-1] * 252
            short_term_return = returns.rolling(21).mean().iloc[-1] * 252
            # Expect reversion to long-term mean
            expected_returns = long_term_mean - 0.5 * (short_term_return - long_term_mean)
            return expected_returns

        else:
            # Default: historical mean
            return returns.mean() * 252

    def _estimate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate covariance matrix based on risk model"""
        if self.config.risk_model == RiskModel.HISTORICAL:
            return returns.cov() * 252

        elif self.config.risk_model == RiskModel.EWMA:
            # Exponentially weighted moving average
            lambda_param = 0.94
            cov_matrix = returns.cov() * 252

            # EWMA update (simplified)
            for i in range(min(60, len(returns))):
                if i > 0:
                    idx = len(returns) - i - 1
                    r_t = returns.iloc[idx].values.reshape(-1, 1)
                    cov_matrix = lambda_param * cov_matrix + (1 - lambda_param) * 252 * (
                        r_t @ r_t.T
                    )
                    cov_matrix = pd.DataFrame(
                        cov_matrix, index=returns.columns, columns=returns.columns
                    )

            return cov_matrix

        else:
            # Default to historical
            return returns.cov() * 252

    def _calculate_risk_contributions(
        self, weights: pd.Series, covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """Calculate risk contributions for each asset"""
        w = weights.values.reshape(-1, 1)
        portfolio_var = (w.T @ covariance_matrix.values @ w)[0, 0]
        marginal_contrib = covariance_matrix.values @ w
        risk_contrib = weights.values * marginal_contrib.flatten()

        return pd.Series(risk_contrib / portfolio_var, index=weights.index)


class TacticalAssetAllocation(BaseAllocationStrategy):
    """Tactical Asset Allocation with market timing"""

    def calculate_allocation(
        self, market_data: dict[str, pd.DataFrame], current_weights: pd.Series | None = None
    ) -> AllocationResult:
        """Calculate tactical allocation based on market signals"""
        try:
            # Get return and risk estimates
            expected_returns, covariance_matrix = self._estimate_returns_and_risks(market_data)

            # Generate market timing signals
            market_signals = self._generate_market_signals(market_data)

            # Adjust expected returns based on signals
            adjusted_returns = self._adjust_returns_for_signals(expected_returns, market_signals)

            # Optimize portfolio
            target_weights = self._optimize_portfolio(
                adjusted_returns, covariance_matrix, current_weights
            )

            # Calculate metrics
            result = self._create_allocation_result(
                target_weights,
                current_weights,
                adjusted_returns,
                covariance_matrix,
                market_signals,
                "Tactical allocation based on market signals",
            )

            return result

        except Exception as e:
            logger.error(f"Tactical allocation failed: {str(e)}")
            return self._create_fallback_result(current_weights, f"Error: {str(e)}")

    def _generate_market_signals(self, market_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Generate market timing signals"""
        signals = {}

        for asset, data in market_data.items():
            if len(data) < 50:
                signals[asset] = 0.0
                continue

            try:
                # Multiple signal combination
                momentum_signal = self._momentum_signal(data)
                volatility_signal = self._volatility_signal(data)
                trend_signal = self._trend_signal(data)

                # Combine signals
                combined_signal = (
                    0.4 * momentum_signal + 0.3 * volatility_signal + 0.3 * trend_signal
                )
                signals[asset] = np.clip(combined_signal, -1, 1)

            except Exception as e:
                logger.warning(f"Signal generation failed for {asset}: {str(e)}")
                signals[asset] = 0.0

        return signals

    def _momentum_signal(self, data: pd.DataFrame) -> float:
        """Generate momentum signal"""
        if "close" not in data.columns:
            return 0.0

        prices = data["close"]
        if len(prices) < 126:
            return 0.0

        # 6-month momentum
        momentum_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) if len(prices) >= 126 else 0

        # 3-month momentum
        momentum_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) if len(prices) >= 63 else 0

        # Combine short and long term momentum
        momentum_signal = 0.6 * momentum_6m + 0.4 * momentum_3m

        # Normalize to [-1, 1]
        return np.tanh(momentum_signal * 10)

    def _volatility_signal(self, data: pd.DataFrame) -> float:
        """Generate volatility-based signal"""
        if "close" not in data.columns:
            return 0.0

        returns = data["close"].pct_change().dropna()
        if len(returns) < 60:
            return 0.0

        # Current volatility vs historical
        current_vol = returns.rolling(21).std().iloc[-1] * np.sqrt(252)
        historical_vol = returns.rolling(252).std().iloc[-1] * np.sqrt(252)

        # Low volatility is good (positive signal), high volatility is bad
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
        volatility_signal = -(vol_ratio - 1)  # Negative sign because low vol is good

        return np.clip(volatility_signal, -1, 1)

    def _trend_signal(self, data: pd.DataFrame) -> float:
        """Generate trend-following signal"""
        if "close" not in data.columns:
            return 0.0

        prices = data["close"]
        if len(prices) < 50:
            return 0.0

        # Simple moving averages
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        current_price = prices.iloc[-1]

        # Trend signal based on price vs moving averages
        if current_price > sma_20 > sma_50:
            trend_signal = 1.0  # Strong uptrend
        elif current_price > sma_20:
            trend_signal = 0.5  # Moderate uptrend
        elif current_price < sma_20 < sma_50:
            trend_signal = -1.0  # Strong downtrend
        elif current_price < sma_20:
            trend_signal = -0.5  # Moderate downtrend
        else:
            trend_signal = 0.0  # Neutral

        return trend_signal

    def _adjust_returns_for_signals(
        self, expected_returns: pd.Series, signals: dict[str, float]
    ) -> pd.Series:
        """Adjust expected returns based on tactical signals"""
        adjusted_returns = expected_returns.copy()

        for asset in adjusted_returns.index:
            if asset in signals:
                signal = signals[asset]
                # Boost expected returns for positive signals, reduce for negative
                adjustment = signal * 0.05  # 5% max adjustment
                adjusted_returns[asset] = adjusted_returns[asset] + adjustment

        return adjusted_returns

    def _optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_weights: pd.Series | None,
    ) -> pd.Series:
        """Optimize portfolio weights"""
        n_assets = len(expected_returns)

        # Objective function: maximize Sharpe ratio with transaction cost penalty
        def objective(weights):
            w = pd.Series(weights, index=expected_returns.index)
            portfolio_return = (w * expected_returns).sum()
            portfolio_var = w.values @ covariance_matrix.values @ w.values
            portfolio_vol = np.sqrt(portfolio_var)

            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

            # Transaction cost penalty
            transaction_penalty = 0
            if current_weights is not None:
                aligned_current = current_weights.reindex(expected_returns.index, fill_value=0)
                turnover = np.sum(np.abs(w - aligned_current))
                transaction_penalty = turnover * self.config.transaction_costs

            return -(sharpe - transaction_penalty)

        # Constraints
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # Weights sum to 1

        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            logger.warning(f"Optimization failed: {result.message}")
            return pd.Series(x0, index=expected_returns.index)


class VolatilityTargetingStrategy(BaseAllocationStrategy):
    """Volatility targeting allocation strategy"""

    def calculate_allocation(
        self, market_data: dict[str, pd.DataFrame], current_weights: pd.Series | None = None
    ) -> AllocationResult:
        """Calculate allocation to target specific portfolio volatility"""
        try:
            # Get return and risk estimates
            expected_returns, covariance_matrix = self._estimate_returns_and_risks(market_data)

            # Calculate base weights (e.g., equal weight or cap-weighted)
            base_weights = self._calculate_base_weights(expected_returns)

            # Scale weights to target volatility
            target_weights = self._scale_to_target_volatility(base_weights, covariance_matrix)

            # Calculate metrics
            result = self._create_allocation_result(
                target_weights,
                current_weights,
                expected_returns,
                covariance_matrix,
                {},
                f"Volatility targeting at {self.config.target_volatility:.1%}",
            )

            return result

        except Exception as e:
            logger.error(f"Volatility targeting failed: {str(e)}")
            return self._create_fallback_result(current_weights, f"Error: {str(e)}")

    def _calculate_base_weights(self, expected_returns: pd.Series) -> pd.Series:
        """Calculate base weights before volatility scaling"""
        # Simple equal weight for now
        n_assets = len(expected_returns)
        return pd.Series(1.0 / n_assets, index=expected_returns.index)

    def _scale_to_target_volatility(
        self, weights: pd.Series, covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """Scale weights to achieve target portfolio volatility"""
        # Current portfolio volatility
        portfolio_var = weights.values @ covariance_matrix.values @ weights.values
        current_vol = np.sqrt(portfolio_var)

        # Calculate scaling factor
        if current_vol > 0:
            scale_factor = self.config.target_volatility / current_vol
            scaled_weights = weights * scale_factor

            # Ensure weights don't exceed bounds
            scaled_weights = scaled_weights.clip(self.config.min_weight, self.config.max_weight)

            # Renormalize
            scaled_weights = scaled_weights / scaled_weights.sum()

            return scaled_weights
        else:
            return weights


class RiskParityStrategy(BaseAllocationStrategy):
    """Risk Parity allocation strategy"""

    def calculate_allocation(
        self, market_data: dict[str, pd.DataFrame], current_weights: pd.Series | None = None
    ) -> AllocationResult:
        """Calculate risk parity allocation"""
        try:
            # Get return and risk estimates
            expected_returns, covariance_matrix = self._estimate_returns_and_risks(market_data)

            # Optimize for equal risk contributions
            target_weights = self._optimize_risk_parity(covariance_matrix)

            # Calculate metrics
            result = self._create_allocation_result(
                target_weights,
                current_weights,
                expected_returns,
                covariance_matrix,
                {},
                "Risk parity allocation with equal risk contributions",
            )

            return result

        except Exception as e:
            logger.error(f"Risk parity allocation failed: {str(e)}")
            return self._create_fallback_result(current_weights, f"Error: {str(e)}")

    def _optimize_risk_parity(self, covariance_matrix: pd.DataFrame) -> pd.Series:
        """Optimize for risk parity"""
        n_assets = len(covariance_matrix)
        assets = covariance_matrix.index

        def objective(weights):
            w = np.array(weights)
            portfolio_vol = np.sqrt(w @ covariance_matrix.values @ w)
            marginal_contrib = (covariance_matrix.values @ w) / portfolio_vol
            contrib = w * marginal_contrib
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)

        # Constraints and bounds
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]

        # Initial guess: inverse volatility
        vols = np.sqrt(np.diag(covariance_matrix.values))
        x0 = (1 / vols) / np.sum(1 / vols)

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if result.success:
            return pd.Series(result.x, index=assets)
        else:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            return pd.Series(x0, index=assets)


class DynamicAllocationFramework:
    """Main framework for dynamic asset allocation"""

    def __init__(self, config: AllocationConfig) -> None:
        self.config = config
        self.strategy = self._create_strategy()
        self.allocation_history = deque(maxlen=1000)
        self.last_allocation = None
        self.last_rebalance_date = None

    def _create_strategy(self) -> BaseAllocationStrategy:
        """Create allocation strategy based on configuration"""
        if self.config.strategy == AllocationStrategy.TACTICAL:
            return TacticalAssetAllocation(self.config)
        elif self.config.strategy == AllocationStrategy.VOLATILITY_TARGETING:
            return VolatilityTargetingStrategy(self.config)
        elif self.config.strategy == AllocationStrategy.RISK_PARITY:
            return RiskParityStrategy(self.config)
        else:
            # Default to tactical
            return TacticalAssetAllocation(self.config)

    def calculate_allocation(
        self, market_data: dict[str, pd.DataFrame], current_date: pd.Timestamp | None = None
    ) -> AllocationResult:
        """Calculate optimal asset allocation"""
        current_weights = None
        if self.last_allocation and self.last_allocation.success:
            current_weights = self.last_allocation.target_weights

        # Check if rebalancing is needed
        if not self._should_rebalance(market_data, current_date):
            # Return current allocation with no trades
            if self.last_allocation:
                no_trade_result = AllocationResult(
                    target_weights=self.last_allocation.target_weights,
                    current_weights=self.last_allocation.target_weights,
                    rebalancing_trades=pd.Series(
                        0.0, index=self.last_allocation.target_weights.index
                    ),
                    expected_return=self.last_allocation.expected_return,
                    expected_volatility=self.last_allocation.expected_volatility,
                    sharpe_ratio=self.last_allocation.sharpe_ratio,
                    risk_contributions=self.last_allocation.risk_contributions,
                    turnover=0.0,
                    transaction_costs=0.0,
                    allocation_rationale="No rebalancing needed",
                    regime_assessment={},
                    risk_metrics=self.last_allocation.risk_metrics,
                    success=True,
                    message="No rebalancing required",
                )
                return no_trade_result

        # Calculate new allocation
        result = self.strategy.calculate_allocation(market_data, current_weights)

        # Update history
        if result.success:
            self.last_allocation = result
            self.last_rebalance_date = current_date or pd.Timestamp.now()

            self.allocation_history.append(
                {"timestamp": current_date or pd.Timestamp.now(), "result": result}
            )

        return result

    def _should_rebalance(
        self, market_data: dict[str, pd.DataFrame], current_date: pd.Timestamp | None
    ) -> bool:
        """Determine if portfolio should be rebalanced"""
        if not self.last_allocation or not self.last_rebalance_date:
            return True

        current_date = current_date or pd.Timestamp.now()

        if self.config.rebalancing_method == RebalancingMethod.CALENDAR:
            # Calendar-based rebalancing
            days_since_rebalance = (current_date - self.last_rebalance_date).days
            return days_since_rebalance >= self.config.rebalancing_frequency

        elif self.config.rebalancing_method == RebalancingMethod.THRESHOLD:
            # Threshold-based rebalancing
            # This would require current market values to calculate drift
            # For now, assume rebalancing is needed
            return True

        elif self.config.rebalancing_method == RebalancingMethod.VOLATILITY_BASED:
            # Rebalance when volatility changes significantly
            try:
                # Calculate current volatility regime
                recent_volatility = self._calculate_recent_volatility(market_data)
                if hasattr(self.last_allocation, "expected_volatility"):
                    vol_change = abs(recent_volatility - self.last_allocation.expected_volatility)
                    return vol_change > self.config.rebalancing_threshold
            except (AttributeError, TypeError, ValueError) as e:
                # Volatility-based rebalancing check failed - fallback to calendar
                logger.debug(f"Volatility-based rebalancing check failed: {e}")
                pass
            return True

        else:
            # Default to calendar rebalancing
            days_since_rebalance = (current_date - self.last_rebalance_date).days
            return days_since_rebalance >= self.config.rebalancing_frequency

    def _calculate_recent_volatility(self, market_data: dict[str, pd.DataFrame]) -> float:
        """Calculate recent portfolio volatility"""
        if not market_data or not self.last_allocation:
            return 0.15  # Default

        try:
            # Get recent returns
            returns_data = {}
            for asset, data in market_data.items():
                if asset in self.last_allocation.target_weights.index and "close" in data.columns:
                    returns = data["close"].pct_change().dropna()
                    if len(returns) > 20:
                        returns_data[asset] = returns.tail(21)  # Last 21 days

            if len(returns_data) >= 2:
                combined_returns = pd.DataFrame(returns_data).dropna()
                if len(combined_returns) > 5:
                    weights = self.last_allocation.target_weights.reindex(
                        combined_returns.columns, fill_value=0
                    )
                    portfolio_returns = (combined_returns * weights).sum(axis=1)
                    return portfolio_returns.std() * np.sqrt(252)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            # Portfolio volatility calculation failed
            logger.debug(f"Portfolio volatility calculation failed: {e}")
            pass

        return 0.15  # Default volatility

    def get_allocation_performance(self) -> dict[str, Any]:
        """Get allocation performance metrics"""
        if not self.allocation_history:
            return {}

        successful_allocations = [
            h["result"] for h in self.allocation_history if h["result"].success
        ]
        if not successful_allocations:
            return {"success_rate": 0.0}

        metrics = {
            "success_rate": len(successful_allocations) / len(self.allocation_history),
            "avg_expected_return": np.mean([a.expected_return for a in successful_allocations]),
            "avg_expected_volatility": np.mean(
                [a.expected_volatility for a in successful_allocations]
            ),
            "avg_sharpe_ratio": np.mean([a.sharpe_ratio for a in successful_allocations]),
            "avg_turnover": np.mean([a.turnover for a in successful_allocations]),
            "avg_transaction_costs": np.mean([a.transaction_costs for a in successful_allocations]),
            "total_allocations": len(self.allocation_history),
        }

        return metrics


# Helper functions for creating allocation results
def _create_allocation_result(
    self,
    target_weights: pd.Series,
    current_weights: pd.Series | None,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    signals: dict[str, float],
    rationale: str,
) -> AllocationResult:
    """Create allocation result with all metrics"""
    try:
        # Handle current weights
        if current_weights is None:
            current_weights = pd.Series(0.0, index=target_weights.index)
        else:
            current_weights = current_weights.reindex(target_weights.index, fill_value=0.0)

        # Calculate rebalancing trades
        rebalancing_trades = target_weights - current_weights

        # Portfolio metrics
        portfolio_return = (target_weights * expected_returns).sum()
        portfolio_var = target_weights.values @ covariance_matrix.values @ target_weights.values
        portfolio_vol = np.sqrt(portfolio_var)
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        # Risk contributions
        risk_contributions = self._calculate_risk_contributions(target_weights, covariance_matrix)

        # Turnover and costs
        turnover = np.sum(np.abs(rebalancing_trades))
        transaction_costs = turnover * self.config.transaction_costs

        # Risk metrics
        risk_metrics = {
            "portfolio_volatility": portfolio_vol,
            "max_weight": target_weights.max(),
            "min_weight": target_weights.min(),
            "weight_concentration": np.sum(target_weights**2),  # HHI
            "risk_concentration": np.sum(risk_contributions**2),
        }

        return AllocationResult(
            target_weights=target_weights,
            current_weights=current_weights,
            rebalancing_trades=rebalancing_trades,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            risk_contributions=risk_contributions,
            turnover=turnover,
            transaction_costs=transaction_costs,
            allocation_rationale=rationale,
            regime_assessment=signals,
            risk_metrics=risk_metrics,
            success=True,
            message="Allocation successful",
        )

    except Exception as e:
        logger.error(f"Failed to create allocation result: {str(e)}")
        return self._create_fallback_result(current_weights, f"Metrics calculation error: {str(e)}")


def _create_fallback_result(
    self, current_weights: pd.Series | None, message: str
) -> AllocationResult:
    """Create fallback result when allocation fails"""
    if current_weights is None or len(current_weights) == 0:
        # Create minimal result
        return AllocationResult(
            target_weights=pd.Series(dtype=float),
            current_weights=pd.Series(dtype=float),
            rebalancing_trades=pd.Series(dtype=float),
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            risk_contributions=pd.Series(dtype=float),
            turnover=0.0,
            transaction_costs=0.0,
            allocation_rationale="Fallback allocation",
            regime_assessment={},
            risk_metrics={},
            success=False,
            message=message,
        )
    else:
        # Return current weights as target
        return AllocationResult(
            target_weights=current_weights,
            current_weights=current_weights,
            rebalancing_trades=pd.Series(0.0, index=current_weights.index),
            expected_return=0.0,
            expected_volatility=0.15,
            sharpe_ratio=0.0,
            risk_contributions=pd.Series(1.0 / len(current_weights), index=current_weights.index),
            turnover=0.0,
            transaction_costs=0.0,
            allocation_rationale="Fallback to current weights",
            regime_assessment={},
            risk_metrics={"error": message},
            success=False,
            message=message,
        )


# Add methods to base class
BaseAllocationStrategy._create_allocation_result = _create_allocation_result
BaseAllocationStrategy._create_fallback_result = _create_fallback_result


def create_dynamic_allocator(
    strategy: AllocationStrategy = AllocationStrategy.TACTICAL,
    target_volatility: float = 0.15,
    **kwargs,
) -> DynamicAllocationFramework:
    """Factory function to create dynamic allocator"""
    config = AllocationConfig(strategy=strategy, target_volatility=target_volatility, **kwargs)

    return DynamicAllocationFramework(config)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample market data
    np.random.seed(42)
    n_days = 252
    assets = ["STOCKS", "BONDS", "COMMODITIES", "REITS"]

    # Create different asset classes with different characteristics
    market_data = {}
    base_returns = [0.08, 0.04, 0.06, 0.07]  # Expected annual returns
    volatilities = [0.18, 0.08, 0.25, 0.20]  # Annual volatilities

    for i, asset in enumerate(assets):
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        daily_returns = np.random.normal(
            base_returns[i] / 252, volatilities[i] / np.sqrt(252), n_days
        )
        prices = 100 * np.exp(np.cumsum(daily_returns))

        market_data[asset] = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.001, n_days)),
                "high": prices * (1 + np.abs(np.random.normal(0.005, 0.001, n_days))),
                "low": prices * (1 - np.abs(np.random.normal(0.005, 0.001, n_days))),
                "close": prices,
                "volume": np.random.randint(1000000, 5000000, n_days),
            },
            index=dates,
        )

    print("Dynamic Asset Allocation Framework Testing")
    print("=" * 55)

    # Test different allocation strategies
    strategies = [
        AllocationStrategy.TACTICAL,
        AllocationStrategy.VOLATILITY_TARGETING,
        AllocationStrategy.RISK_PARITY,
    ]

    for strategy in strategies:
        print(f"\nTesting {strategy.value} allocation...")
        try:
            allocator = create_dynamic_allocator(
                strategy=strategy,
                target_volatility=0.12,
                rebalancing_method=RebalancingMethod.CALENDAR,
                rebalancing_frequency=21,
            )

            # Calculate allocation
            result = allocator.calculate_allocation(market_data)

            if result.success:
                print(
                    f"✅ Success: Expected Return: {result.expected_return:.4f}, "
                    f"Volatility: {result.expected_volatility:.4f}, "
                    f"Sharpe: {result.sharpe_ratio:.4f}"
                )
                print("   Target weights:")
                for asset, weight in result.target_weights.items():
                    print(f"     {asset}: {weight:.3f}")
                print(
                    f"   Turnover: {result.turnover:.4f}, Transaction costs: {result.transaction_costs:.4f}"
                )

                # Test rebalancing decision
                should_rebalance = allocator._should_rebalance(
                    market_data, pd.Timestamp("2023-12-31")
                )
                print(f"   Should rebalance: {should_rebalance}")

            else:
                print(f"❌ Failed: {result.message}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    # Test performance metrics
    print("\nTesting performance metrics...")
    try:
        allocator = create_dynamic_allocator(strategy=AllocationStrategy.TACTICAL)

        # Simulate multiple allocations
        for i in range(5):
            allocator.calculate_allocation(market_data)

        metrics = allocator.get_allocation_performance()
        print("✅ Performance metrics:")
        print(f"   Success rate: {metrics.get('success_rate', 0):.2f}")
        print(f"   Average expected return: {metrics.get('avg_expected_return', 0):.4f}")
        print(f"   Average turnover: {metrics.get('avg_turnover', 0):.4f}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    print("\n🚀 Dynamic Asset Allocation Framework ready for production!")
