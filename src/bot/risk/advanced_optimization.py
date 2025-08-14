"""
Risk-Adjusted Portfolio Optimization for Multi-Asset Strategy Enhancement

This module implements sophisticated risk-adjusted optimization techniques including:
- Conditional Value at Risk (CVaR) optimization
- Maximum Diversification optimization
- Robust optimization under uncertainty
- Factor-based risk modeling
- Tail risk management
- Downside deviation optimization
- Multi-objective optimization (return vs risk vs ESG)
- Dynamic risk budgeting
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Optional dependencies with graceful fallback
try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("CVXPY not available. Advanced optimization features will be limited.")

try:
    from sklearn.covariance import EmpiricalCovariance, LedoitWolf
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some risk modeling features will be limited.")

logger = logging.getLogger(__name__)


class RiskMeasure(Enum):
    """Risk measures for optimization"""

    VARIANCE = "variance"
    CVAR = "cvar"
    SEMI_VARIANCE = "semi_variance"
    DOWNSIDE_DEVIATION = "downside_deviation"
    MAX_DRAWDOWN = "max_drawdown"
    TAIL_RISK = "tail_risk"
    TRACKING_ERROR = "tracking_error"


class OptimizationType(Enum):
    """Types of risk-adjusted optimization"""

    MEAN_VARIANCE = "mean_variance"
    MEAN_CVAR = "mean_cvar"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_CORRELATION = "min_correlation"
    ROBUST_MEAN_VARIANCE = "robust_mean_variance"
    MULTI_OBJECTIVE = "multi_objective"
    FACTOR_RISK_PARITY = "factor_risk_parity"


class UncertaintyModel(Enum):
    """Models for handling parameter uncertainty"""

    NO_UNCERTAINTY = "no_uncertainty"
    ELLIPSOIDAL = "ellipsoidal"
    BOX = "box"
    FACTOR_UNCERTAINTY = "factor_uncertainty"
    BOOTSTRAP = "bootstrap"


@dataclass
class RiskOptimizationConfig:
    """Configuration for risk-adjusted optimization"""

    optimization_type: OptimizationType = OptimizationType.MEAN_CVAR
    risk_measure: RiskMeasure = RiskMeasure.CVAR
    uncertainty_model: UncertaintyModel = UncertaintyModel.ELLIPSOIDAL
    confidence_level: float = 0.95
    target_return: float | None = None
    max_weight: float = 0.3
    min_weight: float = 0.0
    max_leverage: float = 1.0
    transaction_costs: float = 0.001
    risk_aversion: float = 1.0
    uncertainty_level: float = 0.1  # Parameter uncertainty level
    robustness_level: float = 0.05  # Robustness constraint level
    downside_threshold: float = 0.0  # Threshold for downside measures
    max_concentration: float = 0.4  # Maximum sector/factor concentration
    lookback_period: int = 252
    simulation_samples: int = 10000  # For Monte Carlo methods
    max_iterations: int = 1000


@dataclass
class RiskOptimizationResult:
    """Result from risk-adjusted optimization"""

    optimal_weights: pd.Series
    expected_return: float
    expected_risk: float
    cvar: float
    max_drawdown: float
    sharpe_ratio: float
    risk_contributions: pd.Series
    factor_exposures: pd.Series | None = None
    tail_risk_metrics: dict[str, float] = field(default_factory=dict)
    robustness_metrics: dict[str, float] = field(default_factory=dict)
    optimization_statistics: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    message: str = ""


class BaseRiskOptimizer(ABC):
    """Base class for risk-adjusted optimizers"""

    def __init__(self, config: RiskOptimizationConfig) -> None:
        self.config = config
        self.factor_model = None
        self.uncertainty_sets = {}

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        benchmark: pd.Series | None = None,
    ) -> RiskOptimizationResult:
        """Perform risk-adjusted optimization"""
        pass

    def _estimate_parameters(self, returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """Estimate expected returns and covariance matrix"""
        # Expected returns
        if self.config.lookback_period and len(returns) > self.config.lookback_period:
            recent_returns = returns.tail(self.config.lookback_period)
        else:
            recent_returns = returns

        expected_returns = recent_returns.mean() * 252  # Annualized

        # Covariance matrix with shrinkage
        if SKLEARN_AVAILABLE:
            lw = LedoitWolf()
            cov_matrix, _shrinkage = lw.fit(recent_returns).covariance_, lw.shrinkage_
            cov_matrix = pd.DataFrame(
                cov_matrix * 252, index=returns.columns, columns=returns.columns
            )
        else:
            cov_matrix = recent_returns.cov() * 252

        return expected_returns, cov_matrix

    def _calculate_cvar(
        self, weights: pd.Series, returns: pd.DataFrame, alpha: float = 0.05
    ) -> float:
        """Calculate Conditional Value at Risk"""
        portfolio_returns = (returns * weights).sum(axis=1)
        var = np.percentile(portfolio_returns, alpha * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return -cvar * np.sqrt(252)  # Annualized positive CVaR

    def _calculate_downside_deviation(
        self, weights: pd.Series, returns: pd.DataFrame, threshold: float = 0.0
    ) -> float:
        """Calculate downside deviation"""
        portfolio_returns = (returns * weights).sum(axis=1)
        downside_returns = portfolio_returns[portfolio_returns < threshold / 252]  # Daily threshold
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, weights: pd.Series, returns: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        portfolio_returns = (returns * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return -drawdown.min()

    def _build_factor_model(self, returns: pd.DataFrame) -> dict[str, Any] | None:
        """Build factor model for risk attribution"""
        if not SKLEARN_AVAILABLE:
            return None

        try:
            # Use PCA for factor model
            n_factors = min(5, len(returns.columns) // 2)

            scaler = StandardScaler()
            scaled_returns = scaler.fit_transform(returns.values)

            pca = PCA(n_components=n_factors)
            factor_returns = pca.fit_transform(scaled_returns)

            # Factor loadings
            loadings = pd.DataFrame(
                pca.components_.T * np.sqrt(pca.explained_variance_),
                index=returns.columns,
                columns=[f"Factor_{i+1}" for i in range(n_factors)],
            )

            # Factor returns
            factor_returns_df = pd.DataFrame(
                factor_returns,
                index=returns.index,
                columns=[f"Factor_{i+1}" for i in range(n_factors)],
            )

            return {
                "loadings": loadings,
                "factor_returns": factor_returns_df,
                "explained_variance": pca.explained_variance_ratio_,
                "idiosyncratic_var": np.var(
                    scaled_returns - pca.inverse_transform(factor_returns), axis=0
                ),
            }

        except Exception as e:
            logger.warning(f"Factor model building failed: {e}")
            return None


class CVaROptimizer(BaseRiskOptimizer):
    """Conditional Value at Risk optimizer"""

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        benchmark: pd.Series | None = None,
    ) -> RiskOptimizationResult:
        """Optimize using CVaR as risk measure"""
        try:
            assets = returns.columns
            len(assets)
            len(returns)

            if expected_returns is None:
                expected_returns, _ = self._estimate_parameters(returns)

            # Use CVXPY if available for CVaR optimization
            if CVXPY_AVAILABLE:
                return self._cvxpy_cvar_optimization(returns, expected_returns, assets)
            else:
                return self._scipy_cvar_optimization(returns, expected_returns, assets)

        except Exception as e:
            logger.error(f"CVaR optimization failed: {str(e)}")
            return self._create_fallback_result(returns.columns, f"Error: {str(e)}")

    def _cvxpy_cvar_optimization(
        self, returns: pd.DataFrame, expected_returns: pd.Series, assets: pd.Index
    ) -> RiskOptimizationResult:
        """CVaR optimization using CVXPY"""
        n_assets = len(assets)
        n_scenarios = len(returns)
        alpha = 1 - self.config.confidence_level

        # Decision variables
        w = cp.Variable(n_assets)  # Portfolio weights
        z = cp.Variable()  # VaR
        u = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR

        # Scenario returns
        R = returns.values  # n_scenarios x n_assets

        # Portfolio returns for each scenario
        portfolio_returns = R @ w

        # CVaR constraints
        constraints = [
            u >= 0,
            u >= -(portfolio_returns) - z,  # Negative because we want losses
            cp.sum(w) == 1,
            w >= self.config.min_weight,
            w <= self.config.max_weight,
        ]

        # CVaR calculation
        cvar = z + cp.sum(u) / (alpha * n_scenarios)

        # Objective: maximize expected return - risk_aversion * CVaR
        portfolio_return = expected_returns.values @ w
        objective = cp.Maximize(portfolio_return - self.config.risk_aversion * cvar)

        # Target return constraint if specified
        if self.config.target_return is not None:
            constraints.append(portfolio_return >= self.config.target_return)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, max_iters=self.config.max_iterations)

        if problem.status == cp.OPTIMAL:
            optimal_weights = pd.Series(w.value, index=assets)

            # Calculate metrics
            result = self._calculate_optimization_metrics(
                optimal_weights, returns, expected_returns, "CVaR optimization successful"
            )
            return result

        else:
            raise ValueError(f"CVXPY optimization failed with status: {problem.status}")

    def _scipy_cvar_optimization(
        self, returns: pd.DataFrame, expected_returns: pd.Series, assets: pd.Index
    ) -> RiskOptimizationResult:
        """CVaR optimization using scipy (approximation)"""
        n_assets = len(assets)

        def objective(weights):
            w = pd.Series(weights, index=assets)
            # Approximate CVaR using historical simulation
            cvar = self._calculate_cvar(w, returns, alpha=1 - self.config.confidence_level)
            portfolio_return = (w * expected_returns).sum()
            return -(portfolio_return - self.config.risk_aversion * cvar)

        # Constraints
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        if self.config.target_return is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: (pd.Series(x, index=assets) * expected_returns).sum()
                    - self.config.target_return,
                }
            )

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
            options={"maxiter": self.config.max_iterations},
        )

        if result.success:
            optimal_weights = pd.Series(result.x, index=assets)
            return self._calculate_optimization_metrics(
                optimal_weights, returns, expected_returns, "Scipy CVaR optimization successful"
            )
        else:
            raise ValueError(f"Scipy optimization failed: {result.message}")


class MaxDiversificationOptimizer(BaseRiskOptimizer):
    """Maximum Diversification optimizer"""

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        benchmark: pd.Series | None = None,
    ) -> RiskOptimizationResult:
        """Optimize for maximum diversification ratio"""
        try:
            assets = returns.columns
            n_assets = len(assets)

            expected_returns, cov_matrix = self._estimate_parameters(returns)

            # Individual asset volatilities
            individual_vols = np.sqrt(np.diag(cov_matrix.values))

            def objective(weights):
                w = np.array(weights)
                # Weighted average of individual volatilities
                weighted_vol = np.sum(w * individual_vols)
                # Portfolio volatility
                portfolio_vol = np.sqrt(w @ cov_matrix.values @ w)
                # Diversification ratio (maximize)
                if portfolio_vol > 0:
                    return -weighted_vol / portfolio_vol
                else:
                    return -1

            # Constraints
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

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
                options={"maxiter": self.config.max_iterations},
            )

            if result.success:
                optimal_weights = pd.Series(result.x, index=assets)
                return self._calculate_optimization_metrics(
                    optimal_weights,
                    returns,
                    expected_returns,
                    "Max diversification optimization successful",
                )
            else:
                raise ValueError(f"Max diversification optimization failed: {result.message}")

        except Exception as e:
            logger.error(f"Max diversification optimization failed: {str(e)}")
            return self._create_fallback_result(returns.columns, f"Error: {str(e)}")


class RobustMeanVarianceOptimizer(BaseRiskOptimizer):
    """Robust Mean-Variance optimizer with parameter uncertainty"""

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        benchmark: pd.Series | None = None,
    ) -> RiskOptimizationResult:
        """Robust optimization under parameter uncertainty"""
        try:
            assets = returns.columns
            len(assets)

            expected_returns, cov_matrix = self._estimate_parameters(returns)

            # Build uncertainty sets
            uncertainty_set = self._build_uncertainty_set(expected_returns, cov_matrix, returns)

            # Robust optimization
            if CVXPY_AVAILABLE and self.config.uncertainty_model == UncertaintyModel.ELLIPSOIDAL:
                return self._robust_cvxpy_optimization(
                    returns, expected_returns, cov_matrix, uncertainty_set
                )
            else:
                return self._robust_scipy_optimization(
                    returns, expected_returns, cov_matrix, uncertainty_set
                )

        except Exception as e:
            logger.error(f"Robust optimization failed: {str(e)}")
            return self._create_fallback_result(returns.columns, f"Error: {str(e)}")

    def _build_uncertainty_set(
        self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, returns: pd.DataFrame
    ) -> dict[str, Any]:
        """Build uncertainty set for parameters"""
        len(expected_returns)
        n_obs = len(returns)

        # Standard errors for expected returns
        return_std_errors = returns.std() / np.sqrt(n_obs)

        # Uncertainty in expected returns
        uncertainty_radius = self.config.uncertainty_level

        if self.config.uncertainty_model == UncertaintyModel.ELLIPSOIDAL:
            # Ellipsoidal uncertainty set
            return {
                "type": "ellipsoidal",
                "mu_center": expected_returns.values,
                "sigma_uncertainty": uncertainty_radius * return_std_errors.values,
                "cov_center": cov_matrix.values,
            }
        elif self.config.uncertainty_model == UncertaintyModel.BOX:
            # Box uncertainty set
            return {
                "type": "box",
                "mu_lower": expected_returns.values - uncertainty_radius * return_std_errors.values,
                "mu_upper": expected_returns.values + uncertainty_radius * return_std_errors.values,
                "cov_center": cov_matrix.values,
            }
        else:
            return {"type": "none"}

    def _robust_cvxpy_optimization(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        uncertainty_set: dict[str, Any],
    ) -> RiskOptimizationResult:
        """Robust optimization using CVXPY"""
        n_assets = len(expected_returns)
        assets = returns.columns

        # Decision variables
        w = cp.Variable(n_assets)

        # Basic constraints
        constraints = [cp.sum(w) == 1, w >= self.config.min_weight, w <= self.config.max_weight]

        # Portfolio variance (deterministic part)
        portfolio_var = cp.quad_form(w, cov_matrix.values)

        # Robust expected return (worst-case)
        if uncertainty_set["type"] == "ellipsoidal":
            mu_center = uncertainty_set["mu_center"]
            sigma_uncertainty = uncertainty_set["sigma_uncertainty"]

            # Worst-case expected return under ellipsoidal uncertainty
            robust_return = (
                mu_center @ w
                - cp.norm(cp.multiply(sigma_uncertainty, w)) * self.config.robustness_level
            )

        else:
            # Fallback to deterministic
            robust_return = expected_returns.values @ w

        # Objective: maximize robust Sharpe ratio approximation
        objective = cp.Maximize(robust_return - 0.5 * self.config.risk_aversion * portfolio_var)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, max_iters=self.config.max_iterations)

        if problem.status == cp.OPTIMAL:
            optimal_weights = pd.Series(w.value, index=assets)
            return self._calculate_optimization_metrics(
                optimal_weights, returns, expected_returns, "Robust CVXPY optimization successful"
            )
        else:
            raise ValueError(f"Robust CVXPY optimization failed with status: {problem.status}")

    def _robust_scipy_optimization(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        uncertainty_set: dict[str, Any],
    ) -> RiskOptimizationResult:
        """Robust optimization using scipy"""
        n_assets = len(expected_returns)
        assets = returns.columns

        def objective(weights):
            w = np.array(weights)

            # Portfolio variance
            portfolio_var = w @ cov_matrix.values @ w

            # Worst-case expected return
            if uncertainty_set["type"] == "box":
                # Use lower bound of expected returns (pessimistic)
                worst_case_return = np.sum(w * uncertainty_set["mu_lower"])
            else:
                # Fallback to deterministic
                worst_case_return = np.sum(w * expected_returns.values)

            # Objective: maximize robust utility
            return -(worst_case_return - 0.5 * self.config.risk_aversion * portfolio_var)

        # Constraints
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
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
            options={"maxiter": self.config.max_iterations},
        )

        if result.success:
            optimal_weights = pd.Series(result.x, index=assets)
            return self._calculate_optimization_metrics(
                optimal_weights, returns, expected_returns, "Robust scipy optimization successful"
            )
        else:
            raise ValueError(f"Robust scipy optimization failed: {result.message}")


class RiskAdjustedOptimizationFramework:
    """Main framework for risk-adjusted portfolio optimization"""

    def __init__(self, config: RiskOptimizationConfig) -> None:
        self.config = config
        self.optimizer = self._create_optimizer()
        self.optimization_history = []

    def _create_optimizer(self) -> BaseRiskOptimizer:
        """Create optimizer based on configuration"""
        if self.config.optimization_type == OptimizationType.MEAN_CVAR:
            return CVaROptimizer(self.config)
        elif self.config.optimization_type == OptimizationType.MAX_DIVERSIFICATION:
            return MaxDiversificationOptimizer(self.config)
        elif self.config.optimization_type == OptimizationType.ROBUST_MEAN_VARIANCE:
            return RobustMeanVarianceOptimizer(self.config)
        else:
            # Default to CVaR
            return CVaROptimizer(self.config)

    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        benchmark: pd.Series | None = None,
    ) -> RiskOptimizationResult:
        """Optimize portfolio with risk adjustment"""
        result = self.optimizer.optimize(returns, expected_returns, benchmark)

        # Store in history
        self.optimization_history.append({"timestamp": pd.Timestamp.now(), "result": result})

        return result

    def get_optimization_metrics(self) -> dict[str, Any]:
        """Get optimization performance metrics"""
        if not self.optimization_history:
            return {}

        recent_results = [h["result"] for h in self.optimization_history[-50:]]
        successful_results = [r for r in recent_results if r.success]

        if not successful_results:
            return {"success_rate": 0.0}

        metrics = {
            "success_rate": len(successful_results) / len(recent_results),
            "avg_expected_return": np.mean([r.expected_return for r in successful_results]),
            "avg_expected_risk": np.mean([r.expected_risk for r in successful_results]),
            "avg_sharpe_ratio": np.mean([r.sharpe_ratio for r in successful_results]),
            "avg_cvar": np.mean([r.cvar for r in successful_results]),
            "avg_max_drawdown": np.mean([r.max_drawdown for r in successful_results]),
            "total_optimizations": len(self.optimization_history),
        }

        return metrics


# Helper methods for base class
def _calculate_optimization_metrics(
    self, weights: pd.Series, returns: pd.DataFrame, expected_returns: pd.Series, message: str
) -> RiskOptimizationResult:
    """Calculate comprehensive optimization metrics"""
    try:
        # Portfolio metrics
        portfolio_return = (weights * expected_returns).sum()

        # Portfolio returns time series
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        # CVaR
        cvar = self._calculate_cvar(weights, returns, alpha=1 - self.config.confidence_level)

        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(weights, returns)

        # Risk contributions (approximate)
        _, cov_matrix = self._estimate_parameters(returns)
        w = weights.values.reshape(-1, 1)
        portfolio_var = (w.T @ cov_matrix.values @ w)[0, 0]
        marginal_contrib = cov_matrix.values @ w
        risk_contrib = weights.values * marginal_contrib.flatten()
        risk_contributions = pd.Series(risk_contrib / portfolio_var, index=weights.index)

        # Factor exposures (if factor model available)
        factor_exposures = None
        if hasattr(self, "factor_model") and self.factor_model:
            factor_exposures = self.factor_model["loadings"].T @ weights

        # Tail risk metrics
        tail_risk_metrics = {
            "cvar_95": self._calculate_cvar(weights, returns, alpha=0.05),
            "cvar_99": self._calculate_cvar(weights, returns, alpha=0.01),
            "downside_deviation": self._calculate_downside_deviation(
                weights, returns, self.config.downside_threshold
            ),
        }

        return RiskOptimizationResult(
            optimal_weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_vol,
            cvar=cvar,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            risk_contributions=risk_contributions,
            factor_exposures=factor_exposures,
            tail_risk_metrics=tail_risk_metrics,
            robustness_metrics={},
            optimization_statistics={
                "portfolio_concentration": np.sum(weights**2),  # HHI
                "max_weight": weights.max(),
                "min_weight": weights.min(),
            },
            success=True,
            message=message,
        )

    except Exception as e:
        logger.error(f"Metrics calculation failed: {str(e)}")
        return self._create_fallback_result(returns.columns, f"Metrics error: {str(e)}")


def _create_fallback_result(self, assets: pd.Index, message: str) -> RiskOptimizationResult:
    """Create fallback result when optimization fails"""
    n_assets = len(assets)
    equal_weights = pd.Series(1.0 / n_assets, index=assets)

    return RiskOptimizationResult(
        optimal_weights=equal_weights,
        expected_return=0.0,
        expected_risk=0.15,
        cvar=0.05,
        max_drawdown=0.1,
        sharpe_ratio=0.0,
        risk_contributions=pd.Series(1.0 / n_assets, index=assets),
        factor_exposures=None,
        tail_risk_metrics={},
        robustness_metrics={},
        optimization_statistics={"error": message},
        success=False,
        message=message,
    )


# Add methods to base class
BaseRiskOptimizer._calculate_optimization_metrics = _calculate_optimization_metrics
BaseRiskOptimizer._create_fallback_result = _create_fallback_result


def create_risk_optimizer(
    optimization_type: OptimizationType = OptimizationType.MEAN_CVAR,
    risk_measure: RiskMeasure = RiskMeasure.CVAR,
    **kwargs,
) -> RiskAdjustedOptimizationFramework:
    """Factory function to create risk-adjusted optimizer"""
    config = RiskOptimizationConfig(
        optimization_type=optimization_type, risk_measure=risk_measure, **kwargs
    )

    return RiskAdjustedOptimizationFramework(config)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data with fat tails and regime changes
    np.random.seed(42)
    n_days = 500
    n_assets = 4
    assets = ["EQUITY", "BONDS", "COMMODITIES", "REITS"]

    # Generate returns with different risk characteristics
    base_returns = [0.08, 0.04, 0.06, 0.07]  # Expected annual returns
    volatilities = [0.18, 0.08, 0.25, 0.20]  # Annual volatilities

    # Add some fat tail events
    returns_data = []
    for _i in range(n_days):
        # Normal returns most of the time
        if np.random.random() < 0.95:
            daily_returns = [
                np.random.normal(base_returns[j] / 252, volatilities[j] / np.sqrt(252))
                for j in range(n_assets)
            ]
        else:
            # Occasional extreme events
            daily_returns = [
                np.random.normal(base_returns[j] / 252 - 0.05, volatilities[j] / np.sqrt(252) * 2)
                for j in range(n_assets)
            ]
        returns_data.append(daily_returns)

    returns_df = pd.DataFrame(
        returns_data, columns=assets, index=pd.date_range("2022-01-01", periods=n_days, freq="D")
    )

    print("Risk-Adjusted Portfolio Optimization Framework Testing")
    print("=" * 65)

    # Test different optimization types
    optimization_types = [
        OptimizationType.MEAN_CVAR,
        OptimizationType.MAX_DIVERSIFICATION,
        OptimizationType.ROBUST_MEAN_VARIANCE,
    ]

    for opt_type in optimization_types:
        print(f"\nTesting {opt_type.value} optimization...")
        try:
            optimizer = create_risk_optimizer(
                optimization_type=opt_type,
                confidence_level=0.95,
                risk_aversion=2.0,
                max_weight=0.6,
                uncertainty_level=0.1,
            )

            result = optimizer.optimize_portfolio(returns_df)

            if result.success:
                print(
                    f"✅ Success: Expected Return: {result.expected_return:.4f}, "
                    f"Risk: {result.expected_risk:.4f}, "
                    f"Sharpe: {result.sharpe_ratio:.4f}"
                )
                print(f"   CVaR (95%): {result.cvar:.4f}, Max DD: {result.max_drawdown:.4f}")
                print("   Optimal weights:")
                for asset, weight in result.optimal_weights.items():
                    print(f"     {asset}: {weight:.3f}")

                # Tail risk metrics
                if result.tail_risk_metrics:
                    print("   Tail Risk Metrics:")
                    for metric, value in result.tail_risk_metrics.items():
                        print(f"     {metric}: {value:.4f}")

            else:
                print(f"❌ Failed: {result.message}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    # Test optimization metrics
    print("\nTesting optimization performance metrics...")
    try:
        optimizer = create_risk_optimizer(optimization_type=OptimizationType.MEAN_CVAR)

        # Run multiple optimizations
        for _i in range(3):
            optimizer.optimize_portfolio(returns_df)

        metrics = optimizer.get_optimization_metrics()
        print("✅ Performance metrics:")
        print(f"   Success rate: {metrics.get('success_rate', 0):.2f}")
        print(f"   Average Sharpe ratio: {metrics.get('avg_sharpe_ratio', 0):.4f}")
        print(f"   Average CVaR: {metrics.get('avg_cvar', 0):.4f}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    print("\n🚀 Risk-Adjusted Portfolio Optimization Framework ready for production!")
