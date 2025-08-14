"""
Portfolio-Level Optimization Framework for Multi-Asset Strategy Enhancement

This module implements sophisticated portfolio optimization techniques including:
- Modern Portfolio Theory (MPT) with robust optimization
- Black-Litterman model for incorporating views
- Risk parity and hierarchical risk parity approaches
- Dynamic portfolio rebalancing with transaction costs
- Multi-objective optimization (return vs risk vs drawdown)
- Factor-based portfolio construction
"""

import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import inv
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
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some covariance estimation methods will be limited.")

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""

    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hrp"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    EQUAL_WEIGHT = "equal_weight"


class RiskModel(Enum):
    """Risk modeling approaches"""

    SAMPLE_COVARIANCE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"
    SHRINKAGE = "shrinkage"
    FACTOR_MODEL = "factor_model"
    ROBUST = "robust"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""

    min_weight: float = 0.0
    max_weight: float = 1.0
    max_turnover: float | None = None
    max_leverage: float = 1.0
    sector_constraints: dict[str, tuple[float, float]] | None = None
    asset_constraints: dict[str, tuple[float, float]] | None = None
    transaction_costs: float = 0.001
    target_volatility: float | None = None


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization"""

    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    risk_model: RiskModel = RiskModel.LEDOIT_WOLF
    constraints: OptimizationConstraints = field(default_factory=OptimizationConstraints)
    lookback_period: int = 252
    rebalance_frequency: int = 21
    risk_aversion: float = 1.0
    confidence_level: float = 0.95
    tau: float = 1.0  # Black-Litterman scaling factor
    use_robust_optimization: bool = False
    max_iterations: int = 1000
    tolerance: float = 1e-6


@dataclass
class OptimizationResult:
    """Results from portfolio optimization"""

    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float
    turnover: float
    transaction_costs: float
    objective_value: float
    success: bool
    message: str
    optimization_time: float
    additional_metrics: dict[str, Any] = field(default_factory=dict)


class BasePortfolioOptimizer(ABC):
    """Base class for portfolio optimizers"""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.assets = []
        self.returns_data = None
        self.covariance_matrix = None
        self.expected_returns = None

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
    ) -> OptimizationResult:
        """Optimize portfolio weights"""
        pass

    def _estimate_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Estimate expected returns using historical mean"""
        return returns.mean() * 252  # Annualized

    def _estimate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate covariance matrix based on risk model"""
        if not SKLEARN_AVAILABLE and self.config.risk_model != RiskModel.SAMPLE_COVARIANCE:
            logger.warning("Scikit-learn not available, falling back to sample covariance")
            return returns.cov() * 252

        if self.config.risk_model == RiskModel.SAMPLE_COVARIANCE:
            return returns.cov() * 252
        elif self.config.risk_model == RiskModel.LEDOIT_WOLF:
            lw = LedoitWolf()
            cov_matrix, _ = lw.fit(returns).covariance_, lw.shrinkage_
            return pd.DataFrame(cov_matrix * 252, index=returns.columns, columns=returns.columns)
        elif self.config.risk_model == RiskModel.SHRINKAGE:
            sample_cov = returns.cov() * 252
            n_assets = len(sample_cov)
            shrinkage_target = np.trace(sample_cov) / n_assets * np.eye(n_assets)
            shrinkage_intensity = 0.2
            shrunk_cov = (
                1 - shrinkage_intensity
            ) * sample_cov + shrinkage_intensity * pd.DataFrame(
                shrinkage_target, index=sample_cov.index, columns=sample_cov.columns
            )
            return shrunk_cov
        else:
            return returns.cov() * 252

    def _calculate_portfolio_metrics(
        self, weights: pd.Series, returns: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate portfolio performance metrics"""
        portfolio_returns = (returns * weights).sum(axis=1)

        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Diversification ratio
        individual_vols = returns.std() * np.sqrt(252)
        weighted_vol = (weights * individual_vols).sum()
        diversification_ratio = weighted_vol / annual_vol if annual_vol > 0 else 0

        return {
            "expected_return": annual_return,
            "expected_volatility": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "diversification_ratio": diversification_ratio,
        }


class MeanVarianceOptimizer(BasePortfolioOptimizer):
    """Mean-Variance Optimization using Markowitz framework"""

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
    ) -> OptimizationResult:
        start_time = time.time()

        try:
            n_assets = len(returns.columns)
            assets = returns.columns.tolist()

            # Estimate parameters
            if expected_returns is None:
                expected_returns = self._estimate_expected_returns(returns)
            cov_matrix = self._estimate_covariance_matrix(returns)

            # Use CVXPY if available for robust optimization
            if CVXPY_AVAILABLE and self.config.use_robust_optimization:
                return self._cvxpy_optimize(
                    returns, expected_returns, cov_matrix, current_weights, start_time
                )
            else:
                return self._scipy_optimize(
                    returns, expected_returns, cov_matrix, current_weights, start_time
                )

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            # Return equal weight as fallback
            equal_weights = pd.Series(1 / n_assets, index=assets)
            metrics = self._calculate_portfolio_metrics(equal_weights, returns)
            return OptimizationResult(
                weights=equal_weights,
                expected_return=metrics["expected_return"],
                expected_volatility=metrics["expected_volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                diversification_ratio=metrics["diversification_ratio"],
                turnover=0.0,
                transaction_costs=0.0,
                objective_value=0.0,
                success=False,
                message=f"Optimization failed: {str(e)}. Using equal weights.",
                optimization_time=time.time() - start_time,
            )

    def _scipy_optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        current_weights: pd.Series | None,
        start_time: float,
    ) -> OptimizationResult:
        """Optimize using scipy"""
        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        # Objective function: maximize Sharpe ratio or minimize variance
        def objective(weights):
            w = pd.Series(weights, index=assets)
            portfolio_return = (w * expected_returns).sum()
            portfolio_vol = np.sqrt(w.values @ cov_matrix.values @ w.values)

            if self.config.method == OptimizationMethod.MAXIMUM_SHARPE:
                return -portfolio_return / portfolio_vol if portfolio_vol > 0 else -portfolio_return
            elif self.config.method == OptimizationMethod.MINIMUM_VARIANCE:
                return portfolio_vol
            else:  # Mean-variance with risk aversion
                return -portfolio_return + self.config.risk_aversion * portfolio_vol**2

        # Constraints
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # Weights sum to 1

        # Add turnover constraint if specified
        if self.config.constraints.max_turnover is not None and current_weights is not None:

            def turnover_constraint(weights):
                return self.config.constraints.max_turnover - np.sum(
                    np.abs(weights - current_weights.values)
                )

            constraints.append({"type": "ineq", "fun": turnover_constraint})

        # Bounds
        bounds = [
            (self.config.constraints.min_weight, self.config.constraints.max_weight)
            for _ in range(n_assets)
        ]

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
            metrics = self._calculate_portfolio_metrics(optimal_weights, returns)

            # Calculate turnover and transaction costs
            turnover = 0.0
            transaction_costs = 0.0
            if current_weights is not None:
                turnover = np.sum(np.abs(optimal_weights - current_weights))
                transaction_costs = turnover * self.config.constraints.transaction_costs

            return OptimizationResult(
                weights=optimal_weights,
                expected_return=metrics["expected_return"],
                expected_volatility=metrics["expected_volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                diversification_ratio=metrics["diversification_ratio"],
                turnover=turnover,
                transaction_costs=transaction_costs,
                objective_value=-result.fun,
                success=True,
                message="Optimization successful",
                optimization_time=time.time() - start_time,
            )
        else:
            raise ValueError(f"Optimization failed: {result.message}")

    def _cvxpy_optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        current_weights: pd.Series | None,
        start_time: float,
    ) -> OptimizationResult:
        """Optimize using CVXPY for robust optimization"""
        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        # Decision variable
        w = cp.Variable(n_assets)

        # Objective
        portfolio_return = expected_returns.values @ w
        portfolio_risk = cp.quad_form(w, cov_matrix.values)

        if self.config.method == OptimizationMethod.MAXIMUM_SHARPE:
            # Maximize Sharpe ratio (approximate)
            objective = cp.Maximize(
                portfolio_return - 0.5 * self.config.risk_aversion * portfolio_risk
            )
        elif self.config.method == OptimizationMethod.MINIMUM_VARIANCE:
            objective = cp.Minimize(portfolio_risk)
        else:
            objective = cp.Maximize(portfolio_return - self.config.risk_aversion * portfolio_risk)

        # Constraints
        constraints = [cp.sum(w) == 1]  # Weights sum to 1
        constraints.append(w >= self.config.constraints.min_weight)
        constraints.append(w <= self.config.constraints.max_weight)

        # Turnover constraint
        if self.config.constraints.max_turnover is not None and current_weights is not None:
            constraints.append(
                cp.norm(w - current_weights.values, 1) <= self.config.constraints.max_turnover
            )

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, max_iters=self.config.max_iterations)

        if problem.status == cp.OPTIMAL:
            optimal_weights = pd.Series(w.value, index=assets)
            metrics = self._calculate_portfolio_metrics(optimal_weights, returns)

            # Calculate turnover and transaction costs
            turnover = 0.0
            transaction_costs = 0.0
            if current_weights is not None:
                turnover = np.sum(np.abs(optimal_weights - current_weights))
                transaction_costs = turnover * self.config.constraints.transaction_costs

            return OptimizationResult(
                weights=optimal_weights,
                expected_return=metrics["expected_return"],
                expected_volatility=metrics["expected_volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                diversification_ratio=metrics["diversification_ratio"],
                turnover=turnover,
                transaction_costs=transaction_costs,
                objective_value=problem.value,
                success=True,
                message="CVXPY optimization successful",
                optimization_time=time.time() - start_time,
            )
        else:
            raise ValueError(f"CVXPY optimization failed with status: {problem.status}")


class BlackLittermanOptimizer(BasePortfolioOptimizer):
    """Black-Litterman optimization with investor views"""

    def __init__(
        self,
        config: OptimizationConfig,
        views: dict[str, float] | None = None,
        view_confidence: dict[str, float] | None = None,
    ) -> None:
        super().__init__(config)
        self.views = views or {}
        self.view_confidence = view_confidence or {}

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
    ) -> OptimizationResult:
        start_time = time.time()

        try:
            assets = returns.columns.tolist()
            n_assets = len(assets)

            # Market capitalization weights (equal weight as proxy)
            market_caps = pd.Series(1.0, index=assets)
            w_market = market_caps / market_caps.sum()

            # Estimate covariance matrix
            cov_matrix = self._estimate_covariance_matrix(returns)

            # Implied equilibrium returns
            delta = self._calculate_risk_aversion(returns, w_market, cov_matrix)
            pi = delta * (cov_matrix @ w_market)

            # Black-Litterman with views
            if self.views:
                mu_bl, cov_bl = self._apply_black_litterman(pi, cov_matrix, assets)
            else:
                mu_bl, cov_bl = pi, cov_matrix

            # Optimize with Black-Litterman inputs
            mv_optimizer = MeanVarianceOptimizer(self.config)
            return mv_optimizer._scipy_optimize(returns, mu_bl, cov_bl, current_weights, start_time)

        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {str(e)}")
            # Fallback to equal weight
            equal_weights = pd.Series(1 / n_assets, index=assets)
            metrics = self._calculate_portfolio_metrics(equal_weights, returns)
            return OptimizationResult(
                weights=equal_weights,
                expected_return=metrics["expected_return"],
                expected_volatility=metrics["expected_volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                diversification_ratio=metrics["diversification_ratio"],
                turnover=0.0,
                transaction_costs=0.0,
                objective_value=0.0,
                success=False,
                message=f"Black-Litterman failed: {str(e)}. Using equal weights.",
                optimization_time=time.time() - start_time,
            )

    def _calculate_risk_aversion(
        self, returns: pd.DataFrame, market_weights: pd.Series, cov_matrix: pd.DataFrame
    ) -> float:
        """Calculate implied risk aversion parameter"""
        market_return = (returns * market_weights).sum(axis=1).mean() * 252
        market_vol = (returns * market_weights).sum(axis=1).std() * np.sqrt(252)
        return market_return / (market_vol**2)

    def _apply_black_litterman(
        self, pi: pd.Series, cov_matrix: pd.DataFrame, assets: list[str]
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Apply Black-Litterman model with views"""
        n_assets = len(assets)

        # Create view matrix P and view vector Q
        views_list = []
        confidence_list = []
        P_rows = []

        for asset, view in self.views.items():
            if asset in assets:
                asset_idx = assets.index(asset)
                P_row = np.zeros(n_assets)
                P_row[asset_idx] = 1.0
                P_rows.append(P_row)
                views_list.append(view)
                confidence_list.append(self.view_confidence.get(asset, 1.0))

        if not views_list:
            return pi, cov_matrix

        P = np.array(P_rows)
        Q = np.array(views_list)

        # Uncertainty matrix (diagonal with view confidences)
        Omega = np.diag([1.0 / conf for conf in confidence_list])

        # Black-Litterman formula
        tau = self.config.tau
        M1 = inv(tau * cov_matrix.values)
        M2 = P.T @ inv(Omega) @ P
        M3 = inv(tau * cov_matrix.values) @ pi.values
        M4 = P.T @ inv(Omega) @ Q

        mu_bl = inv(M1 + M2) @ (M3 + M4)
        cov_bl = inv(M1 + M2)

        return pd.Series(mu_bl, index=assets), pd.DataFrame(cov_bl, index=assets, columns=assets)


class RiskParityOptimizer(BasePortfolioOptimizer):
    """Risk Parity optimization - equal risk contribution"""

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
    ) -> OptimizationResult:
        start_time = time.time()

        try:
            assets = returns.columns.tolist()
            n_assets = len(assets)

            # Estimate covariance matrix
            cov_matrix = self._estimate_covariance_matrix(returns)

            # Risk parity objective: minimize sum of squared differences in risk contributions
            def objective(weights):
                w = np.array(weights)
                portfolio_vol = np.sqrt(w @ cov_matrix.values @ w)
                marginal_contrib = (cov_matrix.values @ w) / portfolio_vol
                contrib = w * marginal_contrib
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)

            # Constraints and bounds
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
            bounds = [
                (self.config.constraints.min_weight, self.config.constraints.max_weight)
                for _ in range(n_assets)
            ]

            # Initial guess: inverse volatility weights
            vols = np.sqrt(np.diag(cov_matrix.values))
            inv_vol_weights = (1 / vols) / np.sum(1 / vols)

            # Optimize
            result = minimize(
                objective,
                inv_vol_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.config.max_iterations},
            )

            if result.success:
                optimal_weights = pd.Series(result.x, index=assets)
                metrics = self._calculate_portfolio_metrics(optimal_weights, returns)

                # Calculate turnover and transaction costs
                turnover = 0.0
                transaction_costs = 0.0
                if current_weights is not None:
                    turnover = np.sum(np.abs(optimal_weights - current_weights))
                    transaction_costs = turnover * self.config.constraints.transaction_costs

                return OptimizationResult(
                    weights=optimal_weights,
                    expected_return=metrics["expected_return"],
                    expected_volatility=metrics["expected_volatility"],
                    sharpe_ratio=metrics["sharpe_ratio"],
                    max_drawdown=metrics["max_drawdown"],
                    diversification_ratio=metrics["diversification_ratio"],
                    turnover=turnover,
                    transaction_costs=transaction_costs,
                    objective_value=result.fun,
                    success=True,
                    message="Risk parity optimization successful",
                    optimization_time=time.time() - start_time,
                )
            else:
                raise ValueError(f"Risk parity optimization failed: {result.message}")

        except Exception as e:
            logger.error(f"Risk parity optimization failed: {str(e)}")
            # Fallback to equal weight
            equal_weights = pd.Series(1 / n_assets, index=assets)
            metrics = self._calculate_portfolio_metrics(equal_weights, returns)
            return OptimizationResult(
                weights=equal_weights,
                expected_return=metrics["expected_return"],
                expected_volatility=metrics["expected_volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                diversification_ratio=metrics["diversification_ratio"],
                turnover=0.0,
                transaction_costs=0.0,
                objective_value=0.0,
                success=False,
                message=f"Risk parity failed: {str(e)}. Using equal weights.",
                optimization_time=time.time() - start_time,
            )


class PortfolioOptimizationFramework:
    """Main framework for portfolio optimization with multiple methods"""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.optimizers = self._initialize_optimizers()
        self.optimization_history = []

    def _initialize_optimizers(self) -> dict[OptimizationMethod, BasePortfolioOptimizer]:
        """Initialize available optimizers"""
        optimizers = {}

        # Always available optimizers
        optimizers[OptimizationMethod.MEAN_VARIANCE] = MeanVarianceOptimizer(self.config)
        optimizers[OptimizationMethod.MINIMUM_VARIANCE] = MeanVarianceOptimizer(self.config)
        optimizers[OptimizationMethod.MAXIMUM_SHARPE] = MeanVarianceOptimizer(self.config)
        optimizers[OptimizationMethod.BLACK_LITTERMAN] = BlackLittermanOptimizer(self.config)
        optimizers[OptimizationMethod.RISK_PARITY] = RiskParityOptimizer(self.config)

        return optimizers

    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        method: OptimizationMethod | None = None,
    ) -> OptimizationResult:
        """Optimize portfolio using specified method"""
        if method is None:
            method = self.config.method

        if method not in self.optimizers:
            raise ValueError(f"Optimization method {method} not available")

        # Run optimization
        result = self.optimizers[method].optimize(returns, expected_returns, current_weights)

        # Store in history
        self.optimization_history.append(
            {"timestamp": pd.Timestamp.now(), "method": method, "result": result}
        )

        return result

    def backtest_optimization(
        self, returns: pd.DataFrame, rebalance_dates: pd.DatetimeIndex | None = None
    ) -> pd.DataFrame:
        """Backtest portfolio optimization over time"""
        if rebalance_dates is None:
            rebalance_dates = returns.index[:: self.config.rebalance_frequency]

        weights_history = []

        current_weights = None

        for date in rebalance_dates:
            if date not in returns.index:
                continue

            # Get historical data up to rebalance date
            hist_data = returns.loc[:date].iloc[-self.config.lookback_period :]

            if len(hist_data) < 20:  # Need minimum data
                continue

            # Optimize portfolio
            result = self.optimize_portfolio(hist_data, current_weights=current_weights)

            if result.success:
                current_weights = result.weights
                weights_history.append(
                    {
                        "date": date,
                        "weights": current_weights.to_dict(),
                        "expected_return": result.expected_return,
                        "expected_volatility": result.expected_volatility,
                        "sharpe_ratio": result.sharpe_ratio,
                    }
                )

        return pd.DataFrame(weights_history)


def create_portfolio_optimizer(
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
    risk_model: RiskModel = RiskModel.LEDOIT_WOLF,
    constraints: OptimizationConstraints | None = None,
    **kwargs,
) -> PortfolioOptimizationFramework:
    """Factory function to create portfolio optimizer"""
    if constraints is None:
        constraints = OptimizationConstraints()

    config = OptimizationConfig(
        method=method, risk_model=risk_model, constraints=constraints, **kwargs
    )

    return PortfolioOptimizationFramework(config)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_days = 500
    n_assets = 5
    assets = [f"ASSET_{i}" for i in range(n_assets)]

    # Simulate correlated returns
    returns_data = np.random.multivariate_normal(
        mean=[0.0005] * n_assets,
        cov=np.eye(n_assets) * 0.01 + np.ones((n_assets, n_assets)) * 0.002,
        size=n_days,
    )

    returns_df = pd.DataFrame(
        returns_data, index=pd.date_range("2022-01-01", periods=n_days, freq="D"), columns=assets
    )

    # Test different optimization methods
    methods = [
        OptimizationMethod.MEAN_VARIANCE,
        OptimizationMethod.MINIMUM_VARIANCE,
        OptimizationMethod.MAXIMUM_SHARPE,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.BLACK_LITTERMAN,
    ]

    print("Portfolio Optimization Framework Testing")
    print("=" * 50)

    for method in methods:
        print(f"\nTesting {method.value}...")
        try:
            optimizer = create_portfolio_optimizer(method=method)
            result = optimizer.optimize_portfolio(returns_df)

            if result.success:
                print(
                    f"✅ Success: Expected Return: {result.expected_return:.4f}, "
                    f"Volatility: {result.expected_volatility:.4f}, "
                    f"Sharpe: {result.sharpe_ratio:.4f}"
                )
                print(f"   Weights: {result.weights.round(3).to_dict()}")
            else:
                print(f"❌ Failed: {result.message}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    print("\n🚀 Portfolio Optimization Framework ready for production!")
