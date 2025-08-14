"""
Markowitz Portfolio Optimizer using cvxpy
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from ...core.base import BaseComponent, ComponentConfig


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_return: Optional[float] = None
    max_risk: Optional[float] = None
    long_only: bool = True
    sum_to_one: bool = True
    max_positions: Optional[int] = None
    sector_limits: Optional[Dict[str, float]] = None
    min_position_size: float = 0.01


class MarkowitzOptimizer(BaseComponent):
    """Modern Portfolio Theory optimizer using convex optimization"""
    
    def __init__(self, 
                 config: Optional[ComponentConfig] = None,
                 db_manager=None,
                 risk_free_rate: float = 0.02):
        """Initialize Markowitz optimizer
        
        Args:
            config: Component configuration
            db_manager: Database manager
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculations
        """
        if config is None:
            config = ComponentConfig(
                component_id='markowitz_optimizer',
                component_type='portfolio_optimizer'
            )
        super().__init__(config, db_manager)
        
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = risk_free_rate
        
        # Cached optimization results
        self.last_weights = None
        self.last_metrics = {}
        self.optimization_history = []
        
    def optimize(self,
                 returns: pd.DataFrame,
                 constraints: Optional[OptimizationConstraints] = None,
                 objective: str = 'max_sharpe',
                 risk_aversion: float = 1.0) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize portfolio weights
        
        Args:
            returns: DataFrame of asset returns (rows: dates, columns: assets)
            constraints: Optimization constraints
            objective: Optimization objective ('max_sharpe', 'min_risk', 'max_return', 'efficient_frontier')
            risk_aversion: Risk aversion parameter for utility maximization
            
        Returns:
            Tuple of (weights dict, metrics dict)
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        self.logger.info(f"Optimizing portfolio with {len(returns.columns)} assets, objective: {objective}")
        
        # Calculate statistics
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Number of assets
        n_assets = len(returns.columns)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Expected return and risk
        portfolio_return = mean_returns.values @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix.values)
        
        # Define objective based on type
        if objective == 'max_sharpe':
            # Max Sharpe requires special handling
            weights_opt, metrics = self._optimize_max_sharpe(
                mean_returns, cov_matrix, constraints
            )
        elif objective == 'min_risk':
            obj = portfolio_risk
            weights_opt, metrics = self._solve_optimization(
                obj, weights, portfolio_return, portfolio_risk,
                mean_returns, cov_matrix, constraints, minimize=True
            )
        elif objective == 'max_return':
            obj = portfolio_return
            weights_opt, metrics = self._solve_optimization(
                obj, weights, portfolio_return, portfolio_risk,
                mean_returns, cov_matrix, constraints, minimize=False
            )
        elif objective == 'efficient_frontier':
            # Generate efficient frontier
            weights_opt, metrics = self._generate_efficient_frontier(
                mean_returns, cov_matrix, constraints
            )
        else:
            # Risk-adjusted utility
            obj = portfolio_return - risk_aversion * portfolio_risk
            weights_opt, metrics = self._solve_optimization(
                obj, weights, portfolio_return, portfolio_risk,
                mean_returns, cov_matrix, constraints, minimize=False
            )
        
        # Create weights dictionary
        weights_dict = {
            asset: float(weight) 
            for asset, weight in zip(returns.columns, weights_opt)
            if abs(weight) > constraints.min_position_size
        }
        
        # Normalize weights to sum to 1
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            weights_dict = {k: v/total_weight for k, v in weights_dict.items()}
        
        # Store results
        self.last_weights = weights_dict
        self.last_metrics = metrics
        
        # Record optimization
        self._record_optimization(weights_dict, metrics, objective)
        
        return weights_dict, metrics
    
    def _solve_optimization(self,
                           objective,
                           weights,
                           portfolio_return,
                           portfolio_risk,
                           mean_returns: pd.Series,
                           cov_matrix: pd.DataFrame,
                           constraints: OptimizationConstraints,
                           minimize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve the optimization problem
        
        Args:
            objective: Optimization objective
            weights: Weight variables
            portfolio_return: Portfolio return expression
            portfolio_risk: Portfolio risk expression
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
            constraints: Optimization constraints
            minimize: Whether to minimize or maximize
            
        Returns:
            Tuple of (optimal weights, metrics)
        """
        # Build constraints list
        constraint_list = []
        
        # Weight bounds
        if constraints.long_only:
            constraint_list.append(weights >= constraints.min_weight)
        constraint_list.append(weights <= constraints.max_weight)
        
        # Sum to one
        if constraints.sum_to_one:
            constraint_list.append(cp.sum(weights) == 1)
        
        # Target return
        if constraints.target_return is not None:
            constraint_list.append(portfolio_return >= constraints.target_return)
        
        # Max risk
        if constraints.max_risk is not None:
            constraint_list.append(portfolio_risk <= constraints.max_risk**2)
        
        # Max positions (cardinality constraint - approximation)
        if constraints.max_positions is not None:
            # This is a non-convex constraint, using L1 norm as approximation
            constraint_list.append(cp.norm(weights, 1) <= constraints.max_positions * constraints.max_weight)
        
        # Create and solve problem
        if minimize:
            problem = cp.Problem(cp.Minimize(objective), constraint_list)
        else:
            problem = cp.Problem(cp.Maximize(objective), constraint_list)
        
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            self.logger.warning(f"Optimization failed with status: {problem.status}")
            # Return equal weights as fallback
            n_assets = len(mean_returns)
            weights_opt = np.ones(n_assets) / n_assets
        else:
            weights_opt = weights.value
        
        # Calculate metrics
        metrics = self._calculate_metrics(weights_opt, mean_returns, cov_matrix)
        
        return weights_opt, metrics
    
    def _optimize_max_sharpe(self,
                            mean_returns: pd.Series,
                            cov_matrix: pd.DataFrame,
                            constraints: OptimizationConstraints) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize for maximum Sharpe ratio
        
        Uses the trick of reformulating as a convex problem
        
        Args:
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
            constraints: Optimization constraints
            
        Returns:
            Tuple of (optimal weights, metrics)
        """
        n_assets = len(mean_returns)
        
        # Use auxiliary variables for Sharpe ratio optimization
        y = cp.Variable(n_assets)
        kappa = cp.Variable()
        
        # Adjusted returns (excess returns over risk-free rate)
        excess_returns = mean_returns.values - self.risk_free_rate/252
        
        # Constraints
        constraint_list = [
            excess_returns @ y == 1,
            kappa >= 0
        ]
        
        if constraints.long_only:
            constraint_list.append(y >= 0)
        
        # Objective: minimize portfolio variance
        objective = cp.quad_form(y, cov_matrix.values)
        
        # Solve
        problem = cp.Problem(cp.Minimize(objective), constraint_list)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            self.logger.warning(f"Sharpe optimization failed with status: {problem.status}")
            # Fallback to equal weights
            weights_opt = np.ones(n_assets) / n_assets
        else:
            # Recover weights
            weights_opt = y.value / cp.sum(y).value
        
        # Calculate metrics
        metrics = self._calculate_metrics(weights_opt, mean_returns, cov_matrix)
        
        return weights_opt, metrics
    
    def _generate_efficient_frontier(self,
                                    mean_returns: pd.Series,
                                    cov_matrix: pd.DataFrame,
                                    constraints: OptimizationConstraints,
                                    n_points: int = 20) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate efficient frontier
        
        Args:
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
            constraints: Optimization constraints
            n_points: Number of points on frontier
            
        Returns:
            Tuple of (optimal weights for max Sharpe, metrics including frontier)
        """
        # Get min and max possible returns
        n_assets = len(mean_returns)
        
        # Min variance portfolio
        weights = cp.Variable(n_assets)
        portfolio_risk = cp.quad_form(weights, cov_matrix.values)
        
        constraint_list = [
            cp.sum(weights) == 1,
            weights >= 0 if constraints.long_only else weights >= -1
        ]
        
        problem = cp.Problem(cp.Minimize(portfolio_risk), constraint_list)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        min_risk_return = mean_returns.values @ weights.value
        
        # Max return portfolio
        max_return = mean_returns.max() if constraints.long_only else mean_returns.abs().max()
        
        # Generate frontier points
        target_returns = np.linspace(min_risk_return, max_return, n_points)
        frontier_risks = []
        frontier_returns = []
        frontier_weights = []
        
        for target_ret in target_returns:
            weights = cp.Variable(n_assets)
            portfolio_return = mean_returns.values @ weights
            portfolio_risk = cp.quad_form(weights, cov_matrix.values)
            
            constraint_list = [
                cp.sum(weights) == 1,
                portfolio_return >= target_ret,
                weights >= 0 if constraints.long_only else weights >= -1
            ]
            
            problem = cp.Problem(cp.Minimize(portfolio_risk), constraint_list)
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status in ["optimal", "optimal_inaccurate"]:
                frontier_returns.append(float(target_ret))
                frontier_risks.append(float(np.sqrt(portfolio_risk.value)))
                frontier_weights.append(weights.value.copy())
        
        # Find max Sharpe portfolio
        sharpe_ratios = [
            (ret - self.risk_free_rate/252) / risk 
            for ret, risk in zip(frontier_returns, frontier_risks)
            if risk > 0
        ]
        
        if sharpe_ratios:
            max_sharpe_idx = np.argmax(sharpe_ratios)
            optimal_weights = frontier_weights[max_sharpe_idx]
        else:
            # Fallback
            optimal_weights = np.ones(n_assets) / n_assets
        
        # Calculate metrics
        metrics = self._calculate_metrics(optimal_weights, mean_returns, cov_matrix)
        metrics['efficient_frontier'] = {
            'returns': frontier_returns,
            'risks': frontier_risks,
            'sharpe_ratios': sharpe_ratios
        }
        
        return optimal_weights, metrics
    
    def _calculate_metrics(self,
                          weights: np.ndarray,
                          mean_returns: pd.Series,
                          cov_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio metrics
        
        Args:
            weights: Portfolio weights
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
            
        Returns:
            Dictionary of metrics
        """
        # Portfolio statistics
        portfolio_return = float(mean_returns.values @ weights)
        portfolio_variance = float(weights @ cov_matrix.values @ weights)
        portfolio_std = float(np.sqrt(portfolio_variance))
        
        # Annualized metrics (assuming daily returns)
        annual_return = portfolio_return * 252
        annual_std = portfolio_std * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_std if annual_std > 0 else 0
        
        # Diversification metrics
        n_positions = np.sum(np.abs(weights) > 0.001)
        concentration = float(np.sum(weights**2))  # Herfindahl index
        max_weight = float(np.max(np.abs(weights)))
        
        metrics = {
            'expected_return': portfolio_return,
            'expected_risk': portfolio_std,
            'annual_return': annual_return,
            'annual_volatility': annual_std,
            'sharpe_ratio': sharpe_ratio,
            'n_positions': int(n_positions),
            'concentration': concentration,
            'max_weight': max_weight,
            'optimization_status': 'success'
        }
        
        return metrics
    
    def backtest_allocation(self,
                           returns: pd.DataFrame,
                           weights: Dict[str, float],
                           rebalance_freq: str = 'M') -> Dict[str, Any]:
        """Backtest a portfolio allocation
        
        Args:
            returns: Historical returns
            weights: Portfolio weights
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Dictionary of backtest results
        """
        # Filter returns to include only allocated assets
        allocated_assets = list(weights.keys())
        returns_subset = returns[allocated_assets]
        
        # Convert weights to array
        weight_array = np.array([weights[asset] for asset in allocated_assets])
        
        # Calculate portfolio returns
        portfolio_returns = returns_subset @ weight_array
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate metrics
        total_return = float(cumulative_returns.iloc[-1] - 1)
        annual_return = float(portfolio_returns.mean() * 252)
        annual_vol = float(portfolio_returns.std() * np.sqrt(252))
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Calculate max drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'cumulative_returns': cumulative_returns.to_dict(),
            'portfolio_returns': portfolio_returns.to_dict()
        }
        
        return results
    
    def calculate_risk_metrics(self,
                              returns: pd.DataFrame,
                              weights: Dict[str, float],
                              confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate risk metrics for portfolio
        
        Args:
            returns: Historical returns
            weights: Portfolio weights
            confidence_level: Confidence level for VaR/CVaR
            
        Returns:
            Dictionary of risk metrics
        """
        # Filter and calculate portfolio returns
        allocated_assets = list(weights.keys())
        returns_subset = returns[allocated_assets]
        weight_array = np.array([weights[asset] for asset in allocated_assets])
        portfolio_returns = returns_subset @ weight_array
        
        # Value at Risk
        var = float(np.percentile(portfolio_returns, (1 - confidence_level) * 100))
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar = float(portfolio_returns[portfolio_returns <= var].mean())
        
        # Downside deviation
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_dev = float(np.sqrt(np.mean(negative_returns**2))) if len(negative_returns) > 0 else 0
        
        # Sortino ratio (using downside deviation)
        annual_return = portfolio_returns.mean() * 252
        sortino = (annual_return - self.risk_free_rate) / (downside_dev * np.sqrt(252)) if downside_dev > 0 else 0
        
        # Tracking error (if benchmark provided)
        # For now, using market (equal weight) as benchmark
        market_returns = returns_subset.mean(axis=1)
        tracking_error = float((portfolio_returns - market_returns).std() * np.sqrt(252))
        
        # Information ratio
        excess_returns = portfolio_returns - market_returns
        info_ratio = float(excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
        
        risk_metrics = {
            'var_95': var,
            'cvar_95': cvar,
            'downside_deviation': downside_dev,
            'sortino_ratio': sortino,
            'tracking_error': tracking_error,
            'information_ratio': info_ratio
        }
        
        return risk_metrics
    
    def _record_optimization(self,
                            weights: Dict[str, float],
                            metrics: Dict[str, Any],
                            objective: str):
        """Record optimization in database
        
        Args:
            weights: Optimized weights
            metrics: Optimization metrics
            objective: Optimization objective
        """
        record = {
            'timestamp': datetime.now(),
            'objective': objective,
            'weights': weights.copy(),
            'metrics': metrics.copy()
        }
        
        self.optimization_history.append(record)
        
        # Keep only last 100 optimizations
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        # Store in database if available
        if self.db_manager:
            try:
                import json
                self.db_manager.execute(
                    """INSERT INTO portfolio_optimizations 
                       (optimization_date, objective, weights, metrics, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        datetime.now(),
                        objective,
                        json.dumps(weights),
                        json.dumps(metrics),
                        datetime.now()
                    )
                )
            except Exception as e:
                self.logger.error(f"Error storing optimization: {e}")