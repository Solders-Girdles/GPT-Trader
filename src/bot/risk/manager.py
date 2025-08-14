"""
Comprehensive Risk Management System for Phase 5 Production Integration.
Integrates position sizing optimization, stop-loss management, portfolio-level risk limits, and stress testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from bot.analytics.risk_decomposition import RiskDecompositionAnalyzer
from bot.portfolio.optimizer import PortfolioAllocation

logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Risk metrics for monitoring."""

    VAR = "var"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"


@dataclass
class RiskLimits:
    """Risk limits for portfolio management."""

    # Portfolio-level limits
    max_portfolio_var: float = 0.02  # 2% VaR
    max_portfolio_drawdown: float = 0.15  # 15% max drawdown
    max_portfolio_volatility: float = 0.25  # 25% volatility
    max_portfolio_beta: float = 1.2  # Max beta

    # Position-level limits
    max_position_size: float = 0.1  # 10% max position
    max_sector_exposure: float = 0.3  # 30% max sector exposure
    max_correlation: float = 0.7  # Max correlation between positions

    # Risk per trade
    max_risk_per_trade: float = 0.01  # 1% risk per trade
    max_daily_loss: float = 0.03  # 3% max daily loss

    # Liquidity limits
    min_liquidity_ratio: float = 0.1  # 10% minimum liquidity
    max_illiquid_exposure: float = 0.2  # 20% max illiquid exposure


@dataclass
class StopLossConfig:
    """Stop-loss configuration."""

    stop_loss_pct: float = 0.05  # 5% stop loss
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    time_stop_days: int = 30  # 30-day time stop
    breakeven_after_pct: float = 0.02  # Move to breakeven after 2% profit


@dataclass
class PositionRisk:
    """Risk metrics for a position."""

    symbol: str
    current_value: float
    position_size: float
    var_95: float
    cvar_95: float
    beta: float
    volatility: float
    correlation_with_portfolio: float
    liquidity_score: float
    concentration_risk: float
    stop_loss_level: float
    risk_contribution: float


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""

    total_value: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    volatility: float
    beta: float
    sharpe_ratio: float
    correlation_matrix: pd.DataFrame
    risk_contributions: dict[str, float]
    concentration_metrics: dict[str, float]
    liquidity_metrics: dict[str, float]
    stress_test_results: dict[str, Any]
    timestamp: datetime


class RiskManager:
    """Comprehensive risk management system."""

    def __init__(
        self,
        risk_limits: RiskLimits,
        stop_loss_config: StopLossConfig,
        portfolio_allocation: PortfolioAllocation | None = None,
    ) -> None:
        self.risk_limits = risk_limits
        self.stop_loss_config = stop_loss_config
        self.portfolio_allocation = portfolio_allocation

        # Initialize risk analyzer
        self.risk_analyzer = RiskDecompositionAnalyzer()

        # Risk state
        self.current_risk: PortfolioRisk | None = None
        self.risk_history: list[PortfolioRisk] = []
        self.position_risks: dict[str, PositionRisk] = {}

        # Stop-loss tracking
        self.stop_losses: dict[str, dict[str, Any]] = {}
        self.trailing_stops: dict[str, dict[str, Any]] = {}

        logger.info("Risk manager initialized")

    def calculate_position_risk(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        market_data: pd.DataFrame,
        portfolio_returns: pd.Series | None = None,
    ) -> PositionRisk:
        """Calculate risk metrics for a position."""

        # Calculate basic risk metrics
        returns = market_data["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)

        # Calculate VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        # Calculate beta if portfolio returns available
        beta = 1.0  # Default
        if portfolio_returns is not None and len(portfolio_returns) > 0:
            correlation = returns.corr(portfolio_returns)
            portfolio_vol = portfolio_returns.std()
            if portfolio_vol > 0:
                beta = correlation * volatility / portfolio_vol

        # Calculate position size
        position_size = position_value / portfolio_value

        # Calculate correlation with portfolio
        correlation_with_portfolio = 0.5  # Default, would be calculated from historical data

        # Calculate liquidity score (simplified)
        avg_volume = market_data.get("Volume", pd.Series([1000000])).mean()
        liquidity_score = min(1.0, avg_volume / 10000000)  # Normalize to 0-1

        # Calculate concentration risk
        concentration_risk = position_size**2  # Square of position size

        # Calculate stop loss level
        current_price = market_data["Close"].iloc[-1]
        stop_loss_level = current_price * (1 - self.stop_loss_config.stop_loss_pct)

        # Calculate risk contribution (simplified)
        risk_contribution = position_size * volatility

        return PositionRisk(
            symbol=symbol,
            current_value=position_value,
            position_size=position_size,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            volatility=volatility,
            correlation_with_portfolio=correlation_with_portfolio,
            liquidity_score=liquidity_score,
            concentration_risk=concentration_risk,
            stop_loss_level=stop_loss_level,
            risk_contribution=risk_contribution,
        )

    def calculate_portfolio_risk(
        self,
        positions: dict[str, PositionRisk],
        portfolio_value: float,
        market_data: dict[str, pd.DataFrame],
        historical_returns: pd.DataFrame | None = None,
    ) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics."""

        if not positions:
            return self._create_empty_portfolio_risk(portfolio_value)

        # Calculate portfolio returns if historical data available
        portfolio_returns = None
        if historical_returns is not None:
            portfolio_returns = self._calculate_portfolio_returns(positions, historical_returns)

        # Calculate basic portfolio metrics
        total_value = sum(pos.current_value for pos in positions.values())

        # Calculate portfolio volatility
        if portfolio_returns is not None and len(portfolio_returns) > 0:
            volatility = portfolio_returns.std() * np.sqrt(252)
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        else:
            # Estimate from position-level metrics
            volatility = self._estimate_portfolio_volatility(positions)
            var_95 = self._estimate_portfolio_var(positions)
            cvar_95 = var_95 * 1.5  # Rough estimate
            max_drawdown = volatility * 2  # Rough estimate

        # Calculate portfolio beta
        portfolio_beta = self._calculate_portfolio_beta(positions)

        # Calculate Sharpe ratio
        if portfolio_returns is not None:
            mean_return = portfolio_returns.mean() * 252
            sharpe_ratio = (mean_return - 0.02) / volatility  # Assuming 2% risk-free rate
        else:
            sharpe_ratio = 0.0

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(positions, market_data)

        # Calculate risk contributions
        risk_contributions = {pos.symbol: pos.risk_contribution for pos in positions.values()}

        # Calculate concentration metrics
        concentration_metrics = self._calculate_concentration_metrics(positions)

        # Calculate liquidity metrics
        liquidity_metrics = self._calculate_liquidity_metrics(positions)

        # Run stress tests
        stress_test_results = self._run_stress_tests(positions, market_data)

        return PortfolioRisk(
            total_value=total_value,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            volatility=volatility,
            beta=portfolio_beta,
            sharpe_ratio=sharpe_ratio,
            correlation_matrix=correlation_matrix,
            risk_contributions=risk_contributions,
            concentration_metrics=concentration_metrics,
            liquidity_metrics=liquidity_metrics,
            stress_test_results=stress_test_results,
            timestamp=datetime.now(),
        )

    def check_risk_limits(
        self, portfolio_risk: PortfolioRisk, positions: dict[str, PositionRisk]
    ) -> list[str]:
        """Check if portfolio violates any risk limits."""
        violations = []

        # Portfolio-level checks
        if abs(portfolio_risk.var_95) > self.risk_limits.max_portfolio_var:
            violations.append(
                f"Portfolio VaR ({portfolio_risk.var_95:.3f}) exceeds limit ({self.risk_limits.max_portfolio_var})"
            )

        if portfolio_risk.max_drawdown > self.risk_limits.max_portfolio_drawdown:
            violations.append(
                f"Portfolio drawdown ({portfolio_risk.max_drawdown:.3f}) exceeds limit ({self.risk_limits.max_portfolio_drawdown})"
            )

        if portfolio_risk.volatility > self.risk_limits.max_portfolio_volatility:
            violations.append(
                f"Portfolio volatility ({portfolio_risk.volatility:.3f}) exceeds limit ({self.risk_limits.max_portfolio_volatility})"
            )

        if portfolio_risk.beta > self.risk_limits.max_portfolio_beta:
            violations.append(
                f"Portfolio beta ({portfolio_risk.beta:.3f}) exceeds limit ({self.risk_limits.max_portfolio_beta})"
            )

        # Position-level checks
        for symbol, position in positions.items():
            if position.position_size > self.risk_limits.max_position_size:
                violations.append(
                    f"Position size for {symbol} ({position.position_size:.3f}) exceeds limit ({self.risk_limits.max_position_size})"
                )

            if position.correlation_with_portfolio > self.risk_limits.max_correlation:
                violations.append(
                    f"Correlation for {symbol} ({position.correlation_with_portfolio:.3f}) exceeds limit ({self.risk_limits.max_correlation})"
                )

        # Concentration checks
        if portfolio_risk.concentration_metrics.get("herfindahl_index", 0) > 0.25:
            violations.append("Portfolio concentration too high")

        return violations

    def optimize_position_sizing(
        self, positions: dict[str, PositionRisk], target_risk: float, method: str = "risk_parity"
    ) -> dict[str, float]:
        """Optimize position sizing to meet risk targets."""

        if method == "risk_parity":
            return self._optimize_risk_parity_sizing(positions, target_risk)
        elif method == "equal_risk":
            return self._optimize_equal_risk_sizing(positions, target_risk)
        else:
            return self._optimize_kelly_sizing(positions, target_risk)

    def update_stop_losses(
        self, symbol: str, current_price: float, entry_price: float, highest_price: float
    ) -> dict[str, Any]:
        """Update stop-loss levels for a position."""

        # Fixed stop loss
        fixed_stop = entry_price * (1 - self.stop_loss_config.stop_loss_pct)

        # Trailing stop
        trailing_stop = highest_price * (1 - self.stop_loss_config.trailing_stop_pct)

        # Breakeven stop (if profitable)
        if current_price > entry_price * (1 + self.stop_loss_config.breakeven_after_pct):
            breakeven_stop = entry_price
        else:
            breakeven_stop = fixed_stop

        # Use the highest of the three stops
        effective_stop = max(fixed_stop, trailing_stop, breakeven_stop)

        stop_info = {
            "symbol": symbol,
            "current_price": current_price,
            "entry_price": entry_price,
            "highest_price": highest_price,
            "fixed_stop": fixed_stop,
            "trailing_stop": trailing_stop,
            "breakeven_stop": breakeven_stop,
            "effective_stop": effective_stop,
            "stop_distance": (current_price - effective_stop) / current_price,
            "timestamp": datetime.now(),
        }

        self.stop_losses[symbol] = stop_info
        return stop_info

    def check_stop_losses(self, current_prices: dict[str, float]) -> list[dict[str, Any]]:
        """Check if any positions have hit their stop losses."""
        triggered_stops = []

        for symbol, stop_info in self.stop_losses.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                effective_stop = stop_info["effective_stop"]

                if current_price <= effective_stop:
                    triggered_stops.append(
                        {
                            "symbol": symbol,
                            "current_price": current_price,
                            "stop_price": effective_stop,
                            "stop_type": self._determine_stop_type(stop_info),
                            "timestamp": datetime.now(),
                        }
                    )

        return triggered_stops

    def _create_empty_portfolio_risk(self, portfolio_value: float) -> PortfolioRisk:
        """Create empty portfolio risk metrics."""
        return PortfolioRisk(
            total_value=portfolio_value,
            var_95=0.0,
            cvar_95=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            beta=0.0,
            sharpe_ratio=0.0,
            correlation_matrix=pd.DataFrame(),
            risk_contributions={},
            concentration_metrics={},
            liquidity_metrics={},
            stress_test_results={},
            timestamp=datetime.now(),
        )

    def _calculate_portfolio_returns(
        self, positions: dict[str, PositionRisk], historical_returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate portfolio returns from historical data."""
        portfolio_returns = pd.Series(0.0, index=historical_returns.index)

        for symbol, position in positions.items():
            if symbol in historical_returns.columns:
                returns = historical_returns[symbol].fillna(0)
                portfolio_returns += position.position_size * returns

        return portfolio_returns

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from return series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _estimate_portfolio_volatility(self, positions: dict[str, PositionRisk]) -> float:
        """Estimate portfolio volatility from position-level metrics."""
        total_vol = 0.0
        for position in positions.values():
            total_vol += position.position_size * position.volatility
        return total_vol

    def _estimate_portfolio_var(self, positions: dict[str, PositionRisk]) -> float:
        """Estimate portfolio VaR from position-level metrics."""
        total_var = 0.0
        for position in positions.values():
            total_var += position.position_size * abs(position.var_95)
        return total_var

    def _calculate_portfolio_beta(self, positions: dict[str, PositionRisk]) -> float:
        """Calculate portfolio beta from position betas."""
        portfolio_beta = 0.0
        for position in positions.values():
            portfolio_beta += position.position_size * position.beta
        return portfolio_beta

    def _calculate_correlation_matrix(
        self, positions: dict[str, PositionRisk], market_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate correlation matrix between positions."""
        symbols = list(positions.keys())
        n_symbols = len(symbols)

        if n_symbols == 0:
            return pd.DataFrame()

        # Create correlation matrix (simplified)
        correlation_matrix = pd.DataFrame(np.eye(n_symbols), index=symbols, columns=symbols)

        # Add correlations based on position metrics
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    # Use correlation from position risk
                    corr = (
                        positions[symbol1].correlation_with_portfolio
                        * positions[symbol2].correlation_with_portfolio
                    )
                    correlation_matrix.loc[symbol1, symbol2] = corr

        return correlation_matrix

    def _calculate_concentration_metrics(
        self, positions: dict[str, PositionRisk]
    ) -> dict[str, float]:
        """Calculate concentration metrics."""
        if not positions:
            return {}

        position_sizes = [pos.position_size for pos in positions.values()]

        # Herfindahl index
        herfindahl_index = sum(size**2 for size in position_sizes)

        # Gini coefficient (simplified)
        sorted_sizes = sorted(position_sizes)
        n = len(sorted_sizes)
        gini = sum((2 * i - n - 1) * size for i, size in enumerate(sorted_sizes)) / (
            n * sum(sorted_sizes)
        )

        return {
            "herfindahl_index": herfindahl_index,
            "gini_coefficient": gini,
            "largest_position": max(position_sizes),
            "top_5_concentration": sum(sorted(position_sizes, reverse=True)[:5]),
        }

    def _calculate_liquidity_metrics(self, positions: dict[str, PositionRisk]) -> dict[str, float]:
        """Calculate liquidity metrics."""
        if not positions:
            return {}

        liquidity_scores = [pos.liquidity_score for pos in positions.values()]
        position_sizes = [pos.position_size for pos in positions.values()]

        # Weighted average liquidity
        weighted_liquidity = sum(
            score * size for score, size in zip(liquidity_scores, position_sizes, strict=False)
        )

        # Illiquid exposure
        illiquid_exposure = sum(
            size
            for score, size in zip(liquidity_scores, position_sizes, strict=False)
            if score < 0.5
        )

        return {
            "weighted_liquidity": weighted_liquidity,
            "illiquid_exposure": illiquid_exposure,
            "min_liquidity": min(liquidity_scores),
            "avg_liquidity": np.mean(liquidity_scores),
        }

    def _run_stress_tests(
        self, positions: dict[str, PositionRisk], market_data: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Run stress tests on the portfolio."""

        stress_scenarios = {
            "market_crash": -0.20,  # 20% market crash
            "volatility_spike": 2.0,  # 2x volatility
            "correlation_breakdown": 0.5,  # 50% correlation increase
            "liquidity_crisis": 0.3,  # 70% liquidity reduction
        }

        results = {}

        for scenario, shock in stress_scenarios.items():
            if scenario == "market_crash":
                # Simulate market crash impact
                portfolio_loss = sum(
                    pos.position_size * pos.beta * shock for pos in positions.values()
                )
                results[scenario] = {
                    "portfolio_loss": portfolio_loss,
                    "var_impact": portfolio_loss * 1.5,
                    "liquidity_impact": "high",
                }
            elif scenario == "volatility_spike":
                # Simulate volatility spike
                vol_impact = sum(
                    pos.position_size * pos.volatility * shock for pos in positions.values()
                )
                results[scenario] = {
                    "volatility_impact": vol_impact,
                    "var_impact": vol_impact * 2.0,
                    "liquidity_impact": "medium",
                }
            else:
                # Simplified stress test
                results[scenario] = {
                    "impact": shock,
                    "severity": "medium" if abs(shock) < 0.5 else "high",
                }

        return results

    def _optimize_risk_parity_sizing(
        self, positions: dict[str, PositionRisk], target_risk: float
    ) -> dict[str, float]:
        """Optimize position sizing for risk parity."""
        if not positions:
            return {}

        # Equal risk contribution
        sum(pos.risk_contribution for pos in positions.values())
        target_risk_per_position = target_risk / len(positions)

        new_sizes = {}
        for symbol, position in positions.items():
            if position.volatility > 0:
                new_size = target_risk_per_position / position.volatility
                new_sizes[symbol] = min(new_size, self.risk_limits.max_position_size)
            else:
                new_sizes[symbol] = 0.0

        return new_sizes

    def _optimize_equal_risk_sizing(
        self, positions: dict[str, PositionRisk], target_risk: float
    ) -> dict[str, float]:
        """Optimize position sizing for equal risk."""
        if not positions:
            return {}

        # Equal risk per position
        risk_per_position = target_risk / len(positions)

        new_sizes = {}
        for symbol, position in positions.items():
            if position.volatility > 0:
                new_size = risk_per_position / position.volatility
                new_sizes[symbol] = min(new_size, self.risk_limits.max_position_size)
            else:
                new_sizes[symbol] = 0.0

        return new_sizes

    def _optimize_kelly_sizing(
        self, positions: dict[str, PositionRisk], target_risk: float
    ) -> dict[str, float]:
        """Optimize position sizing using Kelly criterion (simplified)."""
        if not positions:
            return {}

        # Simplified Kelly sizing based on Sharpe ratio
        new_sizes = {}
        total_kelly = 0.0

        for symbol, position in positions.items():
            # Estimate Kelly fraction from Sharpe ratio
            kelly_fraction = max(0, position.volatility * 0.1)  # Simplified
            new_sizes[symbol] = kelly_fraction
            total_kelly += kelly_fraction

        # Normalize to target risk
        if total_kelly > 0:
            for symbol in new_sizes:
                new_sizes[symbol] = (new_sizes[symbol] / total_kelly) * target_risk
                new_sizes[symbol] = min(new_sizes[symbol], self.risk_limits.max_position_size)

        return new_sizes

    def _determine_stop_type(self, stop_info: dict[str, Any]) -> str:
        """Determine which type of stop was triggered."""
        current_price = stop_info["current_price"]

        if current_price <= stop_info["breakeven_stop"]:
            return "breakeven"
        elif current_price <= stop_info["trailing_stop"]:
            return "trailing"
        else:
            return "fixed"

    def get_risk_summary(self) -> dict[str, Any]:
        """Get a summary of current risk metrics."""
        if self.current_risk is None:
            return {"status": "no_risk_data"}

        return {
            "timestamp": self.current_risk.timestamp,
            "total_value": self.current_risk.total_value,
            "var_95": self.current_risk.var_95,
            "volatility": self.current_risk.volatility,
            "beta": self.current_risk.beta,
            "sharpe_ratio": self.current_risk.sharpe_ratio,
            "n_positions": len(self.position_risks),
            "risk_limits_violated": len(
                self.check_risk_limits(self.current_risk, self.position_risks)
            ),
        }

    def get_risk_history(self) -> list[PortfolioRisk]:
        """Get the risk history."""
        return self.risk_history
