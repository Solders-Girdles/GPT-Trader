"""Risk Management Integration Layer for GPT-Trader.

This module provides the risk management integration layer that validates all allocations
against risk limits and ensures portfolio safety during trading operations.

Features:
- Real-time allocation validation against risk limits
- Position sizing adjustments based on risk parameters
- Portfolio-level exposure monitoring
- Stop-loss and take-profit calculations
- Risk metrics reporting and alerts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd
from bot.config import get_config
from bot.portfolio.allocator import PortfolioRules
from bot.risk.simple_risk_manager import RiskLimits, RiskManager, StopLossConfig

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Comprehensive risk configuration for integration."""

    # Position limits
    max_position_size: float = 0.10  # 10% max per position
    max_sector_exposure: float = 0.30  # 30% max sector exposure
    max_correlation: float = 0.70  # Max correlation between positions

    # Portfolio limits
    max_portfolio_exposure: float = 0.95  # 95% max total exposure
    max_portfolio_var: float = 0.02  # 2% VaR limit
    max_portfolio_volatility: float = 0.25  # 25% volatility limit
    max_drawdown: float = 0.15  # 15% max drawdown

    # Risk per trade
    max_risk_per_trade: float = 0.01  # 1% risk per trade
    max_daily_loss: float = 0.03  # 3% max daily loss

    # Stop-loss parameters
    default_stop_loss_pct: float = 0.05  # 5% stop loss
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    take_profit_pct: float = 0.10  # 10% take profit

    # Risk monitoring
    risk_check_frequency: int = 300  # seconds
    enable_realtime_monitoring: bool = True
    alert_on_limit_breach: bool = True

    # Advanced risk features
    use_dynamic_sizing: bool = True
    correlation_lookback_days: int = 60
    volatility_lookback_days: int = 30
    stress_test_enabled: bool = True


@dataclass
class AllocationResult:
    """Result of allocation validation with risk adjustments."""

    original_allocations: Dict[str, int] = field(default_factory=dict)
    adjusted_allocations: Dict[str, int] = field(default_factory=dict)
    warnings: Dict[str, str] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    stop_levels: Dict[str, Dict[str, float]] = field(default_factory=dict)
    total_exposure: float = 0.0
    risk_budget_used: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    passed_validation: bool = True

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def allocation_changed(self) -> bool:
        """Check if allocations were modified."""
        return self.original_allocations != self.adjusted_allocations


class RiskIntegration:
    """Risk management integration layer for portfolio allocation validation."""

    def __init__(
        self, risk_config: RiskConfig | None = None, portfolio_rules: PortfolioRules | None = None
    ):
        """Initialize risk integration layer.

        Args:
            risk_config: Risk configuration parameters
            portfolio_rules: Portfolio allocation rules
        """
        self.config = get_config()
        self.risk_config = risk_config or RiskConfig()
        self.portfolio_rules = portfolio_rules or PortfolioRules()

        # Create risk limits from config
        risk_limits = RiskLimits(
            max_portfolio_var=self.risk_config.max_portfolio_var,
            max_portfolio_drawdown=self.risk_config.max_drawdown,
            max_portfolio_volatility=self.risk_config.max_portfolio_volatility,
            max_position_size=self.risk_config.max_position_size,
            max_sector_exposure=self.risk_config.max_sector_exposure,
            max_correlation=self.risk_config.max_correlation,
            max_risk_per_trade=self.risk_config.max_risk_per_trade,
            max_daily_loss=self.risk_config.max_daily_loss,
        )

        # Create stop-loss config
        stop_loss_config = StopLossConfig(
            stop_loss_pct=self.risk_config.default_stop_loss_pct,
            trailing_stop_pct=self.risk_config.trailing_stop_pct,
        )

        # Initialize risk manager
        self.risk_manager = RiskManager(risk_limits=risk_limits, stop_loss_config=stop_loss_config)

        # Risk state tracking
        self.current_portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.last_risk_check = datetime.now()
        self.risk_violations = []

        logger.info("Risk integration layer initialized")

    def validate_allocations(
        self,
        allocations: Dict[str, int],
        current_prices: Dict[str, float],
        portfolio_value: float,
        market_data: Dict[str, pd.DataFrame] | None = None,
        current_positions: Dict[str, int] | None = None,
    ) -> AllocationResult:
        """
        Validate allocations against comprehensive risk limits.

        Args:
            allocations: Proposed allocations {symbol: shares}
            current_prices: Current market prices {symbol: price}
            portfolio_value: Total portfolio value
            market_data: Historical market data for risk calculations
            current_positions: Current position sizes

        Returns:
            AllocationResult with adjusted allocations and risk metrics
        """
        logger.info(f"Validating allocations for {len(allocations)} symbols")

        result = AllocationResult(
            original_allocations=allocations.copy(), adjusted_allocations=allocations.copy()
        )

        try:
            # Update portfolio state
            self.current_portfolio_value = portfolio_value

            # Phase 1: Basic position size validation
            self._validate_position_sizes(result, current_prices, portfolio_value)

            # Phase 2: Portfolio exposure validation
            self._validate_portfolio_exposure(result, current_prices, portfolio_value)

            # Phase 3: Risk budget validation
            self._validate_risk_budget(result, current_prices, portfolio_value)

            # Phase 4: Calculate stop-loss levels
            self._calculate_stop_levels(result, current_prices)

            # Phase 5: Advanced risk checks (if market data available)
            if market_data:
                self._advanced_risk_validation(result, market_data, current_prices, portfolio_value)

            # Phase 6: Generate risk metrics
            self._calculate_risk_metrics(result, current_prices, portfolio_value)

            # Final validation check
            result.passed_validation = len(result.warnings) == 0 or all(
                "reduced" in warning.lower() or "scaled" in warning.lower()
                for warning in result.warnings.values()
            )

            logger.info(
                f"Risk validation complete. "
                f"Adjusted {len([k for k, v in result.adjusted_allocations.items()
                              if v != result.original_allocations.get(k, 0)])} positions"
            )

        except Exception as e:
            logger.error(f"Error during risk validation: {e}")
            result.warnings["validation_error"] = f"Risk validation failed: {str(e)}"
            result.passed_validation = False

        return result

    def _validate_position_sizes(
        self, result: AllocationResult, current_prices: Dict[str, float], portfolio_value: float
    ) -> None:
        """Validate individual position sizes against limits."""
        for symbol, shares in result.adjusted_allocations.copy().items():
            if symbol not in current_prices:
                result.warnings[symbol] = "No price data available"
                result.adjusted_allocations[symbol] = 0
                continue

            position_value = shares * current_prices[symbol]
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0

            # Check position size limit
            if position_pct > self.risk_config.max_position_size:
                max_shares = int(
                    (portfolio_value * self.risk_config.max_position_size) / current_prices[symbol]
                )
                result.adjusted_allocations[symbol] = max_shares
                result.warnings[symbol] = (
                    f"Position size reduced from {shares} to {max_shares} shares "
                    f"({position_pct:.1%} -> {self.risk_config.max_position_size:.1%})"
                )
                logger.warning(
                    f"Position size limit breach for {symbol}: {result.warnings[symbol]}"
                )

    def _validate_portfolio_exposure(
        self, result: AllocationResult, current_prices: Dict[str, float], portfolio_value: float
    ) -> None:
        """Validate total portfolio exposure."""
        total_exposure = sum(
            result.adjusted_allocations[sym] * current_prices[sym]
            for sym in result.adjusted_allocations
            if sym in current_prices
        )

        result.total_exposure = total_exposure / portfolio_value if portfolio_value > 0 else 0

        # Check total exposure limit
        if result.total_exposure > self.risk_config.max_portfolio_exposure:
            scale_factor = self.risk_config.max_portfolio_exposure / result.total_exposure

            for symbol in result.adjusted_allocations:
                if symbol in current_prices:
                    result.adjusted_allocations[symbol] = int(
                        result.adjusted_allocations[symbol] * scale_factor
                    )

            result.warnings["portfolio"] = (
                f"Total exposure scaled down by {scale_factor:.1%} "
                f"({result.total_exposure:.1%} -> {self.risk_config.max_portfolio_exposure:.1%})"
            )

            # Recalculate exposure after scaling
            result.total_exposure = self.risk_config.max_portfolio_exposure

            logger.warning(f"Portfolio exposure limit breach: {result.warnings['portfolio']}")

    def _validate_risk_budget(
        self, result: AllocationResult, current_prices: Dict[str, float], portfolio_value: float
    ) -> None:
        """Validate risk budget allocation."""
        total_risk = 0.0

        for symbol, shares in result.adjusted_allocations.items():
            if symbol in current_prices and shares > 0:
                position_value = shares * current_prices[symbol]
                # Simplified risk calculation based on position size
                position_risk = (
                    position_value / portfolio_value
                ) * self.risk_config.max_risk_per_trade
                total_risk += position_risk

        result.risk_budget_used = total_risk

        # Check if total risk exceeds reasonable limits
        max_total_risk = self.risk_config.max_risk_per_trade * len(result.adjusted_allocations)
        if total_risk > max_total_risk:
            risk_scale_factor = max_total_risk / total_risk

            for symbol in result.adjusted_allocations:
                if symbol in current_prices:
                    result.adjusted_allocations[symbol] = int(
                        result.adjusted_allocations[symbol] * risk_scale_factor
                    )

            result.warnings["risk_budget"] = (
                f"Risk budget exceeded, positions scaled by {risk_scale_factor:.1%}"
            )

    def _calculate_stop_levels(
        self, result: AllocationResult, current_prices: Dict[str, float]
    ) -> None:
        """Calculate stop-loss and take-profit levels."""
        for symbol, shares in result.adjusted_allocations.items():
            if symbol in current_prices and shares > 0:
                current_price = current_prices[symbol]

                # Calculate stop levels
                stop_loss = current_price * (1 - self.risk_config.default_stop_loss_pct)
                trailing_stop = current_price * (1 - self.risk_config.trailing_stop_pct)
                take_profit = current_price * (1 + self.risk_config.take_profit_pct)

                result.stop_levels[symbol] = {
                    "stop_loss": stop_loss,
                    "trailing_stop": trailing_stop,
                    "take_profit": take_profit,
                    "current_price": current_price,
                    "risk_per_share": current_price - stop_loss,
                }

    def _advanced_risk_validation(
        self,
        result: AllocationResult,
        market_data: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        portfolio_value: float,
    ) -> None:
        """Perform advanced risk validation using historical data."""
        try:
            # Calculate portfolio volatility and correlations
            returns_data = {}
            for symbol in result.adjusted_allocations:
                if symbol in market_data:
                    df = market_data[symbol]
                    if "Close" in df.columns and len(df) > 1:
                        returns = df["Close"].pct_change().dropna()
                        if len(returns) >= self.risk_config.volatility_lookback_days:
                            returns_data[symbol] = returns.tail(
                                self.risk_config.volatility_lookback_days
                            )

            if len(returns_data) > 1:
                # Calculate correlation matrix
                returns_df = pd.DataFrame(returns_data)
                correlation_matrix = returns_df.corr()

                # Check for high correlations
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        corr = correlation_matrix.iloc[i, j]
                        if abs(corr) > self.risk_config.max_correlation:
                            symbol1 = correlation_matrix.columns[i]
                            symbol2 = correlation_matrix.columns[j]
                            high_corr_pairs.append((symbol1, symbol2, corr))

                if high_corr_pairs:
                    result.warnings["correlation"] = (
                        f"High correlation detected: {len(high_corr_pairs)} pairs exceed "
                        f"{self.risk_config.max_correlation:.1%} threshold"
                    )

                # Calculate portfolio volatility
                portfolio_weights = {}
                total_value = sum(
                    result.adjusted_allocations[sym] * current_prices[sym]
                    for sym in result.adjusted_allocations
                    if sym in current_prices
                )

                for symbol in result.adjusted_allocations:
                    if symbol in current_prices:
                        weight = (
                            result.adjusted_allocations[symbol] * current_prices[symbol]
                        ) / total_value
                        portfolio_weights[symbol] = weight

                # Simplified portfolio volatility calculation
                portfolio_vol = sum(
                    portfolio_weights.get(symbol, 0) * returns_data[symbol].std() * (252**0.5)
                    for symbol in returns_data
                )

                result.risk_metrics["portfolio_volatility"] = portfolio_vol

                if portfolio_vol > self.risk_config.max_portfolio_volatility:
                    result.warnings["volatility"] = (
                        f"Portfolio volatility ({portfolio_vol:.1%}) exceeds limit "
                        f"({self.risk_config.max_portfolio_volatility:.1%})"
                    )

        except Exception as e:
            logger.warning(f"Advanced risk validation failed: {e}")
            result.warnings["advanced_risk"] = "Advanced risk calculations unavailable"

    def _calculate_risk_metrics(
        self, result: AllocationResult, current_prices: Dict[str, float], portfolio_value: float
    ) -> None:
        """Calculate comprehensive risk metrics."""
        metrics = {
            "total_positions": len([s for s, q in result.adjusted_allocations.items() if q > 0]),
            "total_exposure_pct": result.total_exposure,
            "largest_position_pct": 0.0,
            "risk_budget_used": result.risk_budget_used,
            "avg_position_size": 0.0,
            "concentration_ratio": 0.0,
        }

        if portfolio_value > 0:
            position_values = [
                result.adjusted_allocations[sym] * current_prices[sym]
                for sym in result.adjusted_allocations
                if sym in current_prices and result.adjusted_allocations[sym] > 0
            ]

            if position_values:
                total_invested = sum(position_values)
                metrics["largest_position_pct"] = max(position_values) / portfolio_value
                metrics["avg_position_size"] = (
                    total_invested / len(position_values) / portfolio_value
                )

                # Herfindahl concentration index
                weights = [pv / total_invested for pv in position_values]
                metrics["concentration_ratio"] = sum(w**2 for w in weights)

        result.risk_metrics.update(metrics)

    def check_daily_loss_limit(self, current_pnl: float) -> bool:
        """Check if daily loss limit has been breached.

        Args:
            current_pnl: Current day's P&L

        Returns:
            True if daily loss limit is breached
        """
        self.daily_pnl = current_pnl
        max_loss = self.current_portfolio_value * self.risk_config.max_daily_loss

        if current_pnl < -max_loss:
            logger.critical(f"Daily loss limit breached: {current_pnl:.2f} < -{max_loss:.2f}")
            return True

        return False

    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": self.current_portfolio_value,
            "daily_pnl": self.daily_pnl,
            "risk_config": {
                "max_position_size": self.risk_config.max_position_size,
                "max_portfolio_exposure": self.risk_config.max_portfolio_exposure,
                "max_daily_loss": self.risk_config.max_daily_loss,
                "default_stop_loss_pct": self.risk_config.default_stop_loss_pct,
            },
            "risk_manager_summary": self.risk_manager.get_risk_summary(),
            "recent_violations": self.risk_violations[-10:],  # Last 10 violations
            "last_risk_check": self.last_risk_check.isoformat(),
        }

    def update_stop_losses(
        self, positions: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """Update stop-loss levels for current positions.

        Args:
            positions: Dict of {symbol: {'current_price': float, 'entry_price': float, 'highest_price': float}}

        Returns:
            Updated stop-loss information for each position
        """
        stop_updates = {}

        for symbol, pos_info in positions.items():
            try:
                stop_info = self.risk_manager.update_stop_losses(
                    symbol=symbol,
                    current_price=pos_info["current_price"],
                    entry_price=pos_info["entry_price"],
                    highest_price=pos_info.get("highest_price", pos_info["current_price"]),
                )
                stop_updates[symbol] = stop_info

            except Exception as e:
                logger.error(f"Error updating stop loss for {symbol}: {e}")
                stop_updates[symbol] = {"error": str(e)}

        return stop_updates

    def check_triggered_stops(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for triggered stop losses.

        Args:
            current_prices: Current market prices

        Returns:
            List of triggered stop-loss events
        """
        return self.risk_manager.check_stop_losses(current_prices)

    def validate_new_position(
        self,
        symbol: str,
        proposed_shares: int,
        current_price: float,
        portfolio_value: float,
        existing_positions: Dict[str, int] | None = None,
    ) -> Tuple[bool, str, int]:
        """Validate a single new position against risk limits.

        Args:
            symbol: Stock symbol
            proposed_shares: Proposed number of shares
            current_price: Current stock price
            portfolio_value: Total portfolio value
            existing_positions: Current positions

        Returns:
            Tuple of (is_valid, reason, adjusted_shares)
        """
        if proposed_shares <= 0:
            return True, "No position", 0

        position_value = proposed_shares * current_price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0

        # Check position size limit
        if position_pct > self.risk_config.max_position_size:
            max_shares = int((portfolio_value * self.risk_config.max_position_size) / current_price)
            return (
                False,
                f"Position size too large: {position_pct:.1%} > {self.risk_config.max_position_size:.1%}",
                max_shares,
            )

        # Check total exposure with new position
        existing_exposure = 0.0
        if existing_positions:
            # Would need current prices for existing positions - simplified for now
            existing_exposure = (
                sum(existing_positions.values()) * current_price * 0.1
            )  # Rough estimate

        total_exposure = (existing_exposure + position_value) / portfolio_value
        if total_exposure > self.risk_config.max_portfolio_exposure:
            return (
                False,
                f"Total exposure too high: {total_exposure:.1%} > {self.risk_config.max_portfolio_exposure:.1%}",
                0,
            )

        return True, "Position validated", proposed_shares
