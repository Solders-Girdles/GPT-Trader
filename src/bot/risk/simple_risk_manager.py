"""Simplified Risk Manager for Integration Testing.

This module provides simplified versions of risk management classes
that don't depend on complex analytics modules, allowing the integration
orchestrator to work without all dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


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
    max_daily_loss: float = 0.03  # 3% daily loss limit

    # Liquidity limits
    min_daily_volume: float = 1_000_000  # Min $1M daily volume
    max_position_pct_volume: float = 0.05  # Max 5% of daily volume


@dataclass
class StopLossConfig:
    """Configuration for stop-loss management."""

    stop_loss_pct: float = 0.05  # 5% stop loss
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    take_profit_pct: float = 0.10  # 10% take profit
    use_atr_stops: bool = True  # Use ATR-based stops
    atr_multiplier: float = 2.0  # ATR multiplier for stops
    min_hold_days: int = 1  # Minimum holding period
    max_hold_days: int = 252  # Maximum holding period


class RiskManager:
    """Simplified risk manager for integration testing."""

    def __init__(self, risk_limits: RiskLimits, stop_loss_config: StopLossConfig):
        """Initialize the risk manager.

        Args:
            risk_limits: Risk limit configuration
            stop_loss_config: Stop loss configuration
        """
        self.risk_limits = risk_limits
        self.stop_loss_config = stop_loss_config

        # State tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.stop_losses: Dict[str, float] = {}
        self.risk_violations: List[Dict[str, Any]] = []

        logger.info("Simplified RiskManager initialized")

    def update_stop_losses(
        self, symbol: str, current_price: float, entry_price: float, highest_price: float
    ) -> Dict[str, Any]:
        """Update stop loss levels for a position.

        Args:
            symbol: Stock symbol
            current_price: Current market price
            entry_price: Entry price of position
            highest_price: Highest price since entry

        Returns:
            Updated stop loss information
        """
        try:
            # Calculate static stop loss
            static_stop = entry_price * (1 - self.stop_loss_config.stop_loss_pct)

            # Calculate trailing stop
            trailing_stop = highest_price * (1 - self.stop_loss_config.trailing_stop_pct)

            # Use the higher of static and trailing stops
            new_stop_loss = max(static_stop, trailing_stop)

            # Update stored stop loss (ratchet up only)
            if symbol not in self.stop_losses:
                self.stop_losses[symbol] = new_stop_loss
            else:
                self.stop_losses[symbol] = max(self.stop_losses[symbol], new_stop_loss)

            # Check if stop is triggered
            stop_triggered = current_price <= self.stop_losses[symbol]

            return {
                "symbol": symbol,
                "current_price": current_price,
                "stop_loss_price": self.stop_losses[symbol],
                "static_stop": static_stop,
                "trailing_stop": trailing_stop,
                "stop_triggered": stop_triggered,
                "entry_price": entry_price,
                "highest_price": highest_price,
                "current_pnl_pct": (current_price - entry_price) / entry_price,
            }

        except Exception as e:
            logger.error(f"Error updating stop loss for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e), "stop_triggered": False}

    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for triggered stop losses.

        Args:
            current_prices: Current market prices

        Returns:
            List of triggered stop loss events
        """
        triggered_stops = []

        for symbol, stop_price in self.stop_losses.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]

                if current_price <= stop_price:
                    triggered_stops.append(
                        {
                            "symbol": symbol,
                            "current_price": current_price,
                            "stop_price": stop_price,
                            "timestamp": datetime.now(),
                            "event_type": "stop_loss_triggered",
                        }
                    )

                    logger.info(
                        f"Stop loss triggered for {symbol}: "
                        f"price {current_price:.2f} <= stop {stop_price:.2f}"
                    )

        return triggered_stops

    def validate_position_size(
        self, symbol: str, position_size: float, current_price: float, portfolio_value: float
    ) -> Dict[str, Any]:
        """Validate a position against size limits.

        Args:
            symbol: Stock symbol
            position_size: Proposed position size (shares)
            current_price: Current stock price
            portfolio_value: Total portfolio value

        Returns:
            Validation result with adjustments if needed
        """
        position_value = abs(position_size) * current_price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0

        result = {
            "symbol": symbol,
            "original_size": position_size,
            "adjusted_size": position_size,
            "position_value": position_value,
            "position_pct": position_pct,
            "valid": True,
            "warnings": [],
            "adjustments": [],
        }

        # Check position size limit
        if position_pct > self.risk_limits.max_position_size:
            max_position_value = portfolio_value * self.risk_limits.max_position_size
            adjusted_size = int(max_position_value / current_price)

            result["adjusted_size"] = adjusted_size
            result["valid"] = False
            result["warnings"].append(
                f"Position size reduced from {position_size} to {adjusted_size} shares "
                f"({position_pct:.1%} -> {self.risk_limits.max_position_size:.1%})"
            )
            result["adjustments"].append("position_size_limit")

        return result

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk state.

        Returns:
            Risk summary dictionary
        """
        return {
            "active_positions": len(self.positions),
            "active_stop_losses": len(self.stop_losses),
            "recent_violations": len(self.risk_violations),
            "risk_limits": {
                "max_position_size": self.risk_limits.max_position_size,
                "max_portfolio_exposure": self.risk_limits.max_sector_exposure,
                "max_daily_loss": self.risk_limits.max_daily_loss,
            },
            "stop_loss_config": {
                "stop_loss_pct": self.stop_loss_config.stop_loss_pct,
                "trailing_stop_pct": self.stop_loss_config.trailing_stop_pct,
                "use_atr_stops": self.stop_loss_config.use_atr_stops,
            },
        }

    def calculate_portfolio_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        returns_data: Dict[str, pd.Series] = None,
    ) -> Dict[str, float]:
        """Calculate basic portfolio risk metrics.

        Args:
            positions: Current positions {symbol: shares}
            prices: Current prices {symbol: price}
            returns_data: Historical returns data (optional)

        Returns:
            Portfolio risk metrics
        """
        if not positions or not prices:
            return {
                "portfolio_value": 0.0,
                "total_exposure": 0.0,
                "largest_position_pct": 0.0,
                "position_count": 0,
                "concentration_risk": 0.0,
            }

        # Calculate position values
        position_values = {}
        total_value = 0.0

        for symbol, shares in positions.items():
            if symbol in prices and shares != 0:
                value = abs(shares) * prices[symbol]
                position_values[symbol] = value
                total_value += value

        # Calculate metrics
        if total_value == 0:
            return {
                "portfolio_value": 0.0,
                "total_exposure": 0.0,
                "largest_position_pct": 0.0,
                "position_count": 0,
                "concentration_risk": 0.0,
            }

        largest_position = max(position_values.values()) if position_values else 0.0
        largest_position_pct = largest_position / total_value

        # Herfindahl concentration index
        weights = [v / total_value for v in position_values.values()]
        concentration_risk = sum(w**2 for w in weights)

        return {
            "portfolio_value": total_value,
            "total_exposure": total_value,  # Simplified - assumes no short positions
            "largest_position_pct": largest_position_pct,
            "position_count": len(position_values),
            "concentration_risk": concentration_risk,
        }

    def clear_position(self, symbol: str) -> None:
        """Clear position and associated risk tracking.

        Args:
            symbol: Symbol to clear
        """
        self.positions.pop(symbol, None)
        self.stop_losses.pop(symbol, None)
        logger.debug(f"Cleared position tracking for {symbol}")

    def reset(self) -> None:
        """Reset all risk tracking state."""
        self.positions.clear()
        self.stop_losses.clear()
        self.risk_violations.clear()
        logger.info("Risk manager state reset")
