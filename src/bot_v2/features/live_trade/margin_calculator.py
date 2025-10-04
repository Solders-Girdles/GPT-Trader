"""
Margin Calculator Component.

Handles margin requirements, leverage calculation, and liquidation risk
assessment with proper warning thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class MarginMetrics:
    """Complete margin and leverage metrics."""

    positions_value: Decimal
    equity: Decimal
    leverage: Decimal

    # Margin requirements
    initial_margin_required: Decimal
    maintenance_margin_required: Decimal
    margin_used: Decimal
    margin_available: Decimal

    # Risk levels
    margin_health: Literal["healthy", "warning", "critical", "liquidation_risk"]
    margin_buffer_pct: Decimal  # % of equity available as buffer

    # Warnings
    warnings: list[str]

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "positions_value": float(self.positions_value),
            "equity": float(self.equity),
            "leverage": float(self.leverage),
            "initial_margin_required": float(self.initial_margin_required),
            "maintenance_margin_required": float(self.maintenance_margin_required),
            "margin_used": float(self.margin_used),
            "margin_available": float(self.margin_available),
            "margin_health": self.margin_health,
            "margin_buffer_pct": float(self.margin_buffer_pct),
            "warnings": self.warnings,
        }


class MarginCalculator:
    """
    Margin and leverage calculator with liquidation risk assessment.

    Uses standard perpetuals margin formulas:
    - Initial margin = positions_value / max_leverage
    - Maintenance margin = positions_value * maintenance_margin_rate
    - Leverage = positions_value / equity
    """

    # Default margin parameters (Coinbase Advanced perpetuals)
    DEFAULT_MAX_LEVERAGE = Decimal("10")  # 10x max leverage
    DEFAULT_MAINTENANCE_MARGIN_RATE = Decimal("0.05")  # 5% maintenance

    # Warning thresholds
    WARNING_THRESHOLD_PCT = Decimal("0.20")  # Warn at 20% buffer
    CRITICAL_THRESHOLD_PCT = Decimal("0.10")  # Critical at 10% buffer

    @staticmethod
    def calculate_margin_metrics(
        positions_value: Decimal,
        cash_balance: Decimal,
        unrealized_pnl: Decimal = Decimal("0"),
        max_leverage: Decimal | None = None,
        maintenance_margin_rate: Decimal | None = None,
    ) -> MarginMetrics:
        """
        Calculate comprehensive margin metrics.

        Args:
            positions_value: Total notional value of positions
            cash_balance: Available cash balance
            unrealized_pnl: Unrealized PnL from positions
            max_leverage: Maximum allowed leverage (default 10x)
            maintenance_margin_rate: Maintenance margin rate (default 5%)

        Returns:
            MarginMetrics with all calculations and warnings
        """
        if max_leverage is None:
            max_leverage = MarginCalculator.DEFAULT_MAX_LEVERAGE
        if maintenance_margin_rate is None:
            maintenance_margin_rate = MarginCalculator.DEFAULT_MAINTENANCE_MARGIN_RATE

        warnings = []

        # Calculate equity
        equity = cash_balance + unrealized_pnl

        # Handle zero or negative equity
        if equity <= 0:
            return MarginCalculator._create_liquidation_metrics(
                positions_value, equity, "Negative or zero equity - liquidation imminent"
            )

        # Handle zero positions (no margin required)
        if positions_value == 0:
            return MarginCalculator._create_zero_position_metrics(cash_balance, unrealized_pnl)

        # Calculate leverage
        leverage = positions_value / equity

        # Initial margin required (for opening positions)
        initial_margin_required = positions_value / max_leverage

        # Maintenance margin required (to avoid liquidation)
        maintenance_margin_required = positions_value * maintenance_margin_rate

        # Margin used (use initial margin as conservative estimate)
        margin_used = initial_margin_required

        # Margin available (equity - maintenance margin)
        margin_available = equity - maintenance_margin_required

        # Calculate margin buffer percentage
        margin_buffer_pct = (
            (margin_available / equity * Decimal("100")) if equity > 0 else Decimal("0")
        )

        # Assess margin health
        margin_health, health_warnings = MarginCalculator._assess_margin_health(
            margin_available, margin_buffer_pct, leverage, max_leverage
        )
        warnings.extend(health_warnings)

        return MarginMetrics(
            positions_value=positions_value,
            equity=equity,
            leverage=leverage,
            initial_margin_required=initial_margin_required,
            maintenance_margin_required=maintenance_margin_required,
            margin_used=margin_used,
            margin_available=margin_available,
            margin_health=margin_health,
            margin_buffer_pct=margin_buffer_pct,
            warnings=warnings,
        )

    @staticmethod
    def _assess_margin_health(
        margin_available: Decimal,
        margin_buffer_pct: Decimal,
        leverage: Decimal,
        max_leverage: Decimal,
    ) -> tuple[Literal["healthy", "warning", "critical", "liquidation_risk"], list[str]]:
        """Assess margin health and generate warnings."""
        warnings = []

        # Liquidation risk
        if margin_available <= 0:
            warnings.append("LIQUIDATION RISK: Margin available is zero or negative")
            return "liquidation_risk", warnings

        # Critical (< 10% buffer)
        if margin_buffer_pct < MarginCalculator.CRITICAL_THRESHOLD_PCT * Decimal("100"):
            warnings.append(
                f"CRITICAL: Margin buffer at {margin_buffer_pct:.1f}% (below {MarginCalculator.CRITICAL_THRESHOLD_PCT * 100}%)"
            )
            return "critical", warnings

        # Warning (< 20% buffer)
        if margin_buffer_pct < MarginCalculator.WARNING_THRESHOLD_PCT * Decimal("100"):
            warnings.append(
                f"WARNING: Margin buffer at {margin_buffer_pct:.1f}% (below {MarginCalculator.WARNING_THRESHOLD_PCT * 100}%)"
            )
            return "warning", warnings

        # Check leverage limits
        if leverage > max_leverage:
            warnings.append(f"WARNING: Leverage {leverage:.2f}x exceeds max {max_leverage}x")
            return "warning", warnings

        # Healthy
        return "healthy", warnings

    @staticmethod
    def _create_liquidation_metrics(
        positions_value: Decimal, equity: Decimal, warning: str
    ) -> MarginMetrics:
        """Create metrics for liquidation scenario."""
        return MarginMetrics(
            positions_value=positions_value,
            equity=equity,
            leverage=Decimal("999"),  # Effectively infinite
            initial_margin_required=Decimal("0"),
            maintenance_margin_required=Decimal("0"),
            margin_used=Decimal("0"),
            margin_available=Decimal("0"),
            margin_health="liquidation_risk",
            margin_buffer_pct=Decimal("0"),
            warnings=[warning],
        )

    @staticmethod
    def _create_zero_position_metrics(
        cash_balance: Decimal, unrealized_pnl: Decimal
    ) -> MarginMetrics:
        """Create metrics when no positions are held."""
        equity = cash_balance + unrealized_pnl
        return MarginMetrics(
            positions_value=Decimal("0"),
            equity=equity,
            leverage=Decimal("0"),
            initial_margin_required=Decimal("0"),
            maintenance_margin_required=Decimal("0"),
            margin_used=Decimal("0"),
            margin_available=equity,
            margin_health="healthy",
            margin_buffer_pct=Decimal("100"),
            warnings=[],
        )

    @staticmethod
    def calculate_max_position_size(
        equity: Decimal,
        price: Decimal,
        max_leverage: Decimal | None = None,
    ) -> Decimal:
        """
        Calculate maximum position size given equity and leverage.

        Args:
            equity: Available equity
            price: Asset price
            max_leverage: Maximum leverage (default 10x)

        Returns:
            Maximum position size in base units
        """
        if max_leverage is None:
            max_leverage = MarginCalculator.DEFAULT_MAX_LEVERAGE

        if equity <= 0 or price <= 0:
            return Decimal("0")

        # Max notional = equity * max_leverage
        max_notional = equity * max_leverage

        # Max quantity = max_notional / price
        return max_notional / price
