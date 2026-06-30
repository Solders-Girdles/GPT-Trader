"""Risk data models for the live risk manager.

Inert value objects describing risk warnings, volatility-check outcomes, and
exposure state, plus RiskValidationError. The enforcement logic that produces
them lives in LiveRiskManager (the risk.manager package); keeping the contracts
here lets callers depend on the shapes without importing the manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class RiskValidationError(Exception):
    """Raised when a trade fails risk validation checks.

    Note: This is specific to risk validation failures. For general validation
    errors, use gpt_trader.errors.ValidationError instead.
    """

    pass


# Transitional alias for backwards compatibility - remove after migration
ValidationError = RiskValidationError


class RiskWarningLevel(Enum):
    """Severity levels for risk warnings."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RiskWarning:
    """A risk warning generated during checks."""

    level: RiskWarningLevel
    message: str
    action: str = ""  # Suggested action: REDUCE_POSITION, CLOSE_POSITION, etc.
    symbol: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "message": self.message,
            "action": self.action,
            "symbol": self.symbol,
            "details": self.details,
        }


@dataclass
class VolatilityCheckOutcome:
    """Result of a volatility circuit breaker check."""

    triggered: bool = False
    symbol: str = ""
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "symbol": self.symbol,
            "reason": self.reason,
        }


@dataclass
class ExposureState:
    """Tracks exposure across spot and CFM markets."""

    spot_exposure: Decimal = field(default_factory=lambda: Decimal("0"))
    cfm_exposure: Decimal = field(default_factory=lambda: Decimal("0"))
    cfm_margin_used: Decimal = field(default_factory=lambda: Decimal("0"))
    cfm_available_margin: Decimal = field(default_factory=lambda: Decimal("0"))
    cfm_buying_power: Decimal = field(default_factory=lambda: Decimal("0"))

    @property
    def total_exposure(self) -> Decimal:
        """Total notional exposure across all markets."""
        return self.spot_exposure + self.cfm_exposure

    @property
    def cfm_margin_utilization(self) -> Decimal:
        """Percentage of CFM margin currently used."""
        if self.cfm_available_margin + self.cfm_margin_used == 0:
            return Decimal("0")
        total_margin = self.cfm_available_margin + self.cfm_margin_used
        return self.cfm_margin_used / total_margin

    def to_payload(self) -> dict[str, Any]:
        return {
            "spot_exposure": str(self.spot_exposure),
            "cfm_exposure": str(self.cfm_exposure),
            "cfm_margin_used": str(self.cfm_margin_used),
            "cfm_available_margin": str(self.cfm_available_margin),
            "cfm_buying_power": str(self.cfm_buying_power),
            "total_exposure": str(self.total_exposure),
            "cfm_margin_utilization": str(self.cfm_margin_utilization),
        }
