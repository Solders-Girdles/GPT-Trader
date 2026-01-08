"""Validation helpers for bot configuration.

This module provides configuration validation utilities including:
- ConfigValidationError: Exception for validation failures
- ConfigValidationResult: Pydantic model for validation results
- format_validation_errors: Helper to format Pydantic errors
- validate_config: Validate BotConfig instances
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from gpt_trader.app.config.bot_config import BotConfig


class ConfigValidationError(Exception):
    """Raised when configuration values fail validation."""

    def __init__(self, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        message = "; ".join(self.errors) if self.errors else "Invalid configuration"
        super().__init__(message)


class ConfigValidationResult(BaseModel):
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if validation failed."""
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return bool(self.warnings)


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Coerce Pydantic validation errors into user-facing messages."""

    errors: list[str] = []
    for error in exc.errors():
        loc = error.get("loc", ())
        field_path = ".".join(str(item) for item in loc)
        message = error.get("msg", "")
        if field_path:
            errors.append(f"{field_path}: {message}")
        else:
            errors.append(message or str(exc))
    if not errors:
        errors.append(str(exc))
    return errors


def validate_config(config: BotConfig) -> list[str]:
    """
    Validate the configuration and return a list of error messages.
    Returns an empty list if the configuration is valid.
    """
    errors = []

    # Validate Risk Config
    risk = config.risk
    if risk.position_fraction <= 0 or risk.position_fraction > 1:
        errors.append(
            f"risk.position_fraction must be between 0 and 1, got {risk.position_fraction}"
        )

    if risk.stop_loss_pct <= 0 or risk.stop_loss_pct >= 1:
        errors.append(f"risk.stop_loss_pct must be between 0 and 1, got {risk.stop_loss_pct}")

    if risk.take_profit_pct <= 0:
        errors.append(f"risk.take_profit_pct must be positive, got {risk.take_profit_pct}")

    if risk.max_leverage < 1:
        errors.append(f"risk.max_leverage must be >= 1, got {risk.max_leverage}")

    # Validate Strategy Config
    strategy = config.strategy
    # Check if it's the expected strategy type before accessing fields
    if hasattr(strategy, "short_ma_period") and hasattr(strategy, "long_ma_period"):
        if strategy.short_ma_period >= strategy.long_ma_period:
            errors.append(
                f"strategy.short_ma_period ({strategy.short_ma_period}) must be less than "
                f"strategy.long_ma_period ({strategy.long_ma_period})"
            )

    # Validate Symbols
    if not config.symbols:
        errors.append("symbols list cannot be empty")

    return errors


__all__ = [
    "ConfigValidationError",
    "ConfigValidationResult",
    "format_validation_errors",
    "validate_config",
]
