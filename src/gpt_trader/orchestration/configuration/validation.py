"""Validation helpers for bot configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError


class ConfigValidationError(Exception):
    """Raised when configuration values fail validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "; ".join(errors) if errors else "Invalid configuration"
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


__all__ = [
    "ConfigValidationError",
    "ConfigValidationResult",
    "format_validation_errors",
]
