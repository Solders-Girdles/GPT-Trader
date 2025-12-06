"""Structured types for API credential validation results.

This module provides dataclasses and enums used by the credential validation
system to report detailed findings about API key configuration, connectivity,
and permission status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationSeverity(Enum):
    """Severity level for validation findings."""

    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationCategory(Enum):
    """Category of validation check."""

    KEY_FORMAT = "key_format"
    CONNECTIVITY = "connectivity"
    PERMISSIONS = "permissions"
    ACCOUNT_STATUS = "account_status"
    MODE_COMPATIBILITY = "mode_compatibility"


@dataclass
class ValidationFinding:
    """A single validation check result.

    Attributes:
        category: The type of check that produced this finding.
        severity: How serious this finding is (success, info, warning, error).
        message: Short description of the finding.
        details: Additional context or error details.
        suggestion: Actionable suggestion to fix the issue.
        raw_data: Optional raw data from the API for debugging.
    """

    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    details: str = ""
    suggestion: str = ""
    raw_data: Any = None


@dataclass
class PermissionDetails:
    """Parsed API key permissions from Coinbase.

    These values come from the /api/v3/brokerage/key_permissions endpoint.

    Attributes:
        can_trade: Whether the key can place/cancel orders.
        can_view: Whether the key can view portfolio and balances.
        portfolio_type: Type of portfolio ("SPOT" or "INTX" for perpetuals).
        portfolio_uuid: UUID of the associated portfolio.
    """

    can_trade: bool = False
    can_view: bool = False
    portfolio_type: str = ""
    portfolio_uuid: str = ""


@dataclass
class CredentialValidationResult:
    """Complete result of credential validation for a trading mode.

    This aggregates all validation findings and provides convenience properties
    to check overall validation status.

    Attributes:
        valid_for_mode: Whether credentials satisfy the selected mode's requirements.
        mode: The trading mode being validated for (demo, paper, read_only, live).
        credentials_configured: Whether any credentials are configured.
        key_format_valid: Whether the API key format is valid.
        connectivity_ok: Whether API connectivity was successful.
        jwt_generation_ok: Whether JWT token generation succeeded.
        api_latency_ms: Measured API latency in milliseconds.
        permissions: Parsed permission details from the API.
        findings: List of individual validation findings.
    """

    valid_for_mode: bool = False
    mode: str = ""
    credentials_configured: bool = False
    key_format_valid: bool = False
    connectivity_ok: bool = False
    jwt_generation_ok: bool = False
    api_latency_ms: float = 0.0
    permissions: PermissionDetails = field(default_factory=PermissionDetails)
    findings: list[ValidationFinding] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if any findings are errors."""
        return any(f.severity == ValidationSeverity.ERROR for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        """Check if any findings are warnings."""
        return any(f.severity == ValidationSeverity.WARNING for f in self.findings)

    @property
    def blocking_issues(self) -> list[ValidationFinding]:
        """Get all findings that are blocking errors."""
        return [f for f in self.findings if f.severity == ValidationSeverity.ERROR]

    @property
    def success_count(self) -> int:
        """Count of successful checks."""
        return sum(1 for f in self.findings if f.severity == ValidationSeverity.SUCCESS)

    @property
    def warning_count(self) -> int:
        """Count of warning checks."""
        return sum(1 for f in self.findings if f.severity == ValidationSeverity.WARNING)

    @property
    def error_count(self) -> int:
        """Count of error checks."""
        return sum(1 for f in self.findings if f.severity == ValidationSeverity.ERROR)

    def add_success(
        self,
        category: ValidationCategory,
        message: str,
        details: str = "",
    ) -> None:
        """Add a successful validation finding."""
        self.findings.append(
            ValidationFinding(
                category=category,
                severity=ValidationSeverity.SUCCESS,
                message=message,
                details=details,
            )
        )

    def add_info(
        self,
        category: ValidationCategory,
        message: str,
        details: str = "",
        suggestion: str = "",
    ) -> None:
        """Add an informational finding."""
        self.findings.append(
            ValidationFinding(
                category=category,
                severity=ValidationSeverity.INFO,
                message=message,
                details=details,
                suggestion=suggestion,
            )
        )

    def add_warning(
        self,
        category: ValidationCategory,
        message: str,
        details: str = "",
        suggestion: str = "",
    ) -> None:
        """Add a warning finding."""
        self.findings.append(
            ValidationFinding(
                category=category,
                severity=ValidationSeverity.WARNING,
                message=message,
                details=details,
                suggestion=suggestion,
            )
        )

    def add_error(
        self,
        category: ValidationCategory,
        message: str,
        details: str = "",
        suggestion: str = "",
    ) -> None:
        """Add an error finding."""
        self.findings.append(
            ValidationFinding(
                category=category,
                severity=ValidationSeverity.ERROR,
                message=message,
                details=details,
                suggestion=suggestion,
            )
        )
