"""Tests for validation result types."""

from gpt_trader.preflight.validation_result import (
    CredentialValidationResult,
    PermissionDetails,
    ValidationCategory,
    ValidationFinding,
    ValidationSeverity,
    normalize_preflight_result,
)


class TestValidationSeverity:
    """Tests for ValidationSeverity enum."""

    def test_severity_values(self):
        """Verify all severity values are defined."""
        assert ValidationSeverity.SUCCESS.value == "success"
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"


class TestValidationCategory:
    """Tests for ValidationCategory enum."""

    def test_category_values(self):
        """Verify all category values are defined."""
        assert ValidationCategory.KEY_FORMAT.value == "key_format"
        assert ValidationCategory.CONNECTIVITY.value == "connectivity"
        assert ValidationCategory.PERMISSIONS.value == "permissions"
        assert ValidationCategory.ACCOUNT_STATUS.value == "account_status"
        assert ValidationCategory.MODE_COMPATIBILITY.value == "mode_compatibility"


class TestValidationFinding:
    """Tests for ValidationFinding dataclass."""

    def test_minimal_finding(self):
        """Create a finding with only required fields."""
        finding = ValidationFinding(
            category=ValidationCategory.KEY_FORMAT,
            severity=ValidationSeverity.SUCCESS,
            message="Key format valid",
        )
        assert finding.category == ValidationCategory.KEY_FORMAT
        assert finding.severity == ValidationSeverity.SUCCESS
        assert finding.message == "Key format valid"
        assert finding.details == ""
        assert finding.suggestion == ""
        assert finding.raw_data is None

    def test_full_finding(self):
        """Create a finding with all fields."""
        finding = ValidationFinding(
            category=ValidationCategory.PERMISSIONS,
            severity=ValidationSeverity.ERROR,
            message="Trade permission missing",
            details="can_trade=False",
            suggestion="Enable trade permission in Developer Portal",
            raw_data={"can_trade": False},
        )
        assert finding.details == "can_trade=False"
        assert finding.suggestion == "Enable trade permission in Developer Portal"
        assert finding.raw_data == {"can_trade": False}


class TestPermissionDetails:
    """Tests for PermissionDetails dataclass."""

    def test_default_values(self):
        """Verify default permission values are False/empty."""
        perms = PermissionDetails()
        assert perms.can_trade is False
        assert perms.can_view is False
        assert perms.portfolio_type == ""
        assert perms.portfolio_uuid == ""

    def test_full_permissions(self):
        """Create permissions with all values set."""
        perms = PermissionDetails(
            can_trade=True,
            can_view=True,
            portfolio_type="INTX",
            portfolio_uuid="uuid-12345",
        )
        assert perms.can_trade is True
        assert perms.can_view is True
        assert perms.portfolio_type == "INTX"
        assert perms.portfolio_uuid == "uuid-12345"


class TestCredentialValidationResult:
    """Tests for CredentialValidationResult dataclass."""

    def test_default_values(self):
        """Verify default result values."""
        result = CredentialValidationResult()
        assert result.valid_for_mode is False
        assert result.mode == ""
        assert result.credentials_configured is False
        assert result.key_format_valid is False
        assert result.connectivity_ok is False
        assert result.jwt_generation_ok is False
        assert result.api_latency_ms == 0.0
        assert result.permissions == PermissionDetails()
        assert result.findings == []

    def test_has_errors_empty(self):
        """has_errors should be False for empty findings."""
        result = CredentialValidationResult()
        assert result.has_errors is False

    def test_has_errors_with_success(self):
        """has_errors should be False with only success findings."""
        result = CredentialValidationResult()
        result.add_success(ValidationCategory.KEY_FORMAT, "Format OK")
        assert result.has_errors is False

    def test_has_errors_with_error(self):
        """has_errors should be True with error findings."""
        result = CredentialValidationResult()
        result.add_error(ValidationCategory.PERMISSIONS, "Trade missing")
        assert result.has_errors is True

    def test_has_warnings(self):
        """has_warnings should detect warning findings."""
        result = CredentialValidationResult()
        assert result.has_warnings is False
        result.add_warning(ValidationCategory.ACCOUNT_STATUS, "Portfolio UUID missing")
        assert result.has_warnings is True

    def test_blocking_issues(self):
        """blocking_issues should return only error findings."""
        result = CredentialValidationResult()
        result.add_success(ValidationCategory.KEY_FORMAT, "OK")
        result.add_warning(ValidationCategory.ACCOUNT_STATUS, "Warning")
        result.add_error(ValidationCategory.PERMISSIONS, "Error 1")
        result.add_error(ValidationCategory.MODE_COMPATIBILITY, "Error 2")

        blocking = result.blocking_issues
        assert len(blocking) == 2
        assert all(f.severity == ValidationSeverity.ERROR for f in blocking)

    def test_counts(self):
        """Verify count properties."""
        result = CredentialValidationResult()
        result.add_success(ValidationCategory.KEY_FORMAT, "OK 1")
        result.add_success(ValidationCategory.CONNECTIVITY, "OK 2")
        result.add_info(ValidationCategory.ACCOUNT_STATUS, "Info")
        result.add_warning(ValidationCategory.PERMISSIONS, "Warning")
        result.add_error(ValidationCategory.MODE_COMPATIBILITY, "Error")

        assert result.success_count == 2
        assert result.warning_count == 1
        assert result.error_count == 1

    def test_add_success(self):
        """add_success should create a success finding."""
        result = CredentialValidationResult()
        result.add_success(
            ValidationCategory.KEY_FORMAT,
            "API key format valid",
            "organizations/123/apiKeys/456",
        )

        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.category == ValidationCategory.KEY_FORMAT
        assert finding.severity == ValidationSeverity.SUCCESS
        assert finding.message == "API key format valid"
        assert finding.details == "organizations/123/apiKeys/456"

    def test_add_info(self):
        """add_info should create an info finding."""
        result = CredentialValidationResult()
        result.add_info(
            ValidationCategory.PERMISSIONS,
            "Trade permission not granted",
            details="View-only key",
            suggestion="Enable trade for live mode",
        )

        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.severity == ValidationSeverity.INFO
        assert finding.suggestion == "Enable trade for live mode"

    def test_add_warning(self):
        """add_warning should create a warning finding."""
        result = CredentialValidationResult()
        result.add_warning(
            ValidationCategory.ACCOUNT_STATUS,
            "Portfolio UUID missing",
            suggestion="Check API key access",
        )

        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.severity == ValidationSeverity.WARNING

    def test_add_error(self):
        """add_error should create an error finding."""
        result = CredentialValidationResult()
        result.add_error(
            ValidationCategory.MODE_COMPATIBILITY,
            "LIVE mode requires trade permission",
            details="can_trade=False",
            suggestion="Create a new API key with trade enabled",
        )

        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.severity == ValidationSeverity.ERROR
        assert finding.details == "can_trade=False"


class TestPreflightResultNormalization:
    """Tests for preflight result normalization helper."""

    def test_normalizes_pass_from_bool(self) -> None:
        result = normalize_preflight_result(True, message="All good")

        assert result["status"] == "pass"
        assert result["message"] == "All good"
        assert result["details"] == {}

    def test_normalizes_warning_from_status_alias(self) -> None:
        result = normalize_preflight_result(
            {"status": "warning", "message": "Heads up", "details": {"checks": 2}}
        )

        assert result["status"] == "warn"
        assert result["message"] == "Heads up"
        assert result["details"] == {"checks": 2}

    def test_normalizes_failure_with_string_details(self) -> None:
        result = normalize_preflight_result(
            False,
            message="Broken",
            details="Missing credentials",
        )

        assert result["status"] == "fail"
        assert result["message"] == "Broken"
        assert result["details"] == {"detail": "Missing credentials"}
