"""Tests for PreflightCheck initialization and mirror properties."""

from __future__ import annotations

from gpt_trader.preflight.core import PreflightCheck


class TestPreflightCheckInit:
    """Test PreflightCheck initialization."""

    def test_default_initialization(self) -> None:
        """PreflightCheck should initialize with defaults."""
        check = PreflightCheck()

        assert check.verbose is False
        assert check.profile == "canary"
        assert check.context is not None
        assert check.context.profile == "canary"

    def test_custom_initialization(self) -> None:
        """PreflightCheck should accept custom values."""
        check = PreflightCheck(verbose=True, profile="prod")

        assert check.verbose is True
        assert check.profile == "prod"
        assert check.context.verbose is True
        assert check.context.profile == "prod"


class TestPreflightCheckCompatibilityMirrors:
    """Test backwards-compatibility properties that mirror context."""

    def test_errors_mirrors_context(self) -> None:
        """errors property should mirror context.errors."""
        check = PreflightCheck()
        check.context.errors.append("test error")

        assert check.errors == ["test error"]
        assert check.errors is check.context.errors

    def test_warnings_mirrors_context(self) -> None:
        """warnings property should mirror context.warnings."""
        check = PreflightCheck()
        check.context.warnings.append("test warning")

        assert check.warnings == ["test warning"]

    def test_successes_mirrors_context(self) -> None:
        """successes property should mirror context.successes."""
        check = PreflightCheck()
        check.context.successes.append("test success")

        assert check.successes == ["test success"]

    def test_config_mirrors_context(self) -> None:
        """config property should mirror context.config."""
        check = PreflightCheck()
        check.context.config["key"] = "value"

        assert check.config == {"key": "value"}
