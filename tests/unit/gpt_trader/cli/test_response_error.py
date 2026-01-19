"""Unit tests for CLI response error models."""

from __future__ import annotations

from gpt_trader.cli.response import CliError, CliErrorCode


class TestCliError:
    """Test CliError dataclass."""

    def test_to_dict_basic(self):
        error = CliError(
            code="RUN_NOT_FOUND",
            message="Run not found",
        )
        result = error.to_dict()
        assert result["code"] == "RUN_NOT_FOUND"
        assert result["message"] == "Run not found"
        assert "details" not in result

    def test_to_dict_with_details(self):
        error = CliError(
            code="RUN_NOT_FOUND",
            message="Run not found",
            details={"run_id": "opt_123"},
        )
        result = error.to_dict()
        assert result["details"]["run_id"] == "opt_123"

    def test_from_code(self):
        error = CliError.from_code(
            CliErrorCode.RUN_NOT_FOUND,
            "Run 'opt_123' not found",
            run_id="opt_123",
        )
        assert error.code == "RUN_NOT_FOUND"
        assert error.message == "Run 'opt_123' not found"
        assert error.details["run_id"] == "opt_123"


class TestCliErrorCode:
    """Test CliErrorCode enum."""

    def test_all_codes_are_strings(self):
        for code in CliErrorCode:
            assert isinstance(code.value, str)
            assert code.value == code.value.upper()

    def test_common_codes_exist(self):
        assert CliErrorCode.RUN_NOT_FOUND
        assert CliErrorCode.FILE_NOT_FOUND
        assert CliErrorCode.CONFIG_INVALID
        assert CliErrorCode.VALIDATION_ERROR
        assert CliErrorCode.API_ERROR
