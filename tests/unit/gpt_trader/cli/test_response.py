"""Unit tests for CLI response envelope."""

from __future__ import annotations

import json

from gpt_trader.cli.response import (
    CliError,
    CliErrorCode,
    CliResponse,
    format_response,
)


class TestCliError:
    """Test CliError dataclass."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        error = CliError(
            code="RUN_NOT_FOUND",
            message="Run not found",
        )
        result = error.to_dict()
        assert result["code"] == "RUN_NOT_FOUND"
        assert result["message"] == "Run not found"
        assert "details" not in result  # Empty details not included

    def test_to_dict_with_details(self):
        """Test to_dict includes details when present."""
        error = CliError(
            code="RUN_NOT_FOUND",
            message="Run not found",
            details={"run_id": "opt_123"},
        )
        result = error.to_dict()
        assert result["details"]["run_id"] == "opt_123"

    def test_from_code(self):
        """Test factory method from error code enum."""
        error = CliError.from_code(
            CliErrorCode.RUN_NOT_FOUND,
            "Run 'opt_123' not found",
            run_id="opt_123",
        )
        assert error.code == "RUN_NOT_FOUND"
        assert error.message == "Run 'opt_123' not found"
        assert error.details["run_id"] == "opt_123"


class TestCliResponse:
    """Test CliResponse dataclass."""

    def test_success_response_basic(self):
        """Test creating basic success response."""
        response = CliResponse.success_response(
            command="optimize list",
            data={"runs": []},
        )
        assert response.success is True
        assert response.exit_code == 0
        assert response.command == "optimize list"
        assert response.data == {"runs": []}
        assert response.errors == []
        assert response.warnings == []

    def test_success_response_with_warnings(self):
        """Test success response with warnings."""
        response = CliResponse.success_response(
            command="optimize view",
            data={"run_id": "opt_123"},
            warnings=["No feasible trials"],
        )
        assert response.success is True
        assert response.warnings == ["No feasible trials"]

    def test_success_response_with_noop(self):
        """Test success response marked as noop."""
        response = CliResponse.success_response(
            command="optimize run",
            data={"preview": True},
            was_noop=True,
        )
        assert response.was_noop is True

    def test_error_response_basic(self):
        """Test creating basic error response."""
        response = CliResponse.error_response(
            command="optimize view",
            code=CliErrorCode.RUN_NOT_FOUND,
            message="Run not found",
        )
        assert response.success is False
        assert response.exit_code == 1
        assert response.data is None
        assert len(response.errors) == 1
        assert response.errors[0].code == "RUN_NOT_FOUND"

    def test_error_response_with_details(self):
        """Test error response with details."""
        response = CliResponse.error_response(
            command="optimize view",
            code=CliErrorCode.RUN_NOT_FOUND,
            message="Run 'opt_123' not found",
            details={"run_id": "opt_123"},
        )
        assert response.errors[0].details["run_id"] == "opt_123"

    def test_to_dict_success(self):
        """Test to_dict for success response."""
        response = CliResponse.success_response(
            command="optimize list",
            data={"runs": [{"id": 1}]},
        )
        result = response.to_dict()

        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["command"] == "optimize list"
        assert result["data"] == {"runs": [{"id": 1}]}
        assert result["errors"] == []
        assert result["warnings"] == []
        assert "metadata" in result
        assert "timestamp" in result["metadata"]
        assert result["metadata"]["was_noop"] is False
        assert result["metadata"]["version"] == "1.0"

    def test_to_dict_error(self):
        """Test to_dict for error response."""
        response = CliResponse.error_response(
            command="optimize view",
            code=CliErrorCode.RUN_NOT_FOUND,
            message="Run not found",
        )
        result = response.to_dict()

        assert result["success"] is False
        assert result["exit_code"] == 1
        assert result["data"] is None
        assert len(result["errors"]) == 1
        assert result["errors"][0]["code"] == "RUN_NOT_FOUND"

    def test_to_json_compact(self):
        """Test JSON serialization compact mode."""
        response = CliResponse.success_response(
            command="optimize list",
            data={"runs": []},
        )
        compact = response.to_json(compact=True)
        normal = response.to_json(compact=False)

        # Compact should be single line (no newlines except in strings)
        assert "\n" not in compact
        # Normal should have indentation
        assert "\n" in normal
        # Both should be valid JSON
        assert json.loads(compact) == json.loads(normal)

    def test_add_warning_fluent(self):
        """Test fluent API for adding warnings."""
        response = (
            CliResponse.success_response(command="test")
            .add_warning("Warning 1")
            .add_warning("Warning 2")
        )
        assert response.warnings == ["Warning 1", "Warning 2"]

    def test_add_error_fluent(self):
        """Test fluent API for adding errors."""
        error = CliError.from_code(CliErrorCode.VALIDATION_ERROR, "Invalid input")
        response = CliResponse.success_response(command="test").add_error(error)

        assert response.success is False
        assert response.exit_code == 1
        assert len(response.errors) == 1

    def test_exit_code_auto_set_on_failure(self):
        """Test exit_code automatically set when success=False."""
        response = CliResponse(
            success=False,
            command="test",
        )
        assert response.exit_code == 1


class TestCliErrorCode:
    """Test CliErrorCode enum."""

    def test_all_codes_are_strings(self):
        """Test all error codes serialize as strings."""
        for code in CliErrorCode:
            assert isinstance(code.value, str)
            # Should be SCREAMING_SNAKE_CASE
            assert code.value == code.value.upper()

    def test_common_codes_exist(self):
        """Test common error codes exist."""
        assert CliErrorCode.RUN_NOT_FOUND
        assert CliErrorCode.FILE_NOT_FOUND
        assert CliErrorCode.CONFIG_INVALID
        assert CliErrorCode.VALIDATION_ERROR
        assert CliErrorCode.API_ERROR


class TestFormatResponse:
    """Test format_response function."""

    def test_json_format(self):
        """Test JSON format output."""
        response = CliResponse.success_response(
            command="test",
            data={"key": "value"},
        )
        output = format_response(response, "json")
        parsed = json.loads(output)
        assert parsed["success"] is True
        assert parsed["data"]["key"] == "value"

    def test_text_format_success_with_data(self):
        """Test text format with data."""
        response = CliResponse.success_response(
            command="test",
            data={"key": "value"},
        )
        output = format_response(response, "text")
        # Should be JSON representation of data for complex data
        assert "key" in output
        assert "value" in output

    def test_text_format_success_no_data(self):
        """Test text format with no data."""
        response = CliResponse.success_response(command="test")
        output = format_response(response, "text")
        assert "Operation completed successfully" in output

    def test_text_format_success_string_data(self):
        """Test text format with string data."""
        response = CliResponse.success_response(
            command="test",
            data="Custom output message",
        )
        output = format_response(response, "text")
        assert output == "Custom output message"

    def test_text_format_error(self):
        """Test text format for errors."""
        response = CliResponse.error_response(
            command="test",
            code=CliErrorCode.RUN_NOT_FOUND,
            message="Run 'opt_123' not found",
            details={"run_id": "opt_123"},
        )
        output = format_response(response, "text")
        assert "RUN_NOT_FOUND" in output
        assert "Run 'opt_123' not found" in output
        assert "run_id" in output


class TestEnvelopeContract:
    """Test the standard envelope contract for AI agents."""

    def test_success_envelope_has_required_fields(self):
        """Test success envelope has all required fields."""
        response = CliResponse.success_response(
            command="optimize list",
            data={"runs": []},
        )
        envelope = response.to_dict()

        # Required top-level fields
        assert "success" in envelope
        assert "exit_code" in envelope
        assert "command" in envelope
        assert "data" in envelope
        assert "errors" in envelope
        assert "warnings" in envelope
        assert "metadata" in envelope

        # Required metadata fields
        assert "timestamp" in envelope["metadata"]
        assert "was_noop" in envelope["metadata"]
        assert "version" in envelope["metadata"]

    def test_error_envelope_has_required_fields(self):
        """Test error envelope has all required fields."""
        response = CliResponse.error_response(
            command="optimize view",
            code=CliErrorCode.RUN_NOT_FOUND,
            message="Not found",
        )
        envelope = response.to_dict()

        assert "success" in envelope
        assert "exit_code" in envelope
        assert "command" in envelope
        assert "data" in envelope
        assert "errors" in envelope
        assert "warnings" in envelope
        assert "metadata" in envelope

        # Errors should have code and message
        assert envelope["errors"][0]["code"]
        assert envelope["errors"][0]["message"]

    def test_envelope_is_json_serializable(self):
        """Test envelope can be JSON serialized."""
        response = CliResponse.success_response(
            command="test",
            data={"nested": {"value": 123}},
        )
        # Should not raise
        json_str = response.to_json()
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["data"]["nested"]["value"] == 123
