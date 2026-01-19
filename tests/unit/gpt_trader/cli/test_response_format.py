"""Unit tests for format_response helper."""

from __future__ import annotations

import json

from gpt_trader.cli.response import CliErrorCode, CliResponse, format_response


class TestFormatResponse:
    """Test format_response function."""

    def test_json_format(self):
        response = CliResponse.success_response(
            command="test",
            data={"key": "value"},
        )
        output = format_response(response, "json")
        parsed = json.loads(output)
        assert parsed["success"] is True
        assert parsed["data"]["key"] == "value"

    def test_text_format_success_with_data(self):
        response = CliResponse.success_response(
            command="test",
            data={"key": "value"},
        )
        output = format_response(response, "text")
        assert "key" in output
        assert "value" in output

    def test_text_format_success_no_data(self):
        response = CliResponse.success_response(command="test")
        output = format_response(response, "text")
        assert "Operation completed successfully" in output

    def test_text_format_success_string_data(self):
        response = CliResponse.success_response(
            command="test",
            data="Custom output message",
        )
        output = format_response(response, "text")
        assert output == "Custom output message"

    def test_text_format_error(self):
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
