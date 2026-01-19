"""Tests for PreflightContext logging methods."""

from __future__ import annotations

import pytest

from gpt_trader.preflight.context import Colors, PreflightContext


class TestPreflightContextLogging:
    """Test logging methods."""

    def test_log_success_appends_to_list(self, capsys: pytest.CaptureFixture) -> None:
        """log_success should append to successes list and print."""
        ctx = PreflightContext()
        ctx.log_success("Test passed")

        assert "Test passed" in ctx.successes
        captured = capsys.readouterr()
        assert "Test passed" in captured.out
        assert Colors.GREEN in captured.out

    def test_log_warning_appends_to_list(self, capsys: pytest.CaptureFixture) -> None:
        """log_warning should append to warnings list and print."""
        ctx = PreflightContext()
        ctx.log_warning("Potential issue")

        assert "Potential issue" in ctx.warnings
        captured = capsys.readouterr()
        assert "Potential issue" in captured.out
        assert Colors.YELLOW in captured.out

    def test_log_error_appends_to_list(self, capsys: pytest.CaptureFixture) -> None:
        """log_error should append to errors list and print."""
        ctx = PreflightContext()
        ctx.log_error("Critical failure")

        assert "Critical failure" in ctx.errors
        captured = capsys.readouterr()
        assert "Critical failure" in captured.out
        assert Colors.RED in captured.out

    def test_log_info_only_prints_when_verbose(self, capsys: pytest.CaptureFixture) -> None:
        """log_info should only print when verbose=True."""
        ctx_quiet = PreflightContext(verbose=False)
        ctx_verbose = PreflightContext(verbose=True)

        ctx_quiet.log_info("Info message")
        captured = capsys.readouterr()
        assert "Info message" not in captured.out

        ctx_verbose.log_info("Info message")
        captured = capsys.readouterr()
        assert "Info message" in captured.out
        assert Colors.CYAN in captured.out

    def test_section_header_prints_formatted(self, capsys: pytest.CaptureFixture) -> None:
        """section_header should print a formatted header."""
        ctx = PreflightContext()
        ctx.section_header("Test Section")

        captured = capsys.readouterr()
        assert "Test Section" in captured.out
        assert "=" in captured.out
        assert Colors.BLUE in captured.out
        assert Colors.BOLD in captured.out
