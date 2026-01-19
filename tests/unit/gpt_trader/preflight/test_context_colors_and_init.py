"""Tests for PreflightContext colors and initialization."""

from __future__ import annotations

from gpt_trader.preflight.context import Colors, PreflightContext


class TestColors:
    """Test terminal color constants."""

    def test_colors_are_ansi_escape_sequences(self) -> None:
        """All color constants should be ANSI escape sequences."""
        assert Colors.RED.startswith("\033[")
        assert Colors.GREEN.startswith("\033[")
        assert Colors.YELLOW.startswith("\033[")
        assert Colors.BLUE.startswith("\033[")
        assert Colors.RESET.startswith("\033[")
        assert Colors.BOLD.startswith("\033[")

    def test_reset_terminates_formatting(self) -> None:
        """RESET should end all formatting."""
        assert Colors.RESET == "\033[0m"


class TestPreflightContextInit:
    """Test PreflightContext initialization."""

    def test_default_initialization(self) -> None:
        """Context should initialize with sensible defaults."""
        ctx = PreflightContext()

        assert ctx.verbose is False
        assert ctx.profile == "canary"
        assert ctx.errors == []
        assert ctx.warnings == []
        assert ctx.successes == []
        assert ctx.config == {}

    def test_custom_initialization(self) -> None:
        """Context should accept custom values."""
        ctx = PreflightContext(verbose=True, profile="prod")

        assert ctx.verbose is True
        assert ctx.profile == "prod"

    def test_lists_are_independent(self) -> None:
        """Each context should have independent lists."""
        ctx1 = PreflightContext()
        ctx2 = PreflightContext()

        ctx1.errors.append("error1")
        ctx2.warnings.append("warning1")

        assert ctx1.errors == ["error1"]
        assert ctx2.errors == []
        assert ctx1.warnings == []
        assert ctx2.warnings == ["warning1"]
