"""Tests for profile configuration preflight checks."""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from gpt_trader.preflight.checks.profile import check_profile_configuration
from gpt_trader.preflight.core import PreflightCheck


class TestCheckProfileConfiguration:
    """Test profile configuration validation."""

    def test_passes_when_profile_exists_and_valid(self) -> None:
        """Should pass when profile file exists and is valid YAML."""
        checker = PreflightCheck(profile="dev")

        yaml_content = """
trading:
  mode: normal
risk_management:
  daily_loss_limit_pct: 0.1
"""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=yaml_content)),
        ):
            result = check_profile_configuration(checker)

        assert result is True
        assert any("validated" in s for s in checker.successes)

    def test_passes_with_canary_profile_correct_settings(self) -> None:
        """Should pass when canary profile has correct restricted settings."""
        checker = PreflightCheck(profile="canary", verbose=True)

        yaml_content = """
trading:
  mode: reduce_only
  position_sizing:
    max_position_size: 0.01
risk_management:
  daily_loss_limit_pct: 0.01
  max_leverage: 1.0
"""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=yaml_content)),
        ):
            result = check_profile_configuration(checker)

        assert result is True
        assert any("validated" in s for s in checker.successes)

    def test_warns_when_canary_has_wrong_values(self, capsys: pytest.CaptureFixture) -> None:
        """Should warn when canary profile has unexpected values."""
        checker = PreflightCheck(profile="canary", verbose=True)

        yaml_content = """
trading:
  mode: normal
  position_sizing:
    max_position_size: 1.0
risk_management:
  daily_loss_limit_pct: 0.2
  max_leverage: 5.0
"""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=yaml_content)),
        ):
            result = check_profile_configuration(checker)

        # Should still pass (warnings, not errors)
        assert result is True
        # Should have warnings about mismatched values
        assert any("expected" in w for w in checker.warnings)

    def test_fails_when_yaml_parse_fails(self) -> None:
        """Should fail when YAML cannot be parsed."""
        checker = PreflightCheck(profile="dev")

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="invalid: yaml: content: [")),
            patch("yaml.safe_load", side_effect=Exception("YAML parse error")),
        ):
            result = check_profile_configuration(checker)

        assert result is False
        assert any("Failed to parse profile" in e for e in checker.errors)

    def test_warns_when_profile_not_found(self) -> None:
        """Should warn when profile file is not found."""
        checker = PreflightCheck(profile="dev")

        with patch("pathlib.Path.exists", return_value=False):
            result = check_profile_configuration(checker)

        # Should still pass with warning
        assert result is True
        assert any("not found" in w for w in checker.warnings)

    def test_shows_canary_defaults_when_missing(self, capsys: pytest.CaptureFixture) -> None:
        """Should show canary defaults when canary profile not found."""
        checker = PreflightCheck(profile="canary", verbose=True)

        with patch("pathlib.Path.exists", return_value=False):
            check_profile_configuration(checker)

        captured = capsys.readouterr()
        assert "Canary defaults" in captured.out or any(
            "canary" in w.lower() for w in checker.warnings
        )

    def test_shows_prod_warning_when_missing(self) -> None:
        """Should warn about testing with canary when prod profile missing."""
        checker = PreflightCheck(profile="prod")

        with patch("pathlib.Path.exists", return_value=False):
            check_profile_configuration(checker)

        assert any("tested with canary" in w for w in checker.warnings)

    def test_prints_section_header(self, capsys: pytest.CaptureFixture) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        with patch("pathlib.Path.exists", return_value=False):
            check_profile_configuration(checker)

        captured = capsys.readouterr()
        assert "PROFILE CONFIGURATION" in captured.out

    def test_handles_nested_key_access_for_canary(self) -> None:
        """Should handle nested key traversal for canary validation."""
        checker = PreflightCheck(profile="canary", verbose=True)

        # Partial config - some keys missing
        yaml_content = """
trading:
  mode: reduce_only
"""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=yaml_content)),
        ):
            result = check_profile_configuration(checker)

        # Should still pass, with warnings for missing keys
        assert result is True
