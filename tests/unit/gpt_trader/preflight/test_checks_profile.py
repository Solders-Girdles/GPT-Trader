"""Tests for profile configuration preflight checks."""

from __future__ import annotations

from pathlib import Path

from typing import Any

import pytest

from gpt_trader.preflight.checks.profile import check_profile_configuration
from gpt_trader.preflight.core import PreflightCheck


def _write_profile(tmp_path: Path, profile: str, content: str) -> Path:
    profile_path = tmp_path / "config" / "profiles" / f"{profile}.yaml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(content, encoding="utf-8")
    return profile_path


def _last_result_details(checker: PreflightCheck) -> dict[str, Any]:
    if not checker.results:
        return {}
    return checker.results[-1]["details"]  # type: ignore[index]


class TestCheckProfileConfiguration:
    """Test profile configuration validation."""

    def test_passes_when_profile_exists_and_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should pass when profile file exists and is valid YAML."""
        monkeypatch.chdir(tmp_path)
        checker = PreflightCheck(profile="dev")

        yaml_content = """
trading:
  mode: normal
risk_management:
  daily_loss_limit_pct: 0.1
"""
        _write_profile(tmp_path, "dev", yaml_content)
        result = check_profile_configuration(checker)

        assert result is True
        assert any("validated" in s for s in checker.successes)

    def test_passes_with_canary_profile_correct_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should pass when canary profile has correct restricted settings."""
        monkeypatch.chdir(tmp_path)
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
        _write_profile(tmp_path, "canary", yaml_content)
        result = check_profile_configuration(checker)

        assert result is True
        assert any("validated" in s for s in checker.successes)

    def test_warns_when_canary_has_wrong_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should warn when canary profile has unexpected values."""
        monkeypatch.chdir(tmp_path)
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
        _write_profile(tmp_path, "canary", yaml_content)
        result = check_profile_configuration(checker)

        # Should still pass (warnings, not errors)
        assert result is True
        # Should have warnings about mismatched values
        assert any("expected" in w for w in checker.warnings)

    def test_fails_when_yaml_parse_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should fail when YAML cannot be parsed."""
        monkeypatch.chdir(tmp_path)
        checker = PreflightCheck(profile="dev")

        # Invalid YAML (unterminated flow sequence).
        _write_profile(tmp_path, "dev", "trading: [")
        result = check_profile_configuration(checker)

        assert result is False
        assert any("Failed to parse profile" in e for e in checker.errors)
        details = _last_result_details(checker)
        assert details.get("category") == "yaml_parse"
        assert details.get("severity") == "error"
        assert details.get("remediation", "").startswith("Inspect")

    def test_warns_when_profile_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should warn when profile file is not found."""
        monkeypatch.chdir(tmp_path)
        checker = PreflightCheck(profile="dev")

        result = check_profile_configuration(checker)

        # Should still pass with warning
        assert result is True
        assert any("not found" in w for w in checker.warnings)
        details = _last_result_details(checker)
        assert details.get("category") == "missing_file"
        assert details.get("severity") == "warning"

    def test_shows_canary_defaults_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should show canary defaults when canary profile not found."""
        monkeypatch.chdir(tmp_path)
        checker = PreflightCheck(profile="canary", verbose=True)

        check_profile_configuration(checker)

        captured = capsys.readouterr()
        assert "Canary defaults" in captured.out or any(
            "canary" in w.lower() for w in checker.warnings
        )

    def test_shows_prod_warning_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should warn about testing with canary when prod profile missing."""
        monkeypatch.chdir(tmp_path)
        checker = PreflightCheck(profile="prod")

        check_profile_configuration(checker)

        assert any("tested with canary" in w for w in checker.warnings)

    def test_prints_section_header(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should print section header."""
        monkeypatch.chdir(tmp_path)
        checker = PreflightCheck(profile="dev")

        check_profile_configuration(checker)

        captured = capsys.readouterr()
        assert "PROFILE CONFIGURATION" in captured.out

    def test_handles_nested_key_access_for_canary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should handle nested key traversal for canary validation."""
        monkeypatch.chdir(tmp_path)
        checker = PreflightCheck(profile="canary", verbose=True)

        # Partial config - some keys missing
        yaml_content = """
trading:
  mode: reduce_only
"""
        _write_profile(tmp_path, "canary", yaml_content)
        result = check_profile_configuration(checker)

        # Should still pass, with warnings for missing keys
        assert result is True
