"""Tests for logging setup environment helpers."""

from __future__ import annotations

import pytest

from gpt_trader.logging.setup import _env_flag


class TestEnvFlag:
    """Test the _env_flag helper function."""

    def test_env_flag_true_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that various true values are recognized."""
        true_values = ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"]
        for value in true_values:
            monkeypatch.setenv("TEST_FLAG", value)
            assert _env_flag("TEST_FLAG") is True, f"Failed for value: {value}"

    def test_env_flag_false_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that various false values are recognized."""
        false_values = ["0", "false", "False", "no", "No", "off", "Off", ""]
        for value in false_values:
            monkeypatch.setenv("TEST_FLAG", value)
            assert _env_flag("TEST_FLAG") is False, f"Failed for value: {value}"

    def test_env_flag_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default value is used when env var not set."""
        monkeypatch.delenv("MISSING_FLAG", raising=False)
        assert _env_flag("MISSING_FLAG", default="0") is False
        assert _env_flag("MISSING_FLAG", default="1") is True

    def test_env_flag_with_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that whitespace is stripped."""
        monkeypatch.setenv("TEST_FLAG", "  true  ")
        assert _env_flag("TEST_FLAG") is True

        monkeypatch.setenv("TEST_FLAG", "  0  ")
        assert _env_flag("TEST_FLAG") is False
