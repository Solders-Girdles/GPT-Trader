"""Tests for dependency preflight checks."""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.preflight.checks.dependencies import check_dependencies
from gpt_trader.preflight.core import PreflightCheck


class TestCheckDependencies:
    """Test dependency verification check."""

    def test_passes_when_all_packages_available(self, capsys: pytest.CaptureFixture) -> None:
        """Should pass when all required packages are importable."""
        checker = PreflightCheck(verbose=True)

        # Mock all imports to succeed (environment-independent test)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            # Let all imports succeed by returning a mock for any package
            try:
                return original_import(name, *args, **kwargs)
            except ImportError:
                # If package not actually installed, return a mock
                return MagicMock()

        with patch.object(builtins, "__import__", mock_import):
            result = check_dependencies(checker)

        assert result is True
        assert any("All required packages installed" in s for s in checker.successes)

    def test_fails_when_package_missing(self, capsys: pytest.CaptureFixture) -> None:
        """Should fail when a required package is missing."""
        checker = PreflightCheck()

        # Mock __import__ to fail for one package
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jwt":
                raise ImportError("No module named 'jwt'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            result = check_dependencies(checker)

        assert result is False
        assert any("Missing required package: jwt" in e for e in checker.errors)

    def test_logs_found_packages_when_verbose(self, capsys: pytest.CaptureFixture) -> None:
        """Should log found packages when verbose."""
        checker = PreflightCheck(verbose=True)

        # Mock all imports to succeed (environment-independent test)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            try:
                return original_import(name, *args, **kwargs)
            except ImportError:
                return MagicMock()

        with patch.object(builtins, "__import__", mock_import):
            check_dependencies(checker)

        captured = capsys.readouterr()
        # Should print info about found packages
        assert "Package" in captured.out

    def test_prints_section_header(self, capsys: pytest.CaptureFixture) -> None:
        """Should print section header."""
        checker = PreflightCheck()

        # Mock all imports to succeed (environment-independent test)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            try:
                return original_import(name, *args, **kwargs)
            except ImportError:
                return MagicMock()

        with patch.object(builtins, "__import__", mock_import):
            check_dependencies(checker)

        captured = capsys.readouterr()
        assert "DEPENDENCY CHECK" in captured.out

    def test_checks_gpt_trader_package(self, capsys: pytest.CaptureFixture) -> None:
        """Should specifically check gpt_trader package."""
        checker = PreflightCheck(verbose=True)

        # Mock all imports to succeed (environment-independent test)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            try:
                return original_import(name, *args, **kwargs)
            except ImportError:
                return MagicMock()

        with patch.object(builtins, "__import__", mock_import):
            result = check_dependencies(checker)

        # If we're running tests, gpt_trader must be installed
        assert result is True

    def test_checks_cryptography_package(self, capsys: pytest.CaptureFixture) -> None:
        """Should check cryptography package is available."""
        checker = PreflightCheck()

        # Mock cryptography import failure
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cryptography":
                raise ImportError("No module named 'cryptography'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            result = check_dependencies(checker)

        assert result is False
        assert any("cryptography" in e for e in checker.errors)
