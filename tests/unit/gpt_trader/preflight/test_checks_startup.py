"""Tests for startup preflight checks: dependencies and basic diagnostics."""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.checks.dependencies import check_dependencies
from gpt_trader.preflight.checks.diagnostics import check_pretrade_diagnostics
from gpt_trader.preflight.core import PreflightCheck


class TestCheckDependencies:
    """Test dependency verification check."""

    def test_passes_when_all_packages_available(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
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

        with monkeypatch.context() as mp:
            mp.setattr(builtins, "__import__", mock_import)
            result = check_dependencies(checker)

        assert result is True
        assert any("All required packages installed" in s for s in checker.successes)

    def test_fails_when_package_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should fail when a required package is missing."""
        checker = PreflightCheck()

        # Mock __import__ to fail for one package
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jwt":
                raise ImportError("No module named 'jwt'")
            return original_import(name, *args, **kwargs)

        with monkeypatch.context() as mp:
            mp.setattr(builtins, "__import__", mock_import)
            result = check_dependencies(checker)

        assert result is False
        assert any("Missing required package: jwt" in e for e in checker.errors)

    def test_logs_found_packages_when_verbose(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should log found packages when verbose."""
        checker = PreflightCheck(verbose=True)

        # Mock all imports to succeed (environment-independent test)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            try:
                return original_import(name, *args, **kwargs)
            except ImportError:
                return MagicMock()

        with monkeypatch.context() as mp:
            mp.setattr(builtins, "__import__", mock_import)
            check_dependencies(checker)

        captured = capsys.readouterr()
        # Should print info about found packages
        assert "Package" in captured.out

    def test_prints_section_header(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should print section header."""
        checker = PreflightCheck()

        # Mock all imports to succeed (environment-independent test)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            try:
                return original_import(name, *args, **kwargs)
            except ImportError:
                return MagicMock()

        with monkeypatch.context() as mp:
            mp.setattr(builtins, "__import__", mock_import)
            check_dependencies(checker)

        captured = capsys.readouterr()
        assert "DEPENDENCY CHECK" in captured.out

    def test_checks_gpt_trader_package(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should specifically check gpt_trader package."""
        checker = PreflightCheck(verbose=True)

        # Mock all imports to succeed (environment-independent test)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            try:
                return original_import(name, *args, **kwargs)
            except ImportError:
                return MagicMock()

        with monkeypatch.context() as mp:
            mp.setattr(builtins, "__import__", mock_import)
            result = check_dependencies(checker)

        # If we're running tests, gpt_trader must be installed
        assert result is True

    def test_checks_cryptography_package(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should check cryptography package is available."""
        checker = PreflightCheck()

        # Mock cryptography import failure
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cryptography":
                raise ImportError("No module named 'cryptography'")
            return original_import(name, *args, **kwargs)

        with monkeypatch.context() as mp:
            mp.setattr(builtins, "__import__", mock_import)
            result = check_dependencies(checker)

        assert result is False
        assert any("cryptography" in e for e in checker.errors)


class TestCheckPretradeDiagnostics:
    """Test pre-trade diagnostics checks."""

    def test_skips_when_remote_checks_bypassed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should skip and succeed when remote checks are bypassed."""
        monkeypatch.setenv("COINBASE_PREFLIGHT_SKIP_REMOTE", "1")

        chk = PreflightCheck(profile="dev")
        assert check_pretrade_diagnostics(chk) is True
        assert any("bypassed" in s for s in chk.successes)

    def test_fails_when_client_build_fails(
        self,
        checker,
        force_remote_env: pytest.MonkeyPatch,
    ) -> None:
        """Should fail when CDP client cannot be built."""
        # Clear credentials to force client build failure
        force_remote_env.delenv("COINBASE_CDP_API_KEY", raising=False)
        force_remote_env.delenv("COINBASE_CDP_PRIVATE_KEY", raising=False)
        assert check_pretrade_diagnostics(checker) is False

    def test_warn_only_returns_true_when_client_build_fails(
        self,
        checker,
        warn_only_env: pytest.MonkeyPatch,
    ) -> None:
        """Should return True in warn-only mode when client build fails."""
        # Clear credentials to force client build failure
        warn_only_env.delenv("COINBASE_CDP_API_KEY", raising=False)
        warn_only_env.delenv("COINBASE_CDP_PRIVATE_KEY", raising=False)
        assert check_pretrade_diagnostics(checker) is True
        assert any("warn-only" in w.lower() for w in checker.warnings)


class TestDiagnosticsSectionHeader:
    """Test section header display."""

    def test_prints_section_header(
        self,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should print section header."""
        monkeypatch.setenv("COINBASE_PREFLIGHT_SKIP_REMOTE", "1")

        chk = PreflightCheck(profile="dev")
        check_pretrade_diagnostics(chk)
        assert "PRE-TRADE DIAGNOSTICS" in capsys.readouterr().out
