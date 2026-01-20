"""Tests for basic pre-trade diagnostics preflight checks."""

from __future__ import annotations

import pytest

from gpt_trader.preflight.checks.diagnostics import check_pretrade_diagnostics
from gpt_trader.preflight.core import PreflightCheck


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


class TestSectionHeader:
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
