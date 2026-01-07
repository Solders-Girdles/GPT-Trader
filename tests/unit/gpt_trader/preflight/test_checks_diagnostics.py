"""Tests for pre-trade diagnostics preflight checks."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.preflight.checks.diagnostics import check_pretrade_diagnostics
from gpt_trader.preflight.core import PreflightCheck

FORCE_REMOTE_ENV = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1", "TRADING_SYMBOLS": "BTC-USD"}
WARN_ONLY_ENV = {**FORCE_REMOTE_ENV, "GPT_TRADER_PREFLIGHT_WARN_ONLY": "1"}


@pytest.fixture
def mock_healthy_client():
    """Create a mock client with healthy defaults."""
    client = MagicMock()
    client.get_resilience_status.return_value = None
    client.get_accounts.return_value = {"accounts": [{"id": "acc1"}]}
    client.list_products.return_value = [{"product_id": "BTC-USD"}]
    client.get_product.return_value = {
        "base_min_size": "0.001",
        "base_increment": "0.001",
        "quote_increment": "0.01",
    }
    client.get_ticker.return_value = {"price": "50000.00"}
    return client


@pytest.fixture
def checker():
    """Create a production checker."""
    return PreflightCheck(profile="prod")


class TestCheckPretradeDiagnostics:
    """Test pre-trade diagnostics checks."""

    def test_skips_when_remote_checks_bypassed(self) -> None:
        """Should skip and succeed when remote checks are bypassed."""
        chk = PreflightCheck(profile="dev")
        with patch.dict(os.environ, {"COINBASE_PREFLIGHT_SKIP_REMOTE": "1"}, clear=True):
            assert check_pretrade_diagnostics(chk) is True
        assert any("bypassed" in s for s in chk.successes)

    def test_fails_when_client_build_fails(self, checker) -> None:
        """Should fail when CDP client cannot be built."""
        with patch.dict(os.environ, FORCE_REMOTE_ENV, clear=True):
            assert check_pretrade_diagnostics(checker) is False

    def test_warn_only_returns_true_when_client_build_fails(self, checker) -> None:
        """Should return True in warn-only mode when client build fails."""
        with patch.dict(os.environ, WARN_ONLY_ENV, clear=True):
            assert check_pretrade_diagnostics(checker) is True
        assert any("warn-only" in w.lower() for w in checker.warnings)


class TestApiHealthCheck:
    """Test API health check within diagnostics."""

    @pytest.mark.parametrize(
        ("resilience_status", "expected_error"),
        [
            (
                {
                    "metrics": {"error_rate": 0.05},
                    "circuit_breakers": {"orders": {"state": "open"}},
                    "rate_limit_usage": 0.5,
                },
                "circuit breakers open",
            ),
            (
                {"metrics": {"error_rate": 0.25}, "circuit_breakers": {}, "rate_limit_usage": 0.5},
                "error rate",
            ),
            (
                {"metrics": {"error_rate": 0.05}, "circuit_breakers": {}, "rate_limit_usage": 0.95},
                "rate limit usage",
            ),
        ],
        ids=["open_breaker", "high_error_rate", "high_rate_limit"],
    )
    def test_triggers_on_api_degradation(
        self, checker, mock_healthy_client, resilience_status, expected_error
    ) -> None:
        """Should fail when API health is degraded."""
        mock_healthy_client.get_resilience_status.return_value = resilience_status
        with (
            patch.dict(os.environ, FORCE_REMOTE_ENV, clear=True),
            patch.object(
                checker, "_build_cdp_client", return_value=(mock_healthy_client, MagicMock())
            ),
        ):
            result = check_pretrade_diagnostics(checker)
        assert result is False
        assert any(expected_error in e for e in checker.errors)

    def test_skips_without_resilience_status(self, checker) -> None:
        """Should skip API health check when resilience status not available."""
        mock_client = MagicMock(spec=["get_accounts", "list_products", "get_product", "get_ticker"])
        mock_client.get_accounts.return_value = {"accounts": [{"id": "acc1"}]}
        mock_client.list_products.return_value = [{"product_id": "BTC-USD"}]
        mock_client.get_product.return_value = {
            "base_min_size": "0.001",
            "base_increment": "0.001",
            "quote_increment": "0.01",
        }
        mock_client.get_ticker.return_value = {"price": "50000.00"}
        with (
            patch.dict(os.environ, FORCE_REMOTE_ENV, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, MagicMock())),
        ):
            assert check_pretrade_diagnostics(checker) is True


class TestBrokerReadinessCheck:
    """Test broker readiness check within diagnostics."""

    @pytest.mark.parametrize(
        ("accounts", "products"),
        [({"accounts": []}, [{"product_id": "BTC-USD"}]), ({"accounts": [{"id": "acc1"}]}, [])],
        ids=["empty_accounts", "empty_products"],
    )
    def test_fails_on_empty_data(self, checker, mock_healthy_client, accounts, products) -> None:
        """Should fail when accounts or products are empty."""
        mock_healthy_client.get_accounts.return_value = accounts
        mock_healthy_client.list_products.return_value = products
        with (
            patch.dict(os.environ, FORCE_REMOTE_ENV, clear=True),
            patch.object(
                checker, "_build_cdp_client", return_value=(mock_healthy_client, MagicMock())
            ),
        ):
            result = check_pretrade_diagnostics(checker)
        assert result is False
        assert any("Empty response" in e for e in checker.errors)

    def test_fails_on_invalid_product_specs(self, checker, mock_healthy_client) -> None:
        """Should fail when product has invalid specs."""
        mock_healthy_client.get_product.return_value = {
            "base_min_size": "0",
            "base_increment": "0.001",
            "quote_increment": "0.01",
        }
        with (
            patch.dict(os.environ, FORCE_REMOTE_ENV, clear=True),
            patch.object(
                checker, "_build_cdp_client", return_value=(mock_healthy_client, MagicMock())
            ),
        ):
            result = check_pretrade_diagnostics(checker)
        assert result is False
        assert any("invalid" in e.lower() for e in checker.errors)


class TestMarketDataCheck:
    """Test market data freshness check within diagnostics."""

    @pytest.mark.parametrize(
        ("ticker", "expected_error"),
        [(None, "No data"), ({"price": "0"}, "Invalid price")],
        ids=["missing_ticker", "invalid_price"],
    )
    def test_fails_on_bad_ticker(
        self, checker, mock_healthy_client, ticker, expected_error
    ) -> None:
        """Should fail when ticker data is missing or invalid."""
        mock_healthy_client.get_ticker.return_value = ticker
        with (
            patch.dict(os.environ, FORCE_REMOTE_ENV, clear=True),
            patch.object(
                checker, "_build_cdp_client", return_value=(mock_healthy_client, MagicMock())
            ),
        ):
            result = check_pretrade_diagnostics(checker)
        assert result is False
        assert any(expected_error in e for e in checker.errors)

    def test_passes_with_valid_ticker(self, checker, mock_healthy_client) -> None:
        """Should pass when ticker has valid price."""
        with (
            patch.dict(os.environ, FORCE_REMOTE_ENV, clear=True),
            patch.object(
                checker, "_build_cdp_client", return_value=(mock_healthy_client, MagicMock())
            ),
        ):
            result = check_pretrade_diagnostics(checker)
        assert result is True
        assert any("50,000.00" in s for s in checker.successes)


class TestWarnOnlyMode:
    """Test warn-only mode behavior."""

    def test_warn_only_converts_errors_to_warnings(self, checker, mock_healthy_client) -> None:
        """Should convert errors to warnings in warn-only mode."""
        mock_healthy_client.get_resilience_status.return_value = {
            "metrics": {"error_rate": 0.25},
            "circuit_breakers": {},
            "rate_limit_usage": 0.5,
        }
        with (
            patch.dict(os.environ, WARN_ONLY_ENV, clear=True),
            patch.object(
                checker, "_build_cdp_client", return_value=(mock_healthy_client, MagicMock())
            ),
        ):
            result = check_pretrade_diagnostics(checker)
        assert result is True
        assert any("error rate" in w for w in checker.warnings)

    def test_warn_only_returns_true_on_failure(self, checker, mock_healthy_client) -> None:
        """Should return True in warn-only mode even when checks fail."""
        mock_healthy_client.get_accounts.return_value = {"accounts": []}
        mock_healthy_client.list_products.return_value = []
        mock_healthy_client.get_product.return_value = None
        mock_healthy_client.get_ticker.return_value = None
        with (
            patch.dict(os.environ, WARN_ONLY_ENV, clear=True),
            patch.object(
                checker, "_build_cdp_client", return_value=(mock_healthy_client, MagicMock())
            ),
        ):
            assert check_pretrade_diagnostics(checker) is True


class TestSectionHeader:
    """Test section header display."""

    def test_prints_section_header(self, capsys: pytest.CaptureFixture) -> None:
        """Should print section header."""
        chk = PreflightCheck(profile="dev")
        with patch.dict(os.environ, {"COINBASE_PREFLIGHT_SKIP_REMOTE": "1"}, clear=True):
            check_pretrade_diagnostics(chk)
        assert "PRE-TRADE DIAGNOSTICS" in capsys.readouterr().out
