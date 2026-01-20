"""Tests for market data and warn-only mode in pre-trade diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.checks.diagnostics import check_pretrade_diagnostics


class TestMarketDataCheck:
    """Test market data freshness check within diagnostics."""

    @pytest.mark.parametrize(
        ("ticker", "expected_error"),
        [(None, "No data"), ({"price": "0"}, "Invalid price")],
        ids=["missing_ticker", "invalid_price"],
    )
    def test_fails_on_bad_ticker(
        self,
        checker,
        mock_healthy_client,
        ticker,
        expected_error,
        force_remote_env: pytest.MonkeyPatch,
    ) -> None:
        """Should fail when ticker data is missing or invalid."""
        mock_healthy_client.get_ticker.return_value = ticker
        force_remote_env.setattr(
            checker, "_build_cdp_client", lambda: (mock_healthy_client, MagicMock())
        )

        result = check_pretrade_diagnostics(checker)
        assert result is False
        assert any(expected_error in e for e in checker.errors)

    def test_passes_with_valid_ticker(
        self,
        checker,
        mock_healthy_client,
        force_remote_env: pytest.MonkeyPatch,
    ) -> None:
        """Should pass when ticker has valid price."""
        force_remote_env.setattr(
            checker, "_build_cdp_client", lambda: (mock_healthy_client, MagicMock())
        )

        result = check_pretrade_diagnostics(checker)
        assert result is True
        assert any("50,000.00" in s for s in checker.successes)


class TestWarnOnlyMode:
    """Test warn-only mode behavior."""

    def test_warn_only_converts_errors_to_warnings(
        self,
        checker,
        mock_healthy_client,
        warn_only_env: pytest.MonkeyPatch,
    ) -> None:
        """Should convert errors to warnings in warn-only mode."""
        mock_healthy_client.get_resilience_status.return_value = {
            "metrics": {"error_rate": 0.25},
            "circuit_breakers": {},
            "rate_limit_usage": 0.5,
        }
        warn_only_env.setattr(
            checker, "_build_cdp_client", lambda: (mock_healthy_client, MagicMock())
        )

        result = check_pretrade_diagnostics(checker)
        assert result is True
        assert any("error rate" in w for w in checker.warnings)

    def test_warn_only_returns_true_on_failure(
        self,
        checker,
        mock_healthy_client,
        warn_only_env: pytest.MonkeyPatch,
    ) -> None:
        """Should return True in warn-only mode even when checks fail."""
        mock_healthy_client.get_accounts.return_value = {"accounts": []}
        mock_healthy_client.list_products.return_value = []
        mock_healthy_client.get_product.return_value = None
        mock_healthy_client.get_ticker.return_value = None
        warn_only_env.setattr(
            checker, "_build_cdp_client", lambda: (mock_healthy_client, MagicMock())
        )

        assert check_pretrade_diagnostics(checker) is True
