"""Tests for API health check within pre-trade diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.checks.diagnostics import check_pretrade_diagnostics


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
        self,
        checker,
        mock_healthy_client,
        resilience_status,
        expected_error,
        force_remote_env: pytest.MonkeyPatch,
    ) -> None:
        """Should fail when API health is degraded."""
        mock_healthy_client.get_resilience_status.return_value = resilience_status
        force_remote_env.setattr(
            checker, "_build_cdp_client", lambda: (mock_healthy_client, MagicMock())
        )

        result = check_pretrade_diagnostics(checker)
        assert result is False
        assert any(expected_error in e for e in checker.errors)

    def test_skips_without_resilience_status(
        self,
        checker,
        force_remote_env: pytest.MonkeyPatch,
    ) -> None:
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
        force_remote_env.setattr(checker, "_build_cdp_client", lambda: (mock_client, MagicMock()))

        assert check_pretrade_diagnostics(checker) is True
