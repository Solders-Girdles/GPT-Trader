"""Tests for broker readiness check within pre-trade diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.checks.diagnostics import check_pretrade_diagnostics


class TestBrokerReadinessCheck:
    """Test broker readiness check within diagnostics."""

    @pytest.mark.parametrize(
        ("accounts", "products"),
        [({"accounts": []}, [{"product_id": "BTC-USD"}]), ({"accounts": [{"id": "acc1"}]}, [])],
        ids=["empty_accounts", "empty_products"],
    )
    def test_fails_on_empty_data(
        self,
        checker,
        mock_healthy_client,
        accounts,
        products,
        force_remote_env: pytest.MonkeyPatch,
    ) -> None:
        """Should fail when accounts or products are empty."""
        mock_healthy_client.get_accounts.return_value = accounts
        mock_healthy_client.list_products.return_value = products
        force_remote_env.setattr(
            checker, "_build_cdp_client", lambda: (mock_healthy_client, MagicMock())
        )

        result = check_pretrade_diagnostics(checker)
        assert result is False
        assert any("Empty response" in e for e in checker.errors)

    def test_fails_on_invalid_product_specs(
        self,
        checker,
        mock_healthy_client,
        force_remote_env: pytest.MonkeyPatch,
    ) -> None:
        """Should fail when product has invalid specs."""
        mock_healthy_client.get_product.return_value = {
            "base_min_size": "0",
            "base_increment": "0.001",
            "quote_increment": "0.01",
        }
        force_remote_env.setattr(
            checker, "_build_cdp_client", lambda: (mock_healthy_client, MagicMock())
        )

        result = check_pretrade_diagnostics(checker)
        assert result is False
        assert any("invalid" in e.lower() for e in checker.errors)
