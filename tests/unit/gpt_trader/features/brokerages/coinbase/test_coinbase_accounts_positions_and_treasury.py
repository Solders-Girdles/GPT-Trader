"""Coinbase positions mapping and treasury delegation tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.client as client_mod
from gpt_trader.core import InvalidRequestError
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from tests.unit.gpt_trader.features.brokerages.coinbase.helpers import (
    CoinbaseBrokerage,
    make_adapter,
)

pytestmark = pytest.mark.endpoints


class TestCoinbasePositionsMapping:
    def test_list_positions_maps_from_cfm(self, monkeypatch) -> None:
        broker = CoinbaseBrokerage(
            APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api")
        )
        broker.connect()

        def fake_positions(self):
            return {
                "positions": [
                    {
                        "product_id": "BTC-USD-PERP",
                        "size": "1.5",
                        "entry_price": "100",
                        "mark_price": "110",
                        "unrealized_pnl": "15",
                        "realized_pnl": "2",
                        "leverage": 5,
                        "side": "long",
                    },
                    {
                        "product_id": "ETH-USD-PERP",
                        "contracts": "2",
                        "avg_entry_price": "2000",
                        "index_price": "1950",
                        "unrealizedPnl": "-100",
                        "realizedPnl": "5",
                        "leverage": 3,
                        "side": "short",
                    },
                ]
            }

        monkeypatch.setattr(client_mod.CoinbaseClient, "list_positions", fake_positions)
        positions = broker.list_positions()
        assert len(positions) == 2
        assert positions[0].symbol == "BTC-USD-PERP" and positions[0].quantity == Decimal("1.5")
        assert positions[1].symbol == "ETH-USD-PERP" and positions[1].quantity == Decimal("2")


class TestCoinbaseTreasuryDelegation:
    def test_adapter_cfm_telemetry_delegates_to_rest_service(self) -> None:
        adapter = make_adapter()
        adapter.rest_service.get_cfm_balance_summary = Mock(
            return_value={"portfolio_value": Decimal("100.5")}
        )
        adapter.rest_service.list_cfm_sweeps = Mock(return_value=[{"amount": Decimal("5.25")}])
        adapter.rest_service.get_cfm_sweeps_schedule = Mock(return_value={"windows": ["08:00Z"]})
        adapter.rest_service.get_cfm_margin_window = Mock(return_value={"margin_window": "TEST"})
        adapter.rest_service.update_cfm_margin_window = Mock(return_value={"status": "ok"})

        summary = adapter.get_cfm_balance_summary()
        sweeps = adapter.list_cfm_sweeps()
        schedule = adapter.get_cfm_sweeps_schedule()
        margin = adapter.get_cfm_margin_window()
        update = adapter.update_cfm_margin_window("TEST")

        assert summary["portfolio_value"] == Decimal("100.5")
        assert sweeps[0]["amount"] == Decimal("5.25")
        assert schedule["windows"] == ["08:00Z"]
        assert margin["margin_window"] == "TEST"
        assert update["status"] == "ok"
        adapter.rest_service.get_cfm_balance_summary.assert_called_once()
        adapter.rest_service.list_cfm_sweeps.assert_called_once()
        adapter.rest_service.get_cfm_sweeps_schedule.assert_called_once()
        adapter.rest_service.get_cfm_margin_window.assert_called_once()
        adapter.rest_service.update_cfm_margin_window.assert_called_once_with(
            "TEST",
            effective_time=None,
            extra_payload=None,
        )

    def test_adapter_cfm_telemetry_respects_derivatives_gating(self) -> None:
        adapter = make_adapter(enable_derivatives=False)
        adapter.client.cfm_balance_summary = Mock(side_effect=AssertionError("should not call"))
        adapter.client.cfm_sweeps = Mock(side_effect=AssertionError("should not call"))
        adapter.client.cfm_sweeps_schedule = Mock(side_effect=AssertionError("should not call"))
        adapter.client.cfm_intraday_current_margin_window = Mock(
            side_effect=AssertionError("should not call")
        )

        summary = adapter.get_cfm_balance_summary()
        sweeps = adapter.list_cfm_sweeps()
        schedule = adapter.get_cfm_sweeps_schedule()
        margin = adapter.get_cfm_margin_window()

        assert summary == {}
        assert sweeps == []
        assert schedule == {}
        assert margin == {}

        with pytest.raises(InvalidRequestError):
            adapter.update_cfm_margin_window("TEST")

    def test_adapter_intx_methods_delegate_to_rest_service(self) -> None:
        adapter = make_adapter()
        adapter.rest_service.intx_allocate = Mock(return_value={"status": "accepted"})
        adapter.rest_service.get_intx_balances = Mock(return_value=[{"asset": "USD"}])
        adapter.rest_service.get_intx_portfolio = Mock(return_value={"uuid": "pf-1"})
        adapter.rest_service.list_intx_positions = Mock(return_value=[{"symbol": "BTC"}])
        adapter.rest_service.get_intx_position = Mock(return_value={"symbol": "ETH"})
        adapter.rest_service.get_intx_multi_asset_collateral = Mock(
            return_value={"ratio": Decimal("0.5")}
        )

        allocation = adapter.intx_allocate({"amount": "10"})
        balances = adapter.get_intx_balances("pf-1")
        portfolio = adapter.get_intx_portfolio("pf-1")
        positions = adapter.list_intx_positions("pf-1")
        position = adapter.get_intx_position("pf-1", "ETH")
        collateral = adapter.get_intx_multi_asset_collateral()

        assert allocation["status"] == "accepted"
        assert balances[0]["asset"] == "USD"
        assert portfolio["uuid"] == "pf-1"
        assert positions[0]["symbol"] == "BTC"
        assert position["symbol"] == "ETH"
        assert collateral["ratio"] == Decimal("0.5")
        adapter.rest_service.intx_allocate.assert_called_once()
        adapter.rest_service.get_intx_balances.assert_called_once_with("pf-1")
        adapter.rest_service.get_intx_portfolio.assert_called_once_with("pf-1")
        adapter.rest_service.list_intx_positions.assert_called_once_with("pf-1")
        adapter.rest_service.get_intx_position.assert_called_once_with("pf-1", "ETH")
        adapter.rest_service.get_intx_multi_asset_collateral.assert_called_once()

    def test_adapter_intx_methods_respect_intx_support(self) -> None:
        adapter = make_adapter(api_mode="exchange")

        balances = adapter.get_intx_balances("pf-1")
        portfolio = adapter.get_intx_portfolio("pf-1")
        positions = adapter.list_intx_positions("pf-1")
        position = adapter.get_intx_position("pf-1", "BTC")
        collateral = adapter.get_intx_multi_asset_collateral()

        assert balances == []
        assert portfolio == {}
        assert positions == []
        assert position == {}
        assert collateral == {}

        with pytest.raises(InvalidRequestError):
            adapter.intx_allocate({"amount": "5"})
