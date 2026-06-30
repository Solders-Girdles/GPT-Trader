"""Coinbase account manager snapshot tests."""

from __future__ import annotations

from typing import Any

import pytest

from gpt_trader.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from tests.unit.gpt_trader.features.brokerages.coinbase.helpers import StubBroker, StubEventStore

pytestmark = pytest.mark.endpoints


class TestCoinbaseAccountManagerSnapshot:
    @staticmethod
    def _freshness(snapshot: dict[str, Any], section: str) -> dict[str, Any]:
        return snapshot["freshness"][section]

    def test_snapshot_collects_all_sections(self) -> None:
        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        snapshot = manager.snapshot()

        assert snapshot["key_permissions"]["can_trade"] is True
        assert snapshot["fee_schedule"]["tier"] == "Advanced"
        assert snapshot["limits"]["max_order"] == "100000"
        assert snapshot["transaction_summary"]["total_volume"] == "12345"
        assert snapshot["payment_methods"][0]["id"] == "pm-1"
        assert snapshot["portfolios"][0]["uuid"] == "pf-1"
        assert snapshot["cfm_balance_summary"]["portfolio_value"] == "250.50"
        assert snapshot["cfm_sweeps"][0]["sweep_id"] == "sweep-1"
        assert snapshot["cfm_sweeps_schedule"]["windows"][0] == "00:00Z"
        assert snapshot["cfm_margin_window"]["margin_window"] == "INTRADAY_STANDARD"
        assert any(
            metric[1].get("event_type") == "account_manager_snapshot" for metric in store.metrics
        )
        assert self._freshness(snapshot, "key_permissions")["status"] == "fresh"
        assert self._freshness(snapshot, "cfm_balance_summary")["status"] == "fresh"

    def test_snapshot_records_error_payloads(self) -> None:
        class FailingFeeScheduleBroker(StubBroker):
            def get_fee_schedule(self):
                raise RuntimeError("boom")

        broker = FailingFeeScheduleBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        snapshot = manager.snapshot()

        assert snapshot["key_permissions"]["can_trade"] is True
        assert snapshot["fee_schedule"]["error"]["message"] == "boom"
        assert snapshot["fee_schedule"]["error"]["type"] == "RuntimeError"
        assert snapshot["portfolios"][0]["uuid"] == "pf-1"
        assert any(
            metric[1].get("event_type") == "account_manager_snapshot" for metric in store.metrics
        )
        metadata = self._freshness(snapshot, "fee_schedule")
        assert metadata["status"] == "error"
        assert metadata["error_code"] == "RuntimeError"
        assert "fetched_at" in metadata

    def test_snapshot_handles_missing_optional_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(StubBroker, "get_cfm_balance_summary")

        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        snapshot = manager.snapshot()

        error_payload = snapshot["cfm_balance_summary"]["error"]
        assert error_payload["type"] == "AttributeError"
        assert "get_cfm_balance_summary" in error_payload["message"]
        assert snapshot["cfm_sweeps"][0]["sweep_id"] == "sweep-1"
        metadata = self._freshness(snapshot, "cfm_balance_summary")
        assert metadata["status"] == "error"
        assert metadata["error_code"] == "AttributeError"

    def test_convert_commits_when_requested(self) -> None:
        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        result = manager.convert({"from": "USD", "to": "USDC", "amount": "100"}, commit=True)

        assert result["trade_id"] == "trade-1"
        assert any(call[0] == "convert_quote" for call in broker.calls)
        assert any(call[0] == "commit_trade" for call in broker.calls)
        assert any(metric[1].get("event_type") == "convert_commit" for metric in store.metrics)

    def test_move_funds_delegates_to_broker(self) -> None:
        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        payload = {"from_portfolio": "pf-1", "to_portfolio": "pf-2", "amount": "5"}
        result = manager.move_funds(payload)

        assert result["status"] == "ok"
        assert ("move_funds", payload) in broker.calls
        assert any(metric[1].get("event_type") == "portfolio_move" for metric in store.metrics)
