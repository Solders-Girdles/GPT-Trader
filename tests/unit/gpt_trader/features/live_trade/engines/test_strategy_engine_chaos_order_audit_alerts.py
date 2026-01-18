"""Chaos tests for TradingEngine order audit alerts."""

from __future__ import annotations

import time

import pytest

pytest_plugins = ["strategy_engine_chaos_fixtures"]


class TestOrderAuditAlerts:
    @pytest.mark.asyncio
    async def test_unfilled_order_alert_emitted_once(self, engine) -> None:
        engine.context.risk_manager.config.unfilled_order_alert_seconds = 1
        engine.context.broker.list_orders.return_value = {
            "orders": [
                {
                    "order_id": "order-1",
                    "product_id": "BTC-USD",
                    "side": "BUY",
                    "status": "OPEN",
                    "created_time": time.time() - 10,
                }
            ]
        }

        await engine._audit_orders()
        events = engine._event_store.list_events()
        alert_events = [e for e in events if e.get("type") == "unfilled_order_alert"]
        assert len(alert_events) == 1

        await engine._audit_orders()
        events = engine._event_store.list_events()
        alert_events = [e for e in events if e.get("type") == "unfilled_order_alert"]
        assert len(alert_events) == 1
