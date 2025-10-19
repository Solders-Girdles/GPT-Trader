from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bot_builder import create_perps_bot


def test_reconcile_updates_stale_local_open_order():
    # Create config without dry_run so reconciliation actually runs
    config = BotConfig.from_profile("dev")
    config.dry_run = False  # Ensure reconciliation runs
    bot = create_perps_bot(config)

    # Seed a local OPEN order not present at broker
    stale = Order(
        id="stale_order_123",
        client_id="stale_order_123",
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Decimal("1"),
        price=Decimal("100"),
        stop_price=None,
        tif=TimeInForce.GTC,
        status=OrderStatus.SUBMITTED,
        filled_quantity=Decimal("0"),
        avg_fill_price=None,
        submitted_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    bot.orders_store.upsert(stale)

    # Broker does not know this order; reconciliation should fetch final status (mock returns CANCELLED)
    asyncio.run(bot.runtime_coordinator.reconcile_state_on_startup())

    record = bot.orders_store.get_by_id("stale_order_123")
    assert record is not None
    assert record.status == OrderStatus.CANCELLED.value
