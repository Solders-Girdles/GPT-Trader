import asyncio
import json
from pathlib import Path
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

import pytest
from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig, Profile
from bot_v2.features.brokerages.core.interfaces import Order, OrderSide, OrderStatus, OrderType, TimeInForce

# Add src to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary data directory for testing."""
    return tmp_path

@pytest.mark.asyncio
async def test_startup_reconciliation(data_dir):
    """
    Verify that a stale 'OPEN' order in the local store is reconciled
    to its true 'CANCELLED' status from the exchange on startup.
    """
    # 1. Seed the order store with a stale order
    stale_order_data = {
        "order_id": "stale_order_123",
        "client_id": "client_stale_123",
        "symbol": "BTC-PERP",
        "side": "BUY",
        "order_type": "LIMIT",
        "qty": "0.1",
        "price": "50000",
        "status": "OPEN", # Stale status
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z"
    }
    orders_file = data_dir / "perps_bot/dev/orders.jsonl"
    orders_file.parent.mkdir(parents=True, exist_ok=True)
    with orders_file.open('w') as f:
        f.write(json.dumps(stale_order_data) + '\n')

    # 2. Configure the bot and mock the broker
    config = BotConfig.from_profile(profile='dev', dry_run=True)
    
    # Set the EVENT_STORE_ROOT to our temporary directory
    os.environ['EVENT_STORE_ROOT'] = str(data_dir)
    
    bot = PerpsBot(config)
    
    # Mock the real order status from the "exchange"
    final_order_status = Order(
        id="stale_order_123",
        client_id="client_stale_123",
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        qty=Decimal("0.1"),
        price=Decimal("50000"),
        tif=TimeInForce.GTC,
        status=OrderStatus.CANCELLED, # True status
        filled_qty=Decimal("0"),
        avg_fill_price=None,
        submitted_at=datetime.fromisoformat("2023-01-01T12:00:00+00:00"),
        updated_at=datetime.fromisoformat("2023-01-01T12:01:00+00:00"),
        stop_price=None
    )
    
    # Mock the broker's methods
    bot.broker = MagicMock()
    bot.broker.list_orders = MagicMock(return_value=[]) # No open orders on exchange
    bot.broker.get_order = MagicMock(return_value=final_order_status)

    # 3. Run the reconciliation process
    await bot._reconcile_state_on_startup()

    # 4. Assertions
    # Verify that the broker was queried for the stale order
    bot.broker.get_order.assert_called_once_with("stale_order_123")

    # Verify that the local order store was updated
    reconciled_order = bot.orders_store.get_by_id("stale_order_123")
    assert reconciled_order is not None
    assert reconciled_order.status == OrderStatus.CANCELLED.value

    # Clean up environment variable
    del os.environ['EVENT_STORE_ROOT']

