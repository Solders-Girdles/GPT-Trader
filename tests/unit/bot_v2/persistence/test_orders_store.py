import unittest
from datetime import datetime, timezone
from decimal import Decimal
from bot_v2.persistence.orders_store import StoredOrder
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


class TestOrdersStore(unittest.TestCase):

    def test_from_order_timestamp_conversion(self):
        """
        Verify that StoredOrder correctly converts datetime objects
        for submitted_at and updated_at into ISO 8601 strings.
        """
        now = datetime.now(timezone.utc)
        order = Order(
            id="test_order_123",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            status=OrderStatus.PENDING,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            tif=TimeInForce.GTC,
            submitted_at=now,
            updated_at=now,
            client_id="test_client_id",
            filled_quantity=Decimal("0.0"),
            avg_fill_price=None,
            stop_price=None,
        )

        stored_order = StoredOrder.from_order(order)

        # Assert that the datetime objects were converted to ISO 8601 strings
        self.assertEqual(stored_order.created_at, now.isoformat())
        self.assertEqual(stored_order.updated_at, now.isoformat())
        self.assertEqual(stored_order.order_id, "test_order_123")


if __name__ == "__main__":
    unittest.main()
