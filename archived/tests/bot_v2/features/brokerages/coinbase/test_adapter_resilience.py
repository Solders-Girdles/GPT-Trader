
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket


@pytest.mark.asyncio
class TestCoinbaseResilience(unittest.TestCase):

    def setUp(self):
        """Set up a CoinbaseBrokerage instance for resilience testing."""
        self.mock_config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_pass",
            base_url="https://api.sandbox.coinbase.com",
            api_mode="sandbox",
            sandbox=True,
            enable_derivatives=True
        )
        self.brokerage = CoinbaseBrokerage(self.mock_config)

    async def test_ws_gap_handling_and_resync(self):
        """
        Verify that a WebSocket sequence gap triggers an order book snapshot
        request to ensure state is resynchronized.
        """
        # 1. Mock the WebSocket client to simulate a sequence gap
        mock_ws = MagicMock(spec=CoinbaseWebSocket)
        
        # Simulate a stream with a gap
        async def mock_stream_messages():
            yield {"type": "ticker", "sequence": 1, "product_id": "BTC-USD", "price": "50000"}
            yield {"type": "ticker", "sequence": 3, "product_id": "BTC-USD", "price": "50001"} # Gap from 1 to 3

        mock_ws.stream_messages = mock_stream_messages
        
        # 2. Override the internal WS factory to inject our mock
        self.brokerage._ws_factory_override = lambda: mock_ws

        # 3. Mock the REST client's get_product_book method
        self.brokerage.client.get_product_book = AsyncMock(return_value={
            "bids": [["49999", "1"]],
            "asks": [["50002", "1"]]
        })

        # 4. Run the stream_orderbook method and collect the output
        output_messages = []
        stream_generator = self.brokerage.stream_orderbook(symbols=["BTC-USD"])
        
        # Consume the generator
        for message in stream_generator:
            output_messages.append(message)
            if len(output_messages) >= 3: # Limit to avoid infinite loop in test
                break

        # 5. Assertions
        # Check that get_product_book was called after the gap
        self.brokerage.client.get_product_book.assert_called_once_with(
            product_id="BTC-USD", level=2
        )

        # Check that a snapshot message was yielded
        snapshot_message = next((m for m in output_messages if m.get("type") == "snapshot"), None)
        self.assertIsNotNone(snapshot_message)
        self.assertEqual(snapshot_message["product_id"], "BTC-USD")
        self.assertIn("book", snapshot_message)
        self.assertEqual(snapshot_message["book"]["bids"], [["49999", "1"]])


if __name__ == '__main__':
    unittest.main()

