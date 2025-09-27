"""Unit tests for CoinbaseBrokerage auth selection logic."""
import unittest
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
from bot_v2.features.brokerages.coinbase.client import CoinbaseAuth

class TestAuthSelection(unittest.TestCase):
    def test_jwt_auth_selection(self):
        """Verify that CDPAuthV2 is used when CDP keys are provided."""
        config = APIConfig(
            api_key="",
            api_secret="",
            passphrase="",
            base_url="",
            cdp_api_key="test_key",
            cdp_private_key="test_secret",
            api_mode='advanced',
            sandbox=True
        )
        broker = CoinbaseBrokerage(config)
        self.assertIsInstance(broker.client.auth, CDPAuthV2)

    def test_hmac_auth_fallback(self):
        """Verify that CoinbaseAuth (HMAC) is used as a fallback."""
        config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test",
            base_url="",
            api_mode='exchange',
            sandbox=True
        )
        broker = CoinbaseBrokerage(config)
        self.assertIsInstance(broker.client.auth, CoinbaseAuth)

if __name__ == '__main__':
    unittest.main()
