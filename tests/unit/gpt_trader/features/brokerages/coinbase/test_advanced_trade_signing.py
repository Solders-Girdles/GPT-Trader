import base64
import hashlib
import hmac

from gpt_trader.features.brokerages.coinbase.client import CoinbaseAuth


def test_advanced_trade_hmac_signing_base64(monkeypatch):
    # Use a base64 secret for 'secret'
    secret_b64 = base64.b64encode(b"secret").decode()
    auth = CoinbaseAuth(api_key="key123", api_secret=secret_b64, passphrase=None)

    # Freeze time for deterministic signature
    monkeypatch.setattr("time.time", lambda: 1_700_000_000)

    method = "GET"
    path = "/api/v3/brokerage/products"
    body = None

    headers = auth.sign(method, path, body)

    # Compute expected signature independently
    prehash = (str(1_700_000_000) + method + path).encode()
    key = base64.b64decode(secret_b64)
    expected = base64.b64encode(hmac.new(key, prehash, hashlib.sha256).digest()).decode()

    assert headers["CB-ACCESS-KEY"] == "key123"
    assert headers["CB-ACCESS-TIMESTAMP"] == str(1_700_000_000)
    assert headers["CB-ACCESS-SIGN"] == expected
