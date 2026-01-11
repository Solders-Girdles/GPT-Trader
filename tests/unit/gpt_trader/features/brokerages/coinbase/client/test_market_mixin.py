from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient


def _make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def test_get_products_appends_filters() -> None:
    client = _make_client()
    urls: list[str] = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"products": []})

    client.set_transport_for_testing(transport)
    client.get_products(product_type="spot", contract_expiry_type="perpetual")

    parsed = urlparse(urls[0])
    params = parse_qs(parsed.query)
    assert params["product_type"] == ["spot"]
    assert params["contract_expiry_type"] == ["perpetual"]


def test_list_products_filters_payload_dict() -> None:
    client = _make_client()
    client.get_products = lambda **_kwargs: {  # type: ignore[method-assign]
        "products": [
            {
                "product_id": "BTC-USD",
                "product_type": "SPOT",
                "contract_expiry_type": "PERPETUAL",
            },
            {
                "product_id": "ETH-USD",
                "product_type": "FUTURE",
                "contract_expiry_type": "EXPIRING",
            },
            "junk",
        ]
    }

    result = client.list_products(product_type="spot", contract_expiry_type="perpetual")

    assert [product["product_id"] for product in result] == ["BTC-USD"]


def test_list_products_filters_payload_list() -> None:
    client = _make_client()
    client.get_products = lambda **_kwargs: [  # type: ignore[method-assign]
        {"product_id": "BTC-USD", "product_type": "SPOT"},
        {"product_id": "ETH-USD", "product_type": "FUTURE"},
        "junk",
    ]

    result = client.list_products(product_type="spot")

    assert [product["product_id"] for product in result] == ["BTC-USD"]


def test_list_products_handles_invalid_shape() -> None:
    client = _make_client()
    client.get_products = lambda **_kwargs: {"products": "oops"}  # type: ignore[method-assign]

    result = client.list_products()

    assert result == []


def test_list_products_filters_contract_expiry() -> None:
    client = _make_client()
    client.get_products = lambda **_kwargs: {  # type: ignore[method-assign]
        "products": [
            {
                "product_id": "BTC-USD",
                "product_type": "FUTURE",
                "contract_expiry_type": "PERPETUAL",
            },
            {
                "product_id": "ETH-USD",
                "product_type": "FUTURE",
                "contract_expiry_type": "EXPIRING",
            },
        ]
    }

    result = client.list_products(contract_expiry_type="expiring")

    assert [product["product_id"] for product in result] == ["ETH-USD"]
