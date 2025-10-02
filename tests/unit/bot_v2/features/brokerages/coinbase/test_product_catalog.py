from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
from bot_v2.features.brokerages.core.interfaces import MarketType, NotFoundError


def product_payload(symbol: str, *, contract_type: str | None = None) -> dict[str, object]:
    base, quote = symbol.split("-", 1)
    payload: dict[str, object] = {
        "product_id": symbol,
        "base_currency": base,
        "quote_currency": quote,
        "base_min_size": "0.01",
        "base_increment": "0.01",
        "quote_increment": "0.01",
        "min_market_funds": "10",
    }
    if contract_type:
        payload["contract_type"] = contract_type
        payload["contract_size"] = "1"
        payload["funding_rate"] = "0.0001"
        payload["next_funding_time"] = datetime.utcnow().isoformat()
    return payload


class DummyClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls = 0

    def get_products(self) -> dict[str, object]:
        self.calls += 1
        return {"products": self.payloads}


def test_product_catalog_refresh_and_get() -> None:
    client = DummyClient([product_payload("BTC-USD")])
    catalog = ProductCatalog(ttl_seconds=60)
    catalog.refresh(client)

    retrieved = catalog.get(client, "BTC-USD")
    assert retrieved.symbol == "BTC-USD"
    assert retrieved.market_type is MarketType.SPOT
    assert client.calls == 1


def test_product_catalog_expires_and_refreshes() -> None:
    client = DummyClient([product_payload("ETH-USD")])
    catalog = ProductCatalog(ttl_seconds=1)
    catalog.refresh(client)

    catalog._last_refresh = datetime.utcnow() - timedelta(seconds=120)
    catalog.get(client, "ETH-USD")
    assert client.calls >= 2


def test_product_catalog_missing_product_raises() -> None:
    catalog = ProductCatalog(ttl_seconds=60)
    client = DummyClient([])
    with pytest.raises(NotFoundError):
        catalog.get(client, "MISSING")


def test_product_catalog_get_funding() -> None:
    perp_payload = product_payload("SOL-USD-PERP", contract_type="perpetual")
    spot_payload = product_payload("SOL-USD")
    client = DummyClient([perp_payload, spot_payload])

    catalog = ProductCatalog(ttl_seconds=60)
    catalog.refresh(client)

    rate, next_time = catalog.get_funding(client, "SOL-USD-PERP")
    assert rate is not None and next_time is not None

    no_rate, no_time = catalog.get_funding(client, "SOL-USD")
    assert no_rate is None and no_time is None
