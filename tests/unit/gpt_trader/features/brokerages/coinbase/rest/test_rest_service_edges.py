"""Edge coverage for CoinbaseRestService facade wiring."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from gpt_trader.features.brokerages.coinbase.rest_service import CoinbaseRestService
from gpt_trader.features.brokerages.coinbase.utilities import PositionState, ProductCatalog
from gpt_trader.persistence.event_store import EventStore


def _make_service() -> CoinbaseRestService:
    config = APIConfig(
        api_key="test_key",
        api_secret="test_secret",
        passphrase="test_passphrase",
        base_url="https://api.coinbase.com",
        sandbox=True,
    )
    endpoints = CoinbaseEndpoints(config)
    client = Mock(spec=CoinbaseClient)
    market_data = Mock(spec=MarketDataService)
    event_store = Mock(spec=EventStore)
    product_catalog = ProductCatalog()
    return CoinbaseRestService(
        client=client,
        endpoints=endpoints,
        config=config,
        product_catalog=product_catalog,
        market_data=market_data,
        event_store=event_store,
        bot_config=None,
    )


def test_product_catalog_setter_updates_dependencies() -> None:
    service = _make_service()
    new_catalog = ProductCatalog()

    service.product_catalog = new_catalog

    assert service.product_catalog is new_catalog
    assert service._product_service._product_catalog is new_catalog
    assert service._core.product_catalog is new_catalog


def test_positions_returns_defensive_copy() -> None:
    service = _make_service()
    service._position_store.set(
        "BTC-USD",
        PositionState(
            symbol="BTC-USD",
            side="long",
            quantity=Decimal("1"),
            entry_price=Decimal("100"),
        ),
    )

    positions = service.positions
    positions.pop("BTC-USD")

    assert "BTC-USD" in service.positions


def test_process_fill_for_pnl_uses_shared_store() -> None:
    service = _make_service()

    service.process_fill_for_pnl(
        {
            "product_id": "BTC-USD",
            "size": "1",
            "price": "100",
            "side": "buy",
        }
    )

    assert "BTC-USD" in service.positions


def test_update_position_metrics_delegates() -> None:
    service = _make_service()
    service._core.update_position_metrics = Mock()

    service.update_position_metrics("BTC-USD")

    service._core.update_position_metrics.assert_called_once_with("BTC-USD")
