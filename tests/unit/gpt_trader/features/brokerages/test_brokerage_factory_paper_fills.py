from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.factory as brokerage_factory
from gpt_trader.app.config import BotConfig


def _make_config() -> BotConfig:
    config = BotConfig()
    config.mock_broker = False
    config.dry_run = False
    config.paper_fills = True
    config.risk.dry_run_equity_usd = Decimal("12345")
    return config


def test_create_brokerage_paper_fills_selects_hybrid_paper_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config()

    creds = SimpleNamespace(
        key_name="test_key",
        private_key="test_private_key",
        warnings=[],
        source="test",
        masked_key_name="test_key",
    )
    monkeypatch.setattr(brokerage_factory, "resolve_coinbase_credentials", lambda: creds)

    auth = object()
    monkeypatch.setattr(brokerage_factory, "SimpleAuth", MagicMock(return_value=auth))

    market_client = object()
    monkeypatch.setattr(brokerage_factory, "CoinbaseClient", MagicMock(return_value=market_client))

    paper_broker = object()
    paper_ctor = MagicMock(return_value=paper_broker)
    monkeypatch.setattr(brokerage_factory, "HybridPaperBroker", paper_ctor)

    event_store = MagicMock()
    market_data = MagicMock()
    product_catalog = MagicMock()

    broker, returned_event_store, returned_market_data, returned_product_catalog = (
        brokerage_factory.create_brokerage(
            event_store=event_store,
            market_data=market_data,
            product_catalog=product_catalog,
            config=config,
        )
    )

    assert broker is paper_broker
    assert returned_event_store is event_store
    assert returned_market_data is market_data
    assert returned_product_catalog is product_catalog

    paper_ctor.assert_called_once()
    kwargs = paper_ctor.call_args.kwargs
    assert kwargs["client"] is market_client
    assert kwargs["initial_equity"] == Decimal("12345")


def test_create_brokerage_paper_fills_mutually_exclusive_with_mock_broker() -> None:
    config = _make_config()
    config.mock_broker = True

    with pytest.raises(ValueError, match="paper_fills.*mock_broker"):
        brokerage_factory.create_brokerage(
            event_store=MagicMock(),
            market_data=MagicMock(),
            product_catalog=MagicMock(),
            config=config,
        )


def test_create_brokerage_paper_fills_mutually_exclusive_with_dry_run() -> None:
    config = _make_config()
    config.dry_run = True

    with pytest.raises(ValueError, match="paper_fills.*dry_run"):
        brokerage_factory.create_brokerage(
            event_store=MagicMock(),
            market_data=MagicMock(),
            product_catalog=MagicMock(),
            config=config,
        )
