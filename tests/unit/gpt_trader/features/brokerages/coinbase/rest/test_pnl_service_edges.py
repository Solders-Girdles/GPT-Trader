"""Edge coverage for Coinbase PnLService behavior."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.rest.pnl_service import PnLService
from gpt_trader.features.brokerages.coinbase.rest.position_state_store import PositionStateStore
from gpt_trader.features.brokerages.coinbase.utilities import PositionState


def _make_service() -> tuple[PnLService, PositionStateStore, Mock]:
    store = PositionStateStore()
    market_data = Mock(spec=MarketDataService)
    return PnLService(position_store=store, market_data=market_data), store, market_data


def test_get_position_pnl_uses_entry_price_when_mark_missing() -> None:
    service, store, market_data = _make_service()
    store.set(
        "BTC-USD",
        PositionState(
            symbol="BTC-USD",
            side="long",
            quantity=Decimal("2"),
            entry_price=Decimal("100"),
        ),
    )
    market_data.get_mark.return_value = None

    pnl = service.get_position_pnl("BTC-USD")

    assert pnl["mark"] == Decimal("100")
    assert pnl["unrealized_pnl"] == Decimal("0")


def test_get_position_pnl_short_position_flips_sign() -> None:
    service, store, market_data = _make_service()
    store.set(
        "BTC-USD",
        PositionState(
            symbol="BTC-USD",
            side="short",
            quantity=Decimal("1"),
            entry_price=Decimal("100"),
        ),
    )
    market_data.get_mark.return_value = Decimal("90")

    pnl = service.get_position_pnl("BTC-USD")

    assert pnl["unrealized_pnl"] == Decimal("10")


def test_get_portfolio_pnl_empty_store_returns_zeros() -> None:
    service, _store, _market_data = _make_service()

    result = service.get_portfolio_pnl()

    assert result["total_realized_pnl"] == Decimal("0")
    assert result["total_unrealized_pnl"] == Decimal("0")
    assert result["total_pnl"] == Decimal("0")
    assert result["positions"] == []
