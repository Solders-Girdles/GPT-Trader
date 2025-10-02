from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.coinbase import market_data_service


class FrozenDateTime(datetime):
    now_value: datetime = datetime(2025, 1, 15, 12, 0)

    @classmethod
    def utcnow(cls) -> datetime:
        return cls.now_value


@pytest.fixture(autouse=True)
def _patch_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(market_data_service, "datetime", FrozenDateTime)


def test_initialise_symbols_and_update_ticker() -> None:
    service = market_data_service.MarketDataService()
    service.initialise_symbols(["BTC-USD"])

    service.update_ticker(
        symbol="BTC-USD",
        bid=Decimal("100"),
        ask=Decimal("101"),
        last=Decimal("102"),
        timestamp=datetime(2025, 1, 15, 12, 0),
    )

    quote = service.get_cached_quote("BTC-USD")
    assert quote is not None
    assert quote["mid"] == Decimal("100.5")
    assert quote["spread_bps"] == pytest.approx(100.0)
    assert quote["last"] == Decimal("102")


def test_record_trade_and_snapshot() -> None:
    service = market_data_service.MarketDataService()
    service.initialise_symbols(["ETH-USD"])

    now = datetime(2025, 1, 15, 12, 0)
    for idx in range(3):
        service.record_trade("ETH-USD", Decimal("0.5"), now + timedelta(seconds=idx))

    snapshot = service.get_snapshot("ETH-USD")
    assert snapshot["vol_1m"] > 0
    assert snapshot["vol_5m"] > 0


def test_update_depth_and_mark_cache() -> None:
    service = market_data_service.MarketDataService()
    service.initialise_symbols(["ETH-USD"])

    service.update_depth(
        "ETH-USD",
        [
            ("buy", "100", "2"),
            ("sell", "101", "3"),
            ("buy", "99", "1"),
        ],
    )

    snapshot = service.get_snapshot("ETH-USD")
    assert snapshot["depth_l1"] == Decimal("503")  # Sum of top-of-book bid/ask notionals
    assert snapshot["depth_l10"] == Decimal("602")

    service.set_mark("ETH-USD", Decimal("150"))
    assert service.get_mark("ETH-USD") == Decimal("150"), "Mark cache should round-trip"


def test_is_stale_and_get_cached_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    service = market_data_service.MarketDataService()
    service.initialise_symbols(["SOL-USD"])
    service.update_ticker(
        "SOL-USD",
        bid=Decimal("10"),
        ask=Decimal("11"),
        last=Decimal("10.5"),
        timestamp=datetime(2025, 1, 15, 12, 0),
    )

    assert not service.is_stale("SOL-USD", threshold_seconds=60)

    FrozenDateTime.now_value = datetime(2025, 1, 15, 12, 2)
    assert service.is_stale("SOL-USD", threshold_seconds=30)

    FrozenDateTime.now_value = datetime(2025, 1, 15, 12, 0)
    assert service.get_cached_quote("SOL-USD") is not None
    assert service.get_cached_quote("MISSING") is None
