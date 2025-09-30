from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.coinbase.market_data_features import RollingWindow
from bot_v2.features.brokerages.coinbase.utilities import MarkCache


class MarketDataService:
    """Maintains cached market data, rolling metrics, and mark prices."""

    def __init__(self) -> None:
        self._market_data: dict[str, dict[str, Any]] = {}
        self._rolling_windows: dict[str, dict[str, RollingWindow]] = {}
        self._mark_cache = MarkCache()

    @property
    def mark_cache(self) -> MarkCache:
        return self._mark_cache

    def initialise_symbols(self, symbols: Iterable[str]) -> None:
        for symbol in symbols:
            symbol = str(symbol)
            if symbol not in self._market_data:
                self._market_data[symbol] = {
                    "mid": Decimal("0"),
                    "spread_bps": 0.0,
                    "depth_l1": Decimal("0"),
                    "depth_l10": Decimal("0"),
                    "last_update": None,
                }
            if symbol not in self._rolling_windows:
                self._rolling_windows[symbol] = {
                    "vol_1m": RollingWindow(60),
                    "vol_5m": RollingWindow(300),
                }

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._market_data

    def update_ticker(
        self,
        symbol: str,
        bid: Decimal | None,
        ask: Decimal | None,
        last: Decimal | None,
        timestamp: datetime,
    ) -> None:
        data = self._market_data.setdefault(symbol, {})
        if bid is not None and ask is not None:
            data["bid"] = bid
            data["ask"] = ask
            data["mid"] = (bid + ask) / 2
            if bid > 0:
                data["spread_bps"] = float((ask - bid) / bid * 10000)
        if last is not None:
            data["last"] = last
        data["last_update"] = timestamp

    def record_trade(self, symbol: str, size: Decimal, timestamp: datetime) -> None:
        windows = self._rolling_windows.get(symbol)
        if not windows:
            return
        for window in windows.values():
            window.add(float(size), timestamp)

    def update_depth(self, symbol: str, changes: Iterable[Iterable[str]]) -> None:
        bid_depth_usd = Decimal("0")
        ask_depth_usd = Decimal("0")
        bid_depth_l1_usd = Decimal("0")
        ask_depth_l1_usd = Decimal("0")

        bid_count = 0
        ask_count = 0

        for change in list(changes)[:10]:
            if len(change) < 3:
                continue
            side, price_str, size_str = change
            price = Decimal(price_str) if price_str else Decimal("0")
            size = Decimal(size_str) if size_str and size_str != "0" else Decimal("0")
            notional = price * size
            if side == "buy":
                bid_depth_usd += notional
                if bid_count == 0:
                    bid_depth_l1_usd = notional
                bid_count += 1
            elif side == "sell":
                ask_depth_usd += notional
                if ask_count == 0:
                    ask_depth_l1_usd = notional
                ask_count += 1

        data = self._market_data.setdefault(symbol, {})
        data["depth_l1"] = bid_depth_l1_usd + ask_depth_l1_usd
        data["depth_l10"] = bid_depth_usd + ask_depth_usd

    def is_stale(self, symbol: str, threshold_seconds: int = 10) -> bool:
        data = self._market_data.get(symbol)
        if not data:
            return True
        last_update = data.get("last_update")
        if not last_update:
            return True
        return (datetime.utcnow() - last_update).total_seconds() > threshold_seconds

    def get_cached_quote(self, symbol: str) -> dict[str, Any] | None:
        data = self._market_data.get(symbol)
        if not data:
            return None
        if not data.get("last_update"):
            return None
        return data

    def get_snapshot(self, symbol: str) -> dict[str, Any]:
        data = self._market_data.get(symbol)
        if not data:
            return {}
        snapshot = dict(data)
        windows = self._rolling_windows.get(symbol)
        if windows:
            snapshot.update(
                {
                    "vol_1m": windows.get("vol_1m", RollingWindow(60)).sum,
                    "vol_5m": windows.get("vol_5m", RollingWindow(300)).sum,
                }
            )
        return snapshot

    def set_mark(self, symbol: str, price: Decimal) -> None:
        self._mark_cache.set_mark(symbol, price)

    def get_mark(self, symbol: str) -> Decimal | None:
        return self._mark_cache.get_mark(symbol)

    def rolling_windows(self, symbol: str) -> dict[str, RollingWindow]:
        return self._rolling_windows.setdefault(symbol, {})
